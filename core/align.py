"""Sentence-level timestamps from the narration audio.

Tiers (best first):
  1. whisper   - faster-whisper word timestamps, matched to the SCRIPT words (script text stays the
                 source of truth, ASR errors are tolerated), then snapped to acoustic pauses.
  2. silence   - FFmpeg silencedetect pauses assigned to sentence boundaries by dynamic programming.
  3. estimate  - word-count proportional. Used only when everything else fails; always flagged.

Whatever the tier, the result is a gap-free, overlap-free partition of [0, audio_duration].
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .util import AUDIO_EXTS, MediaError, find_binary, log, probe

MIN_SCENE = 0.10          # seconds; keeps every scene at least ~3 frames
MIN_MATCH_RATIO = 0.60    # below this the Whisper result is not trusted
SOURCE_RANK = {"whisper": 3, "silence": 2, "estimate": 1}


class AlignError(Exception):
    pass


@dataclass
class AlignResult:
    sentences: list[dict]
    backend: str
    duration: float
    match_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- text helpers
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("’", "'").replace("'", ""))


def audio_duration(path: str | Path) -> float:
    info = probe(path)
    if info["duration"] <= 0:
        raise MediaError("Narration file has zero duration")
    return info["duration"]


# ----------------------------------------------------------------------------- finalize
def finalize(sentences: list[str], starts: list[float], sources: list[str], duration: float) -> list[dict]:
    """starts[i] = start time of sentence i (starts[0] forced to 0). Produces contiguous scenes."""
    n = len(sentences)
    starts = list(starts)
    starts[0] = 0.0
    for i in range(1, n):
        starts[i] = max(starts[i], starts[i - 1] + MIN_SCENE)
    for i in range(n - 1, 0, -1):
        starts[i] = min(starts[i], duration - MIN_SCENE * (n - i))
    for i in range(1, n):  # final forward pass in case backward pass pushed things below zero
        starts[i] = max(starts[i], starts[i - 1] + 1e-3)
    starts = [round(s, 3) for s in starts]
    out = []
    for i, text in enumerate(sentences):
        end = starts[i + 1] if i + 1 < n else round(duration, 3)
        # A scene is bounded by boundary i (its start) and boundary i+1 (its end); its timing is only
        # as trustworthy as the weaker of the two. Boundary 0 is always t=0 (exact by definition).
        bounds = ([sources[i]] if i > 0 else []) + ([sources[i + 1]] if i + 1 < n else [])
        weakest = min(bounds, key=lambda s: SOURCE_RANK[s]) if bounds else sources[0]
        out.append({
            "id": i + 1,
            "text": text,
            "start": starts[i],
            "end": end,
            "duration": round(end - starts[i], 3),
            "timing_source": weakest,
        })
    return out


# ----------------------------------------------------------------------------- tier 1: whisper
def transcribe_words(audio_path, model_size="small.en", device="cpu", language="en", prompt=None,
                     progress: Callable[[float], None] | None = None) -> list[tuple[str, float, float]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise AlignError("faster-whisper is not installed (pip install faster-whisper)")
    try:
        model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")
    except Exception as e:  # download failure, bad model name, missing CUDA libs ...
        raise AlignError(f"Could not load Whisper model '{model_size}': {type(e).__name__}: {str(e)[:200]}")
    try:
        segments, info = model.transcribe(
            str(audio_path), language=language or None, word_timestamps=True, vad_filter=False,
            beam_size=5, condition_on_previous_text=False, initial_prompt=prompt)
        words: list[tuple[str, float, float]] = []
        total = max(info.duration, 1e-6)
        for seg in segments:
            for w in (seg.words or []):
                words.append((w.word, float(w.start), float(w.end)))
            if progress:
                progress(min(seg.end / total, 1.0))
    except Exception as e:
        raise AlignError(f"Transcription failed: {type(e).__name__}: {str(e)[:200]}")
    if not words:
        raise AlignError("Whisper returned no words (silent or unsupported audio?)")
    return words


def align_from_words(sentences: list[str], asr_words: list[tuple[str, float, float]], duration: float,
                     ) -> AlignResult:
    """Pure function: map ASR words onto script sentences. Levenshtein alignment on normalized tokens,
    so mis-heard words still anchor their position and dropped/extra words don't shift sentences."""
    from rapidfuzz.distance import Levenshtein

    script_toks: list[str] = []
    script_sent: list[int] = []
    for i, s in enumerate(sentences):
        for t in tokens(s):
            script_toks.append(t)
            script_sent.append(i)
    asr_toks: list[str] = []
    asr_time: list[tuple[float, float]] = []
    for w, st, en in asr_words:
        for t in tokens(w):
            asr_toks.append(t)
            asr_time.append((st, en))
    if not script_toks or not asr_toks:
        raise AlignError("Nothing to align (empty script or transcript)")

    mapping: dict[int, int] = {}
    equal = 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(script_toks, asr_toks):
        if tag == "equal":
            equal += i2 - i1
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            span_a, span_b = i2 - i1, j2 - j1
            for k in range(span_a):
                mapping[i1 + k] = j1 + min(span_b - 1, int(k * span_b / span_a))
    ratio = equal / len(script_toks)

    n = len(sentences)
    # first mapped token per sentence -> its ASR start time
    best: dict[int, tuple[int, float]] = {}
    for tok_idx in sorted(mapping):
        si = script_sent[tok_idx]
        if si not in best:
            best[si] = (tok_idx, asr_time[mapping[tok_idx]][0])

    starts: list[float] = [0.0] * n
    sources: list[str] = ["whisper"] * n
    known = [i for i in range(n) if i in best]
    for i in range(n):
        if i in best:
            starts[i] = best[i][1]
        else:  # interpolate between neighbours that are known; flag as estimate
            prev = max((k for k in known if k < i), default=None)
            nxt = min((k for k in known if k > i), default=None)
            lo = best[prev][1] if prev is not None else 0.0
            hi = best[nxt][1] if nxt is not None else duration
            lo_i = prev if prev is not None else -1
            hi_i = nxt if nxt is not None else n
            starts[i] = lo + (hi - lo) * (i - lo_i) / (hi_i - lo_i)
            sources[i] = "estimate"
    # keep monotonic (ASR jitter) before finalize
    for i in range(1, n):
        starts[i] = max(starts[i], starts[i - 1])
    res = finalize(sentences, starts, sources, duration)
    return AlignResult(res, "whisper", duration, match_ratio=round(ratio, 3))


# ----------------------------------------------------------------------------- silence detection
def detect_pauses(audio_path, noise_db=-35, min_dur=0.2) -> list[tuple[float, float]]:
    """Return silent intervals (start, end) from FFmpeg silencedetect, including leading/trailing."""
    cmd = [find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-i", str(audio_path),
           "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        raise AlignError("Silence detection timed out")
    pauses, cur = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            cur = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and cur is not None:
            pauses.append((cur, float(m.group(1))))
            cur = None
    return pauses


def _weights(sentences: list[str]) -> list[float]:
    ws = []
    for s in sentences:
        words = len(tokens(s))
        commas = s.count(",") + s.count(";") + s.count(":")
        ws.append(max(1.0, words + 0.7 * commas))
    return ws


def _expected_boundaries(sentences, speech_start, speech_end) -> list[float]:
    w = _weights(sentences)
    total = sum(w)
    out, acc = [], 0.0
    for i in range(len(sentences) - 1):
        acc += w[i]
        out.append(speech_start + (speech_end - speech_start) * acc / total)
    return out


def keep_sentence_scale_pauses(pauses: list[tuple[float, float]], n_boundaries: int) -> list[tuple[float, float]]:
    """Comma/breath pauses are much shorter than sentence gaps. If there are at least as many long pauses as
    boundaries, discard the short ones so a comma can never be mistaken for a sentence end."""
    if n_boundaries <= 0 or len(pauses) <= n_boundaries:
        return pauses
    lengths = sorted((b - a for a, b in pauses), reverse=True)
    ref = lengths[n_boundaries - 1]          # length of the weakest pause we still need
    thresh = 0.5 * ref
    kept = [p for p in pauses if (p[1] - p[0]) >= thresh]
    return kept if len(kept) >= n_boundaries else pauses


def assign_pauses(expected: list[float], pauses: list[tuple[float, float]],
                  skip_cost: float = 3.0, len_bonus: float = 3.0) -> list[int | None]:
    """Choose, for each expected boundary, a distinct pause in increasing order (or None) minimizing
    distance to the estimate, with a bonus for longer pauses (sentence gaps are longer than commas)."""
    n, P = len(expected), len(pauses)
    INF = float("inf")
    center = [(a + b) / 2 for a, b in pauses]
    length = [b - a for a, b in pauses]
    f = [0.0] + [INF] * P               # state s: last used pause index is s-1 (0 = none)
    back: list[list[tuple[int, bool] | None]] = []
    for j in range(n):
        g = [INF] * (P + 1)
        b: list[tuple[int, bool] | None] = [None] * (P + 1)
        for s in range(P + 1):
            if f[s] < INF:
                g[s] = f[s] + skip_cost
                b[s] = (s, False)
        best, best_s = INF, -1
        for k in range(P):
            if f[k] < best:
                best, best_s = f[k], k
            if best < INF:
                cand = best + abs(center[k] - expected[j]) - len_bonus * min(length[k], 1.0)
                if cand < g[k + 1]:
                    g[k + 1] = cand
                    b[k + 1] = (best_s, True)
        back.append(b)
        f = g
    s = min(range(P + 1), key=lambda i: f[i])
    assign: list[int | None] = [None] * n
    for j in range(n - 1, -1, -1):
        prev, used = back[j][s]  # type: ignore[misc]
        if used:
            assign[j] = s - 1
        s = prev
    return assign


def align_from_silence(sentences: list[str], audio_path, duration: float,
                       progress: Callable[[float], None] | None = None) -> AlignResult:
    n = len(sentences)
    if n == 1:
        return AlignResult(finalize(sentences, [0.0], ["silence"], duration), "silence", duration)
    best_pauses, best_count = [], -1
    for noise in (-38, -32, -45, -28):
        pauses = detect_pauses(audio_path, noise_db=noise, min_dur=0.2)
        inner = [p for p in pauses if p[0] > 0.05 and p[1] < duration - 0.05]
        if len(inner) > best_count:
            best_pauses, best_count = inner, len(inner)
        if len(inner) >= n - 1:
            break
        if progress:
            progress(0.5)
    lead = detect_pauses(audio_path, noise_db=-38, min_dur=0.1)
    speech_start = lead[0][1] if lead and lead[0][0] <= 0.05 else 0.0
    speech_end = lead[-1][0] if lead and lead[-1][1] >= duration - 0.05 else duration
    if speech_end <= speech_start:
        speech_start, speech_end = 0.0, duration
    expected = _expected_boundaries(sentences, speech_start, speech_end)
    best_pauses = keep_sentence_scale_pauses(best_pauses, n - 1)
    assign = assign_pauses(expected, best_pauses)
    starts, sources = [0.0], ["silence"]
    for j, a in enumerate(assign):
        if a is None:
            starts.append(expected[j])
            sources.append("estimate")
        else:
            starts.append(best_pauses[a][1])  # end of pause = onset of next sentence's speech
            sources.append("silence")
    return AlignResult(finalize(sentences, starts, sources, duration), "silence", duration)


def refine_with_pauses(result: AlignResult, pauses: list[tuple[float, float]], window: float = 0.4) -> None:
    """Snap whisper boundaries to the nearest acoustic speech onset (end of a pause) within `window`."""
    if not pauses:
        return
    scenes = result.sentences
    starts = [s["start"] for s in scenes]
    changed = False
    for i in range(1, len(scenes)):
        if scenes[i]["timing_source"] == "estimate":
            continue
        t = starts[i]
        cand = [(abs(e - t), e) for (s, e) in pauses if s - window <= t <= e + window]
        if cand:
            d, e = min(cand)
            if d <= window and abs(e - t) > 1e-3:
                starts[i] = e
                changed = True
    if changed:
        sources = [s["timing_source"] for s in scenes]
        new = finalize([s["text"] for s in scenes], starts, sources, result.duration)
        result.sentences[:] = new


# ----------------------------------------------------------------------------- tier 3 + orchestration
def align_from_estimate(sentences: list[str], duration: float | None, wpm: float = 150.0) -> AlignResult:
    w = _weights(sentences)
    if duration is None:
        duration = round(sum(len(tokens(s)) for s in sentences) / wpm * 60 + 0.4 * len(sentences), 3)
    total, acc, starts = sum(w), 0.0, [0.0]
    for x in w[:-1]:
        acc += x
        starts.append(duration * acc / total)
    return AlignResult(finalize(sentences, starts, ["estimate"] * len(sentences), duration), "estimate", duration)


def align_narration(sentences: list[str], audio_path: str | Path | None, backend: str = "auto",
                    model: str = "small.en", device: str = "cpu", language: str = "en", wpm: float = 150.0,
                    progress: Callable[[float], None] | None = None) -> AlignResult:
    if not sentences:
        raise AlignError("The script has no sentences")
    if not audio_path:
        res = align_from_estimate(sentences, None, wpm)
        res.warnings.append("No narration file: timing is ESTIMATED from word count. Import your narration and re-align.")
        return res
    if Path(audio_path).suffix.lower() not in AUDIO_EXTS:
        raise AlignError(f"Unsupported narration format: {Path(audio_path).suffix}")
    duration = audio_duration(audio_path)
    warnings: list[str] = []
    if backend in ("auto", "whisper"):
        try:
            prompt = " ".join(sentences[:3])[:400]
            words = transcribe_words(audio_path, model, device, language, prompt, progress)
            res = align_from_words(sentences, words, duration)
            if res.match_ratio is not None and res.match_ratio >= MIN_MATCH_RATIO:
                try:
                    refine_with_pauses(res, detect_pauses(audio_path, -38, 0.15))
                except AlignError:
                    pass
                res.warnings = warnings
                return res
            warnings.append(f"Whisper matched only {res.match_ratio:.0%} of the script words - "
                            "does the narration match the script?")
            if backend == "whisper":
                res.warnings = warnings
                return res
        except AlignError as e:
            warnings.append(str(e))
            if backend == "whisper":
                raise
        warnings.append("Falling back to silence-based alignment.")
    if backend in ("auto", "whisper", "silence"):
        try:
            res = align_from_silence(sentences, audio_path, duration, progress)
            res.warnings = warnings
            return res
        except (AlignError, MediaError) as e:
            warnings.append(f"Silence-based alignment failed: {e}")
            if backend == "silence":
                raise AlignError(str(e))
    res = align_from_estimate(sentences, duration, wpm)
    res.warnings = warnings + ["Timing is ESTIMATED from word count (audio analysis failed)."]
    return res
