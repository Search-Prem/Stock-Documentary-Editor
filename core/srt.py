"""SRT subtitles from the same timeline that drives the video (so they can never disagree)."""
from __future__ import annotations

import textwrap

from .util import srt_ts


def _wrap(text: str, width: int = 42) -> str:
    lines = textwrap.wrap(text, width=width) or [text]
    return "\n".join(lines)


def build_srt(tl: dict, max_cue_chars: int = 0) -> str:
    """One cue per sentence (start/end = the scene). With max_cue_chars > 0, long sentences are split at word
    boundaries and the scene's time is shared in proportion to text length."""
    cues: list[tuple[float, float, str]] = []
    for sc in tl["scenes"]:
        text = sc["text"].strip()
        if max_cue_chars and len(text) > max_cue_chars:
            words, chunks, cur = text.split(), [], ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_cue_chars:
                    chunks.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                chunks.append(cur)
            total = sum(len(c) for c in chunks)
            t = sc["start"]
            for i, c in enumerate(chunks):
                end = sc["end"] if i == len(chunks) - 1 else round(t + sc["duration"] * len(c) / total, 3)
                cues.append((t, end, c))
                t = end
        else:
            cues.append((sc["start"], sc["end"], text))
    out = []
    for i, (a, b, txt) in enumerate(cues, 1):
        out.append(f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{_wrap(txt)}\n")
    return "\n".join(out)
