"""Choose footage for a scene: gather candidates -> score -> fill the sentence's duration with clips."""
from __future__ import annotations

import math
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .cache import Cache, DownloadError, link_into_project, make_thumb
from .planner import _words, content_terms, stem
from .providers.base import Candidate, Provider, ProviderError, RateLimitError
from .providers.pexels import PexelsProvider
from .providers.pixabay import PixabayProvider
from .providers.unsplash import UnsplashProvider
from .timeline import layout_clips, make_clip
from .util import MediaError, log

HEAD = 0.2           # skip the first fraction of a second of stock clips (encoder warm-up / fades)
FILL_EPS = 0.02
BAD_SLUG = re.compile(r"\b(logo|watermark|text|title|intro|template|advert\w*|promo\w*|sale|banner|"
                      r"subscribe|countdown|lower third|green screen|greenscreen)\b", re.I)
LOW_RELEVANCE = 2.2


def make_providers(settings: dict, cache: Cache) -> list[Provider]:
    table = {"pexels": PexelsProvider, "pixabay": PixabayProvider, "unsplash": UnsplashProvider}
    out = []
    for name in settings["providers"]:
        cls = table.get(name)
        if cls:
            p = cls(cache)
            if p.configured():
                out.append(p)
    return out


# ---------------------------------------------------------------------------------- pure planning
@dataclass
class Seg:
    idx: int            # index into the ordered candidate list
    src_in: float
    length: float
    reuse: bool = False
    loop: bool = False


def _usable(c: dict, reserve: float) -> float:
    if c["kind"] == "image":
        return math.inf
    return c["duration"] - HEAD - reserve


def plan_fill(D: float, cands: list[dict], min_clip: float, max_clip: float, target: float,
              reserve: float) -> tuple[list[Seg], float]:
    """Cover D seconds with segments from `cands` (best first; dicts with kind/duration).
    Relevance order is preserved; pacing aims at ~`target` seconds per clip; never leaves a sliver shorter
    than `min_clip`; reuse of a source and looping happen only after fresh sources run out.
    Returns (segments, uncovered_seconds)."""
    segs: list[Seg] = []
    remaining = D
    used: set[int] = set()
    guard = 0
    while remaining > FILL_EPS and guard < 100:
        guard += 1
        n = max(1, int(remaining / target + 0.5), math.ceil(remaining / max_clip - 1e-9))
        slot = remaining / n
        pick = None
        for i, c in enumerate(cands):
            if i in used:
                continue
            u = _usable(c, reserve)
            if u < min(min_clip, remaining) - 1e-6:
                continue
            take = min(u, slot)
            left = remaining - take
            if 0 < left < min_clip - 1e-6:
                if u >= remaining and remaining <= max_clip * 1.25:
                    take = remaining                       # absorb the sliver into this clip
                elif take - (min_clip - left) >= min_clip:
                    take = remaining - min_clip            # shorten so the leftover is a proper clip
            pick = (i, take)
            break
        if pick is None:
            break
        i, take = pick
        used.add(i)
        segs.append(Seg(i, HEAD if cands[i]["kind"] == "video" else 0.0, take))
        remaining -= take
    # pass 2: unused tail of sources already in this scene (different footage from the same file)
    if remaining > FILL_EPS:
        for s in list(segs):
            c = cands[s.idx]
            if c["kind"] != "video" or remaining <= FILL_EPS:
                continue
            spare = c["duration"] - reserve - (s.src_in + s.length)
            if spare >= min(min_clip, remaining) - 1e-6:
                take = min(spare, remaining, max_clip)
                segs.append(Seg(s.idx, s.src_in + s.length, take, reuse=True))
                remaining -= take
    # pass 3 (last resort): loop the best available source, clearly flagged
    if remaining > FILL_EPS and cands:
        segs.append(Seg(0, HEAD if cands[0]["kind"] == "video" else 0.0, remaining, loop=True))
        remaining = 0.0
    return segs, remaining


# ---------------------------------------------------------------------------------- selector
class Selector:
    def __init__(self, project, cache: Cache, providers: list[Provider], progress: Callable[[str], None] | None = None):
        self.project = project
        self.cache = cache
        self.providers = providers
        self.s = project.settings
        self.say = progress or (lambda m: None)

    # ------------------------------------------------------------ helpers
    def _reserve(self) -> float:
        return float(self.s["crossfade"]) + 0.05

    @staticmethod
    def used_counter(tl: dict, exclude_scene: int | None = None) -> Counter:
        used: Counter = Counter()
        for sc in tl["scenes"]:
            if sc["id"] == exclude_scene:
                continue
            for c in sc["clips"]:
                if c.get("source") and c["source"] != "local":
                    used[f"{c['source']}:{c['media_id']}"] += 1
        return used

    def _suitable(self, c: Candidate, scene: dict) -> bool:
        key = c.key()
        if key in scene.get("rejected", []) or self.cache.is_bad(key):
            return False
        ow, oh = c.orig_width or c.width, c.orig_height or c.height
        if ow and oh and ow < oh * 1.2:                       # portrait / square
            return False
        if (c.height or oh) and max(c.height, oh) < 720:
            return False
        if c.kind == "video" and c.duration < 1.5:
            return False
        if BAD_SLUG.search(c.slug or ""):
            return False
        return True

    def _score(self, c: Candidate, qidx: int, q_stems: list[set], sent_stems: set, used: Counter) -> tuple[float, float]:
        slug = {stem(w) for w in _words(c.slug)}
        rank_prior = 1.0 - min(c.rank, 14) / 15
        if slug:
            q_frac = 0.0
            for i, qs in enumerate(q_stems):
                if qs:
                    q_frac = max(q_frac, len(qs & slug) / len(qs) * (1 - 0.04 * min(i, 10)))
            s_frac = min(1.0, len(sent_stems & slug) / max(1, min(len(sent_stems), 6)))
            rel = 4.0 * q_frac + 1.0 * s_frac + 1.5 * rank_prior * (1 - 0.04 * min(qidx, 10))
        else:
            rel = 2.0 + 1.5 * rank_prior                      # no text to judge: trust the search ranking
        total = rel
        h = max(c.height, 0)
        if c.width >= 1920 and h >= 1080:
            total += 1.0
        elif c.width >= 1280 and h >= 720:
            total += 0.3
        else:
            total -= 1.5
        ow, oh = c.orig_width or c.width, c.orig_height or c.height
        if ow and oh and abs(ow / oh - 16 / 9) < 0.03:
            total += 0.7
        if c.kind == "video":
            total += 0.3 if c.duration >= 5 else 0.0
            total -= 2.0 if c.duration < float(self.s["min_clip"]) else 0.0
        else:
            total -= 1.5                                       # stills are a fallback
        if c.source == "pexels":
            total += 0.5                                       # Pexels first
        total -= 6.0 * used.get(c.key(), 0)
        return total, rel

    def _query_iter(self, scene: dict):
        base = scene["queries"]
        n = len(base)
        if not n:
            return
        start = scene.get("query_round", 0) * int(self.s["queries_per_round"])
        for i in range(max(n, int(self.s["queries_per_round"]))):
            idx = start + i
            yield idx % n, base[idx % n], 1 + idx // n

    def _enough(self, scored: list[tuple[float, float, Candidate]], D: float) -> bool:
        """Is there enough distinct, usable footage to fill D seconds with some choice left over?"""
        res, cap = self._reserve(), float(self.s["max_clip"])
        vids = [c for _, _, c in scored if c.kind == "video" and c.duration - HEAD - res > 0]
        imgs = [c for _, _, c in scored if c.kind == "image"]
        supply = sum(min(c.duration - HEAD - res, cap) for c in vids)
        if len(vids) >= math.ceil(D / cap) + 1 and supply >= 1.5 * D:
            return True
        return len(imgs) >= math.ceil(D / float(self.s["target_clip"])) + 1

    # ------------------------------------------------------------ gather + rank
    def gather(self, scene: dict, used: Counter, budget: int | None = None) -> list[tuple[float, float, Candidate]]:
        """Collect candidates. Order: first video provider (Pexels) -> other video providers -> stock photos.
        Stops as soon as there is enough footage, and never makes more than `budget` LIVE API calls
        (cache hits are free)."""
        D = scene["duration"]
        budget = int(self.s["max_api_calls_per_scene"]) if budget is None else budget
        q_stems = [{stem(w) for w in _words(q)} for q in scene["queries"]]
        sent_stems = {stem(w) for w in content_terms(scene["text"])}
        pool: dict[str, tuple[int, Candidate]] = {}
        live = 0

        def ranked():
            return self._rank(pool, q_stems, sent_stems, used)

        video_provs = [p for p in self.providers if p.supports_video]
        phases: list[tuple[list[Provider], str]] = []
        if video_provs:
            phases.append((video_provs[:1], "video"))
            if len(video_provs) > 1:
                phases.append((video_provs[1:], "video"))
        phases.append((self.providers, "image"))          # stills only if video could not fill the sentence
        for provs, kind in phases:
            # once video footage alone can fill the sentence, later phases (other providers, stills) are skipped
            if live >= budget or self._enough([t for t in ranked() if t[2].kind == "video"], D):
                break
            for prov in provs:
                for qidx, q, page in self._query_iter(scene):
                    if self._enough(ranked(), D) or live >= budget:
                        break
                    before = prov.api_calls
                    try:
                        res = prov.search_videos(q, page=page) if kind == "video" else prov.search_images(q, page=page)
                    except RateLimitError:
                        raise
                    except ProviderError as e:
                        scene["error"] = str(e)
                        break                              # this provider is failing; try the next one
                    live += prov.api_calls - before
                    for c in res:
                        if self._suitable(c, scene):
                            key = c.key()
                            if key not in pool or (qidx, c.rank) < (pool[key][0], pool[key][1].rank):
                                pool[key] = (qidx, c)
        return ranked()

    def _rank(self, pool, q_stems, sent_stems, used) -> list[tuple[float, float, Candidate]]:
        scored = []
        for qidx, c in pool.values():
            total, rel = self._score(c, qidx, q_stems, sent_stems, used)
            scored.append((total, rel, c))
        scored.sort(key=lambda t: -t[0])
        return scored

    # ------------------------------------------------------------ build one scene
    def build_scene(self, scene: dict, used: Counter) -> None:
        """Fill scene['clips']. Never raises for per-scene problems (marks NEEDS ATTENTION instead);
        RateLimitError propagates so the caller can pause the whole batch."""
        scene["clips"], scene["flags"], scene["error"] = [], [], ""
        D = scene["duration"]
        try:
            ranked = self.gather(scene, used)
        except RateLimitError:
            scene["status"], scene["flags"] = "pending", ["rate_limited"]
            scene["error"] = "Pexels rate limit reached. Cached results are kept; resume later."
            raise
        except Exception as e:  # anything unexpected must not take down the project
            ranked = []
            scene["error"] = f"{type(e).__name__}: {e}"
        best_rel = {c.key(): rel for _, rel, c in ranked}
        ordered = [c for _, _, c in ranked]
        resolved: dict[str, tuple] = {}
        segs: list[Seg] = []
        remaining = D
        for _ in range(6):
            dicts = [{"kind": c.kind, "duration": c.duration} for c in ordered]
            segs, remaining = plan_fill(D, dicts, float(self.s["min_clip"]), float(self.s["max_clip"]),
                                        float(self.s["target_clip"]), self._reserve()) if ordered else ([], D)
            stable = True
            for sg in segs:
                c = ordered[sg.idx]
                if c.key() in resolved:
                    continue
                prov = next((p for p in self.providers if p.name == c.source), None)
                try:
                    path, info = self.cache.get_media(c, prov)
                except (DownloadError, MediaError) as e:
                    scene["error"] = str(e)
                    ordered = [x for x in ordered if x.key() != c.key()]
                    stable = False
                    break
                # trust the real file, not the API metadata: reject portrait and low-resolution downloads
                why = ("portrait" if info["kind"] == "video" and info["height"] > info["width"]
                       else "low resolution" if info["kind"] == "video" and info["height"] < 720
                       else "")
                if why:
                    self.cache.mark_bad(c.key(), why)
                    ordered = [x for x in ordered if x.key() != c.key()]
                    stable = False
                    break
                if info["kind"] == "video" and info["duration"] < c.duration - 0.05:
                    c.duration = info["duration"]          # real file is shorter than the API claimed: re-plan
                    stable = False
                resolved[c.key()] = (path, info)
                if not stable:
                    break
            if stable:
                break
        if not ordered or not segs:
            scene["status"], scene["flags"] = "needs_attention", ["no_footage"]
            scene["error"] = scene["error"] or "No suitable footage found. Edit the queries, Search Again, or Replace with your own clip."
            return
        clips = []
        for sg in segs:
            c = ordered[sg.idx]
            path, info = resolved[c.key()]
            clip = make_clip({
                "source": c.source, "media_id": c.media_id, "kind": c.kind, "page_url": c.page_url,
                "download_url": c.download_url, "local_path": str(path), "thumb": c.thumb,
                "credit": c.extra.get("credit", ""), "width": info["width"], "height": info["height"],
                "src_duration": info["duration"] if c.kind == "video" else 0.0, "query": c.query,
                "score": round(best_rel.get(c.key(), 0.0), 2), "loop": sg.loop, "fallback": c.kind == "image",
            }, sg.src_in, sg.length)
            clips.append(clip)
        # exact tiling: the last clip absorbs float dust so clip durations sum to the scene duration
        clips[-1]["duration"] = round(D - sum(c["duration"] for c in clips[:-1]), 3)
        scene["clips"] = clips
        layout_clips(scene)
        for cl in clips:
            link_into_project(Path(cl["local_path"]), self.project.scene_dir(scene["id"]))
        flags = []
        if any(c["kind"] == "image" for c in clips):
            flags.append("image_fallback")
        if any(s.reuse for s in segs):
            flags.append("reused_source")
        if any(s.loop for s in segs):
            flags.append("looped")
        if min((c["score"] for c in clips), default=0) < LOW_RELEVANCE:
            flags.append("low_relevance")
        if any(used.get(f"{c['source']}:{c['media_id']}", 0) for c in clips):
            flags.append("repeat")
        scene["flags"] = flags
        scene["error"] = ""
        scene["status"] = "needs_attention" if "looped" in flags else "ok"
        if "looped" in flags:
            scene["error"] = "Not enough distinct footage for this sentence; a clip is looped. Replace or Search Again."
        for c in clips:
            used[f"{c['source']}:{c['media_id']}"] += 1

    # ------------------------------------------------------------ candidates for the Replace picker
    def candidates(self, scene: dict, used: Counter, query: str | None = None, limit: int = 30) -> list[dict]:
        if query:
            sc = dict(scene, queries=[query], query_round=0)
            sc["rejected"] = []
            ranked = self.gather(sc, used, budget=2)
        else:
            ranked = self.gather(dict(scene, rejected=[]), used, budget=2)
        out = []
        for total, rel, c in ranked[:limit]:
            d = c.to_dict()
            d["score"] = round(rel, 2)
            out.append(d)
        return out
