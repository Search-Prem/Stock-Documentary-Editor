"""Timeline model: narration segment -> semantic scene -> one or more clips. Gap/overlap validation."""
from __future__ import annotations

from .align import AlignResult
from .planner import expand_queries, plan_queries
from .util import frames, sha1

EPS = 1e-3


def new_scene(a: dict, topic: str) -> dict:
    base = plan_queries(a["text"], topic)
    return {
        "id": a["id"], "text": a["text"], "start": a["start"], "end": a["end"], "duration": a["duration"],
        "timing_source": a["timing_source"],
        "queries": expand_queries(base, topic), "custom_queries": False, "query_round": 0,
        "rejected": [], "status": "pending", "flags": [], "error": "", "clips": [],
    }


def build_timeline(result: AlignResult, settings: dict, old: dict | None = None) -> dict:
    """Create the master timeline from alignment output. Scenes whose text AND duration are unchanged keep
    their previously chosen footage, so re-aligning doesn't throw away work."""
    old_by_text: dict[str, list[dict]] = {}
    for s in (old or {}).get("scenes", []):
        old_by_text.setdefault(sha1(s["text"]), []).append(s)
    scenes = []
    for a in result.sentences:
        sc = new_scene(a, settings["topic_hint"])
        cands = old_by_text.get(sha1(a["text"]), [])
        prev = cands.pop(0) if cands else None
        if prev and abs(prev["duration"] - a["duration"]) <= 0.05 and prev.get("clips"):
            for k in ("queries", "custom_queries", "query_round", "rejected", "status", "flags", "error", "clips"):
                sc[k] = prev[k]
            layout_clips(sc)
        scenes.append(sc)
    return {"version": 1, "audio_duration": round(result.duration, 3), "fps": settings["fps"],
            "alignment": {"backend": result.backend, "match_ratio": result.match_ratio, "warnings": result.warnings},
            "scenes": scenes}


def layout_clips(scene: dict) -> None:
    """Recompute absolute clip start/end from durations; the last clip absorbs rounding so clips tile the scene."""
    t = scene["start"]
    clips = scene["clips"]
    for i, c in enumerate(clips):
        c["start"] = round(t, 3)
        c["end"] = round(t + c["duration"], 3)
        t = c["end"]
    if clips:
        clips[-1]["end"] = scene["end"]
        clips[-1]["duration"] = round(scene["end"] - clips[-1]["start"], 3)


def validate_timeline(tl: dict) -> list[str]:
    """Return a list of problems (empty = perfectly gap-free / overlap-free / exact)."""
    issues: list[str] = []
    scenes = tl.get("scenes", [])
    if not scenes:
        return ["timeline has no scenes"]
    if abs(scenes[0]["start"]) > EPS:
        issues.append(f"first scene starts at {scenes[0]['start']}, expected 0")
    dur = tl.get("audio_duration")
    if dur is not None and abs(scenes[-1]["end"] - dur) > EPS:
        issues.append(f"last scene ends at {scenes[-1]['end']}, narration ends at {dur}")
    for i, s in enumerate(scenes):
        if s["end"] - s["start"] <= 0:
            issues.append(f"scene {s['id']} has non-positive duration")
        if abs((s["end"] - s["start"]) - s["duration"]) > EPS:
            issues.append(f"scene {s['id']} duration field disagrees with start/end")
        if i and abs(s["start"] - scenes[i - 1]["end"]) > EPS:
            kind = "gap" if s["start"] > scenes[i - 1]["end"] else "overlap"
            issues.append(f"{kind} between scene {scenes[i - 1]['id']} and {s['id']}")
        clips = s.get("clips", [])
        if clips:
            if abs(clips[0]["start"] - s["start"]) > EPS:
                issues.append(f"scene {s['id']}: first clip does not start with the scene")
            if abs(clips[-1]["end"] - s["end"]) > EPS:
                issues.append(f"scene {s['id']}: clips end at {clips[-1]['end']}, scene ends at {s['end']}")
            for a, b in zip(clips, clips[1:]):
                if abs(a["end"] - b["start"]) > EPS:
                    issues.append(f"scene {s['id']}: gap/overlap between clips")
    return issues


def frame_plan(tl: dict, fps: int) -> list[dict]:
    """Snap every scene and clip boundary to the output frame grid. Total frames = round(duration*fps), and
    every scene/clip boundary is an integer frame, so the concatenated video cannot drift."""
    total = frames(tl["audio_duration"], fps)
    out = []
    for i, s in enumerate(tl["scenes"]):
        f0 = frames(s["start"], fps)
        f1 = total if i == len(tl["scenes"]) - 1 else frames(s["end"], fps)
        f0 = out[-1]["f1"] if out else 0
        f1 = max(f1, f0 + 1)
        bounds = [f0]
        for c in s["clips"][:-1]:
            bounds.append(min(max(frames(c["end"], fps), bounds[-1] + 1), f1 - 1))
        bounds.append(f1)
        out.append({"scene": s, "f0": f0, "f1": f1, "clip_bounds": bounds})
    return out


def make_clip(cand_or_local: dict, src_in: float, duration: float) -> dict:
    c = {
        "source": "", "media_id": "", "kind": "video", "page_url": "", "download_url": "", "local_path": "",
        "thumb": "", "credit": "", "width": 0, "height": 0, "src_duration": 0.0, "query": "",
        "score": 0.0, "loop": False, "pinned": False, "fallback": False,
    }
    c.update(cand_or_local)
    c["src_in"] = round(src_in, 3)
    c["duration"] = round(duration, 3)
    c["start"] = 0.0
    c["end"] = round(duration, 3)
    return c
