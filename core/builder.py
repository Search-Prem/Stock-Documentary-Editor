"""High-level operations used by the web UI, the CLI and the tests. Each one loads the timeline from disk,
does its work, and saves after every scene - so a crash or a closed browser never loses finished work."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .align import AlignError, align_narration
from .cache import Cache, DownloadError, link_into_project, make_thumb
from .config import redact
from .planner import expand_queries, plan_queries
from .project import Project
from .providers.base import ProviderError, RateLimitError
from .render import RenderError, mux_narration, render_timeline
from .selector import Selector, make_providers
from .srt import build_srt
from .timeline import build_timeline, layout_clips, make_clip, validate_timeline
from .util import IMAGE_EXTS, VIDEO_EXTS, MediaError, check_decodes, log, probe, sha1

Progress = Callable[[float, str], None]
_noop: Progress = lambda f, m: None


class BuildError(Exception):
    """Something the user can fix (message is safe to display)."""


# ------------------------------------------------------------------------------------- align
def align_project(project: Project, progress: Progress = _noop) -> dict:
    sentences = project.sentences()
    if not sentences:
        raise BuildError("Add a script first.")
    st = project.settings
    narration = project.narration_path()
    progress(0.05, "Analysing narration timing")
    try:
        res = align_narration(sentences, narration, backend=st["align_backend"], model=st["whisper_model"],
                              device=st["whisper_device"], language=st["language"], wpm=float(st["estimate_wpm"]),
                              progress=lambda f: progress(0.05 + 0.9 * f, "Transcribing narration"))
    except (AlignError, MediaError) as e:
        raise BuildError(redact(e))
    tl = build_timeline(res, st, old=project.load_timeline())
    project.save_timeline(tl)
    project.srt_path.write_text(build_srt(tl, int(st["srt_max_cue_chars"])), encoding="utf-8")
    issues = validate_timeline(tl)
    progress(1.0, "Timeline ready")
    return {"scenes": len(tl["scenes"]), "backend": res.backend, "match_ratio": res.match_ratio,
            "warnings": res.warnings, "duration": tl["audio_duration"], "issues": issues}


# ------------------------------------------------------------------------------------- visuals
def _selector(project: Project, progress_msg=None) -> tuple[Selector, Cache]:
    cache = Cache()
    provs = make_providers(project.settings, cache)
    if not provs:
        raise BuildError("No stock provider is configured. Add PEXELS_API_KEY to your .env file and restart.")
    return Selector(project, cache, provs, progress_msg), cache


def _has_pinned(scene: dict) -> bool:
    return any(c.get("pinned") for c in scene["clips"])


def build_visuals(project: Project, progress: Progress = _noop, force: bool = False,
                  only: list[int] | None = None) -> dict:
    """Find footage for every scene that still needs it. User-chosen (pinned) scenes are never overwritten
    unless force=True. Stops cleanly (keeping everything finished) if the API rate limit is hit."""
    tl = project.load_timeline()
    if not tl["scenes"]:
        raise BuildError("Align the narration first.")
    sel, cache = _selector(project)
    used = Selector.used_counter(tl)
    todo = [s for s in tl["scenes"]
            if (only is None or s["id"] in only)
            and (force or s["status"] in ("pending", "needs_attention"))
            and (force or not _has_pinned(s))]
    done = failed = 0
    paused = ""
    provider_failures = 0
    for i, sc in enumerate(todo):
        progress(i / max(len(todo), 1), f"Finding footage: scene {sc['id']} of {len(tl['scenes'])}")
        for c in sc["clips"]:  # rebuilding: release this scene's earlier picks from the 'used' tally
            k = f"{c['source']}:{c['media_id']}"
            if used.get(k):
                used[k] -= 1
        try:
            sel.build_scene(sc, used)
        except RateLimitError as e:
            paused = redact(e)
            project.save_timeline(tl)
            break
        except Exception as e:
            sc["status"], sc["error"], sc["flags"] = "needs_attention", redact(f"{type(e).__name__}: {e}")[:300], ["error"]
        if sc["status"] == "ok":
            done += 1
            provider_failures = 0
        else:
            failed += 1
            if "network" in sc.get("error", "").lower():
                provider_failures += 1
        project.save_timeline(tl)
        if provider_failures >= 3:
            paused = "Stock provider unreachable (3 scenes in a row). Check your connection and press Build again."
            break
    progress(1.0, "Finished")
    return {"built": done, "needs_attention": failed, "paused": paused,
            "pending": sum(1 for s in tl["scenes"] if s["status"] == "pending"),
            "api_calls": {p.name: p.api_calls for p in sel.providers}, "cache": dict(cache.stats)}


def _find(tl: dict, scene_id: int) -> dict:
    sc = next((s for s in tl["scenes"] if s["id"] == scene_id), None)
    if sc is None:
        raise BuildError(f"Scene {scene_id} does not exist")
    return sc


def search_again(project: Project, scene_id: int, progress: Progress = _noop) -> dict:
    """Different footage for one scene: remember what was rejected, advance to the next queries/page."""
    tl = project.load_timeline()
    sc = _find(tl, scene_id)
    sel, _ = _selector(project)
    used = Selector.used_counter(tl, exclude_scene=scene_id)
    for c in sc["clips"]:
        if c["source"] != "local":
            k = f"{c['source']}:{c['media_id']}"
            if k not in sc["rejected"]:
                sc["rejected"].append(k)
    sc["query_round"] += 1
    progress(0.2, f"Searching again for scene {scene_id}")
    try:
        sel.build_scene(sc, used)
    except RateLimitError as e:
        project.save_timeline(tl)
        raise BuildError(redact(e))
    project.save_timeline(tl)
    return {"scene": scene_id, "status": sc["status"], "clips": len(sc["clips"])}


def set_queries(project: Project, scene_id: int, queries: list[str]) -> None:
    tl = project.load_timeline()
    sc = _find(tl, scene_id)
    qs = [" ".join(q.split()) for q in queries if q and q.strip()][:12]
    if not qs:
        sc["queries"] = expand_queries(plan_queries(sc["text"], project.settings["topic_hint"]), project.settings["topic_hint"])
        sc["custom_queries"] = False
    else:
        sc["queries"], sc["custom_queries"] = qs, True
    sc["query_round"], sc["rejected"] = 0, []
    project.save_timeline(tl)


def list_candidates(project: Project, scene_id: int, query: str | None = None) -> list[dict]:
    tl = project.load_timeline()
    sc = _find(tl, scene_id)
    sel, _ = _selector(project)
    try:
        return sel.candidates(sc, Selector.used_counter(tl, exclude_scene=scene_id), query)
    except RateLimitError as e:
        raise BuildError(redact(e))


# ------------------------------------------------------------------------------------- manual replace
def _refresh_status(sc: dict) -> None:
    flags = [f for f in sc.get("flags", []) if f not in ("no_footage", "error", "render_failed")]
    if not sc["clips"]:
        sc["status"], flags = "needs_attention", flags + ["no_footage"]
    elif any(c.get("loop") and not c.get("pinned") for c in sc["clips"]):
        sc["status"] = "needs_attention"
    else:
        sc["status"], sc["error"] = "ok", ""
        flags = [f for f in flags if f != "looped"]
    if any(c.get("loop") for c in sc["clips"]) and "looped" not in flags:
        flags.append("looped")
    sc["flags"] = flags


def _apply_replacement(project: Project, sc: dict, idx: int | None, clip_base: dict, path: Path, info: dict) -> None:
    whole = idx is None or not sc["clips"] or not (0 <= idx < len(sc["clips"]))
    slot = sc["duration"] if whole else sc["clips"][idx]["duration"]
    loop = info["kind"] == "video" and info["duration"] < slot + 0.05
    clip = make_clip({**clip_base, "local_path": str(path), "kind": info["kind"], "width": info["width"],
                      "height": info["height"], "src_duration": info["duration"] if info["kind"] == "video" else 0.0,
                      "loop": loop, "pinned": True}, 0.0, slot)
    if not clip.get("thumb"):
        sdir = project.scene_dir(sc["id"])
        thumb = sdir / f"thumb_{sha1(str(path))[:8]}.jpg"
        if make_thumb(path, thumb, is_image=info["kind"] == "image"):
            clip["thumb"] = thumb.relative_to(project.dir).as_posix()
    if whole:
        sc["clips"] = [clip]
    else:
        sc["clips"][idx] = clip
    layout_clips(sc)
    _refresh_status(sc)


def replace_with_local(project: Project, scene_id: int, clip_index: int | None, path: str) -> dict:
    p = Path(path).expanduser()
    if p.suffix.lower() not in VIDEO_EXTS | IMAGE_EXTS:
        raise BuildError("Use an MP4, MOV, WebM, JPG or PNG file.")
    try:
        info = probe(p)
        if info["kind"] not in ("video", "image"):
            raise MediaError("not a video or image")
        if not check_decodes(p):
            raise MediaError("the file does not decode (corrupt?)")
    except MediaError as e:
        raise BuildError(f"Cannot use that file: {redact(e)}")
    tl = project.load_timeline()
    sc = _find(tl, scene_id)
    _apply_replacement(project, sc, clip_index, {"source": "local", "media_id": sha1(str(p.resolve()))[:10],
                                                 "page_url": "", "query": ""}, p.resolve(), info)
    project.save_timeline(tl)
    return {"status": sc["status"], "flags": sc["flags"]}


def replace_with_candidate(project: Project, scene_id: int, clip_index: int | None, cand: dict) -> dict:
    from .providers.base import Candidate
    c = Candidate.from_dict(cand)
    sel, cache = _selector(project)
    prov = next((p for p in sel.providers if p.name == c.source), None)
    try:
        path, info = cache.get_media(c, prov)
    except (DownloadError, MediaError) as e:
        raise BuildError(f"Could not use that clip: {redact(e)}")
    tl = project.load_timeline()
    sc = _find(tl, scene_id)
    _apply_replacement(project, sc, clip_index, {"source": c.source, "media_id": c.media_id, "page_url": c.page_url,
                                                 "download_url": c.download_url, "thumb": c.thumb, "query": c.query,
                                                 "fallback": c.kind == "image"}, path, info)
    link_into_project(path, project.scene_dir(scene_id))
    project.save_timeline(tl)
    return {"status": sc["status"], "flags": sc["flags"]}


# ------------------------------------------------------------------------------------- output
def credits_text(tl: dict) -> str:
    seen, lines = set(), []
    for sc in tl["scenes"]:
        for c in sc["clips"]:
            if c["source"] not in ("local", "") and (c["source"], c["media_id"]) not in seen:
                seen.add((c["source"], c["media_id"]))
                lines.append(f"Scene {sc['id']}: {c['source']} {c.get('page_url') or c['media_id']} {c.get('credit', '')}".strip())
    return "\n".join(lines) + ("\n" if lines else "")


def export_project(project: Project, with_narration: bool, progress: Progress = _noop) -> dict:
    tl = project.load_timeline()
    if not tl["scenes"]:
        raise BuildError("Align the narration first.")
    if with_narration and not project.narration_path():
        raise BuildError("No narration file to add. Import narration first, or export video-only.")
    st = project.settings
    try:
        rep = render_timeline(project, tl, mode="final", progress=lambda f, m: progress(f * 0.9, m))
    except (RenderError, MediaError) as e:
        raise BuildError(f"Render failed: {redact(e)}")
    project.save_timeline(tl)  # render may have flagged failed scenes
    out = {"video_only": rep["video"], "report": rep}
    if with_narration:
        progress(0.92, "Adding narration")
        dest = project.final_dir / "documentary_with_narration.mp4"
        music = st["music_path"] if st["music_enabled"] and st["music_path"] and os.path.isfile(st["music_path"]) else None
        try:
            mux_narration(Path(rep["video"]), project.narration_path(), dest, tl["audio_duration"], music,
                          float(st["music_volume"]))
        except MediaError as e:
            raise BuildError(f"Adding narration failed: {redact(e)}")
        out["with_narration"] = str(dest)
        out["audio_duration"] = probe(dest)["duration"]
    project.srt_path.write_text(build_srt(tl, int(st["srt_max_cue_chars"])), encoding="utf-8")
    (project.final_dir / "subtitles.srt").write_text(project.srt_path.read_text(encoding="utf-8"), encoding="utf-8")
    (project.final_dir / "credits.txt").write_text(credits_text(tl), encoding="utf-8")
    out["srt"] = str(project.final_dir / "subtitles.srt")
    progress(1.0, "Export complete")
    return out


def preview_project(project: Project, progress: Progress = _noop) -> dict:
    tl = project.load_timeline()
    if not tl["scenes"]:
        raise BuildError("Align the narration first.")
    try:
        rep = render_timeline(project, tl, mode="preview", progress=progress)
    except (RenderError, MediaError) as e:
        raise BuildError(f"Preview failed: {redact(e)}")
    project.save_timeline(tl)
    out = project.preview_dir / "preview.mp4"
    narr = project.narration_path()
    if narr:
        mux_narration(Path(rep["video"]), narr, out, tl["audio_duration"])
    else:
        import shutil
        shutil.copyfile(rep["video"], out)
    rep["preview"] = str(out)
    return rep
