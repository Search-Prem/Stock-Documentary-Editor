"""FFmpeg assembly. Every scene is rendered as its own segment with an EXACT integer frame count, then the
segments are concatenated without re-encoding. Boundaries are snapped to the frame grid once, up front, so
nothing can drift; the result is verified against the narration length."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from .config import redact
from .timeline import frame_plan, validate_timeline
from .util import (FFmpegError, MediaError, find_binary, frames, log, probe, run_ffmpeg, sha1)

PLACEHOLDER_COLOR = "0x1c1c1e"


class RenderError(Exception):
    pass


# ------------------------------------------------------------------------------- helpers
def _filter_has_option(filter_name: str, option: str) -> bool:
    try:
        r = subprocess.run([find_binary("ffmpeg"), "-hide_banner", "-h", f"filter={filter_name}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        return option in r.stdout
    except Exception:
        return False


def count_frames(path: str | Path) -> int:
    info = probe(path)
    if info["nb_frames"]:
        return info["nb_frames"]
    r = subprocess.run([find_binary("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    try:
        return int(r.stdout.strip())
    except ValueError:
        raise RenderError("Could not count frames in rendered video")


def _spec(scene: dict, bounds: list[int], W: int, H: int, fps: int, crf: int, preset: str, xfade: float) -> dict:
    clips = []
    for c in scene["clips"]:
        p = c["local_path"]
        try:
            st = os.stat(p)
            sig = [st.st_size, int(st.st_mtime)]
        except OSError:
            sig = [-1, -1]
        clips.append([p, sig, c["kind"], c["src_in"], bool(c.get("loop"))])
    return {"clips": clips, "bounds": bounds, "W": W, "H": H, "fps": fps, "crf": crf, "preset": preset, "xfade": xfade}


# ------------------------------------------------------------------------------- one scene
def scene_ffmpeg_args(scene: dict, bounds: list[int], W: int, H: int, fps: int, crf: int, preset: str,
                      crossfade: float, out: Path) -> list[str]:
    clips = scene["clips"]
    F = bounds[-1] - bounds[0]
    args: list[str] = []
    filters: list[str] = []
    if not clips:  # placeholder slate so the scene keeps its exact duration and the project still renders
        args += ["-f", "lavfi", "-i", f"color=c={PLACEHOLDER_COLOR}:s={W}x{H}:r={fps}:d={F / fps + 1:.3f}"]
        filters.append(f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=0xd83b3b@0.8:t=10,format=yuv420p,"
                       f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={F},setpts=PTS-STARTPTS[out]")
    else:
        n = len(clips)
        slots = [bounds[i + 1] - bounds[i] for i in range(n)]
        hf = int(round(crossfade * fps / 2)) if n > 1 and crossfade > 0 else 0
        hf = min(hf, min(slots) // 2) if n > 1 else 0
        lens = [slots[i] + (hf if i > 0 else 0) + (hf if i < n - 1 else 0) for i in range(n)]
        for i, c in enumerate(clips):
            L = lens[i]
            if c["kind"] == "image":
                args += ["-i", c["local_path"]]
                dz = 0.10 / max(L, 1)
                filters.append(
                    f"[{i}:v]scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase,crop={2 * W}:{2 * H},"
                    f"zoompan=z='min(zoom+{dz:.6f},1.10)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={L}:s={W}x{H}:fps={fps},"
                    f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=1,"
                    f"trim=end_frame={L},setpts=PTS-STARTPTS[c{i}]")
            else:
                if c.get("loop"):
                    args += ["-stream_loop", "-1"]
                args += ["-ss", f"{c['src_in']:.3f}", "-t", f"{L / fps + 0.6:.3f}", "-i", c["local_path"]]
                filters.append(
                    f"[{i}:v]fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,"
                    f"format=yuv420p,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=3,"
                    f"trim=end_frame={L},setpts=PTS-STARTPTS[c{i}]")
        if n == 1:
            prev = "c0"
        elif hf == 0:
            filters.append("".join(f"[c{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cat]")
            prev = "cat"
        else:
            prev = "c0"
            for k in range(1, n):
                off = (bounds[k] - bounds[0] - hf) / fps
                filters.append(f"[{prev}][c{k}]xfade=transition=fade:duration={2 * hf / fps:.6f}:offset={off:.6f}[x{k}]")
                prev = f"x{k}"
        filters.append(f"[{prev}]tpad=stop_mode=clone:stop_duration=1,trim=end_frame={F},setpts=PTS-STARTPTS[out]")
    args += ["-filter_complex", ";".join(filters), "-map", "[out]", "-frames:v", str(F),
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
             "-an", "-video_track_timescale", str(fps * 1000), str(out)]
    return args


def render_scene(scene, bounds, W, H, fps, crf, preset, crossfade, out: Path) -> None:
    F = bounds[-1] - bounds[0]
    tmp = out.with_suffix(".part.mp4")
    try:
        run_ffmpeg(scene_ffmpeg_args(scene, bounds, W, H, fps, crf, preset, crossfade, tmp))
        got = count_frames(tmp)
        if got != F:
            raise RenderError(f"scene {scene['id']}: rendered {got} frames, expected {F}")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------------------- whole timeline
def render_timeline(project, tl: dict, mode: str = "final", progress: Callable[[float, str], None] | None = None,
                    keep_segments: bool | None = None) -> dict:
    """mode 'preview' (low-res, segments cached across runs) or 'final' (full quality). Silent video output."""
    say = progress or (lambda f, m: None)
    st = project.settings
    fps = int(st["fps"])
    if mode == "preview":
        W, H, crf, preset = int(st["preview_width"]), int(st["preview_height"]), int(st["preview_crf"]), "ultrafast"
        seg_dir, out = project.preview_dir / "segments", project.preview_dir / "preview_silent.mp4"
    else:
        W, H, crf, preset = int(st["width"]), int(st["height"]), int(st["crf"]), st["preset"]
        seg_dir, out = project.final_dir / "_segments", project.final_dir / "documentary_video.mp4"
    issues = validate_timeline(tl)
    if issues:
        raise RenderError("Timeline is not valid: " + "; ".join(issues[:5]))
    seg_dir.mkdir(parents=True, exist_ok=True)
    plan = frame_plan(tl, fps)
    total_frames = plan[-1]["f1"]
    crossfade = float(st["crossfade"])
    segs, failed, report_scenes = [], [], []
    for i, item in enumerate(plan):
        sc = item["scene"]
        say(i / len(plan), f"Rendering scene {sc['id']} of {len(plan)}")
        spec = _spec(sc, item["clip_bounds"], W, H, fps, crf, preset, crossfade)
        seg = seg_dir / f"scene_{sc['id']:03d}_{sha1(json.dumps(spec))[:10]}.mp4"
        if not (seg.exists() and seg.stat().st_size > 0):
            for old in seg_dir.glob(f"scene_{sc['id']:03d}_*.mp4"):
                old.unlink(missing_ok=True)
            try:
                render_scene(sc, item["clip_bounds"], W, H, fps, crf, preset, crossfade, seg)
                if sc.get("status") == "needs_attention" and "render_failed" in sc.get("flags", []):
                    sc["flags"].remove("render_failed")
            except (MediaError, RenderError) as e:
                log.warning("scene %s failed to render: %s", sc["id"], redact(e))
                failed.append({"scene": sc["id"], "error": redact(e)})
                sc["status"] = "needs_attention"
                sc["error"] = f"Render failed: {redact(e)}"[:300]
                sc.setdefault("flags", [])
                if "render_failed" not in sc["flags"]:
                    sc["flags"].append("render_failed")
                stub = dict(sc, clips=[])
                render_scene(stub, item["clip_bounds"][:1] + [item["clip_bounds"][-1]], W, H, fps, crf, preset, 0, seg)
        segs.append(seg)
        report_scenes.append({"id": sc["id"], "f0": item["f0"], "f1": item["f1"],
                              "start": item["f0"] / fps, "end": item["f1"] / fps})
    say(0.95, "Joining scenes")
    lst = seg_dir / "concat.txt"
    lst.write_text("".join("file '{}'\n".format(str(s.resolve()).replace("\\", "/").replace("'", "'\\''")) for s in segs),
                   encoding="utf-8")
    tmp = out.with_suffix(".part.mp4")
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-an", "-movflags", "+faststart", str(tmp)])
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    actual = count_frames(out)
    info = probe(out)
    report = {"video": str(out), "fps": fps, "frames_expected": total_frames, "frames_actual": actual,
              "duration_video": info["duration"], "duration_narration": tl["audio_duration"],
              "failed_scenes": failed, "scenes": report_scenes}
    if actual != total_frames:
        raise RenderError(f"Rendered {actual} frames but the narration needs {total_frames}")
    if mode == "final" and not (keep_segments or False):
        for s in segs:
            s.unlink(missing_ok=True)
        lst.unlink(missing_ok=True)
        try:
            seg_dir.rmdir()
        except OSError:
            pass
    say(1.0, "Done")
    return report


# ------------------------------------------------------------------------------- audio
def mux_narration(video: Path, narration: Path, out: Path, duration: float, music: str | None = None,
                  music_volume: float = 0.12) -> None:
    """Add the EXISTING narration (and optional quiet local music). Video stream is copied, never re-encoded."""
    args = ["-i", str(video), "-i", str(narration)]
    if music:
        args += ["-stream_loop", "-1", "-i", music]
        norm = ":normalize=0" if _filter_has_option("amix", "normalize") else ""
        comp = "" if norm else ",volume=2"
        fade_start = max(0.0, duration - 2.0)
        fc = (f"[1:a]aresample=48000[n];"
              f"[2:a]aresample=48000,volume={music_volume:.3f},atrim=duration={duration:.3f},"
              f"afade=t=out:st={fade_start:.3f}:d=2[m];"
              f"[n][m]amix=inputs=2:duration=first:dropout_transition=0{norm}{comp}[a]")
        args += ["-filter_complex", fc, "-map", "0:v", "-map", "[a]"]
    else:
        args += ["-map", "0:v", "-map", "1:a"]
    tmp = out.with_suffix(".part.mp4")
    try:
        run_ffmpeg(args + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(tmp)])
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def scene_preview(project, tl: dict, scene_id: int, with_audio: bool = True) -> Path:
    """Render one scene at preview quality (reuses cached segments) with its slice of the narration."""
    st = project.settings
    fps = int(st["fps"])
    plan = frame_plan(tl, fps)
    item = next((p for p in plan if p["scene"]["id"] == scene_id), None)
    if item is None:
        raise RenderError(f"No scene {scene_id}")
    sc = item["scene"]
    W, H, crf = int(st["preview_width"]), int(st["preview_height"]), int(st["preview_crf"])
    seg_dir = project.preview_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec(sc, item["clip_bounds"], W, H, fps, crf, "ultrafast", float(st["crossfade"]))
    seg = seg_dir / f"scene_{sc['id']:03d}_{sha1(json.dumps(spec))[:10]}.mp4"
    if not seg.exists():
        for old in seg_dir.glob(f"scene_{sc['id']:03d}_*.mp4"):
            old.unlink(missing_ok=True)
        render_scene(sc, item["clip_bounds"], W, H, fps, crf, "ultrafast", float(st["crossfade"]), seg)
    out = project.preview_dir / f"scene_{sc['id']:03d}.mp4"
    narr = project.narration_path()
    if with_audio and narr:
        run_ffmpeg(["-i", str(seg), "-ss", f"{item['f0'] / fps:.3f}", "-t", f"{(item['f1'] - item['f0']) / fps:.3f}",
                    "-i", str(narr), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(out)])
    else:
        import shutil
        shutil.copyfile(seg, out)
    return out
