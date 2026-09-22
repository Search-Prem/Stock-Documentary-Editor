"""Shared helpers: ffmpeg/ffprobe discovery, probing, time formatting, atomic JSON."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import get_config, redact

log = logging.getLogger("stockdoc")

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


class MediaError(Exception):
    """A media file is missing, corrupt, or unsupported."""


class FFmpegError(MediaError):
    """FFmpeg/ffprobe failed or is not installed."""


def find_binary(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    configured = get_config().ffmpeg_path
    if configured:
        p = Path(configured)
        if p.is_dir():
            for cand in (p / exe, p / "bin" / exe):
                if cand.exists():
                    return str(cand)
        elif p.exists():
            if name == "ffmpeg":
                return str(p)
            sib = p.parent / exe
            if sib.exists():
                return str(sib)
    found = shutil.which(name)
    if found:
        return found
    raise FFmpegError(f"{name} not found. Install FFmpeg and add it to PATH, or set FFMPEG_PATH in .env")


def ffmpeg_available() -> bool:
    try:
        find_binary("ffmpeg")
        find_binary("ffprobe")
        return True
    except FFmpegError:
        return False


def run_ffmpeg(args: list[str], timeout: float | None = None, loglevel: str = "error") -> str:
    cmd = [find_binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-loglevel", loglevel, *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FFmpegError("FFmpeg timed out")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-8:])
        raise FFmpegError(redact(f"FFmpeg failed (exit {r.returncode}): {tail}"))
    return r.stderr


def _parse_rate(s: str | None) -> float:
    try:
        if not s or s == "0/0":
            return 0.0
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> dict[str, Any]:
    """Return {kind, duration, width, height, fps, nb_frames, has_audio}. Raises MediaError."""
    p = str(path)
    if not os.path.isfile(p):
        raise MediaError(f"File not found: {p}")
    cmd = [find_binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", p]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except subprocess.TimeoutExpired:
        raise MediaError(f"Timed out reading {os.path.basename(p)}")
    if r.returncode != 0:
        raise MediaError(f"Cannot read {os.path.basename(p)} (corrupt or unsupported): {r.stderr.strip()[-200:]}")
    try:
        data = json.loads(r.stdout or "{}")
    except ValueError:
        raise MediaError(f"Unreadable metadata for {os.path.basename(p)}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    ext = Path(p).suffix.lower()
    try:
        dur = float(fmt.get("duration") or (v or a or {}).get("duration") or 0)
    except ValueError:
        dur = 0.0
    if ext in IMAGE_EXTS and v:
        kind = "image"
    elif v:
        kind = "video"
    elif a:
        kind = "audio"
    else:
        raise MediaError(f"No audio or video stream in {os.path.basename(p)}")
    nb = 0
    if v:
        try:
            nb = int(v.get("nb_frames") or 0)
        except ValueError:
            nb = 0
    info = {
        "kind": kind,
        "duration": dur,
        "width": int((v or {}).get("width") or 0),
        "height": int((v or {}).get("height") or 0),
        "fps": _parse_rate((v or {}).get("avg_frame_rate")) or _parse_rate((v or {}).get("r_frame_rate")),
        "nb_frames": nb,
        "has_audio": a is not None,
    }
    if kind == "video" and dur <= 0:
        raise MediaError(f"{os.path.basename(p)} has no usable duration")
    return info


def check_decodes(path: str | Path, seconds: float = 1.5) -> bool:
    """Quick decode test to catch truncated/corrupt files before they break a render."""
    try:
        run_ffmpeg(["-xerror", "-t", str(seconds), "-i", str(path), "-f", "null", "-"], timeout=60)
        return True
    except FFmpegError:
        return False


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}" if h else f"{m:02d}:{s:02d}.{ms:03d}"


def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def frames(t: float, fps: int) -> int:
    """Time -> frame index on the output grid (nearest frame)."""
    return int(round(t * fps + 1e-9))
