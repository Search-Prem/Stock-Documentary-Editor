"""Project folders + persistence (script, narration, settings, timeline). Everything resumes from disk."""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from .config import get_config
from .script_split import split_sentences
from .util import AUDIO_EXTS, MediaError, atomic_write_json, probe, read_json, sha1

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

DEFAULT_SETTINGS = {
    "topic_hint": "plants",          # appended to weak single-word queries to keep results on-topic
    "align_backend": "auto",         # auto | whisper | silence | estimate
    "whisper_model": "small.en",
    "whisper_device": "cpu",         # cpu | cuda
    "language": "en",
    "estimate_wpm": 150,
    "min_clip": 2.5,                 # seconds
    "max_clip": 10.0,
    "target_clip": 6.0,              # preferred pacing for long sentences
    "crossfade": 0.35,               # seconds, only between clips inside one sentence; 0 = hard cuts
    "queries_per_round": 3,
    "max_api_calls_per_scene": 4,
    "providers": ["pexels", "pixabay", "unsplash"],
    "music_enabled": False,
    "music_path": "",
    "music_volume": 0.12,
    "srt_max_cue_chars": 0,          # 0 = one subtitle per sentence
    "fps": 30, "width": 1920, "height": 1080, "crf": 18, "preset": "veryfast",
    "preview_width": 960, "preview_height": 540, "preview_crf": 30,
}


def empty_timeline() -> dict:
    return {"version": 1, "audio_duration": None, "fps": 30, "alignment": {}, "scenes": []}


class Project:
    def __init__(self, name: str, base: Path | None = None):
        if not NAME_RE.match(name or ""):
            raise ValueError("Project name may only contain letters, numbers, '-' and '_' (max 64 chars)")
        self.name = name
        self.base = Path(base or get_config().projects_dir)
        self.dir = self.base / name
        self.script_dir = self.dir / "script"
        self.narration_dir = self.dir / "narration"
        self.timeline_dir = self.dir / "timeline"
        self.media_dir = self.dir / "media"
        self.preview_dir = self.dir / "preview"
        self.final_dir = self.dir / "final"
        self.meta_path = self.dir / "project.json"
        self.timeline_path = self.timeline_dir / "timeline.json"
        self.srt_path = self.timeline_dir / "subtitles.srt"
        self.script_path = self.script_dir / "script.txt"

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, name: str, base: Path | None = None) -> "Project":
        p = cls(name, base)
        if p.dir.exists():
            raise FileExistsError(f"Project '{name}' already exists")
        for d in (p.script_dir, p.narration_dir, p.timeline_dir, p.media_dir, p.preview_dir, p.final_dir):
            d.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p.meta_path, {"name": name, "created": time.time(), "settings": {}, "narration": None})
        return p

    @classmethod
    def open(cls, name: str, base: Path | None = None) -> "Project":
        p = cls(name, base)
        if not p.meta_path.exists():
            raise FileNotFoundError(f"Project '{name}' not found")
        return p

    @classmethod
    def list_names(cls, base: Path | None = None) -> list[str]:
        b = Path(base or get_config().projects_dir)
        if not b.exists():
            return []
        return sorted(d.name for d in b.iterdir() if (d / "project.json").exists())

    # ------------------------------------------------------------ meta / settings
    def meta(self) -> dict:
        return read_json(self.meta_path, {}) or {}

    def save_meta(self, meta: dict) -> None:
        atomic_write_json(self.meta_path, meta)

    @property
    def settings(self) -> dict:
        s = dict(DEFAULT_SETTINGS)
        s.update(self.meta().get("settings", {}))
        return s

    def update_settings(self, patch: dict) -> dict:
        allowed = {k: v for k, v in patch.items() if k in DEFAULT_SETTINGS}
        for k, v in allowed.items():  # coerce to the default's type so a bad UI value can't corrupt things
            d = DEFAULT_SETTINGS[k]
            if isinstance(d, bool):
                allowed[k] = bool(v)
            elif isinstance(d, int):
                allowed[k] = int(float(v))
            elif isinstance(d, float):
                allowed[k] = float(v)
            elif isinstance(d, list):
                allowed[k] = [str(x) for x in v]
            else:
                allowed[k] = str(v)
        m = self.meta()
        m.setdefault("settings", {}).update(allowed)
        self.save_meta(m)
        return self.settings

    # ------------------------------------------------------------ script
    def save_script(self, text: str) -> list[str]:
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.script_path.write_text(text, encoding="utf-8")
        m = self.meta()
        m["script_sha"] = sha1(text)
        self.save_meta(m)
        return split_sentences(text)

    def script_text(self) -> str:
        return self.script_path.read_text(encoding="utf-8") if self.script_path.exists() else ""

    def sentences(self) -> list[str]:
        return split_sentences(self.script_text())

    # ------------------------------------------------------------ narration
    def narration_path(self) -> Path | None:
        name = (self.meta().get("narration") or {}).get("file")
        if name and (self.narration_dir / name).exists():
            return self.narration_dir / name
        return None

    def import_narration(self, filename: str, stream: BinaryIO | None = None, src_path: str | None = None) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise MediaError(f"Unsupported narration format '{ext}'. Use WAV or MP3.")
        self.narration_dir.mkdir(parents=True, exist_ok=True)
        dest = self.narration_dir / f"narration{ext}"
        for old in self.narration_dir.glob("narration.*"):
            if old != dest:
                old.unlink(missing_ok=True)
        if stream is not None:
            with open(dest, "wb") as f:
                shutil.copyfileobj(stream, f)
        elif src_path:
            shutil.copyfile(src_path, dest)
        else:
            raise MediaError("No narration data supplied")
        try:
            info = probe(dest)
            if info["kind"] != "audio" and not info["has_audio"]:
                raise MediaError("The file contains no audio")
            if info["duration"] <= 0:
                raise MediaError("The narration has zero duration")
        except MediaError:
            dest.unlink(missing_ok=True)
            raise
        m = self.meta()
        m["narration"] = {"file": dest.name, "duration": info["duration"]}
        self.save_meta(m)
        return m["narration"]

    # ------------------------------------------------------------ timeline
    def load_timeline(self) -> dict:
        return read_json(self.timeline_path) or empty_timeline()

    def save_timeline(self, tl: dict) -> None:
        atomic_write_json(self.timeline_path, tl)

    def scene_dir(self, scene_id: int) -> Path:
        d = self.media_dir / f"scene_{scene_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d
