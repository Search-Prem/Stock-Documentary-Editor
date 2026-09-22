"""Configuration: .env values, folders, and secret redaction."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _e(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


@dataclass
class Config:
    pexels_key: str
    pixabay_key: str
    unsplash_key: str
    ffmpeg_path: str
    pexels_base: str
    pixabay_base: str
    unsplash_base: str
    projects_dir: Path
    cache_dir: Path

    def secrets(self) -> list[str]:
        return [s for s in (self.pexels_key, self.pixabay_key, self.unsplash_key) if s]


def get_config() -> Config:
    """Read config fresh each call so tests / .env edits are picked up."""
    return Config(
        pexels_key=_e("PEXELS_API_KEY"),
        pixabay_key=_e("PIXABAY_API_KEY"),
        unsplash_key=_e("UNSPLASH_ACCESS_KEY"),
        ffmpeg_path=_e("FFMPEG_PATH"),
        pexels_base=_e("PEXELS_BASE_URL", "https://api.pexels.com").rstrip("/"),
        pixabay_base=_e("PIXABAY_BASE_URL", "https://pixabay.com").rstrip("/"),
        unsplash_base=_e("UNSPLASH_BASE_URL", "https://api.unsplash.com").rstrip("/"),
        projects_dir=Path(_e("PROJECTS_DIR") or ROOT / "projects"),
        cache_dir=Path(_e("CACHE_DIR") or ROOT / "cache"),
    )


def redact(text: object) -> str:
    """Remove API keys from any text before it reaches logs or the UI."""
    s = str(text)
    for secret in get_config().secrets():
        if len(secret) >= 6:
            s = s.replace(secret, "***")
    return s
