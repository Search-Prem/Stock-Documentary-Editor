"""Local caches: (1) provider search results, (2) downloaded media (validated). Nothing is fetched twice."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import get_config, redact
from .providers.base import Candidate
from .util import IMAGE_EXTS, MediaError, atomic_write_json, check_decodes, log, probe, read_json, run_ffmpeg, sha1

SEARCH_TTL = 30 * 24 * 3600
MAX_DOWNLOAD_BYTES = 2_000_000_000
BACKOFF = 1.5


class DownloadError(MediaError):
    pass


def _check_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme == "https":
        return
    if u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost"):  # local test servers only
        return
    raise DownloadError("Refusing to download from a non-HTTPS URL")


class Cache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or get_config().cache_dir)
        self.search_dir = self.root / "search"
        self.media_dir = self.root / "media"
        self.bad_path = self.root / "bad_media.json"
        self.stats = {"search_hits": 0, "downloads": 0, "download_hits": 0}
        for d in (self.search_dir, self.media_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ search results
    def _spath(self, provider: str, kind: str, query: str, page: int) -> Path:
        h = sha1(json.dumps([provider, kind, query.lower().strip(), page]))
        return self.search_dir / provider / f"{h}.json"

    def search_get(self, provider, kind, query, page):
        p = self._spath(provider, kind, query, page)
        data = read_json(p)
        if data and time.time() - data.get("saved", 0) < SEARCH_TTL:
            self.stats["search_hits"] += 1
            return data["results"]
        return None

    def search_put(self, provider, kind, query, page, results):
        atomic_write_json(self._spath(provider, kind, query, page),
                          {"saved": time.time(), "query": query, "results": results})

    # ------------------------------------------------------------------ bad media registry
    def _bad(self) -> dict:
        return read_json(self.bad_path, {}) or {}

    def is_bad(self, key: str) -> bool:
        return key in self._bad()

    def mark_bad(self, key: str, reason: str) -> None:
        bad = self._bad()
        bad[key] = reason[:200]
        atomic_write_json(self.bad_path, bad)

    # ------------------------------------------------------------------ media
    def media_path(self, cand: Candidate) -> Path:
        ext = Path(urlparse(cand.download_url).path).suffix.lower()
        if cand.kind == "image":
            ext = ext if ext in IMAGE_EXTS else ".jpg"
        else:
            ext = ext if ext in (".mp4", ".mov", ".webm", ".m4v") else ".mp4"
        safe_id = "".join(c for c in cand.media_id if c.isalnum() or c in "-_")
        return self.media_dir / f"{cand.source}_{safe_id}{ext}"

    def get_media(self, cand: Candidate, provider=None) -> tuple[Path, dict]:
        """Return (path, probe_info). Downloads once, validates, and never re-downloads a good file."""
        dest = self.media_path(cand)
        if dest.exists() and dest.stat().st_size > 0:
            try:
                info = probe(dest)
                self.stats["download_hits"] += 1
                return dest, info
            except MediaError:
                dest.unlink(missing_ok=True)
        if not cand.download_url:
            raise DownloadError("Candidate has no download URL")
        if provider is not None:
            provider.on_selected(cand)
        self._download(cand.download_url, dest)
        self.stats["downloads"] += 1
        try:
            info = probe(dest)
            if info["kind"] not in ("video", "image"):
                raise MediaError("not a video/image")
            if info["kind"] == "video" and info["width"] < info["height"] * 1.0 and cand.kind == "video":
                pass  # portrait handled by the selector; still a valid file
            if not check_decodes(dest):
                raise MediaError("file does not decode (truncated or corrupt)")
        except MediaError as e:
            dest.unlink(missing_ok=True)
            self.mark_bad(cand.key(), str(e))
            raise DownloadError(f"{cand.key()} is corrupt or unsupported: {e}")
        return dest, info

    def _download(self, url: str, dest: Path, attempts: int = 3) -> None:
        _check_url(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        last = ""
        for attempt in range(attempts):
            try:
                with requests.get(url, stream=True, timeout=(8, 30), headers={"User-Agent": "stockdoc-local-editor/1.0"}) as r:
                    if r.status_code != 200:
                        raise DownloadError(f"download failed: HTTP {r.status_code}")
                    total = int(r.headers.get("Content-Length") or 0)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise DownloadError("file is unreasonably large; skipped")
                    written = 0
                    with open(part, "wb") as f:
                        for chunk in r.iter_content(256 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if written > MAX_DOWNLOAD_BYTES:
                                    raise DownloadError("file is unreasonably large; aborted")
                    if total and written != total:
                        raise DownloadError(f"incomplete download ({written}/{total} bytes)")
                os.replace(part, dest)
                return
            except DownloadError as e:
                last = str(e)
                part.unlink(missing_ok=True)
                if "HTTP 4" in last or "unreasonably" in last:
                    break  # not worth retrying
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                last = f"network error ({type(e).__name__})"
                part.unlink(missing_ok=True)
            except OSError as e:
                part.unlink(missing_ok=True)
                raise DownloadError(redact(f"cannot write to cache: {e}"))
            if attempt < attempts - 1:
                time.sleep(BACKOFF * (attempt + 1))
        raise DownloadError(redact(last or "download failed"))


def make_thumb(src: str | Path, dest: str | Path, is_image: bool = False, at: float = 0.5, width: int = 320) -> bool:
    try:
        args = ([] if is_image else ["-ss", str(at)]) + ["-i", str(src), "-frames:v", "1",
                                                          "-vf", f"scale={width}:-2", "-q:v", "4", str(dest)]
        run_ffmpeg(args, timeout=60)
        return Path(dest).exists()
    except MediaError:
        return False


def link_into_project(cache_file: Path, scene_dir: Path) -> None:
    """Expose a cached file inside media/scene_NNN/ without duplicating it (hard link; pointer file fallback)."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    target = scene_dir / cache_file.name
    if target.exists():
        return
    try:
        os.link(cache_file, target)
    except OSError:
        (scene_dir / (cache_file.name + ".ref.json")).write_text(
            json.dumps({"cached_file": str(cache_file)}, indent=2), encoding="utf-8")
