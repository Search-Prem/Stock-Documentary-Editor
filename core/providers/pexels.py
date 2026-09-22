"""Pexels (PRIMARY source): videos and photos."""
from __future__ import annotations

import time

from ..config import get_config
from .base import Candidate, Provider, RateLimitError, http_get_json, slug_from_url

MIN_INTERVAL = 0.25  # seconds between live API calls


def pick_video_file(files: list[dict]) -> dict | None:
    """Prefer the smallest file that is >= 1920x1080 (so 1080p wins over 4K: less disk/RAM), else the largest."""
    mp4 = [f for f in files if f.get("link") and (f.get("file_type") or "video/mp4") == "video/mp4"]
    if not mp4:
        return None

    def px(f):
        return int(f.get("width") or 0) * int(f.get("height") or 0)

    hd = [f for f in mp4 if int(f.get("width") or 0) >= 1920 and int(f.get("height") or 0) >= 1080]
    if hd:
        return min(hd, key=px)
    return max(mp4, key=px)


class PexelsProvider(Provider):
    name = "pexels"

    def __init__(self, cache=None):
        super().__init__(cache)
        self.remaining: int | None = None
        self.reset_at: float | None = None
        self._last_call = 0.0

    def configured(self) -> bool:
        return bool(get_config().pexels_key)

    def budget_left(self):
        return self.remaining

    def _get(self, path: str, params: dict) -> dict:
        if self.remaining is not None and self.remaining <= 0 and (self.reset_at or 0) > time.time():
            raise RateLimitError("Pexels: hourly request limit used up; try again after it resets")
        wait = MIN_INTERVAL - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        cfg = get_config()
        try:
            data, headers = http_get_json(self.session, cfg.pexels_base + path, params=params,
                                          headers={"Authorization": cfg.pexels_key}, provider="Pexels")
        finally:
            self._last_call = time.time()
            self.api_calls += 1
        try:
            if "X-Ratelimit-Remaining" in headers:
                self.remaining = int(headers["X-Ratelimit-Remaining"])
            if "X-Ratelimit-Reset" in headers:
                self.reset_at = float(headers["X-Ratelimit-Reset"])
        except ValueError:
            pass
        return data

    def search_videos(self, query, per_page=15, page=1):
        def fetch():
            data = self._get("/videos/search", {"query": query, "orientation": "landscape",
                                                "size": "large", "per_page": per_page, "page": page})
            out = []
            for rank, v in enumerate(data.get("videos", [])):
                f = pick_video_file(v.get("video_files", []))
                if not f or not v.get("id"):
                    continue
                pics = v.get("video_pictures") or []
                out.append(Candidate(
                    source="pexels", media_id=str(v["id"]), kind="video", page_url=v.get("url", ""),
                    download_url=f["link"], thumb=v.get("image") or (pics[0].get("picture", "") if pics else ""),
                    duration=float(v.get("duration") or 0), width=int(f.get("width") or 0),
                    height=int(f.get("height") or 0), orig_width=int(v.get("width") or 0),
                    orig_height=int(v.get("height") or 0), fps=float(f.get("fps") or 0),
                    slug=slug_from_url(v.get("url", "")), query=query, rank=rank))
            return out
        return self._cached("video", query, page, fetch)

    def search_images(self, query, per_page=10, page=1):
        def fetch():
            data = self._get("/v1/search", {"query": query, "orientation": "landscape",
                                            "size": "large", "per_page": per_page, "page": page})
            out = []
            for rank, p in enumerate(data.get("photos", [])):
                src = p.get("src") or {}
                url = src.get("original") or src.get("large2x")
                if not url or not p.get("id"):
                    continue
                out.append(Candidate(
                    source="pexels", media_id=f"photo{p['id']}", kind="image", page_url=p.get("url", ""),
                    download_url=url, thumb=src.get("medium") or src.get("small", ""),
                    width=int(p.get("width") or 0), height=int(p.get("height") or 0),
                    orig_width=int(p.get("width") or 0), orig_height=int(p.get("height") or 0),
                    slug=(p.get("alt") or slug_from_url(p.get("url", ""))), query=query, rank=rank))
            return out
        return self._cached("image", query, page, fetch)
