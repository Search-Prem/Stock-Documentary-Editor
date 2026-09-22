"""Pixabay (optional fallback): videos and photos."""
from __future__ import annotations

import time

from ..config import get_config
from .base import Candidate, Provider, http_get_json


class PixabayProvider(Provider):
    name = "pixabay"

    def __init__(self, cache=None):
        super().__init__(cache)
        self._last_call = 0.0

    def configured(self) -> bool:
        return bool(get_config().pixabay_key)

    def _get(self, path: str, params: dict) -> dict:
        wait = 0.7 - (time.time() - self._last_call)  # Pixabay allows ~100 requests/minute
        if wait > 0:
            time.sleep(wait)
        cfg = get_config()
        try:
            data, _ = http_get_json(self.session, cfg.pixabay_base + path,
                                    params={**params, "key": cfg.pixabay_key, "safesearch": "true"},
                                    provider="Pixabay")
        finally:
            self._last_call = time.time()
            self.api_calls += 1
        return data

    def search_videos(self, query, per_page=15, page=1):
        def fetch():
            data = self._get("/api/videos/", {"q": query[:100], "per_page": max(3, per_page), "page": page,
                                              "video_type": "film"})
            out = []
            for rank, h in enumerate(data.get("hits", [])):
                vids = h.get("videos") or {}
                f = next((vids[k] for k in ("large", "medium", "small") if (vids.get(k) or {}).get("url")), None)
                if not f:
                    continue
                pic = h.get("picture_id") or ""
                out.append(Candidate(
                    source="pixabay", media_id=str(h["id"]), kind="video", page_url=h.get("pageURL", ""),
                    download_url=f["url"], thumb=f"https://i.vimeocdn.com/video/{pic}_640x360.jpg" if pic else "",
                    duration=float(h.get("duration") or 0), width=int(f.get("width") or 0),
                    height=int(f.get("height") or 0), orig_width=int(f.get("width") or 0),
                    orig_height=int(f.get("height") or 0), slug=(h.get("tags") or "").replace(",", " "),
                    query=query, rank=rank))
            return out
        return self._cached("video", query, page, fetch)

    def search_images(self, query, per_page=10, page=1):
        def fetch():
            data = self._get("/api/", {"q": query[:100], "per_page": max(3, per_page), "page": page,
                                       "image_type": "photo", "orientation": "horizontal",
                                       "min_width": 1920, "min_height": 1080})
            out = []
            for rank, h in enumerate(data.get("hits", [])):
                url = h.get("fullHDURL") or h.get("largeImageURL")
                if not url:
                    continue
                out.append(Candidate(
                    source="pixabay", media_id=f"img{h['id']}", kind="image", page_url=h.get("pageURL", ""),
                    download_url=url, thumb=h.get("webformatURL", ""), width=int(h.get("imageWidth") or 0),
                    height=int(h.get("imageHeight") or 0), orig_width=int(h.get("imageWidth") or 0),
                    orig_height=int(h.get("imageHeight") or 0), slug=(h.get("tags") or "").replace(",", " "),
                    query=query, rank=rank))
            return out
        return self._cached("image", query, page, fetch)
