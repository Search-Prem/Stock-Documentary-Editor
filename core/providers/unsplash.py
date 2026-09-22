"""Unsplash (optional fallback): photos only."""
from __future__ import annotations

from ..config import get_config
from .base import Candidate, Provider, ProviderError, http_get_json


class UnsplashProvider(Provider):
    name = "unsplash"
    supports_video = False

    def configured(self) -> bool:
        return bool(get_config().unsplash_key)

    def _headers(self):
        return {"Authorization": f"Client-ID {get_config().unsplash_key}", "Accept-Version": "v1"}

    def search_images(self, query, per_page=10, page=1):
        def fetch():
            cfg = get_config()
            try:
                data, _ = http_get_json(self.session, cfg.unsplash_base + "/search/photos", headers=self._headers(),
                                        params={"query": query, "orientation": "landscape",
                                                "per_page": per_page, "page": page}, provider="Unsplash")
            finally:
                self.api_calls += 1
            out = []
            for rank, p in enumerate(data.get("results", [])):
                urls = p.get("urls") or {}
                raw = urls.get("raw")
                if not raw:
                    continue
                sep = "&" if "?" in raw else "?"
                out.append(Candidate(
                    source="unsplash", media_id=str(p["id"]), kind="image", page_url=(p.get("links") or {}).get("html", ""),
                    download_url=f"{raw}{sep}w=2560&q=80&fm=jpg", thumb=urls.get("small", ""),
                    width=int(p.get("width") or 0), height=int(p.get("height") or 0),
                    orig_width=int(p.get("width") or 0), orig_height=int(p.get("height") or 0),
                    slug=" ".join(filter(None, [p.get("alt_description"), p.get("description")])), query=query, rank=rank,
                    extra={"download_location": (p.get("links") or {}).get("download_location", ""),
                           "credit": (p.get("user") or {}).get("name", "")}))
            return out
        return self._cached("image", query, page, fetch)

    def on_selected(self, cand):
        """Unsplash API guidelines: ping the download endpoint when a photo is actually used."""
        loc = cand.extra.get("download_location")
        if not loc:
            return
        try:
            self.session.get(loc, headers=self._headers(), timeout=(5, 10))
        except Exception:
            pass  # non-critical
