"""Provider interface, the Candidate model, and shared HTTP helpers."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import requests

from ..config import redact

BACKOFF = 1.5  # seconds; tests set this to 0


class ProviderError(Exception):
    """API failure (network, bad key, bad response). Message is safe to show (no secrets)."""


class RateLimitError(ProviderError):
    pass


@dataclass
class Candidate:
    source: str
    media_id: str
    kind: str                  # "video" | "image"
    page_url: str = ""
    download_url: str = ""
    thumb: str = ""
    duration: float = 0.0      # seconds (0 for images)
    width: int = 0             # dimensions of the file we would download
    height: int = 0
    orig_width: int = 0        # dimensions of the original (orientation check)
    orig_height: int = 0
    fps: float = 0.0
    slug: str = ""             # descriptive text (url slug / tags / alt text) used for relevance
    query: str = ""
    rank: int = 0              # position within its search results (0 = best)
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.source}:{self.media_id}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Candidate":
        fields = Candidate.__dataclass_fields__
        return Candidate(**{k: v for k, v in d.items() if k in fields})


def slug_from_url(url: str) -> str:
    """https://www.pexels.com/video/a-woman-watering-plants-855564/ -> 'a woman watering plants'"""
    m = re.search(r"/(?:video|photo)/([^/?#]+)", url or "")
    if not m:
        return ""
    slug = re.sub(r"-?\d+$", "", m.group(1))
    return slug.replace("-", " ").strip()


def http_get_json(session: requests.Session, url: str, *, headers=None, params=None,
                  timeout=(8, 25), retries=2, provider="API") -> tuple[dict, requests.structures.CaseInsensitiveDict]:
    last = "unknown error"
    for attempt in range(retries + 1):
        try:
            r = session.get(url, headers=headers, params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = f"network error ({type(e).__name__})"
            if attempt < retries:
                time.sleep(BACKOFF * (attempt + 1))
                continue
            raise ProviderError(f"{provider}: {last}")
        except requests.RequestException as e:
            raise ProviderError(redact(f"{provider}: request failed ({type(e).__name__})"))
        if r.status_code == 429:
            raise RateLimitError(f"{provider}: rate limit reached")
        if r.status_code in (401, 403):
            raise ProviderError(f"{provider}: the API key was rejected (HTTP {r.status_code}). Check your .env")
        if r.status_code >= 500:
            last = f"server error HTTP {r.status_code}"
            if attempt < retries:
                time.sleep(BACKOFF * (attempt + 1))
                continue
            raise ProviderError(f"{provider}: {last}")
        if r.status_code != 200:
            raise ProviderError(f"{provider}: unexpected HTTP {r.status_code}")
        try:
            return r.json(), r.headers
        except ValueError:
            raise ProviderError(f"{provider}: response was not valid JSON")
    raise ProviderError(f"{provider}: {last}")


class Provider:
    name = "base"
    supports_video = True
    supports_images = True

    def __init__(self, cache=None):
        self.cache = cache
        self.api_calls = 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "stockdoc-local-editor/1.0"

    def configured(self) -> bool:
        raise NotImplementedError

    def _cached(self, kind: str, query: str, page: int, fetch) -> list[Candidate]:
        if self.cache is not None:
            hit = self.cache.search_get(self.name, kind, query, page)
            if hit is not None:
                return [Candidate.from_dict(d) for d in hit]
        cands: list[Candidate] = fetch()
        if self.cache is not None:
            self.cache.search_put(self.name, kind, query, page, [c.to_dict() for c in cands])
        return cands

    def search_videos(self, query: str, per_page: int = 15, page: int = 1) -> list[Candidate]:
        return []

    def search_images(self, query: str, per_page: int = 10, page: int = 1) -> list[Candidate]:
        return []

    def on_selected(self, cand: Candidate) -> None:
        """Hook (e.g. Unsplash requires a download-tracking ping)."""

    def budget_left(self) -> Optional[int]:
        return None
