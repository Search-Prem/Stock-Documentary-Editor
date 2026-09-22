"""Split a narration script into sentences (one sentence = one scene)."""
from __future__ import annotations

import re

_ABBREV = [
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc", "e.g", "i.e", "approx",
    "No", "Fig", "Inc", "Ltd", "Co", "U.S", "U.K", "a.m", "p.m", "Jan", "Feb", "Mar", "Apr",
    "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "cf", "ca", "spp", "sp", "var",
]
_PLACEHOLDER = "\u0001"
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
# Genus-style abbreviations: "A. thaliana", "E. coli" (single capital, dot, lowercase word)
_GENUS_RE = re.compile(r"\b([A-Z])\.(?=\s+[a-z])")
_BOUNDARY = re.compile(r"""([.!?…]+["'”’)\]]*)\s+(?=["'“‘(\[]*[A-Z0-9])""")


def _protect(s: str) -> str:
    s = _ABBREV_RE.sub(lambda m: m.group(1) + _PLACEHOLDER, s)
    s = _DECIMAL_RE.sub(_PLACEHOLDER, s)
    s = _GENUS_RE.sub(lambda m: m.group(1) + _PLACEHOLDER, s)
    return s


def split_sentences(text: str) -> list[str]:
    """Blank lines separate paragraphs (always a boundary); single newlines are joined."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[*_`#]+", "", text)  # strip markdown emphasis / headings
    paragraphs = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for para in paragraphs:
        para = " ".join(para.split())
        if not para:
            continue
        para = _protect(para)
        parts, last = [], 0
        for m in _BOUNDARY.finditer(para):
            parts.append(para[last:m.end(1)])
            last = m.end()
        parts.append(para[last:])
        out.extend(p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip())
    # Script with no punctuation at all: fall back to one sentence per line
    if len(out) <= 1 and "\n" in text.strip() and not re.search(r"[.!?]", text):
        out = [" ".join(l.split()) for l in text.splitlines() if l.strip()]
    return out
