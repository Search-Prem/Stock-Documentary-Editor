"""Sentence -> visual search queries. Rule-based (no LLM, no network).

Strategy: (1) domain concept rules turn narration jargon into things a camera can film
("overwatering" -> "watering indoor plant", "wet plant soil"); (2) generic noun-phrase extraction covers
everything else; (3) synonym/suffix expansion supplies fresh queries for "Search Again".
"""
from __future__ import annotations

import re

STOP = set("""a an the and or but if then than that this these those there here of to in on at by for with from
into onto over under about above below up down out off again further once as is are was were be been being am
do does did doing have has had having i me my we our you your he she it its they them their what which who whom
whose when where why how all any both each few more most other some such no nor not only own same so too very
can will just should would could may might must shall also even ever still yet already always never often
sometimes usually really actually simply basically maybe perhaps quite rather many much lot lots one two three
first second next last let lets us get gets got getting make makes made making take takes took taking go goes
went going come comes came coming see sees saw seen look looks looking say says said tell tells told know
knows known think thinks thought want wants need needs needed use uses used using way ways thing things
something anything everything nothing mean means meant common commonly cause causes caused causing reason
reasons result results lead leads leading eventually turn turns turned turning become becomes became
happen happens happened important like well right now today while during before after between because
without within through across along around per via however therefore thus whether either neither
another every different certain able whole part parts kind type sort key main major real true good bad
better best worse worst noticed notice find finds found give gives gave keep keeps kept put puts set sets
let stay stays remain remains cannot dont doesnt isnt arent wasnt didnt wont cant yes yeah okay hold holds held contain contains
include includes including known called call calls call""".split())

# (all-of regexes, queries) ; more regexes matched = more specific = ranked first
CONCEPTS: list[tuple[list[str], list[str]]] = [
    ([r"over-?water"], ["overwatered houseplant", "watering indoor plant", "wet plant soil", "waterlogged plant pot"]),
    ([r"under-?water", r"\bdry\b|thirsty"], ["wilting plant", "dry cracked soil", "drooping houseplant"]),
    ([r"wilt|droop|sag"], ["wilting plant", "drooping houseplant leaves", "dry plant soil"]),
    ([r"yellow", r"leaf|leaves|foliage"], ["yellowing houseplant leaves", "yellow plant leaves close up", "sick plant leaves"]),
    ([r"yellow|brown|discolo|spots?\b"], ["leaf discoloration", "yellowing houseplant", "unhealthy houseplant"]),
    ([r"root", r"absorb|uptake|drink|take up"], ["roots absorbing water", "plant roots in soil", "watering plant roots"]),
    ([r"root", r"rot|suffocat|breathe|oxygen|air"], ["root rot plant", "plant roots close up", "roots in wet soil"]),
    ([r"\broots?\b"], ["plant roots soil", "root system close up", "roots growing"]),
    ([r"photosynth|chlorophyll"], ["sunlight through green leaves", "leaf close up sunlight", "green leaves macro"]),
    ([r"sun|light", r"leaf|leaves|plant|window"], ["sunlight through leaves", "plant near sunny window", "sun rays forest leaves"]),
    ([r"grow light|artificial light|led light"], ["grow light plants", "indoor plant grow lights"]),
    ([r"\bsoil\b|potting"], ["plant soil close up", "hands potting soil", "garden soil"]),
    ([r"water(?:ing|s|ed)?\b"], ["watering houseplant", "watering can pouring water", "water droplets on leaves"]),
    ([r"fertili[sz]|nutrient|nitrogen|potassium|phosph"], ["fertilizing houseplant", "plant nutrients soil", "liquid fertilizer plant"]),
    ([r"aphid|mite|mealybug|gnat|pest|insect|bug"], ["aphids on plant leaf", "pest on houseplant", "insect on leaf macro"]),
    ([r"fung|mold|mould|mildew|rot\b|disease|blight"], ["plant disease leaf", "mold on plant soil", "fungus on plant"]),
    ([r"seed|germinat|sprout"], ["seed germination", "seedling sprouting", "sprout growing timelapse"]),
    ([r"flower|bloom|blossom|petal"], ["flower blooming", "flower close up", "flower bud opening"]),
    ([r"repot|\bpot\b|container|planter"], ["repotting houseplant", "potting plant hands", "plant pots windowsill"]),
    ([r"stomata|microscop|cell|chloroplast"], ["plant cells microscope", "microscope laboratory", "leaf macro"]),
    ([r"prun|trim|cutting|propagat"], ["pruning houseplant", "plant cuttings in water", "trimming plant scissors"]),
    ([r"humid|mist|moisture"], ["misting houseplants", "water mist on leaves", "humidifier plants"]),
    ([r"cold|frost|freez|winter"], ["frost on plants", "winter garden", "cold weather plants"]),
    ([r"cactus|cacti|succulent"], ["succulent plants", "cactus close up", "desert cactus"]),
    ([r"forest|tree|canopy|trunk"], ["forest trees canopy", "tree bark close up", "forest sunlight"]),
    ([r"pollinat|\bbees?\b|butterfl"], ["bee pollinating flower", "pollination flower", "butterfly on flower"]),
    ([r"garden"], ["gardening hands", "vegetable garden", "garden plants"]),
    ([r"oxygen|carbon dioxide|\bco2\b|breath"], ["plant leaves in breeze", "green leaves fresh air", "aquatic plant bubbles"]),
    ([r"scien|research|experiment|study|laborator"], ["scientist laboratory plants", "laboratory research plants"]),
    ([r"indoor|house ?plant|home"], ["indoor plants home", "houseplants windowsill", "houseplant collection"]),
    ([r"leaf|leaves|foliage"], ["green leaves close up", "houseplant leaves", "leaf macro"]),
    ([r"\bplants?\b"], ["houseplants", "green plant close up"]),
]

SYNONYMS = {
    "yellow": ["yellowing", "discolored"], "yellowing": ["yellow"], "leaves": ["foliage", "leaf"], "leaf": ["leaves"],
    "plant": ["houseplant", "indoor plant"], "plants": ["houseplants", "indoor plants"],
    "houseplant": ["indoor plant"], "soil": ["dirt", "potting soil"], "water": ["watering", "irrigation"],
    "watering": ["pouring water"], "roots": ["root system"], "sick": ["unhealthy", "diseased"],
    "unhealthy": ["sick", "dying"], "wet": ["soaked", "saturated"], "dry": ["parched", "cracked"],
    "growing": ["growth", "sprouting"], "sunlight": ["sun rays", "natural light"], "tree": ["forest"],
}
SUFFIXES = ["close up", "macro", "slow motion", "timelapse"]


def stem(w: str) -> str:
    w = w.lower()
    irregular = {"leaves": "leaf", "leafs": "leaf", "roots": "root", "dying": "die"}
    if w in irregular:
        return irregular[w]
    for suf in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf in ("ing", "ed") and len(base) >= 4 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            return base
    return w


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower().replace("’", "'").replace("'", ""))


def content_terms(sentence: str) -> list[str]:
    return [w for w in _words(sentence) if w not in STOP and len(w) >= 3]


def _generic_queries(sentence: str, topic: str) -> list[str]:
    runs: list[list[str]] = []
    cur: list[str] = []
    for w in _words(sentence):
        if w in STOP or len(w) < 3:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        runs.append(cur)
    ranked = sorted(runs, key=lambda r: (-min(len(r), 3), -max(len(x) for x in r)))
    out: list[str] = []
    topic_stems = {stem(t) for t in _words(topic)}
    for r in ranked:
        phrase = r[-3:] if len(r) > 3 else r
        if len(phrase) >= 2:
            out.append(" ".join(phrase))
        else:
            w = phrase[0]
            if stem(w) not in topic_stems:
                out.append(f"{w} {topic}".strip())
                if len(w) >= 6:
                    out.append(w)
    return out


def plan_queries(sentence: str, topic: str = "plants", limit: int = 8) -> list[str]:
    """Ordered visual search queries for one sentence (most specific/visual first)."""
    low = sentence.lower()
    matched: list[tuple[int, int, list[str]]] = []
    for order, (pats, queries) in enumerate(CONCEPTS):
        if all(re.search(p, low) for p in pats):
            matched.append((-len(pats), order, queries))
    matched.sort()
    domain: list[str] = []
    for _, _, qs in matched:
        # take the top of each matched concept so several ideas are represented, not just one
        for q in qs[:2]:
            domain.append(q)
    generic = _generic_queries(sentence, topic)
    ordered = domain[:5] + generic[:3] + domain[5:] + generic[3:]
    seen, out = set(), []
    for q in ordered:
        key = frozenset(stem(w) for w in _words(q) if w not in STOP)
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    if not out:
        out = [topic or "nature"]
    return out[:limit]


def expand_queries(base: list[str], topic: str = "plants", limit: int = 24) -> list[str]:
    """base queries + synonym variants + style suffixes: fresh material for 'Search Again'."""
    out = list(base)
    seen = {frozenset(stem(w) for w in _words(q) if w not in STOP) for q in out}

    def add(q: str):
        key = frozenset(stem(w) for w in _words(q) if w not in STOP)
        if key and key not in seen and len(out) < limit:
            seen.add(key)
            out.append(q)

    for q in base:
        words = q.split()
        for i, w in enumerate(words):
            for syn in SYNONYMS.get(w.lower(), [])[:2]:
                add(" ".join(words[:i] + [syn] + words[i + 1:]))
    for q in base[:3]:
        for suf in SUFFIXES[:2]:
            add(f"{q} {suf}")
    for suf in SUFFIXES:
        add(f"{topic} {suf}")
    return out[:limit]
