"""End-to-end: script + narration audio -> alignment -> (mock) Pexels -> timeline -> FFmpeg -> verified MP4 + SRT."""
import os, re, subprocess, json
from pathlib import Path

import pytest

from core import builder, cache as cache_mod
from core.config import get_config
from core.project import Project
from core.providers import base as prov_base
from core.providers import pexels as pexels_mod
from core.render import count_frames, render_timeline
from core.timeline import frame_plan, validate_timeline
from core.util import find_binary, probe
from tests.fixtures import SAMPLE_SCRIPT, frame_colors, make_color_clip, make_narration
from tests.mock_pexels import API_KEY, MockPexels

FPS = 30
CATALOG = [  # id, slug, colour, duration, w, h, fps
    (1, "yellowing-houseplant-leaves-close-up", "0xff0000", 14, 1920, 1080, 30),
    (2, "woman-watering-indoor-plant", "0x00ff00", 12, 1920, 1080, 25),
    (3, "plant-roots-in-wet-soil", "0x0000ff", 10, 1280, 720, 30),
    (4, "overwatered-houseplant-drooping-leaves", "0xffff00", 9, 1920, 1080, 24),
    (5, "hands-touching-soil-in-plant-pot", "0xff00ff", 11, 1920, 1080, 30),
    (6, "sunlight-through-green-leaves", "0x00ffff", 13, 1920, 1080, 60),
    (7, "wilting-plant-dry-soil", "0xff8000", 8, 1920, 1080, 30),
    (8, "watering-can-pouring-water-on-plant", "0x8000ff", 10, 1920, 1080, 30),
    (9, "green-plant-growing-in-pot", "0xff0080", 12, 1440, 1080, 30),
    (10, "roots-of-plant-underground", "0x00ff80", 7, 1920, 1080, 30),
    (11, "city-traffic-at-night", "0x808080", 10, 1920, 1080, 30),
    (12, "vertical-woman-watering-plants", "0x804000", 10, 1080, 1920, 30),   # portrait -> must be excluded
    (13, "leaf-macro-yellow-plant", "0x408000", 1, 1920, 1080, 30),           # too short -> excluded
    (14, "yellow-plant-leaves-houseplant-yellowing", "0x004080", 10, 1920, 1080, 30),  # CORRUPT, ranks first
    (15, "plant-leaves-low-quality", "0x800040", 10, 640, 360, 30),           # low res -> excluded
]
COLOR = {c[0]: tuple(int(c[2][i:i + 2], 16) for i in (2, 4, 6)) for c in CATALOG}


def nearest(rgb, tol=70):
    best = min(COLOR.items(), key=lambda kv: sum((a - b) ** 2 for a, b in zip(kv[1], rgb)))
    return best[0] if sum((a - b) ** 2 for a, b in zip(best[1], rgb)) ** 0.5 < tol else None


def audio_stream_duration(path):
    r = subprocess.run([find_binary("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    media = root / "stock"; media.mkdir()
    cat = []
    for (i, slug, col, dur, w, h, fps) in CATALOG:
        f = media / f"clip{i}.mp4"
        make_color_clip(f, col, dur, w, h, fps, corrupt=(i == 14))
        cat.append({"id": i, "slug": slug, "path": str(f), "duration": dur, "width": w, "height": h})
    mock = MockPexels(cat).start()
    saved = {k: os.environ.get(k) for k in ("PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_ACCESS_KEY", "PEXELS_BASE_URL",
                                            "PROJECTS_DIR", "CACHE_DIR")}
    os.environ.update({"PEXELS_API_KEY": API_KEY, "PEXELS_BASE_URL": f"http://127.0.0.1:{mock.port}",
                       "PROJECTS_DIR": str(root / "projects"), "CACHE_DIR": str(root / "cache")})
    os.environ.pop("PIXABAY_API_KEY", None); os.environ.pop("UNSPLASH_ACCESS_KEY", None)
    prov_base.BACKOFF = 0; cache_mod.BACKOFF = 0; pexels_mod.MIN_INTERVAL = 0
    p = Project.create("plants_doc")
    p.update_settings({"align_backend": "silence"})
    sentences = p.save_script(SAMPLE_SCRIPT)
    dur, truth = make_narration(sentences, root / "narr.wav")
    p.import_narration("narration.wav", src_path=str(root / "narr.wav"))
    a = builder.align_project(p)
    b = builder.build_visuals(p)
    tl = p.load_timeline()
    c = {"root": root, "mock": mock, "project": p, "sentences": sentences, "truth": truth, "dur": dur,
         "align": a, "build": b, "tl": tl}
    yield c
    mock.stop()
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ----------------------------------------------------------------------------- alignment + timeline
def test_alignment_matches_ground_truth(ctx):
    scenes = ctx["tl"]["scenes"]
    assert len(scenes) == len(ctx["sentences"])
    worst = max(abs(s["start"] - t) for s, t in zip(scenes[1:], ctx["truth"][1:]))
    assert worst < 0.12, worst
    assert scenes[0]["start"] == 0 and scenes[-1]["end"] == round(ctx["dur"], 3)


def test_timeline_has_no_gaps_or_overlaps(ctx):
    assert validate_timeline(ctx["tl"]) == []
    assert ctx["align"]["issues"] == []


# ----------------------------------------------------------------------------- footage selection
def test_every_scene_got_footage_and_bad_media_was_rejected(ctx):
    tl = ctx["tl"]
    assert ctx["build"]["paused"] == ""
    used_ids = {int(c["media_id"]) for s in tl["scenes"] for c in s["clips"]}
    assert not used_ids & {12, 13, 14, 15}, ("excluded clip used:", used_ids & {12, 13, 14, 15}, sorted(used_ids))           # portrait / too short / corrupt / low-res
    assert Path(ctx["project"].dir / "media").exists()
    bad = json.loads((Path(os.environ["CACHE_DIR"]) / "bad_media.json").read_text())
    assert "pexels:14" in bad                                     # corrupt file detected on download, never used
    for s in tl["scenes"]:
        assert s["clips"], f"scene {s['id']} has no clips: {s['error']}"
        assert abs(sum(c["duration"] for c in s["clips"]) - s["duration"]) < 0.002


def test_long_sentences_use_multiple_clips(ctx):
    long_scenes = [s for s in ctx["tl"]["scenes"] if s["duration"] > 9]
    assert long_scenes, "fixture should contain a long sentence"
    assert all(len(s["clips"]) >= 2 for s in long_scenes)
    assert all("looped" not in s["flags"] for s in ctx["tl"]["scenes"]), [(s["id"], s["flags"]) for s in ctx["tl"]["scenes"]]


def test_api_budget_is_respected(ctx):
    calls = ctx["mock"].requests["search"]
    assert calls <= len(ctx["tl"]["scenes"]) * ctx["project"].settings["max_api_calls_per_scene"]
    print("search calls for", len(ctx["tl"]["scenes"]), "scenes:", calls, "downloads:", ctx["mock"].requests["download"])


# ----------------------------------------------------------------------------- rendering + sync
@pytest.fixture(scope="module")
def final(ctx):
    p = ctx["project"]
    p.update_settings({"crossfade": 0.35})
    return builder.export_project(p, with_narration=True)


def test_video_duration_equals_narration(ctx, final):
    rep = final["report"]
    assert rep["frames_actual"] == rep["frames_expected"] == round(ctx["dur"] * FPS)
    vid = probe(final["video_only"])
    assert (vid["width"], vid["height"]) == (1920, 1080) and round(vid["fps"]) == 30
    assert abs(vid["duration"] - ctx["dur"]) <= 1 / FPS
    aud = audio_stream_duration(final["with_narration"])
    assert abs(aud - ctx["dur"]) < 0.05
    assert abs(probe(final["with_narration"])["duration"] - ctx["dur"]) <= 1 / FPS + 0.03
    assert not probe(final["video_only"])["has_audio"] and probe(final["with_narration"])["has_audio"]


def test_every_scene_and_clip_lands_on_its_timestamp(ctx, final):
    tl = ctx["project"].load_timeline()
    cols = frame_colors(final["video_only"])
    plan = frame_plan(tl, FPS)
    hf = 5   # crossfade 0.35s -> 5 frames each side of an in-scene cut
    rows, checked = [], 0
    for item in plan:
        sc = item["scene"]
        bounds = item["clip_bounds"]
        for k, clip in enumerate(sc["clips"]):
            lo, hi = bounds[k], bounds[k + 1]
            lo_safe = lo + (hf + 1 if k > 0 else 0)
            hi_safe = hi - (hf + 1 if k < len(sc["clips"]) - 1 else 0)
            want = int(clip["media_id"])
            for f in range(lo_safe, hi_safe):
                got = nearest(cols[f])
                assert got == want, f"scene {sc['id']} clip {k} frame {f}: expected clip {want}, got {got}"
                checked += 1
        first_clip_id = int(sc["clips"][0]["media_id"])
        rows.append((sc["id"], sc["start"], item["f0"] / FPS, sc["end"], item["f1"] / FPS, len(sc["clips"])))
    assert checked > 0.8 * len(cols)
    print("\nscene  timeline_start  video_start  timeline_end  video_end  clips")
    for r in rows:
        print(f"{r[0]:>5}  {r[1]:>14.3f}  {r[2]:>11.3f}  {r[3]:>12.3f}  {r[4]:>9.3f}  {r[5]}")
    print(f"frames verified against expected clip colour: {checked}/{len(cols)}")


def test_hard_cuts_at_scene_boundaries_are_exact(ctx, final):
    tl = ctx["project"].load_timeline()
    cols = frame_colors(final["video_only"])
    for item in frame_plan(tl, FPS)[1:]:
        f0 = item["f0"]
        prev_id = nearest(cols[f0 - 1]); new_id = nearest(cols[f0])
        want_new = int(item["scene"]["clips"][0]["media_id"])
        assert new_id == want_new, (item["scene"]["id"], new_id, want_new)


def test_srt_matches_timeline(ctx, final):
    srt = Path(final["srt"]).read_text(encoding="utf-8")
    cues = re.findall(r"(\d+)\n(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})\n", srt)
    tl = ctx["project"].load_timeline()
    assert len(cues) == len(tl["scenes"])
    def sec(t): h, m, rest = t.split(":"); s, ms = rest.split(","); return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    for (n, a, b), s in zip(cues, tl["scenes"]):
        assert abs(sec(a) - s["start"]) < 0.001 and abs(sec(b) - s["end"]) < 0.001
    assert (Path(final["srt"]).parent / "credits.txt").read_text().count("pexels") >= 1


# ----------------------------------------------------------------------------- resume / cache
def test_resume_does_not_redownload_or_research(ctx, final):
    mock = ctx["mock"]
    before = dict(mock.requests)
    p2 = Project.open("plants_doc")                       # "reopen the app"
    tl = p2.load_timeline()
    assert validate_timeline(tl) == [] and all(s["clips"] for s in tl["scenes"])
    out = builder.build_visuals(p2, force=True)           # rebuild EVERYTHING
    assert mock.requests == before, (before, mock.requests)   # zero new searches, zero new downloads
    assert out["cache"]["search_hits"] > 0 and out["cache"]["downloads"] == 0


# ----------------------------------------------------------------------------- Search Again / Replace
def test_search_again_gives_different_footage(ctx, final):
    p = ctx["project"]
    tl = p.load_timeline()
    sid = 2
    before = {c["media_id"] for c in tl["scenes"][sid - 1]["clips"]}
    builder.search_again(p, sid)
    sc = p.load_timeline()["scenes"][sid - 1]
    after = {c["media_id"] for c in sc["clips"]}
    assert sc["query_round"] == 1 and before <= {r.split(":")[1] for r in sc["rejected"]}
    assert not before & after, (before, after)
    assert validate_timeline(p.load_timeline()) == []


def test_replace_clip_with_local_media_keeps_timing(ctx, final):
    p = ctx["project"]
    mine = make_color_clip(ctx["root"] / "mine.mp4", "0x123456", 30, 1920, 1080, 30)
    sid = 4
    n_before = len(p.load_timeline()["scenes"][sid - 1]["clips"])
    builder.replace_with_local(p, sid, 0, str(mine))
    tl = p.load_timeline()
    sc = tl["scenes"][sid - 1]
    assert sc["clips"][0]["source"] == "local" and sc["clips"][0]["pinned"] and len(sc["clips"]) == n_before
    assert validate_timeline(tl) == []
    # pinned scenes are protected from a normal (non-forced) Build
    builder.build_visuals(p)
    assert p.load_timeline()["scenes"][sid - 1]["clips"][0]["source"] == "local"
    rep = render_timeline(p, tl, mode="preview")
    assert rep["frames_actual"] == rep["frames_expected"]


def test_too_short_local_video_is_looped_and_flagged_but_timing_holds(ctx, final):
    p = ctx["project"]
    short = make_color_clip(ctx["root"] / "short.mp4", "0x654321", 1.2, 1920, 1080, 30)
    sid = 6
    builder.replace_with_local(p, sid, None, str(short))
    tl = p.load_timeline(); sc = tl["scenes"][sid - 1]
    assert sc["clips"][0]["loop"] and "looped" in sc["flags"]
    rep = render_timeline(p, tl, mode="preview")
    assert rep["frames_actual"] == rep["frames_expected"]


def test_bad_local_file_is_rejected_cleanly(ctx, final):
    p = ctx["project"]
    junk = ctx["root"] / "junk.mp4"; junk.write_bytes(b"not a video at all")
    with pytest.raises(builder.BuildError):
        builder.replace_with_local(p, 3, 0, str(junk))
    assert validate_timeline(p.load_timeline()) == []


# ----------------------------------------------------------------------------- failure handling
def test_missing_media_file_renders_placeholder_not_crash(ctx, final):
    p = ctx["project"]
    tl = p.load_timeline()
    victim = tl["scenes"][-1]                             # last scene (3 clips)
    for c in victim["clips"]:
        Path(c["local_path"]).rename(Path(c["local_path"]).with_suffix(".gone"))
    try:
        rep = render_timeline(p, tl, mode="preview")
        assert rep["frames_actual"] == rep["frames_expected"]           # whole project still renders, exact length
        assert [f["scene"] for f in rep["failed_scenes"]] == [victim["id"]]
        assert victim["status"] == "needs_attention" and "render_failed" in victim["flags"]
    finally:
        for c in victim["clips"]:
            g = Path(c["local_path"]).with_suffix(".gone")
            if g.exists(): g.rename(c["local_path"])


def test_rate_limit_pauses_and_keeps_progress(ctx, final):
    p = Project.create("ratelimit")
    p.save_script("Cactus flowers bloom in the desert after rain. Baobab trees store thousands of litres of water.")
    p.update_settings({"align_backend": "estimate"})
    builder.align_project(p)
    ctx["mock"].fail_mode = "429"
    try:
        out = builder.build_visuals(p)
    finally:
        ctx["mock"].fail_mode = None
    assert "rate limit" in out["paused"].lower() and out["pending"] == 2
    assert [s["status"] for s in p.load_timeline()["scenes"]] == ["pending", "pending"]
    out = builder.build_visuals(p)                        # resume after the limit resets
    assert out["paused"] == "" and out["built"] == 2


def test_no_api_key_gives_clear_error(ctx, final):
    key = os.environ.pop("PEXELS_API_KEY")
    try:
        with pytest.raises(builder.BuildError, match="PEXELS_API_KEY"):
            builder.build_visuals(ctx["project"])
    finally:
        os.environ["PEXELS_API_KEY"] = key


def test_missing_narration_falls_back_to_flagged_estimate(ctx, final):
    p = Project.create("nonarr")
    p.save_script("First sentence here. Second sentence follows it.")
    p.update_settings({"align_backend": "auto"})
    out = builder.align_project(p)
    assert out["backend"] == "estimate" and any("ESTIMATED" in w for w in out["warnings"])
    assert {s["timing_source"] for s in p.load_timeline()["scenes"]} == {"estimate"}
    with pytest.raises(builder.BuildError, match="[Nn]arration"):
        builder.export_project(p, with_narration=True)
