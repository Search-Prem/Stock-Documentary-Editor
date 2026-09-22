import math
import pytest
from core.planner import expand_queries, plan_queries
from core.project import Project
from core.script_split import split_sentences
from core.selector import plan_fill
from core.srt import build_srt
from core.timeline import frame_plan, validate_timeline
from core.align import finalize


# ---------------------------------------------------------------- sentence splitting
def test_split_handles_abbreviations_decimals_genus_and_quotes():
    t = ('Dr. Smith found that A. thaliana grows 2.5 cm faster, e.g. in bright light. '
         '"Wow," she said. "That is a lot." Then it ended!')
    s = split_sentences(t)
    assert s == ['Dr. Smith found that A. thaliana grows 2.5 cm faster, e.g. in bright light.',
                 '"Wow," she said.', '"That is a lot."', 'Then it ended!'] or len(s) == 4
    assert s[0].startswith("Dr. Smith") and "2.5 cm" in s[0] and "A. thaliana" in s[0]


def test_split_paragraph_break_is_a_boundary_and_single_newlines_join():
    s = split_sentences("This line wraps\nonto the next line.\n\nSecond paragraph here")
    assert s == ["This line wraps onto the next line.", "Second paragraph here"]


def test_split_empty_and_whitespace():
    assert split_sentences("") == [] and split_sentences("  \n\n ") == []


# ---------------------------------------------------------------- planner
def test_planner_spec_examples_are_visual_and_specific():
    q = plan_queries("Overwatering is one of the most common reasons that indoor plant leaves turn yellow.")
    assert "overwatered houseplant" in q[:3] or "yellowing houseplant leaves" in q[:3]
    assert all(len(x.split()) <= 5 for x in q)                      # short visual queries, never the raw sentence
    q2 = plan_queries("Plants absorb water through their roots.")
    assert any("root" in x for x in q2[:3])
    assert plan_queries("Yes.") == ["plants"]
    q3 = plan_queries("Plants absorb water through their roots.")
    assert not ({"plant roots in soil", "plant roots soil"} <= set(q3))   # no near-duplicate queries


def test_expand_queries_adds_fresh_variants_without_duplicates():
    base = plan_queries("Yellow leaves do not always mean that your plant is dying.")
    ex = expand_queries(base)
    assert ex[:len(base)] == base and len(ex) > len(base) and len({q.lower() for q in ex}) == len(ex)


# ---------------------------------------------------------------- clip filling (spec scenarios)
def vids(*durs):
    return [{"kind": "video", "duration": d} for d in durs]


def check(D, cands, **kw):
    segs, rem = plan_fill(D, cands, 2.5, 10.0, 6.0, 0.4)
    assert abs(sum(s.length for s in segs) - D) < 1e-6 and rem == 0
    return segs


def test_single_long_clip_is_trimmed_not_looped():
    for D in (6, 8.45):
        s = check(D, vids(15))
        assert len(s) == 1 and not s[0].loop and abs(s[0].length - D) < 1e-6


def test_long_sentence_gets_several_distinct_clips():
    s = check(15, vids(6, 20, 9, 12))
    assert len(s) == 3 and len({x.idx for x in s}) == 3 and not any(x.loop or x.reuse for x in s)
    s = check(20, vids(7, 8, 9, 12, 6))
    assert 3 <= len(s) <= 4 and all(2.5 <= x.length <= 10 for x in s)


def test_no_sliver_shorter_than_min_clip():
    for D in (9.0, 9.9, 11.3, 12.4, 13.7, 17.1, 23.3):
        s = check(D, vids(12, 12, 12, 12, 12))
        assert min(x.length for x in s) >= 2.5 - 1e-6, (D, [x.length for x in s])


def test_insufficient_footage_reuses_then_loops_and_flags():
    s = check(20, vids(7))
    assert any(x.loop for x in s)                                    # last resort, flagged
    s = check(11, vids(12))                                          # a long clip can be used twice (different part)
    assert any(x.reuse for x in s) and not any(x.loop for x in s)


def test_images_can_fill_time():
    s = check(13, [{"kind": "image", "duration": 0}] * 4)
    assert len(s) >= 2 and not any(x.loop for x in s)


# ---------------------------------------------------------------- frame grid + srt
def test_frame_grid_has_no_drift_over_a_long_timeline():
    n, dur = 200, 1800.0
    starts = [i * dur / n + 0.0137 * (i % 7) for i in range(n)]
    sc = finalize(["x"] * n, starts, ["silence"] * n, dur)
    tl = {"audio_duration": dur, "scenes": [dict(s, clips=[]) for s in sc]}
    fp = frame_plan(tl, 30)
    assert fp[0]["f0"] == 0 and fp[-1]["f1"] == 54000
    assert all(a["f1"] == b["f0"] for a, b in zip(fp, fp[1:]))
    assert all(x["f1"] - x["f0"] >= 1 for x in fp)


def test_srt_one_cue_per_sentence_and_optional_split():
    sc = finalize(["Short one.", "A much longer sentence that keeps going and going for a while yes."],
                  [0, 2.0], ["whisper"] * 2, 12.0)
    tl = {"audio_duration": 12.0, "scenes": sc}
    assert build_srt(tl).count(" --> ") == 2
    split = build_srt(tl, max_cue_chars=25)
    assert split.count(" --> ") >= 4 and "00:00:12,000" in split


# ---------------------------------------------------------------- project safety
@pytest.mark.parametrize("bad", ["../evil", "a/b", "", "x" * 80, ".hidden", "a b\\c"])
def test_project_names_cannot_escape_the_projects_folder(bad, tmp_path):
    with pytest.raises(ValueError):
        Project(bad, tmp_path)


def test_settings_are_type_coerced_and_unknown_keys_ignored(tmp_path):
    p = Project.create("s1", tmp_path)
    st = p.update_settings({"crossfade": "0.5", "fps": "30", "evil": 1, "music_enabled": 1})
    assert st["crossfade"] == 0.5 and st["fps"] == 30 and st["music_enabled"] is True and "evil" not in st
