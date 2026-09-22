import random
from core.align import (align_from_words, assign_pauses, finalize, align_from_estimate, tokens, MIN_SCENE,
                        keep_sentence_scale_pauses)
from core.script_split import split_sentences

SCRIPT = ("Have you ever noticed the leaves of your plant turning yellow? "
          "Yellow leaves do not always mean that your plant is dying. "
          "One of the most common causes is overwatering. "
          "Overwatering can prevent oxygen from reaching the roots and eventually cause the leaves to turn yellow. "
          "Plants absorb water through their roots. "
          "Let us look at how to fix it.")


def synth_words(sentences, gap=0.6, wdur=0.32, lead=0.4, mutate=None, seed=1):
    """Fake ASR: every word wdur long, `gap` silence between sentences. Returns (words, true_starts)."""
    rnd = random.Random(seed)
    t, words, true_starts = lead, [], []
    for si, s in enumerate(sentences):
        true_starts.append(t)
        for w in s.split():
            if mutate and rnd.random() < mutate:
                w = rnd.choice(["banana", "", "xyzzy"])  # mis-heard or dropped
            if w:
                words.append((w, t, t + wdur))
            t += wdur
        t += gap
    return words, true_starts, t


def test_finalize_is_contiguous_and_exact():
    s = ["a b", "c d", "e f"]
    out = finalize(s, [0, 8.45, 15.92], ["silence"] * 3, 24.6)
    assert [x["start"] for x in out] == [0.0, 8.45, 15.92]
    assert [x["end"] for x in out] == [8.45, 15.92, 24.6]
    assert out[0]["duration"] == 8.45 and out[1]["duration"] == 7.47 and out[2]["duration"] == 8.68
    for a, b in zip(out, out[1:]):
        assert a["end"] == b["start"]


def test_finalize_enforces_minimum_scene_and_bounds():
    out = finalize(["a"] * 4, [0, 1.0, 1.0, 99.0], ["silence"] * 4, 5.0)
    assert out[-1]["end"] == 5.0
    assert all(x["duration"] >= MIN_SCENE - 1e-9 for x in out)
    assert all(a["end"] == b["start"] for a, b in zip(out, out[1:]))


def test_whisper_tier_exact_when_transcript_is_perfect():
    sents = split_sentences(SCRIPT)
    words, true_starts, end = synth_words(sents)
    res = align_from_words(sents, words, end)
    assert res.match_ratio == 1.0
    for got, want in zip(res.sentences, true_starts):
        assert abs(got["start"] - want) < 0.011 or got["id"] == 1
    assert res.sentences[0]["start"] == 0.0
    assert res.sentences[-1]["end"] == round(end, 3)
    assert {s["timing_source"] for s in res.sentences} == {"whisper"}


def test_whisper_tier_tolerates_misheard_and_dropped_words():
    sents = split_sentences(SCRIPT)
    words, true_starts, end = synth_words(sents, mutate=0.15, seed=7)
    res = align_from_words(sents, words, end)
    assert 0.5 < res.match_ratio < 1.0
    for got, want in zip(res.sentences[1:], true_starts[1:]):
        # a word mis-heard at a sentence start still anchors by position; allow a couple of words
        assert abs(got["start"] - want) < 0.7, (got["id"], got["start"], want)
    assert all(a["end"] == b["start"] for a, b in zip(res.sentences, res.sentences[1:]))


def test_sentence_missing_from_transcript_is_interpolated_and_flagged():
    sents = split_sentences(SCRIPT)
    words, true_starts, end = synth_words(sents)
    # drop every word of sentence 3 (index 2)
    n3 = len(sents[2].split())
    first = sum(len(s.split()) for s in sents[:2])
    words = words[:first] + words[first + n3:]
    res = align_from_words(sents, words, end)
    assert res.sentences[2]["timing_source"] == "estimate"
    assert res.sentences[1]["timing_source"] == "estimate"  # bounded by an estimated boundary
    assert true_starts[1] - 0.05 < res.sentences[2]["start"] < true_starts[3]


def test_assign_pauses_prefers_sentence_gaps_over_comma_pauses():
    # 4 sentences -> 3 boundaries. Real gaps 0.7s, plus 0.25s comma pauses close to the estimates.
    pauses = [(4.0, 4.25), (9.8, 10.5), (14.9, 15.15), (19.6, 20.3), (25.0, 25.25), (29.8, 30.5)]
    expected = [10.0, 20.0, 30.0]
    assign = assign_pauses(expected, pauses)
    assert assign == [1, 3, 5]


def test_assign_pauses_falls_back_to_estimate_when_no_pause_is_near():
    assign = assign_pauses([10.0, 20.0, 30.0], [(10.1, 10.6)])
    assert assign.count(None) == 2 and 0 in assign


def test_estimate_tier_sums_to_duration():
    res = align_from_estimate(["one two three", "four five", "six"], 12.0)
    assert res.sentences[-1]["end"] == 12.0 and res.backend == "estimate"
    assert {s["timing_source"] for s in res.sentences} == {"estimate"}


def test_tokens_normalise_punctuation_and_apostrophes():
    assert tokens("Don't over-water, 2.5 cm!") == ["dont", "over", "water", "2", "5", "cm"]


def test_long_comma_heavy_sentence_does_not_pull_boundary_onto_a_comma():
    """Regression (found in e2e): the estimate for the boundary before a 55-word sentence was ~1.6s early,
    and a 0.21s comma pause beat the real 1.0s sentence gap."""
    pauses = [(3.7, 4.78), (8.13, 9.37), (12.17, 13.32), (19.64, 20.97), (23.18, 24.47),
              (26.97, 27.18), (28.61, 29.61), (32.27, 32.48), (39.8, 40.01), (44.18, 44.43)]
    expected = [4.3, 8.9, 12.9, 20.5, 24.0, 27.3]      # last one is early because the long sentence skews estimates
    kept = keep_sentence_scale_pauses(pauses, 6)
    assert (26.97, 27.18) not in kept and (28.61, 29.61) in kept
    got = assign_pauses(expected, kept)
    assert [kept[i][1] for i in got] == [4.78, 9.37, 13.32, 20.97, 24.47, 29.61]


def test_short_pause_filter_leaves_pauses_alone_when_there_are_too_few_long_ones():
    pauses = [(1.0, 1.5), (5.0, 5.2), (9.0, 9.25)]
    assert keep_sentence_scale_pauses(pauses, 3) == pauses
