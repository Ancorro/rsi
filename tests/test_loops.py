from wsp.loops import hit_limit, repeated_ngram_fraction


def test_no_repeats():
    assert repeated_ngram_fraction(list(range(100)), n=5) == 0.0


def test_loop_is_mostly_repeats():
    toks = list(range(10)) * 20
    assert repeated_ngram_fraction(toks, n=5) > 0.9


def test_short_sequence_is_zero():
    assert repeated_ngram_fraction([1, 2, 3], n=20) == 0.0


def test_hit_limit():
    assert hit_limit(100, 100, finished=False)
    assert not hit_limit(100, 100, finished=True)
    assert not hit_limit(40, 100, finished=False)
