from fairaudit.bootstrap import bootstrap_metric, percentile_ci


def test_percentile_ci_basic():
    lo, hi = percentile_ci(list(range(100)), alpha=0.10)
    assert lo <= hi
    assert 0 <= lo <= 10
    assert 89 <= hi <= 99


def test_percentile_ci_empty():
    lo, hi = percentile_ci([])
    assert lo != lo and hi != hi  # nan, nan


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def test_bootstrap_is_deterministic():
    units = list(range(50))
    a = bootstrap_metric(units, _mean, n_boot=200, seed=0)
    b = bootstrap_metric(units, _mean, n_boot=200, seed=0)
    assert a == b
    assert a["ci_low"] <= a["point"] <= a["ci_high"]


def test_bootstrap_empty_units():
    out = bootstrap_metric([], lambda xs: None, n_boot=10)
    assert out["n_boot"] == 0
    assert out["ci_low"] is None
