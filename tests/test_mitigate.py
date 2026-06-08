from fairaudit.cohort import generate_cohort
from fairaudit.metrics import fairness_summary, predictions
from fairaudit.mitigate import group_thresholds_for_parity


def _pre_post(**kw):
    recs = generate_cohort(400, seed="v0.1", **kw)
    pre = fairness_summary(recs, predictions(recs, 0.5))
    target = sum(predictions(recs, 0.5)) / len(recs)
    thr = group_thresholds_for_parity(recs, target)
    post = fairness_summary(recs, predictions(recs, thr))
    return pre, post, thr


def test_group_thresholds_shrink_dp_gap():
    pre, post, _ = _pre_post(rate_gap=0.30, score_bias=0.0)
    assert post["demographic_parity_diff"] < pre["demographic_parity_diff"]
    assert post["demographic_parity_diff"] < 0.06  # near zero after equalizing selection


def test_thresholds_returned_per_group():
    _, _, thr = _pre_post(rate_gap=0.30, score_bias=0.0)
    assert set(thr) == {"A", "B"}
    # lower-base-rate group B must use a LOWER threshold to reach the same rate
    assert thr["B"] <= thr["A"]


def test_mitigation_has_accuracy_cost():
    # Forcing equal selection on groups with different base rates costs accuracy.
    pre, post, _ = _pre_post(rate_gap=0.30, score_bias=0.0)
    assert post["accuracy"] <= pre["accuracy"] + 1e-9


def test_forcing_parity_breaks_equalized_odds():
    # The impossibility, concretely: a calibrated model has a small EO gap; forcing
    # demographic parity drives the DP gap to ~0 but INCREASES the EO gap.
    pre, post, _ = _pre_post(rate_gap=0.30, score_bias=0.0)
    assert pre["equalized_odds_gap"] < 0.12
    assert post["equalized_odds_gap"] > pre["equalized_odds_gap"]
