from fairaudit.cohort import generate_cohort
from fairaudit.metrics import fairness_summary, predictions
from fairaudit.mitigate import (
    calibrate_by_group,
    group_thresholds_for_equal_opportunity,
    group_thresholds_for_parity,
)


def _pre_post(**kw):
    recs = generate_cohort(400, seed="v0.1", **kw)
    pre = fairness_summary(recs, predictions(recs, 0.5))
    target = sum(predictions(recs, 0.5)) / len(recs)
    thr = group_thresholds_for_parity(recs, target)
    post = fairness_summary(recs, predictions(recs, thr))
    return pre, post, thr


def _overall_tpr(recs, preds):
    pos = [i for i, r in enumerate(recs) if r.y_true == 1]
    return sum(preds[i] for i in pos) / len(pos)


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


# ---- v0.2: equal-opportunity target, calibration, and the DP<->EO conflict ----

def test_equal_opportunity_thresholds_keep_tpr_gap_small():
    recs = generate_cohort(400, rate_gap=0.30, score_bias=0.0, seed="v0.1")
    g = predictions(recs, 0.5)
    thr = group_thresholds_for_equal_opportunity(recs, _overall_tpr(recs, g))
    post = fairness_summary(recs, predictions(recs, thr))
    assert post["tpr_gap"] < 0.06  # equalized opportunity achieved


def test_dp_and_eo_thresholds_conflict():
    # On one cohort: DP-thresholds zero the DP gap; EO-thresholds leave it large.
    # No single threshold rule gets BOTH the DP gap and the TPR gap to ~0.
    recs = generate_cohort(400, rate_gap=0.30, score_bias=0.0, seed="v0.1")
    g = predictions(recs, 0.5)
    dp_thr = group_thresholds_for_parity(recs, sum(g) / len(g))
    eo_thr = group_thresholds_for_equal_opportunity(recs, _overall_tpr(recs, g))
    dp = fairness_summary(recs, predictions(recs, dp_thr))
    eo = fairness_summary(recs, predictions(recs, eo_thr))
    assert dp["demographic_parity_diff"] < 0.06   # DP rule fixes DP
    assert dp["tpr_gap"] > 0.10                    # ...but breaks TPR equality
    assert eo["tpr_gap"] < 0.06                    # EO rule fixes TPR equality
    assert eo["demographic_parity_diff"] > 0.10    # ...but leaves DP gap


def test_calibration_fixes_measurement_bias_no_tradeoff():
    # Equal base rates + a per-group score offset: calibration removes the offset,
    # improving fairness AND accuracy at once (unlike the base-rate trade-off).
    recs = generate_cohort(400, rate_gap=0.0, score_bias=0.20, seed="v0.1")
    biased = fairness_summary(recs, predictions(recs, 0.5))
    cal = calibrate_by_group(recs)
    fixed = fairness_summary(cal, predictions(cal, 0.5))
    assert fixed["demographic_parity_diff"] < biased["demographic_parity_diff"]
    assert fixed["equalized_odds_gap"] < biased["equalized_odds_gap"]
    assert fixed["accuracy"] >= biased["accuracy"]  # no accuracy cost


def test_calibrate_preserves_labels_and_groups():
    recs = generate_cohort(50, score_bias=0.20, seed="v0.1")
    cal = calibrate_by_group(recs)
    assert len(cal) == len(recs)
    assert [r.y_true for r in cal] == [r.y_true for r in recs]
    assert [r.group for r in cal] == [r.group for r in recs]
    assert all(0.0 <= r.score <= 1.0 for r in cal)


def test_calibrate_empty():
    assert calibrate_by_group([]) == []
