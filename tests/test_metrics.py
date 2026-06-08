from fairaudit.cohort import Record
from fairaudit.metrics import fairness_summary, group_rates, predictions


def _toy():
    # Group A: 2 selected of 2; Group B: 0 selected of 2 -> max DP gap.
    return [
        Record("a1", "A", 1, 0.9),
        Record("a2", "A", 0, 0.8),
        Record("b1", "B", 1, 0.2),
        Record("b2", "B", 0, 0.1),
    ]


def test_predictions_global_threshold():
    recs = _toy()
    assert predictions(recs, 0.5) == [1, 1, 0, 0]


def test_predictions_per_group_thresholds():
    recs = _toy()
    preds = predictions(recs, {"A": 0.85, "B": 0.15})
    assert preds == [1, 0, 1, 0]


def test_group_rates_selection_and_tpr():
    recs = _toy()
    gr = group_rates(recs, predictions(recs, 0.5))
    assert gr["A"]["selection_rate"] == 1.0
    assert gr["B"]["selection_rate"] == 0.0
    assert gr["A"]["tpr"] == 1.0  # the one A positive is selected
    assert gr["B"]["tpr"] == 0.0  # the one B positive is missed


def test_fairness_summary_gaps():
    recs = _toy()
    summ = fairness_summary(recs, predictions(recs, 0.5))
    assert summ["demographic_parity_diff"] == 1.0
    assert summ["tpr_gap"] == 1.0
    assert summ["fpr_gap"] == 1.0
    assert summ["equalized_odds_gap"] == 1.0
    # acc: a1 correct(1==1), a2 wrong(1 vs 0), b1 wrong(0 vs 1), b2 correct -> 0.5
    assert summ["accuracy"] == 0.5


def test_missing_group_rate_is_none_not_crash():
    # A group with no negatives -> fpr undefined (None), summary stays computable.
    recs = [Record("a1", "A", 1, 0.9), Record("b1", "B", 1, 0.2)]
    summ = fairness_summary(recs, predictions(recs, 0.5))
    assert summ["fpr_gap"] is None
    assert summ["tpr_gap"] == 1.0
    assert summ["equalized_odds_gap"] == 1.0  # max of available candidates
