from fairaudit.cohort import generate_cohort
from fairaudit.metrics import fairness_summary, predictions


def test_deterministic_given_seed():
    a = generate_cohort(200, seed="x")
    b = generate_cohort(200, seed="x")
    assert [r.score for r in a] == [r.score for r in b]
    assert [r.rid for r in a] == [r.rid for r in b]


def test_groups_balanced_and_scores_in_range():
    recs = generate_cohort(400, seed="v0.1")
    groups = [r.group for r in recs]
    assert groups.count("A") == groups.count("B") == 200
    assert all(0.0 <= r.score <= 1.0 for r in recs)
    assert all(r.y_true in (0, 1) for r in recs)


def test_base_rate_gap_creates_dp_gap():
    # Higher-base-rate group A is selected more often at a shared threshold,
    # so a positive base-rate gap yields a positive demographic-parity gap.
    recs = generate_cohort(400, rate_gap=0.30, score_bias=0.0, seed="v0.1")
    summ = fairness_summary(recs, predictions(recs, 0.5))
    assert summ["per_group"]["A"]["selection_rate"] > summ["per_group"]["B"]["selection_rate"]
    assert summ["demographic_parity_diff"] > 0.05


def test_calibrated_model_has_small_eo_gap_pre():
    # With score_bias=0 the model is calibrated, so at a shared threshold the
    # equalized-odds gap is small EVEN THOUGH demographic parity is violated.
    recs = generate_cohort(400, rate_gap=0.30, score_bias=0.0, seed="v0.1")
    summ = fairness_summary(recs, predictions(recs, 0.5))
    assert summ["equalized_odds_gap"] < 0.12
    assert summ["demographic_parity_diff"] > 0.12  # the two metrics disagree


def test_no_gap_means_small_dp():
    # Equal base rates and no measurement bias -> near-zero disparity (control).
    recs = generate_cohort(400, rate_gap=0.0, score_bias=0.0, seed="v0.1")
    summ = fairness_summary(recs, predictions(recs, 0.5))
    assert summ["demographic_parity_diff"] < 0.10
    assert summ["equalized_odds_gap"] < 0.10
