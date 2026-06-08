"""Post-processing mitigations — and what each one costs.

Three transparent interventions, so the audit can show that **the choice of fairness
criterion is itself a decision with consequences**:

- ``group_thresholds_for_parity`` equalizes *selection rate* (demographic parity).
- ``group_thresholds_for_equal_opportunity`` equalizes *true-positive rate*
  (equal opportunity, the TPR half of equalized odds).
- ``calibrate_by_group`` removes a per-group additive *score offset* (measurement
  bias) before a single shared threshold.

On a cohort whose disparity comes from a genuine base-rate gap, the two
threshold mitigations **conflict**: each zeroes its own gap and widens the other —
you cannot satisfy both. Calibration is the right tool for a *different* problem
(additive measurement bias); it does NOT resolve the base-rate impossibility. The
runner shows all of this side by side rather than picking a winner.
"""
from __future__ import annotations

from fairaudit.cohort import Record


def _groups(records: list[Record]) -> list[str]:
    return sorted({r.group for r in records})


def group_thresholds_for_parity(records: list[Record], target_rate: float) -> dict[str, float]:
    """Per-group threshold so ~``target_rate`` of each group is selected (score >= thr)."""
    out: dict[str, float] = {}
    for g in _groups(records):
        scores = sorted((r.score for r in records if r.group == g), reverse=True)
        n = len(scores)
        if n == 0:
            out[g] = 0.5
            continue
        k = max(1, min(n, round(target_rate * n)))
        out[g] = scores[k - 1]  # k-th highest score => ~k/n selected at ">="
    return out


def group_thresholds_for_equal_opportunity(
    records: list[Record], target_tpr: float
) -> dict[str, float]:
    """Per-group threshold so each group's TPR (recall on positives) ~= ``target_tpr``.

    Chosen over each group's POSITIVES only, so it constrains true-positive rate
    irrespective of how many negatives that threshold also admits.
    """
    out: dict[str, float] = {}
    for g in _groups(records):
        pos = sorted((r.score for r in records if r.group == g and r.y_true == 1), reverse=True)
        n = len(pos)
        if n == 0:
            out[g] = 0.5
            continue
        k = max(1, min(n, round(target_tpr * n)))
        out[g] = pos[k - 1]  # k-th highest positive score => TPR ~= k/n at ">="
    return out


def calibrate_by_group(records: list[Record]) -> list[Record]:
    """Remove each group's additive score offset (shift its mean to the pooled mean).

    Appropriate when the disparity is a per-group *measurement bias* (a constant shift
    in scores). It is NOT a fix for genuine base-rate differences — there, shifting a
    group's scores misrepresents its real outcome rate. Returns recalibrated records;
    the original is untouched.
    """
    if not records:
        return []
    pooled = sum(r.score for r in records) / len(records)
    means: dict[str, float] = {}
    for g in _groups(records):
        gs = [r.score for r in records if r.group == g]
        means[g] = sum(gs) / len(gs)
    out: list[Record] = []
    for r in records:
        s = r.score + (pooled - means[r.group])
        out.append(Record(r.rid, r.group, r.y_true, round(max(0.0, min(1.0, s)), 4)))
    return out
