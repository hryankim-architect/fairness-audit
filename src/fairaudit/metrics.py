"""Group-fairness metrics from predictions on a cohort.

Reports several metrics because they can conflict: **demographic parity**
(equal selection rates) and **equalized odds** (equal TPR and FPR) cannot both
hold in general when error trade-offs differ — the audit shows both rather than
collapsing to one number.
"""
from __future__ import annotations

from fairaudit.cohort import Record


def predictions(records: list[Record], thresholds: float | dict[str, float]) -> list[int]:
    """Binarize scores at a global threshold (float) or per-group thresholds (dict)."""
    def thr(g: str) -> float:
        return thresholds if isinstance(thresholds, int | float) else thresholds[g]
    return [1 if r.score >= thr(r.group) else 0 for r in records]


def _rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


def group_rates(records: list[Record], preds: list[int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in sorted({r.group for r in records}):
        idx = [i for i, r in enumerate(records) if r.group == g]
        pos = [i for i in idx if records[i].y_true == 1]
        neg = [i for i in idx if records[i].y_true == 0]
        out[g] = {
            "n": len(idx),
            "selection_rate": _rate(sum(preds[i] for i in idx), len(idx)),
            "tpr": _rate(sum(preds[i] for i in pos), len(pos)),
            "fpr": _rate(sum(preds[i] for i in neg), len(neg)),
            "accuracy": _rate(sum(1 for i in idx if preds[i] == records[i].y_true), len(idx)),
        }
    return out


def fairness_summary(records: list[Record], preds: list[int]) -> dict:
    gr = group_rates(records, preds)
    groups = sorted(gr)
    a, b = groups[0], groups[-1]

    def gap(key: str) -> float | None:
        va, vb = gr[a][key], gr[b][key]
        return abs(va - vb) if (va is not None and vb is not None) else None

    tpr_gap, fpr_gap = gap("tpr"), gap("fpr")
    eo_candidates = [x for x in (tpr_gap, fpr_gap) if x is not None]
    overall_acc = _rate(sum(1 for i, r in enumerate(records) if preds[i] == r.y_true), len(records))
    return {
        "groups": groups,
        "per_group": gr,
        "demographic_parity_diff": gap("selection_rate"),
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "equalized_odds_gap": max(eo_candidates) if eo_candidates else None,
        "accuracy": overall_acc,
    }
