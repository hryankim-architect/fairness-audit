"""Demographic-parity post-processing: per-group thresholds equalizing selection.

A simple, transparent mitigation: pick a per-group score threshold so each group's
selection rate matches a shared target. This shrinks the demographic-parity gap —
but it is NOT free: matching selection rates while groups are scored differently
generally costs overall accuracy and can shift the equalized-odds gap. The audit
reports both sides so the trade-off is explicit, not hidden.
"""
from __future__ import annotations

from fairaudit.cohort import Record


def group_thresholds_for_parity(records: list[Record], target_rate: float) -> dict[str, float]:
    """Per-group threshold so ~``target_rate`` of each group is selected (score >= thr)."""
    out: dict[str, float] = {}
    for g in sorted({r.group for r in records}):
        scores = sorted((r.score for r in records if r.group == g), reverse=True)
        n = len(scores)
        if n == 0:
            out[g] = 0.5
            continue
        k = max(1, min(n, round(target_rate * n)))
        out[g] = scores[k - 1]  # k-th highest score => ~k/n selected at ">="
    return out
