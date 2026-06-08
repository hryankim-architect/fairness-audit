"""fairness-audit — group-fairness metrics + mitigation trade-offs, honestly.

Clean-room / synthetic. A tunable mock cohort carries a known disparity between two
groups; the audit reports **multiple** fairness metrics (they can conflict —
demographic parity and equalized odds cannot both be satisfied with unequal base
rates), then compares **three mitigations** so the metric conflict is explicit:
demographic-parity thresholds, equal-opportunity thresholds, and per-group
calibration. On a base-rate disparity the two threshold rules conflict (each zeroes
its own gap, widens the other) and parity costs accuracy; on a measurement-bias
disparity calibration fixes fairness and accuracy together. The deliverable is the
methodology + the honest framing, not a single "fair" number.
"""
from __future__ import annotations

# Fairness metric keys reported per run.
METRICS = ("demographic_parity_diff", "equalized_odds_gap", "accuracy")

__all__ = ["METRICS"]
