"""fairness-audit — group-fairness metrics + mitigation trade-off, honestly.

Clean-room / synthetic. A tunable mock cohort carries an injected disparity between
two groups; the audit reports **multiple** fairness metrics (they can conflict —
demographic parity and equalized odds cannot both be perfectly satisfied with
unequal base rates), then a group-threshold mitigation that **shrinks the
demographic-parity gap at an accuracy cost**, reported as a trade-off rather than a
free fix. The deliverable is the methodology + the honest framing.
"""
from __future__ import annotations

# Fairness metric keys reported per run.
METRICS = ("demographic_parity_diff", "equalized_odds_gap", "accuracy")

__all__ = ["METRICS"]
