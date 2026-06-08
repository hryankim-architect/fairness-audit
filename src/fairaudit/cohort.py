"""Synthetic cohort with a tunable group difference (clean-room, deterministic).

The two groups A and B differ in **outcome base rate** by ``rate_gap`` (group A
higher). The mock model is otherwise *calibrated* — it scores positives and
negatives the same way in both groups — so at a single shared threshold the model
roughly equalizes TPR/FPR but NOT selection rate. That is the canonical setup for
the fairness impossibility (Chouldechova 2017; Kleinberg et al. 2016): with unequal
base rates you cannot have demographic parity and equalized odds at once, and
forcing one costs the other (and accuracy).

An optional ``score_bias`` adds pure measurement bias against group B on top, for
experimenting with a different source of disparity. Everything is deterministic
given ``seed``. No real data; groups are abstract labels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    rid: str
    group: str      # "A" | "B"
    y_true: int     # 0 | 1
    score: float    # mock model risk score in [0, 1]


def _u(seed: str, *keys: str) -> float:
    h = hashlib.sha256("|".join((seed, *keys)).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0


def generate_cohort(
    n: int = 400,
    *,
    base_rate: float = 0.45,
    rate_gap: float = 0.30,
    score_bias: float = 0.0,
    noise: float = 0.20,
    seed: str = "v0.1",
) -> list[Record]:
    """Generate ``n`` records, groups balanced A/B.

    Group A base rate = ``base_rate + rate_gap/2``; group B = ``base_rate - rate_gap/2``.
    Scores are calibrated to ``y_true`` with ``noise``; ``score_bias`` (default 0)
    subtracts a constant from group B's scores to model measurement bias.
    """
    rate_a = base_rate + rate_gap / 2.0
    rate_b = base_rate - rate_gap / 2.0
    recs: list[Record] = []
    for i in range(n):
        g = "A" if i % 2 == 0 else "B"
        r = rate_a if g == "A" else rate_b
        y = 1 if _u(seed, str(i), "y") < r else 0
        s = (0.65 if y == 1 else 0.35) + (_u(seed, str(i), "n") - 0.5) * 2.0 * noise
        if g == "B":
            s -= score_bias
        recs.append(Record(f"r{i}", g, y, round(max(0.0, min(1.0, s)), 4)))
    return recs
