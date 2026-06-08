"""Percentile bootstrap confidence intervals (stdlib only, deterministic).

Resample the underlying units (probes/items) with replacement, recompute the
metric on each resample, and report the point estimate + a percentile CI. Used to
turn a single demo number into a reportable estimate with uncertainty.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any


def percentile_ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1-alpha) percentile interval from bootstrap replicates."""
    s = sorted(samples)
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"))
    lo_i = max(0, int((alpha / 2.0) * n))
    hi_i = min(n - 1, int((1.0 - alpha / 2.0) * n))
    return (s[lo_i], s[hi_i])


def bootstrap_metric(
    units: Sequence[Any],
    metric_fn: Callable[[Sequence[Any]], float | None],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap a scalar metric over ``units``.

    ``metric_fn`` maps a (resampled) unit list to a scalar (or None when undefined,
    e.g. empty resample); None replicates are dropped. Deterministic given ``seed``.
    """
    point = metric_fn(units)
    n = len(units)
    if n == 0:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot": 0, "alpha": alpha}
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        v = metric_fn(sample)
        if v is not None:
            reps.append(v)
    lo, hi = percentile_ci(reps, alpha) if reps else (None, None)
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": len(reps), "alpha": alpha}
