#!/usr/bin/env python3
"""End-to-end fairness audit on a synthetic cohort with a group base-rate gap.

The mock model is calibrated, so at a shared global threshold it roughly satisfies
equalized odds but NOT demographic parity (the base-rate gap shows up as a
selection-rate gap). The pipeline then: applies per-group thresholds equalizing
selection (post-mitigation) -> bootstraps a CI on the demographic-parity gap ->
writes both summaries to a hash-chained ledger -> prints pre/post side-by-side.
The point is the trade-off: forcing demographic parity shrinks the DP gap but
breaks equalized odds and costs accuracy. Deterministic; no real data.

Usage:
    python scripts/run_fairness_audit.py [--n 400] [--rate-gap 0.30] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairaudit import METRICS
from fairaudit.audit import DEFAULT_LEDGER, emit, verify
from fairaudit.bootstrap import bootstrap_metric
from fairaudit.cohort import generate_cohort
from fairaudit.metrics import fairness_summary, predictions
from fairaudit.mitigate import group_thresholds_for_parity

GLOBAL_THRESHOLD = 0.5


def _fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--base-rate", type=float, default=0.45)
    ap.add_argument("--rate-gap", type=float, default=0.30,
                    help="group A base rate minus group B base rate (the disparity source)")
    ap.add_argument("--score-bias", type=float, default=0.0,
                    help="extra measurement bias subtracted from group B scores (default 0)")
    ap.add_argument("--noise", type=float, default=0.20)
    ap.add_argument("--seed", default="v0.1")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--reset-ledger", action="store_true",
                    help="truncate the ledger before this run (demo reproducibility)")
    args = ap.parse_args()

    if args.reset_ledger and args.ledger.exists():
        args.ledger.write_text("")  # truncate, not unlink (sandbox mount safe)

    cohort = generate_cohort(
        args.n, base_rate=args.base_rate, rate_gap=args.rate_gap,
        score_bias=args.score_bias, noise=args.noise, seed=args.seed,
    )

    # --- pre-mitigation: one shared threshold for everyone ---
    pre_preds = predictions(cohort, GLOBAL_THRESHOLD)
    pre = fairness_summary(cohort, pre_preds)

    # --- mitigation: per-group thresholds equalizing selection to the global rate ---
    target = sum(pre_preds) / len(pre_preds)
    thresholds = group_thresholds_for_parity(cohort, target)
    post_preds = predictions(cohort, thresholds)
    post = fairness_summary(cohort, post_preds)

    # --- uncertainty on the pre-mitigation demographic-parity gap ---
    boot = bootstrap_metric(
        cohort,
        lambda recs: fairness_summary(recs, predictions(recs, GLOBAL_THRESHOLD))[
            "demographic_parity_diff"],
        n_boot=args.n_boot, seed=0,
    )

    emit("fairness_pre", "fairness-audit",
         {"threshold": GLOBAL_THRESHOLD, "summary": pre}, ledger_path=args.ledger)
    emit("fairness_post", "fairness-audit",
         {"thresholds": thresholds, "target_rate": round(target, 4), "summary": post},
         ledger_path=args.ledger)
    ok, n = verify(args.ledger)

    print("=" * 66)
    print("fairness-audit — synthetic cohort, group base-rate gap (calibrated model)")
    print(f"  n={args.n}  base_rate={args.base_rate}  rate_gap={args.rate_gap}  "
          f"score_bias={args.score_bias}  seed={args.seed}")
    print("=" * 66)
    print(f"{'metric':<26}{'pre':>10}{'post':>10}   (lower gap = fairer)")
    for key in ("demographic_parity_diff", "tpr_gap", "fpr_gap",
                "equalized_odds_gap", "accuracy"):
        print(f"{key:<26}{_fmt(pre[key]):>10}{_fmt(post[key]):>10}")
    print("-" * 66)
    dp_cost = (pre["accuracy"] or 0) - (post["accuracy"] or 0)
    print(f"demographic-parity gap (pre): point={_fmt(boot['point'])}  "
          f"95% CI [{_fmt(boot['ci_low'])}, {_fmt(boot['ci_high'])}]  "
          f"(n_boot={boot['n_boot']})")
    print(f"mitigation shrinks DP gap {_fmt(pre['demographic_parity_diff'])} -> "
          f"{_fmt(post['demographic_parity_diff'])}, "
          f"accuracy cost {dp_cost:+.2f}, "
          f"equalized-odds gap {_fmt(pre['equalized_odds_gap'])} -> "
          f"{_fmt(post['equalized_odds_gap'])}")
    print(f"reported metrics: {', '.join(METRICS)}")
    print(f"ledger: {args.ledger}  chain_ok={ok}  entries={n}")
    print(json.dumps({"pre": pre["per_group"], "post": post["per_group"]}, indent=2))


if __name__ == "__main__":
    main()
