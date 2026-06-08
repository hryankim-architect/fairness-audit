#!/usr/bin/env python3
"""End-to-end fairness audit showing that the *choice of criterion* has consequences.

Two clearly-labelled scenarios on synthetic data (no real data):

1. BASE-RATE DISPARITY (calibrated model, groups differ in outcome rate). At a shared
   threshold the model satisfies equalized odds but not demographic parity. We then
   apply two mitigations on the SAME cohort:
     - demographic-parity thresholds  -> DP gap ~0, but equalized-odds gap blows up;
     - equal-opportunity thresholds    -> TPR gap ~0, but the DP gap persists.
   They conflict: you cannot have both (Chouldechova 2017; Kleinberg et al. 2016).

2. MEASUREMENT BIAS (equal base rates, a per-group score offset). Here per-group
   calibration removes the offset and improves fairness AND accuracy at once — the
   right tool for a different problem, and NOT a fix for scenario 1.

Bootstrap CIs accompany the headline gaps. Deterministic; results go to a
hash-chained ledger.

Usage:
    python scripts/run_fairness_audit.py [--n 400] [--rate-gap 0.30] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fairaudit.audit import DEFAULT_LEDGER, emit, verify
from fairaudit.bootstrap import bootstrap_metric
from fairaudit.cohort import generate_cohort
from fairaudit.metrics import fairness_summary, predictions
from fairaudit.mitigate import (
    calibrate_by_group,
    group_thresholds_for_equal_opportunity,
    group_thresholds_for_parity,
)

GLOBAL_THRESHOLD = 0.5


def _fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


def _overall_tpr(records, preds) -> float:
    pos = [i for i, r in enumerate(records) if r.y_true == 1]
    return sum(preds[i] for i in pos) / len(pos) if pos else 0.0


def _dp_after_parity(recs):
    g = predictions(recs, GLOBAL_THRESHOLD)
    thr = group_thresholds_for_parity(recs, sum(g) / len(g))
    return fairness_summary(recs, predictions(recs, thr))["demographic_parity_diff"]


def _eo_after_parity(recs):
    g = predictions(recs, GLOBAL_THRESHOLD)
    thr = group_thresholds_for_parity(recs, sum(g) / len(g))
    return fairness_summary(recs, predictions(recs, thr))["equalized_odds_gap"]


def _table(title, columns):
    print(title)
    print(f"{'metric':<26}" + "".join(f"{name:>14}" for name, _ in columns))
    for key in ("demographic_parity_diff", "tpr_gap", "fpr_gap", "equalized_odds_gap", "accuracy"):
        print(f"{key:<26}" + "".join(f"{_fmt(s[key]):>14}" for _, s in columns))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--base-rate", type=float, default=0.45)
    ap.add_argument("--rate-gap", type=float, default=0.30,
                    help="group A base rate minus group B base rate (scenario 1 disparity)")
    ap.add_argument("--score-bias", type=float, default=0.0,
                    help="measurement bias on the MAIN cohort (default 0; scenario 2 uses its own)")
    ap.add_argument("--cal-score-bias", type=float, default=0.20,
                    help="per-group score offset for the calibration scenario")
    ap.add_argument("--noise", type=float, default=0.20)
    ap.add_argument("--seed", default="v0.1")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--reset-ledger", action="store_true")
    args = ap.parse_args()

    if args.reset_ledger and args.ledger.exists():
        args.ledger.write_text("")  # truncate, not unlink (sandbox mount safe)

    # ---------- Scenario 1: base-rate disparity, DP vs EO conflict ----------
    cohort = generate_cohort(args.n, base_rate=args.base_rate, rate_gap=args.rate_gap,
                             score_bias=args.score_bias, noise=args.noise, seed=args.seed)
    g = predictions(cohort, GLOBAL_THRESHOLD)
    base = fairness_summary(cohort, g)
    dp_thr = group_thresholds_for_parity(cohort, sum(g) / len(g))
    eo_thr = group_thresholds_for_equal_opportunity(cohort, _overall_tpr(cohort, g))
    dp_summ = fairness_summary(cohort, predictions(cohort, dp_thr))
    eo_summ = fairness_summary(cohort, predictions(cohort, eo_thr))

    print("=" * 68)
    print("fairness-audit v0.2 — synthetic, no real data")
    print(f"  n={args.n}  base_rate={args.base_rate}  rate_gap={args.rate_gap}  "
          f"score_bias={args.score_bias}  seed={args.seed}")
    print("=" * 68)
    print("[Scenario 1] base-rate disparity — demographic parity vs equal opportunity")
    _table("", [("baseline", base), ("DP-thresh", dp_summ), ("EO-thresh", eo_summ)])

    boot_dp = bootstrap_metric(
        cohort, lambda r: fairness_summary(r, predictions(r, GLOBAL_THRESHOLD))[
            "demographic_parity_diff"], n_boot=args.n_boot, seed=0)
    boot_eo_cost = bootstrap_metric(cohort, _eo_after_parity, n_boot=args.n_boot, seed=0)
    print("-" * 68)
    print(f"baseline DP gap   {_fmt(boot_dp['point'])}  95% CI "
          f"[{_fmt(boot_dp['ci_low'])}, {_fmt(boot_dp['ci_high'])}]")
    print(f"EO gap AFTER forcing parity   {_fmt(boot_eo_cost['point'])}  95% CI "
          f"[{_fmt(boot_eo_cost['ci_low'])}, {_fmt(boot_eo_cost['ci_high'])}]  "
          f"(was {_fmt(base['equalized_odds_gap'])} pre)")
    print(f"=> DP-thresholds: DP {_fmt(base['demographic_parity_diff'])}->"
          f"{_fmt(dp_summ['demographic_parity_diff'])} but EO "
          f"{_fmt(base['equalized_odds_gap'])}->{_fmt(dp_summ['equalized_odds_gap'])}, "
          f"acc {_fmt(base['accuracy'])}->{_fmt(dp_summ['accuracy'])}")
    print(f"=> EO-thresholds: TPRgap {_fmt(base['tpr_gap'])}->{_fmt(eo_summ['tpr_gap'])} but DP "
          f"{_fmt(base['demographic_parity_diff'])}->"
          f"{_fmt(eo_summ['demographic_parity_diff'])}  (the two cannot both be ~0)")

    # ---------- Scenario 2: measurement bias, per-group calibration ----------
    cal_cohort = generate_cohort(args.n, base_rate=0.45, rate_gap=0.0,
                                 score_bias=args.cal_score_bias, noise=args.noise, seed=args.seed)
    biased = fairness_summary(cal_cohort, predictions(cal_cohort, GLOBAL_THRESHOLD))
    calibrated = calibrate_by_group(cal_cohort)
    cal_summ = fairness_summary(calibrated, predictions(calibrated, GLOBAL_THRESHOLD))
    print()
    print(f"[Scenario 2] measurement bias (rate_gap=0, score_bias={args.cal_score_bias}) "
          f"— per-group calibration")
    _table("", [("biased", biased), ("calibrated", cal_summ)])
    print("-" * 68)
    print(f"=> calibration: DP {_fmt(biased['demographic_parity_diff'])}->"
          f"{_fmt(cal_summ['demographic_parity_diff'])}, EO "
          f"{_fmt(biased['equalized_odds_gap'])}->{_fmt(cal_summ['equalized_odds_gap'])}, "
          f"acc {_fmt(biased['accuracy'])}->{_fmt(cal_summ['accuracy'])} "
          f"(fairness AND accuracy improve — no trade-off)")

    # ---------- audit ----------
    emit("baseline", "fairness-audit", {"summary": base}, ledger_path=args.ledger)
    emit("mitigate_demographic_parity", "fairness-audit",
         {"thresholds": dp_thr, "summary": dp_summ}, ledger_path=args.ledger)
    emit("mitigate_equal_opportunity", "fairness-audit",
         {"thresholds": eo_thr, "summary": eo_summ}, ledger_path=args.ledger)
    emit("calibration_biased", "fairness-audit", {"summary": biased}, ledger_path=args.ledger)
    emit("calibration_calibrated", "fairness-audit", {"summary": cal_summ},
         ledger_path=args.ledger)
    ok, n = verify(args.ledger)
    print("-" * 68)
    print(f"ledger: {args.ledger}  chain_ok={ok}  entries={n}")


if __name__ == "__main__":
    main()
