# fairness-audit

A small, honest, clean-room harness for **group-fairness auditing** of a binary
classifier. It measures several fairness metrics at once — because they conflict —
then compares **three mitigations** and reports the **metric conflict** and the
**fairness/accuracy trade-off** rather than claiming a free fix. Everything runs on
a synthetic cohort with a *tunable, known* disparity, so the signal is checkable and
the methodology — not a leaderboard number — is the deliverable.

> Part of a small suite of clean-room AI-safety / evaluation harnesses
> (refusal, sycophancy, sandbagging, CoT-faithfulness, fairness). Same house
> style: stdlib-only, deterministic mock, hash-chained audit ledger, bootstrap
> CIs, CI-gated, defensive-by-design.

## Why this exists

"Is the model fair?" has no single answer. Two of the most common criteria are
provably in tension:

- **Demographic parity** — equal *selection rate* across groups.
- **Equalized odds** — equal *true-positive* and *false-positive* rates across groups.

When groups differ in outcome rates or in how the model scores them, you generally
**cannot satisfy both at once** (Kleinberg et al., 2016; Chouldechova, 2017). A
serious audit therefore reports *multiple* metrics and is explicit about what any
mitigation costs. This repo demonstrates that discipline end to end on data where
the ground-truth disparity is injected and therefore checkable.

## What it measures

For a cohort of records `(group, y_true, score)` and a decision threshold, the
audit computes, per group and overall:

| metric | meaning |
|---|---|
| `selection_rate` | P(predicted positive) — basis of demographic parity |
| `tpr`, `fpr` | true / false positive rates — basis of equalized odds |
| `demographic_parity_diff` | \|selection_rate_A − selection_rate_B\| |
| `tpr_gap`, `fpr_gap` | per-rate group gaps |
| `equalized_odds_gap` | max(tpr_gap, fpr_gap) |
| `accuracy` | overall, and per group |

All gaps are reported with a **percentile bootstrap CI** so a single demo number is
never mistaken for a precise estimate.

## Three mitigations (and what each one costs)

Which fairness criterion you pick is itself a decision. The audit applies all three
and shows the consequences:

- `group_thresholds_for_parity` — per-group thresholds equalizing **selection rate**
  (demographic parity).
- `group_thresholds_for_equal_opportunity` — per-group thresholds equalizing
  **true-positive rate** (equal opportunity, the TPR half of equalized odds).
- `calibrate_by_group` — removes a per-group additive **score offset** (measurement
  bias) before a single shared threshold.

On a base-rate disparity the two threshold rules **conflict**; calibration is the
right tool for a *different* problem (measurement bias) and does not resolve that
conflict.

## Quickstart

```bash
pip install -e ".[dev]"      # or: uv run --extra dev ...
pytest -q                    # unit tests
python scripts/run_fairness_audit.py     # full audit (both scenarios)
```

Real output (default flags; synthetic, `seed=v0.1`):

```
[Scenario 1] base-rate disparity — demographic parity vs equal opportunity

metric                          baseline     DP-thresh     EO-thresh
demographic_parity_diff             0.25          0.00          0.22
tpr_gap                             0.05          0.28          0.01
fpr_gap                             0.01          0.21          0.03
equalized_odds_gap                  0.05          0.28          0.03
accuracy                            0.89          0.84          0.89
--------------------------------------------------------------------
baseline DP gap    0.25  95% CI [0.15, 0.34]
EO gap AFTER forcing parity    0.28  95% CI [0.18, 0.34]  (was 0.05 pre)

[Scenario 2] measurement bias (rate_gap=0, score_bias=0.2) — per-group calibration

metric                            biased    calibrated
demographic_parity_diff             0.30          0.02
tpr_gap                             0.50          0.00
fpr_gap                             0.09          0.00
equalized_odds_gap                  0.50          0.00
accuracy                            0.82          0.85
```

Read scenario 1 as the **impossibility in action**: the calibrated model satisfies
equalized odds (EO gap 0.05) but not demographic parity (DP gap 0.25).
*Demographic-parity* thresholds drive DP to 0.00 — but push the equalized-odds gap to
**0.28** (CI excludes the 0.05 baseline, so the breakage is real) and drop accuracy 5
points. *Equal-opportunity* thresholds keep the TPR gap ~0 but **leave DP at 0.22**.
No single threshold rule gets both to zero. Scenario 2 is the contrast: when the
disparity is a pure measurement offset, **calibration fixes fairness *and* accuracy
together** — no trade-off, because it is the right tool for that problem.

### Knobs

`--n`, `--base-rate`, `--rate-gap` (the base-rate difference that drives scenario 1),
`--score-bias` (measurement bias on the main cohort), `--cal-score-bias` (the offset
for the scenario-2 calibration demo), `--noise`, `--seed`, `--n-boot`. Set
`--rate-gap 0` to confirm near-zero gaps when there is no disparity (a negative
control).

## Audit ledger

Each run appends one entry per condition (baseline, demographic-parity,
equal-opportunity, and the two calibration states) to a hash-chained NDJSON ledger
(`audit/local-demo.ndjson`); editing any past entry breaks the chain, which
`verify()` reports. The measurement is itself auditable.

## What this is NOT

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md). In short: this is
a **methodology demo on synthetic data**, not a fairness certification, not a claim
about any real model or population, and not a substitute for context-specific
fairness work with stakeholders and real outcome data.

## License

MIT — see [`LICENSE`](LICENSE).
