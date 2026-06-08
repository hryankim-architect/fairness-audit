# fairness-audit

A small, honest, clean-room harness for **group-fairness auditing** of a binary
classifier. It measures several fairness metrics at once — because they conflict —
applies a transparent group-threshold mitigation, and reports the
**fairness/accuracy trade-off** rather than claiming a free fix. Everything runs on
a synthetic cohort with a *tunable, injected* disparity, so the signal is known and
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

## The mitigation (and its cost)

`mitigate.group_thresholds_for_parity` picks a **per-group score threshold** so each
group's selection rate matches a shared target. This shrinks the
demographic-parity gap toward zero — but on groups with different base rates it
**costs overall accuracy** and **breaks equalized odds** (the very criterion the
calibrated model satisfied before mitigation). The runner prints pre- vs
post-mitigation side by side so the trade-off is visible, never hidden.

## Quickstart

```bash
pip install -e ".[dev]"      # or: uv run --extra dev ...
pytest -q                    # unit tests
python scripts/run_fairness_audit.py     # full audit on the synthetic cohort
```

Example run (default scenario: calibrated model, base-rate gap, `score_bias=0`):

```
metric                           pre      post   (lower gap = fairer)
demographic_parity_diff         0.25      0.00
tpr_gap                         0.05      0.28
fpr_gap                         0.01      0.21
equalized_odds_gap              0.05      0.28
accuracy                        0.89      0.84
------------------------------------------------------------------
demographic-parity gap (pre): point= 0.25  95% CI [0.15, 0.34]
mitigation shrinks DP gap 0.25 -> 0.00, accuracy cost +0.05,
  equalized-odds gap 0.05 -> 0.28
```

Read that as the impossibility in action: **pre-mitigation the model satisfies
equalized odds** (EO gap 0.05) **but not demographic parity** (DP gap 0.25). Forcing
demographic parity drives the DP gap to ~0 — and in doing so **drives the
equalized-odds gap up to 0.28 and drops accuracy 5 points.** You cannot have both;
the audit shows you the trade-off instead of hiding it behind one number.

### Knobs

`--n`, `--base-rate`, `--rate-gap` (the group base-rate difference that drives the
disparity), `--score-bias` (optional extra measurement bias against group B),
`--noise`, `--seed`, `--n-boot`. Set `--rate-gap 0` to confirm the audit reports
near-zero gaps when there is no disparity (a negative control).

## Audit ledger

Each run appends its pre- and post-mitigation summaries to a hash-chained NDJSON
ledger (`audit/local-demo.ndjson`); editing any past entry breaks the chain, which
`verify()` reports. The measurement is itself auditable.

## What this is NOT

See [`docs/what-is-out-of-scope.md`](docs/what-is-out-of-scope.md). In short: this is
a **methodology demo on synthetic data**, not a fairness certification, not a claim
about any real model or population, and not a substitute for context-specific
fairness work with stakeholders and real outcome data.

## License

MIT — see [`LICENSE`](LICENSE).
