# Roadmap, `fairness-audit`

Clean-room group-fairness eval. Synthetic cohort with a known disparity; the
methodology — multi-metric reporting + honest metric-conflict + trade-off — is the
deliverable.

> **Status: v0.2 (shipped).** v0.1 metrics + demographic-parity mitigation + bootstrap
> CIs + hash-chained audit. v0.2 adds an **equal-opportunity (TPR) mitigation** and
> **per-group calibration**, a **two-scenario runner** that shows the DP↔EO conflict
> on one cohort (each threshold rule zeroes its own gap, widens the other) and that
> calibration fixes a measurement-bias cohort with no trade-off, plus a **CI on the
> post-parity equalized-odds gap**. Runs end-to-end with offline unit tests + CI-green.

## v0.1 — methodology harness (done)
- [x] `cohort.py` — deterministic synthetic cohort with a group base-rate gap and a
  calibrated model (optional measurement-bias knob); the known ground-truth source
  of the fairness impossibility.
- [x] `metrics.py` — per-group rates + demographic parity / TPR-gap / FPR-gap /
  equalized-odds gap / accuracy; undefined rates returned as `None`, never faked.
- [x] `mitigate.py` — per-group thresholds equalizing selection to a target rate.
- [x] Bootstrap CIs on the demographic-parity gap (percentile, deterministic).
- [x] `scripts/run_fairness_audit.py` → pre/post side-by-side + accuracy cost,
  written to a hash-chained ledger.
- [x] Tests (cohort / metrics / mitigate / bootstrap / audit) + ruff + CI + scope doc.
- [x] Negative control: `--rate-gap 0` yields a near-zero gap.

## v0.2 — depth (shipped)
- [x] **Equal-opportunity (TPR) mitigation** (`group_thresholds_for_equal_opportunity`)
  alongside demographic parity, making the metric-conflict explicit on one cohort.
- [x] **Per-group calibration** (`calibrate_by_group`) — removes a measurement-bias
  offset; the contrasting scenario where fairness and accuracy improve together.
- [x] **Two-scenario runner**: base-rate (DP vs EO conflict) + measurement-bias
  (calibration), with a **CI on the post-parity equalized-odds gap**.
- [ ] Real-model / real-prediction adapter: feed in `(group, y_true, score)` from an
  external classifier so the *same metrics + audit* apply unchanged.
- [ ] Calibration-target metric (per-group reliability / ECE) as a fourth lens.

## Sibling evals
Part of an AI-safety eval set: `agent-refusal-eval` (defensive screening),
`sycophancy-eval` (pressure-robustness), `sandbagging-eval` (capability honesty),
`cot-faithfulness-audit` (reasoning–action consistency), and this
(bias / fairness).
