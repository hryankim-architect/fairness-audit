# Roadmap, `fairness-audit`

Clean-room group-fairness eval. Synthetic cohort with an injected disparity; the
methodology — multi-metric reporting + honest trade-off — is the deliverable.

> **Status: v0.1 (shipped).** Synthetic cohort generator with a tunable, injected
> group-B under-scoring; metrics (per-group selection rate / TPR / FPR, demographic
> parity, equalized-odds gap, accuracy); a group-threshold mitigation that shrinks
> the parity gap at a reported accuracy cost; bootstrap CIs on the headline gap; and
> a hash-chained audit of every pre/post summary. Runs end-to-end with offline unit
> tests + CI-green.

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
- [x] Negative control: `--score-bias 0` yields a near-zero gap.

## v0.2 — depth (planned)
- [ ] Second mitigation (e.g. score calibration per group) to contrast trade-offs.
- [ ] Equal-opportunity (TPR-only) target alongside demographic parity, to make the
  metric-conflict explicit on the same cohort.
- [ ] CIs on the post-mitigation gaps and on the accuracy cost.
- [ ] Real-model / real-prediction adapter: feed in `(group, y_true, score)` from an
  external classifier so the *same metrics + audit* apply unchanged.

## Sibling evals
Part of an AI-safety eval set: `agent-refusal-eval` (defensive screening),
`sycophancy-eval` (pressure-robustness), `sandbagging-eval` (capability honesty),
`cot-faithfulness-audit` (reasoning–action consistency), and this
(bias / fairness).
