# What this is — and what it is NOT

This repo is a **clean-room methodology demo** for group-fairness auditing. Being
explicit about its limits is part of the point.

## It IS

- A demonstration that fairness is **multi-metric**: it computes demographic parity,
  equalized-odds components (TPR/FPR gaps), and accuracy together, because optimizing
  one can worsen another.
- A demonstration that mitigation has a **cost**: the group-threshold post-processing
  shrinks the demographic-parity gap but is reported alongside its accuracy cost and
  its effect on the equalized-odds gap.
- **Deterministic and synthetic**: the cohort is generated from a seed with a
  *known, injected* disparity, so the audit's output can be checked against ground
  truth. No real individuals, no real protected-attribute data.
- **Auditable**: results are written to a hash-chained ledger.

## It is NOT

- **Not a fairness certification.** A near-zero demographic-parity gap on this
  synthetic cohort says nothing about whether any real system is fair. Real audits
  require real outcome data, the right population, and the relevant legal/ethical
  context.
- **Not a claim about any real model, dataset, or group.** Groups are labelled
  abstractly (`A`, `B`). No real demographic, racial, gender, or other protected
  category is represented, modelled, or implied.
- **Not a definition of "the" fair decision.** Which fairness criterion is
  appropriate is a normative, domain- and stakeholder-specific question this code
  does not answer. It surfaces trade-offs; it does not adjudicate them.
- **Not a mitigation recommendation.** Group-specific thresholds can themselves be
  legally or ethically contentious (disparate treatment). The mitigation here is an
  illustrative post-processing method, not advice to deploy one.
- **Not statistically complete.** Bootstrap CIs quantify sampling noise on a toy
  cohort only; they do not cover measurement bias, label bias, or distribution
  shift — often the dominant real-world problems.

## Honest framing

The deliverable is the **method and the discipline**: report several metrics, never
collapse fairness to one number, show what mitigation costs, and make the
measurement reproducible and tamper-evident. Numbers in the README are illustrative
outputs of the synthetic generator, not findings about the world.
