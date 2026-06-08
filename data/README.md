# data/

There is **no data file** here, by design.

The cohort is generated deterministically in code
(`src/fairaudit/cohort.py::generate_cohort`) from a seed, with a tunable, injected
group disparity. This keeps the audit fully reproducible and makes the
ground-truth unfairness *known*, so metrics can be checked against it.

To regenerate / inspect the cohort:

```python
from fairaudit.cohort import generate_cohort
recs = generate_cohort(n=400, score_bias=0.15, seed="v0.1")
```

No real data, and no real protected-attribute information, is used anywhere in this
repo. Groups are abstract labels (`A`, `B`).
