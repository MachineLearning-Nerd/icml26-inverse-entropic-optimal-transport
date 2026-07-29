# Current cumulative verification

Revision status: provisional baseline setup.

The exact current claim contracts are
`.openresearch/artifacts/baseline/claim_2/claim_contract.json` and
`.openresearch/artifacts/baseline/claim_3/claim_contract.json`. The fixed
command is:

```bash
uv run --frozen python -m reproduction.run
```

Before the new baseline run, the only raw numbers shown here are explicitly
historical: Claim 2 closed-form and grid values were both `73.298` with relative
error `2.47e-11`; Claim 3's maximum pointwise identity error was `2.0e-16`.
These came from the immutable judged Space revision `3c31d94f…`.

The current verifier supersedes the historical `verify.py` by adding a locked
environment, exact v5 source anchors, fail-closed thresholds, an independent
quadrature checker, and negative controls. It will not be labeled current
evidence until the formal baseline run exits zero.

