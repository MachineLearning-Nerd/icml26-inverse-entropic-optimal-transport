# Current cumulative verification

Revision status: Claim 1 formal run pending; Claims 2 and 3 freshly verified.

The exact current claim contracts are
`.openresearch/artifacts/baseline/claim_2/claim_contract.json` and
`.openresearch/artifacts/baseline/claim_3/claim_contract.json`. The fixed
command is:

```bash
uv run --frozen python -m reproduction.run
```

The formal baseline run `ae3c8b1c-0ad4-4f9b-9ba7-62bd489591fa` at Git
`41e78e57…` freshly verified Claim 2 at relative error `3.63e-9` and Claim 3
at maximum pointwise error `8.13e-16` and normalized error `4.54e-13`.
Malformed-formula controls failed by `0.911` and `0.221–0.444`, respectively.

The current verifier supersedes the historical `verify.py` by adding a locked
environment, exact v5 source anchors, fail-closed thresholds, an independent
quadrature checker, and negative controls. It will not be labeled current
Claim 1 evidence until its formal child run exits zero.
