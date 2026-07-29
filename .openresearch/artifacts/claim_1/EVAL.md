# Claim 1 evaluation

Current verdict: **BLOCKED pending the repaired formal execution**.

The verifier exits nonzero unless the symbolic certificate, direct likelihood
parity, all three data-stream gradient checks, and both negative controls pass.

Rejected run `0a179058-f6d6-4567-a28e-cf5944068369` found all three positive
routes passing, but the original wrong-sign control used a dimensionful
`0.01` cutoff and was therefore not accepted. The repaired verifier checks the
observed discrepancy against its exact analytic residual and requires
million-fold separation from the correct implementation error.
