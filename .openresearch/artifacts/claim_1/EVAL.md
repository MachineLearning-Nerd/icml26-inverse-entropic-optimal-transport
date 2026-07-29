# Claim 1 evaluation

Current verdict: **VERIFIED**.

The verifier exits nonzero unless the symbolic certificate, direct likelihood
parity, all three data-stream gradient checks, and both negative controls pass.

Rejected run `0a179058-f6d6-4567-a28e-cf5944068369` found all three positive
routes passing, but the original wrong-sign control used a dimensionful
`0.01` cutoff and was therefore not accepted. The repaired verifier checks the
observed discrepancy against its exact analytic residual and requires
million-fold separation from the correct implementation error.

Accepted run `2cf4da04-c18a-4339-93cf-56279246e6b3` at Git
`6e9188957c8eede08256fb044ec55783e4e0fb33` passed all routes and controls on
HF `cpu-upgrade`. Scientific runtime was 37.644 seconds with 64 CPUs visible.
