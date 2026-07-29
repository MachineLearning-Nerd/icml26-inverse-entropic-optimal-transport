# Claim 6 Route 1 method

This route runs the actual Figure-2 regime (`P=128`, `Q=R=1024`), not the
separate 16K Figure-5 sanity check. It reuses the authors' released
Regression, UGAN+L2, and CGAN trainer classes, architectures, objectives,
optimizers, batch size, and 250,000-step configs.

All methods receive the same deterministic per-seed data. Pair generation
follows Appendix D.1 exactly: for every paired observation, draw a fresh
source/target minibatch of 64, solve the stated regularized `rotation-v2`
Sinkhorn plan, and sample one pair. The repaired solver audits every plan's
row/column marginal residual.

Each model is evaluated against the same 512-sample conditional reference at
each of the paper's three displayed probes and the same 1,024-sample marginal
reference. Sliced W2 and independently coded energy distance are reported.
Matched-seed differences against the promoted EBiEOT model include paired
t-based 95% intervals.
