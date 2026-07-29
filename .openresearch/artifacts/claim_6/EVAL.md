# Claim 6 Route 1 evaluation

Formal result pending. `reproduction/swiss_baselines.py` is the executable
route verifier. The cumulative entrypoint exits nonzero for missing/nonfinite
runs, failed metric controls, or a non-deterministic Regression control.

This route will be recorded as `ROUTE_1_COMPLETE`, not prematurely promoted
to a final `VERIFIED`/`FALSIFIED` verdict. Conditional-flow and independent
external-method routes remain available if evidence confidence stays low.
