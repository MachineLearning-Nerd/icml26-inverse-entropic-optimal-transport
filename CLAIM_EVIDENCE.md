# Claim-to-evidence ledger

The paper’s six evidence surfaces are separated from the repository’s older
judged labels. Each row states how the result is produced and where its scope
ends.

| Claim | Paper anchor | How the result is produced | Evidence and controls | Status |
| --- | --- | --- | --- | --- |
| C1 — likelihood / inverse-EOT equivalence | arXiv v5 Sections 3.1–3.2, Eqs. 13–14 | Build the symbolic certificate, compare direct and decomposed conditional NLL, check paired and both marginal gradients, then test the corrected wrong-sign residual. | `.openresearch/artifacts/claim_1/formal_raw.json`, `reproduction/claim_1.py`, and the claim-1 source audit; the wrong-sign route is rejected with separation ratio `6.10e8`. | **VERIFIED_SCOPED** |
| C2 — closed-form normalizer | Proposition 3.1 | Evaluate the finite Gaussian-mixture closed form and compare it with independent midpoint quadrature under positive weights and positive-definite covariances. | `.openresearch/artifacts/baseline/claim_2/formal_raw.json`, `reproduction/verifiers.py`, and `reproduction/independent_checker.py`; malformed-formula relative error is `0.9105`. | **VERIFIED_SCOPED** |
| C3 — exact conditional Gaussian mixture | Proposition 3.2 | Compare the displayed component means/covariances pointwise and compare the normalized closed mixture with independently integrated energy density. | `.openresearch/artifacts/baseline/claim_3/formal_raw.json`, `reproduction/verifiers.py`, and the covariance negative control; maximum pointwise error is `8.13e-16`. | **VERIFIED_SCOPED** |
| C4 — practical cost/dual parameterization | Section 3.3, Eqs. 15–18 | Audit the source anchors and compare the log-sum-exp cost and Gaussian-mixture dual forms with finite closed-form definitions; the Swiss calibration code is the intended executable route. | `.openresearch/artifacts/baseline/source_audit.md`, `reproduction/core.py`, `reproduction/swiss_calibration.py`, and `.openresearch/artifacts/claim_4/`. No accepted current raw run is recorded. | **SOURCE_AUDITED** |
| C5 — real-weather benchmark | Section 5.2, Tables 1–2 | Train/evaluate the weather models over the external TabRED files and compare test log-likelihoods across unpaired-sample counts. | `.openresearch/artifacts/claim_4/EVAL.md`, `notebooks/weather/README.md`, and `audit/weather-tables`; the required external data and full training are absent. | **DEFERRED_EXTERNAL_DATA** |
| C6 — Swiss Roll recovery / universality route | Theorem 3.3, Section 5.1, Figure 2 | Fit the clean-room multimodal-GMM route and then run the fixed architecture calibration and baseline comparison with finite metrics and negative controls. | `reproduction/swiss_calibration.py`, `reproduction/swiss_baselines.py`, `.openresearch/artifacts/claim_6/`, and the historical judged Space; the current trained route has no accepted final result. | **PENDING_ROUTE** |

## Evidence boundaries

1. The arXiv HTML source is pinned by SHA-256 in [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md).
2. Claims 1–3 are finite or symbolic checks with machine-readable controls;
   they do not replace the paper’s analytic proofs.
3. The historical Space contains a clean-room multimodal-GMM universality
   check, but it is not silently promoted to the current full Swiss Roll
   experiment.
4. The weather route is deferred rather than falsified because its external
   data and full training are not available in this repository.

The complete path inventory is [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json).
