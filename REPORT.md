# Audit report

## Executive result

Claims 1–3 have accepted scoped evidence. The practical cost/dual
parameterization is source-audited, the current Swiss Roll route is pending,
and the weather benchmark is deferred for external data and full training.

Overall status:

`PARTIAL_C1_C2_C3_VERIFIED_C4_SOURCE_AUDITED_C5_WEATHER_DEFERRED_C6_SWISS_ROLL_PENDING_NO_CURRENT_SCORE`

## Claim matrix

| Claim | Result | Main boundary |
| --- | --- | --- |
| C1 | `VERIFIED_SCOPED` | Symbolic and finite parity checks; not a training-result claim. |
| C2 | `VERIFIED_SCOPED` | Finite Gaussian-mixture normalization under stated assumptions. |
| C3 | `VERIFIED_SCOPED` | Finite pointwise and normalized-density identity checks. |
| C4 | `SOURCE_AUDITED` | No accepted current raw route for the complete practical parameterization. |
| C5 | `DEFERRED_EXTERNAL_DATA` | TabRED weather files and full training/evaluation are unavailable. |
| C6 | `PENDING_ROUTE` | Historical multimodal-GMM support is preserved; current Swiss Roll training/baseline route is unfinished. |

## Quantitative evidence

- C1 direct conditional NLL parity absolute error: `1.00e-12`; all three data-term gradients are nonzero; wrong-sign separation ratio: `6.10e8`.
- C2 closed-form normalizer relative error: `3.63e-9`; malformed-formula control relative error: `0.9105`.
- C3 maximum pointwise identity error: `8.13e-16`; normalized relative error: `4.54e-13`; covariance control range: `0.221–0.444`.
- The historical clean-room Swiss Roll-support route reports multimodal target NLL decreasing from approximately `4.233` at `K=1` to `0.412` at `K=8`; this is not the current full experiment verdict.

## Publication boundary

- Historical external score: not recorded in this repository
- Current score claim: `false`
- Publication allowed: `false`
- Official author endorsement: `false` / not claimed

Use [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md) for production paths and
[`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) for source/version scope.
