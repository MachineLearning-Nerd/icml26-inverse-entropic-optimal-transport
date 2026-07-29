# Source audit

- Source: `https://ar5iv.labs.arxiv.org/html/2410.02628`
- Resolved paper version: arXiv `2410.02628v5` (arXiv API, updated 2026-06-04)
- Retrieval: 2026-07-29 07:47:10 UTC with an explicit browser User-Agent
- HTML SHA-256: `8139af4bd89f96a5ef4ba271997eaf1108d20bcc61c1ddee06d50d27dc9bb75f`
- Official code: `https://github.com/MuXauJl11110/EBiEOT`
- Official code commit imported here: `f1c69157d8e1874bb8696f101d54dd9eb3cd2f04`

## Exact anchors

- Claim 2: Proposition 3.1, HTML anchor `#S3.Thmproposition1`.
- Claim 3: Proposition 3.2, HTML anchor `#S3.Thmproposition2`.
- Loss derivation: Section 3.1, HTML anchor `#S3.SS1`.
- Inverse-EOT equivalence: Section 3.2, HTML anchor `#S3.SS2`.
- Practical parameterization: Section 3.3, HTML anchor `#S3.SS3`.
- Swiss Roll: Section 5.1 / Figure 2, anchors `#S5.SS1` and `#S5.F2`.
- Weather: Section 5.2 / Tables 1 and 2, anchors `#S5.SS2`, `#S5.T1`, and `#S5.T2`.

## Assumptions and quantifiers used by the baseline

Proposition 3.1 assumes positive mixture weights, positive-definite covariance
matrices, and the cost/potential forms in Eqs. (15) and (16). It states an exact
identity for every input and every finite component count.

Proposition 3.2 has the same assumptions and states that the normalized
conditional is exactly the displayed finite Gaussian mixture, with means
`a_n + A_n b_m(x)` and covariances `epsilon A_n`.

