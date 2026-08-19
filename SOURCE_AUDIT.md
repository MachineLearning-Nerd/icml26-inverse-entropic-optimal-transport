# Source audit entry point

## Primary source

- Paper: *Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization*
- Authors: Mikhail Persiianov, Arip Asadulaev, Nikita Andreev, Nikita Starodubcev, Dmitry Baranchuk, Anastasis Kratsios, Evgeny Burnaev, and Alexander Korotin
- arXiv: [2410.02628](https://arxiv.org/abs/2410.02628), version 5
- HTML source: <https://ar5iv.labs.arxiv.org/html/2410.02628>
- Retrieved: `2026-07-29 07:47:10 UTC`
- Source SHA-256: `8139af4bd89f96a5ef4ba271997eaf1108d20bcc61c1ddee06d50d27dc9bb75f`
- ICML submission identifier: `0p617sK4Z4`
- Official implementation: <https://github.com/MuXauJl11110/EBiEOT>
- Imported official-code commit: `f1c69157d8e1874bb8696f101d54dd9eb3cd2f04`

## Claim anchors

- C1: Sections 3.1–3.2, Eqs. 13–14 — likelihood and inverse-EOT equivalence.
- C2: Proposition 3.1 — closed-form normalization `Zθ(x)`.
- C3: Proposition 3.2 — exact conditional Gaussian mixture.
- C4: Section 3.3, Eqs. 15–18 — practical cost and dual parameterization.
- C5: Section 5.2, Tables 1–2 — real-weather benchmark.
- C6: Theorem 3.3, Section 5.1, Figure 2, and Appendices C.2/D.1–D.2 — Swiss Roll and universality route.

## Version and scope rules

Propositions 3.1 and 3.2 are checked under positive mixture weights,
positive-definite covariance matrices, and the displayed finite cost/potential
forms. Claim 1 is an algebraic identity under the displayed Gibbs-Boltzmann
parameterization. The practical and experimental claims require their own
training/data protocols; pending and deferred routes are not treated as final
verification.
