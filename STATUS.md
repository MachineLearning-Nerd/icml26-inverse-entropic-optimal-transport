# Audit status

**State:** Claims 1–3 have accepted finite or symbolic evidence. The practical
parameterization is source-audited, the current Swiss Roll route is pending,
and the real-weather benchmark is deferred.

- Paper: [Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization](https://arxiv.org/abs/2410.02628)
- Authors: Mikhail Persiianov, Arip Asadulaev, Nikita Andreev, Nikita Starodubcev, Dmitry Baranchuk, Anastasis Kratsios, Evgeny Burnaev, and Alexander Korotin
- ICML submission: `0p617sK4Z4`
- Repository: [MachineLearning-Nerd/icml26-inverse-entropic-optimal-transport](https://github.com/MachineLearning-Nerd/icml26-inverse-entropic-optimal-transport)
- Overall status: `PARTIAL_C1_C2_C3_VERIFIED_C4_SOURCE_AUDITED_C5_WEATHER_DEFERRED_C6_SWISS_ROLL_PENDING_NO_CURRENT_SCORE`
- C1: `VERIFIED_SCOPED` by accepted symbolic, likelihood-parity, gradient, and wrong-sign-control evidence
- C2: `VERIFIED_SCOPED` by accepted finite Gaussian-mixture normalization and midpoint-quadrature evidence
- C3: `VERIFIED_SCOPED` by accepted pointwise Gaussian-mixture identity, normalized-density, and covariance-control evidence
- C4: `SOURCE_AUDITED` for the Section 3.3 cost/dual parameterization; no separate current verdict artifact
- C5: `DEFERRED_EXTERNAL_DATA` for Tables 1–2; TabRED weather files and full training are not present
- C6: `PENDING_ROUTE`; historical clean-room multimodal-GMM evidence is preserved, but the current trained Swiss Roll/calibration route has no accepted final result
- Historical external score: not recorded in this repository
- Current score claim: `false`
- Publication allowed: `false`
- Official author endorsement: `false` / not claimed
- Commit identity: all reachable history uses `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`
- Recovery bundle SHA-256: `75eead4967d5d054eb0594d42bd91a76ac55dd17d45c770657a6d10f7f4be284`

The accepted evidence is finite or route-scoped. It does not certify the
paper’s complete Swiss Roll or weather experiments, and the cumulative runner
remains intentionally fail-closed until its unfinished routes are completed.
