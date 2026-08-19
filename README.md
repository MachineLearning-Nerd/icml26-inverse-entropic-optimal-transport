# Inverse Entropic Optimal Transport for Semi-supervised Learning

Independent reproduction audit for [“Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization”](https://arxiv.org/abs/2410.02628).

The repository is published as [`icml26-inverse-entropic-optimal-transport`](https://github.com/MachineLearning-Nerd/icml26-inverse-entropic-optimal-transport).

> **Audit status:** `PARTIAL_C1_C2_C3_VERIFIED_C4_SOURCE_AUDITED_C5_WEATHER_DEFERRED_C6_SWISS_ROLL_PENDING_NO_CURRENT_SCORE`
>
> Claims 1–3 have accepted machine-readable evidence under explicit finite or
> symbolic contracts. The practical parameterization is source-audited, the
> current Swiss Roll route remains pending, and the weather benchmark is
> deferred for external data and full training. No current judge score or
> author endorsement is claimed. See [`STATUS.md`](STATUS.md),
> [`CLAIM_EVIDENCE.md`](CLAIM_EVIDENCE.md), and [`REPORT.md`](REPORT.md).

## What the paper does

The paper introduces Energy-Based Inverse Entropic Optimal Transport (EBiEOT) for semi-supervised learning. Its objective combines three data streams—paired `(x, y)` observations, unpaired `x` observations, and unpaired `y` observations—through a likelihood objective rather than a separately estimated transport map or an ad-hoc consistency regularizer.

The paper develops two practical families:

- **EBiEOT-GMM**, which uses Gaussian-mixture structure to obtain tractable costs, potentials, normalizers, and conditional distributions.
- **EBiEOT-NN**, which parameterizes the energy/cost with neural networks and uses sampling-based training.

The theoretical sections establish closed-form identities for finite Gaussian mixtures and a universal-approximation route. The experiments study synthetic Swiss Roll data and a real weather-forecasting benchmark.

## Current reproduction assessment

The table below records what was actually checked, how the result is produced, and the current evidence boundary. “Verified” refers to this repository’s executable checks, not to an official author or venue certification.

| Paper surface | How the claim is produced | Evidence in this repository | Status |
| --- | --- | --- | --- |
| Section 3.1–3.2, Eqs. 13–14: data likelihood equals the inverse-EOT objective | Build a symbolic certificate, compare the direct likelihood with the EBiEOT loss, check gradients for paired and both unpaired streams, and require corrected wrong-sign controls to fail | [`reproduction/claim_1.py`](reproduction/claim_1.py), [`claim_1/EVAL.md`](.openresearch/artifacts/claim_1/EVAL.md) | **VERIFIED** — accepted formal run `2cf4da04-c18a-4339-93cf-56279246e6b3` |
| Proposition 3.1: closed-form normalizer `Zθ(x)` | Compare the finite Gaussian-mixture formula with independent midpoint quadrature under the proposition’s positive-weight/positive-definite-covariance assumptions; run a malformed-formula control | [`reproduction/verifiers.py`](reproduction/verifiers.py), [`baseline/claim_2/`](.openresearch/artifacts/baseline/claim_2/), [`baseline/source_audit.md`](.openresearch/artifacts/baseline/source_audit.md) | **VERIFIED_SCOPED** — accepted baseline run `ae3c8b1c-0ad4-4f9b-9ba7-62bd489591fa`, relative error `3.63e-9`, malformed-formula control rejected |
| Proposition 3.2: exact conditional Gaussian mixture | Compare the displayed component means/covariances pointwise and after normalization; run an independent quadrature check and covariance negative control | [`reproduction/verifiers.py`](reproduction/verifiers.py), [`baseline/claim_3/`](.openresearch/artifacts/baseline/claim_3/) | **VERIFIED_SCOPED** — accepted baseline run `ae3c8b1c-0ad4-4f9b-9ba7-62bd489591fa`, maximum pointwise error `8.13e-16`, covariance control rejected |
| Section 3.3, Eqs. 15–18: practical cost/dual parameterization | Audit the source anchors and compare the log-sum-exp cost and Gaussian-mixture dual forms against their finite closed-form definitions | [`reproduction/core.py`](reproduction/core.py), [`baseline/source_audit.md`](.openresearch/artifacts/baseline/source_audit.md) | **SOURCE-AUDITED** — no separate current verdict artifact |
| Theorem 3.3 and Figure 2: Swiss Roll recovery / universality route | Use the clean-room multimodal-GMM likelihood route and the current Swiss Roll calibration/baseline suite; keep unfinished practical routes fail-closed | [`reproduction/swiss_calibration.py`](reproduction/swiss_calibration.py), [`reproduction/swiss_baselines.py`](reproduction/swiss_baselines.py), [`claim_6/EVAL.md`](.openresearch/artifacts/claim_6/EVAL.md) | **PENDING_ROUTE** — historical clean-room multimodal evidence is preserved, but the current trained Swiss Roll/calibration route has no accepted final result |
| Section 5.2, Tables 1–2: real weather benchmark | Train/evaluate the weather models with the external TabRED weather files and compare log-likelihoods against the paper’s baselines | [`claim_4/EVAL.md`](.openresearch/artifacts/claim_4/EVAL.md), [`notebooks/weather/README.md`](notebooks/weather/README.md) | **BLOCKED / DEFERRED** — the formal verifier has no verdict until the data and full training run are available |

The current cumulative runner exposes the exact executable route:

```bash
uv run --frozen python -m reproduction.run
```

It is intentionally fail-closed: a failed parity check, missing/non-finite result, failed negative control, or unfinished formal route prevents a passing summary. The configured hosted run estimates approximately 54 CPU cores, so this is not a lightweight smoke test.

The machine-readable claim ledger is [`claims.json`](claims.json), the
production-path manifest is [`EVIDENCE_MANIFEST.json`](EVIDENCE_MANIFEST.json),
and [`verify_final.py`](verify_final.py) checks the published documentation,
source pin, branch set, and attribution without launching the heavy suite.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`src/`](src/) | EBiEOT-GMM, EBiEOT-NN, costs, potentials, samplers, networks, and baselines |
| [`conf/`](conf/) | Hydra datasets, models, training, and experiment configurations |
| [`scripts/`](scripts/) | Training, baseline, sweep, and notebook entry points |
| [`reproduction/`](reproduction/) | Focused claim verifiers, Swiss Roll calibration, baselines, and cumulative runner |
| [`.openresearch/artifacts/`](.openresearch/artifacts/) | Claim contracts, source audit, evaluation notes, and machine-readable evidence metadata |
| [`candidate_space/`](candidate_space/) | Current evidence state and reproducibility notes |
| [`notebooks/`](notebooks/) | Swiss Roll, MNIST, colored-MNIST, ALAE, and weather experiments |
| [`tests/`](tests/) | Project tests for the implementation |

The implementation preserved here originated from the authors’ public [EBiEOT repository](https://github.com/MuXauJl11110/EBiEOT). The reproduction layer and evidence notes are maintained independently; this repository does not replace the authors’ original publication or claim to be an official author release.

## Running the project

The project requires Python `>=3.10,<3.13` and uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync --frozen
uv run --frozen python -m reproduction.run
```

For implementation experiments, examples include:

```bash
# EBiEOT on synthetic Swiss Roll data
uv run python scripts/train.py experiment=egeot_swiss_roll

# EBiEOT-GMM on Swiss Roll data
uv run python scripts/train.py experiment=gmm_swiss_roll

# Conditional GAN baseline
uv run python scripts/train_baseline.py experiment=baseline_cgan_swiss_roll_16k
```

Data requirements vary by experiment:

- Swiss Roll is generated locally.
- MNIST and colored-MNIST are downloaded into `data/` when their experiments run.
- ALAE experiments require externally prepared FFHQ latent files.
- Weather experiments require the external TabRED files under `../tabred/kal/weather` and remain deferred in the current audit.

## Branch organization

The original branches were generated under `orx/*`. They are being renamed to describe their scientific role. The complete old-to-new map, branch purposes, and provenance rules are in [`branch-audit.md`](branch-audit.md).

| Branch | Role |
| --- | --- |
| `main` | Canonical implementation, current evidence layer, README, and branch map |
| `historical/judged-baseline` | Frozen historical judged state; retained for provenance |
| `audit/loss-inverse-eot-equivalence` | Section 3.1–3.2 objective/loss equivalence certificate |
| `audit/swiss-roll-benchmark` | Faithful Swiss Roll benchmark and practical GMM route |
| `audit/swiss-roll-linear-calibration` | Released linear Swiss Roll architecture calibration |
| `audit/swiss-roll-baselines` | Swiss Roll baseline comparison and practical likelihood suite |
| `audit/swiss-roll-conditional-flow` | Conditional-flow Swiss Roll baselines |
| `audit/weather-tables` | Weather Tables 1–2 benchmark route; currently deferred |

Branch names describe the evidence route, not a claim that every branch has a completed result.

## Paper metadata

- **Title:** Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization
- **Authors:** Mikhail Persiianov, Arip Asadulaev, Nikita Andreev, Nikita Starodubcev, Dmitry Baranchuk, Anastasis Kratsios, Evgeny Burnaev, and Alexander Korotin
- **Paper:** [arXiv:2410.02628](https://arxiv.org/abs/2410.02628) ([version 5 source audit](.openresearch/artifacts/baseline/source_audit.md))
- **OpenReview:** [0p617sK4Z4](https://openreview.net/forum?id=0p617sK4Z4)
- **Official implementation:** [MuXauJl11110/EBiEOT](https://github.com/MuXauJl11110/EBiEOT)

The complete source/version audit is in [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md);
the repository citation is in [`CITATION.cff`](CITATION.cff), and the author
thank-you note is in [`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

### Citation

```bibtex
@misc{persiianov2026inverse,
  title         = {Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization},
  author        = {Persiianov, Mikhail and Asadulaev, Arip and Andreev, Nikita and Starodubcev, Nikita and Baranchuk, Dmitry and Kratsios, Anastasis and Burnaev, Evgeny and Korotin, Alexander},
  year          = {2026},
  eprint        = {2410.02628},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {arXiv:2410.02628v5}
}
```

## Thank you to the authors

Thank you to Mikhail Persiianov, Arip Asadulaev, Nikita Andreev, Nikita Starodubcev, Dmitry Baranchuk, Anastasis Kratsios, Evgeny Burnaev, and Alexander Korotin for developing EBiEOT, publishing the paper, and releasing the implementation that makes independent study possible. This repository is maintained as a reproducibility and documentation companion, with respect for the authors’ original work and attribution.

## Maintenance attribution

Repository documentation, branch naming, audit notes, and maintenance commits in this collection are attributed to **MachineLearning-Nerd** with canonical identity `MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>`. Scientific authorship and ownership of the paper’s ideas remain with the paper authors.
