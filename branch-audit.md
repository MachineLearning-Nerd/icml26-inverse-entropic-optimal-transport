# Branch audit

This file records the transition from generated `orx/*` branch names to descriptive names. The old names are retained only as provenance references during the rename; the live repository should expose the clean names.

## Branch map

| Historical branch | Clean branch | What it does | Current evidence boundary |
| --- | --- | --- | --- |
| `orx/frozen-judged-state-baseline` | `historical/judged-baseline` | Preserves the previously judged baseline state and its evidence context | Historical reference only; not the current cumulative verdict |
| `orx/exact-loss-and-inverse-eot-equivalence` | `audit/loss-inverse-eot-equivalence` | Checks the likelihood/inverse-EOT identity from Sections 3.1–3.2, including gradients and wrong-sign controls | Claim 1 has an accepted formal run; see [`claim_1/EVAL.md`](.openresearch/artifacts/claim_1/EVAL.md) |
| `orx/faithful-swiss-roll-benchmark-and-practical-gmm` | `audit/swiss-roll-benchmark` | Runs the faithful Swiss Roll benchmark and practical Gaussian-mixture route | Historical and practical evidence are kept separate from a final paper-level verdict |
| `orx/released-linear-swiss-roll-architecture-calibrat` | `audit/swiss-roll-linear-calibration` | Calibrates the released linear Swiss Roll architecture and its likelihood behavior | Calibration evidence; not a replacement for the full neural experiment |
| `orx/swiss-roll-baselines-and-practical-likelihood` | `audit/swiss-roll-baselines` | Compares Swiss Roll baselines and evaluates the practical parameterization route | Current cumulative route remains fail-closed until all controls pass |
| `orx/swiss-roll-conditional-flow-baselines` | `audit/swiss-roll-conditional-flow` | Provides conditional-flow baseline comparisons on Swiss Roll data | Supplementary baseline route; not by itself a paper-claim verdict |
| `orx/weather-tables-1-2-exact-benchmark` | `audit/weather-tables` | Targets the real-weather benchmark behind Tables 1–2 | Deferred: external TabRED data and full training/evaluation are still required |

## Claim numbering note

The older judged Space used six surfaces named C0–C5:

1. the likelihood/inverse-EOT objective identity;
2. Proposition 3.1;
3. Proposition 3.2;
4. the Section 3.3 practical cost/dual parameterization;
5. the weather benchmark; and
6. the Swiss Roll/universal-approximation route.

The newer executable suite uses its own route-oriented labels: Claim 1 for the objective identity, baseline Claims 2–3 for the two closed-form propositions, a practical parameterization result, and a Swiss Roll baseline route recorded as Claim 6. This README uses paper section names instead of silently treating those two numbering schemes as identical.

## Provenance rules

- `main` is the canonical branch for the implementation plus current audit documentation.
- Every clean branch must contain the same `README.md` and `branch-audit.md` documentation once the branch is published.
- Historical `orx/*` names may appear in this file as old-name provenance, but should not remain in live GitHub links or branch names.
- Maintenance commits are authored and committed as `MachineLearning-Nerd <37579156+MachineLearning-Nerd@users.noreply.github.com>`.
- A pending, historical, or deferred route must not be rewritten as a final `VERIFIED` or `FALSIFIED` result.

