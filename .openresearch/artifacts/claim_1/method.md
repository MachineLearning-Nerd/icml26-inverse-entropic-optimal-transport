# Claim 1 method

Three non-circular routes are cumulative:

1. SymPy independently simplifies Eq. (13) minus Eq. (14)/`epsilon` to zero.
2. On the authors' official `EbieotGmm`, direct `MixtureSameFamily.log_prob`
   is compared with the separately computed cost, potential, and normalization
   terms.
3. At Figure-2 scale (`P=128`, `Q=R=1024`, `N=50`, `M=25`), gradients from
   the paired joint and both marginal data streams must each be nonzero.

Negative controls use the wrong sign for `f^c` and remove both marginal terms.
The wrong-sign discrepancy must match its independently derived exact residual
`|2 mean(f^c)/epsilon|` within `2e-12`, be non-vacuous (`>1e-8`), and exceed
the correct implementation error by a factor of at least one million.

The first formal run (`0a179058-f6d6-4567-a28e-cf5944068369`) was rejected
because it compared the wrong-sign discrepancy with an arbitrary absolute
threshold of `0.01`. The control was then redesigned around the exact residual;
the scientific calculation was not accepted or reclassified from that run.
The fixed command and image are:

```bash
uv run --frozen python -m reproduction.run
ghcr.io/astral-sh/uv:python3.12-bookworm-slim
```
