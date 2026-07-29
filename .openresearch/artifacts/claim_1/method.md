# Claim 1 method

Three non-circular routes are cumulative:

1. SymPy independently simplifies Eq. (13) minus Eq. (14)/`epsilon` to zero.
2. On the authors' official `EbieotGmm`, direct `MixtureSameFamily.log_prob`
   is compared with the separately computed cost, potential, and normalization
   terms.
3. At Figure-2 scale (`P=128`, `Q=R=1024`, `N=50`, `M=25`), gradients from
   the paired joint and both marginal data streams must each be nonzero.

Negative controls use the wrong sign for `f^c` and remove both marginal terms.
The fixed command and image are:

```bash
uv run --frozen python -m reproduction.run
ghcr.io/astral-sh/uv:python3.12-bookworm-slim
```

