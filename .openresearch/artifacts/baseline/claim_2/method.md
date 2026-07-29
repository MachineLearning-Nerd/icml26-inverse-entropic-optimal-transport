# Method

The verifier evaluates the paper's closed form and an independent midpoint
quadrature of the unnormalized energy density over a 401×401 grid. The checker
does not call the closed-form routine. The negative control deletes the
quadratic completion term from `z_mn`; it must exceed the failure threshold.

Command: `uv run --frozen python -m reproduction.run`

Run image: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
