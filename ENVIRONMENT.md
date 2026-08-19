# Environment and reproduction boundary

## Locked command

Run the cumulative suite with:

```bash
uv sync --frozen
uv run --frozen python -m reproduction.run
```

The project supports Python `>=3.10,<3.13` and pins dependencies in
[`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock). The suite selects
the hosted `cpu-upgrade` profile, estimates 54 CPU cores, and is not a
lightweight smoke test.

## Accepted evidence

- Claim 1 accepted formal run: `2cf4da04-c18a-4339-93cf-56279246e6b3`, 37.644 seconds, 64 visible CPUs.
- Claims 2–3 accepted baseline run: `ae3c8b1c-0ad4-4f9b-9ba7-62bd489591fa`, recorded against Git `41e78e57…`.
- The accepted artifacts include explicit thresholds, independent quadrature,
  and invalid-formula or wrong-covariance controls.

## Runtime boundary

No current full cumulative rerun is claimed in this release. The practical
Swiss Roll route is pending, and the weather route needs external TabRED data
and full training. Do not interpret the accepted finite checks as a complete
paper-scale reproduction.
