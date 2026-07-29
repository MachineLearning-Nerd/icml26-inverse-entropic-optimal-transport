# Method

The verifier compares the energy kernel and completed-square mixture pointwise,
then compares the normalized closed form with independent 401×401 quadrature.
The negative control uses `A_n` instead of the required `epsilon A_n`
covariance and must fail.

Command: `uv run --frozen python -m reproduction.run`

