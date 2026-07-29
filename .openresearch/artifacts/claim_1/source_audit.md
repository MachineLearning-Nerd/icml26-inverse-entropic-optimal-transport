# Claim 1 source audit

Section 3.1 derives Eq. (13) from conditional maximum likelihood using the
Gibbs-Boltzmann parameterization and
`E_theta(y|x)=(c_theta(x,y)-f_theta(y))/epsilon`. The three population terms
require paired joint samples, unpaired `y` marginal samples, and unpaired `x`
marginal samples respectively.

Section 3.2 substitutes the EOT semi-dual into inverse EOT to obtain Eq. (14).
With `(f_theta)^c_theta(x)=-epsilon log Z_theta(x)`, Eq. (14) divided by
`epsilon` is Eq. (13). The statement is an algebraic identity under the
displayed parameterization, not an asymptotic scaling claim.

Source and hash are recorded in
`.openresearch/artifacts/baseline/source_audit.md`.

