# Claim 6 Route 1 limitations and deviations

The authors' released baseline sampler uses one 128-by-128 plan, contrary to
Appendix D.1's fresh minibatch per pair. This route repairs the data procedure
while leaving baseline models/objectives unchanged.

CondNF and CondNF(SS) are deferred to a distinct route because the released
supervised config uses 2.5M steps while the semi-supervised config uses 250K,
and the released Eq. 25 implementation repeats only diagonal pairs instead of
the stated Cartesian Monte Carlo marginal. External DCPEME, parOT, OTCS,
FSBM, and GNOT implementations are not bundled. CGMM(SS) is described in the
paper but absent from the release.

Figure 2 reports qualitative plots rather than numeric values. This route
adds preregistered clean-room numerical metrics without inventing a claimed
paper number.
