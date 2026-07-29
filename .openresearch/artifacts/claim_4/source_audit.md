# Claim 4 source audit

The v5 source defines the practical cost in Section 3.3 Eq. 15, the Gaussian
mixture dual potential in Eq. 16, the closed normalizer in Proposition 3.1,
and the conditional `N*M` Gaussian mixture in Proposition 3.2 Eq. 17. The
imported judge prompt uses the older Eq. 16/Eq. 17 numbering; this campaign
uses the current v5 anchors.

The Swiss-Roll scale is fixed by Section 5.1/Figure 2 (`P=128`,
`Q=R=1024`) and Appendix C.2 (`N=50`, `M=25`, 25,000 updates). The
contract tests likelihood and sampling only after fitting that task-scale
model. Throughput is reported as a measured CPU property, not as an
asymptotic theorem.
