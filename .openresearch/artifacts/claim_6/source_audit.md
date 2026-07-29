# Claim 6 source audit

Section 5.1 and Figure 2 define a Gaussian-to-Swiss-Roll experiment with
`P=128` paired samples and `Q=R=1024` unpaired samples. Appendix D.1 constructs
the stochastic two-mode ground-truth plan by repeatedly solving a
minibatch-64 entropic Sinkhorn problem under
`min(C_{+90 degrees}, C_{-90 degrees})`.

Appendix C.2 fixes `N=50`, `M=25`, `epsilon=1`, paired learning rate `3e-4`,
unpaired learning rate `1e-3`, and 25,000 iterations. It says `a_m(x)` uses a
two-layer MLP and `v_m(x)` a single-layer MLP. The released
`conf/experiment/gmm_swiss_roll.yaml` instead overrides both hidden-channel
lists to empty, making both towers linear. This campaign therefore calibrates
both interpretations as sibling nodes and promotes the one with lower
independently measured conditional sliced Wasserstein distance.

The paper reports only plots and qualitative comparisons for Figure 2, not a
numeric benchmark table. Numeric sliced-Wasserstein and energy-distance
measurements in this reproduction are independent quantifications of the
displayed conditional and marginal distribution-matching statements.
