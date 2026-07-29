# Claim 4 method

The representative full semi-supervised seed trains the Appendix-C.2 model
for exactly 25,000 updates. A clean-room evaluator reconstructs all 1,250
mixture weights, means, and diagonal variances from trained tensors and
compares its log density with PyTorch's independent `MixtureSameFamily`.

Sampling is checked with 8,192 draws at a fixed Figure-2 probe. Empirical
mean and covariance are compared with independently calculated mixture
moments. A collapsed-at-the-mean sampler is the negative sampling control.
The likelihood control deliberately omits mixture-weight normalization.
Likelihood and sampling wall times use five repetitions and report medians.
