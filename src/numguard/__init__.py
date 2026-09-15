"""numguard: audit naive vs numerically-stable ML math kernels.

Compares naive and textbook-stable formulations of softmax, log-sum-exp,
cross-entropy, variance, layer_norm, rms_norm, kl_divergence,
online_softmax, masked_softmax, sum, rope_cos, int8_add, hll_register,
focal_loss_grad, pearson_correlation, and weighted_sampling_key against
an independent high-precision (decimal.Decimal) reference on adversarial
fixtures, at float16/float32/float64 precision (int8_add and
hll_register are scored at float64 only, since their inputs are integer
codes/register values, not a dtype-swept float array).
"""

__version__ = "0.13.0"
