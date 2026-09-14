"""numguard: audit naive vs numerically-stable ML math kernels.

Compares naive and textbook-stable formulations of softmax, log-sum-exp,
cross-entropy, variance, layer_norm, rms_norm, kl_divergence,
online_softmax, masked_softmax, sum, rope_cos, and int8_add against an
independent high-precision (decimal.Decimal) reference on adversarial
fixtures, at float16/float32/float64 precision (int8_add is scored at
float64 only, since its inputs are integer quantization codes, not a
dtype-swept float array).
"""

__version__ = "0.9.0"
