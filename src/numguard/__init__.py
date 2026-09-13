"""numguard: audit naive vs numerically-stable ML math kernels.

Compares naive and textbook-stable formulations of softmax, log-sum-exp,
cross-entropy, variance, layer_norm, rms_norm, and kl_divergence against
an independent high-precision (decimal.Decimal) reference on adversarial
fixtures, at float16/float32/float64 precision.
"""

__version__ = "0.4.0"
