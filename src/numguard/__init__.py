"""numguard: audit naive vs numerically-stable ML math kernels.

Compares naive and textbook-stable formulations of softmax, log-sum-exp,
cross-entropy, variance, layer_norm, and rms_norm against an independent
high-precision (decimal.Decimal) reference on adversarial fixtures, at
float16/float32/float64 precision.
"""

__version__ = "0.3.2"
