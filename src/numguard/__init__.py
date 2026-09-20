"""numguard: audit naive vs numerically-stable ML math kernels.

Compares naive and textbook-stable formulations of softmax, log-sum-exp,
cross-entropy, variance, layer_norm, rms_norm, kl_divergence,
online_softmax, masked_softmax, sum, rope_cos, int8_add, hll_register,
focal_loss_grad, pearson_correlation, explained_variance,
weighted_sampling_key,
geometric_mean, p2_quantile, repetition_penalty, speculative_reject,
weight_decay, gradient_accumulation_bias, longrope_factor_select,
squared_euclidean_distance, bpe_pair_count_overflow,
beam_search_length_penalty, int32_dequant_overflow, norm,
incremental_mean, genlaguerre, mannwhitney_u, i0, and lambertw0 against an
independent high-precision (decimal.Decimal) reference on adversarial
fixtures, at float16/float32/float64 precision (int8_add and hll_register
are scored at float64 only, since their inputs are integer codes/register
values, not a dtype-swept float array; bpe_pair_count_overflow and
int32_dequant_overflow are likewise float64-only, since they audit
integer accumulator/subtraction arithmetic, not a dtype-swept float
array). mannwhitney_u audits scipy.stats.mannwhitneyu's rank-sum
dtype-cast catastrophic-cancellation bug (scipy/scipy#24777). i0 audits
the shared numpy.i0 / scipy.special.i0 / jax.scipy.special.i0
premature-overflow bug (numpy/numpy#32209, scipy/scipy#25823,
jax-ml/jax#39771): all three compute exp(|x|) before dividing by
sqrt(|x|), overflowing for large-but-representable |x|. lambertw0 audits
scipy.special.lambertw's branch-point bug (scipy/scipy#24770, open): the
Halley iteration's initial guess lands exactly on the branch point at
z = -1/e, zeroing a later step's denominator and returning nan where the
true value is exactly -1. sorted_search audits numpy.searchsorted's
uint64-to-float64 promotion bug (numpy/numpy#29727, open): searching a
sorted array of large integers with a plain Python int needle silently
promotes both operands to float64, returning a wrong insertion index
once magnitudes exceed float64's 2**53 exact-integer limit, with no
error, warning, or NaN.
"""

__version__ = "0.32.0"
