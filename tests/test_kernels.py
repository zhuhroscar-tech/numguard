"""Kernel-level regression tests: each test here is a direct assertion
that a SPECIFIC naive formula breaks in a SPECIFIC documented way and
that the corresponding stable formula does not. These are stronger than
the audit engine's aggregate pass/fail (core.py) because they pin the
exact expected qualitative failure (inf, NaN, or negative variance), so
a change that "fixes" the naive kernel or weakens the fixture would be
caught here even if someone loosened the tolerance in core.py.
"""
from __future__ import annotations

import math

import pytest

from numguard import kernels, reference


class TestLogSumExpOverflow:
    def test_naive_overflows_stable_does_not(self):
        values = [1000.0, 1001.0, 1002.0]
        naive = kernels.naive_logsumexp(values, "float32")
        stable = kernels.stable_logsumexp(values, "float32")
        assert math.isinf(naive)
        assert math.isfinite(stable)
        assert stable == pytest.approx(1002.407605964, rel=1e-6)

    def test_naive_underflows_to_negative_infinity(self):
        values = [-1000.0, -1001.0, -1002.0]
        naive = kernels.naive_logsumexp(values, "float32")
        stable = kernels.stable_logsumexp(values, "float32")
        assert math.isinf(naive) and naive < 0
        assert math.isfinite(stable)


class TestSoftmaxOverflow:
    def test_naive_produces_nan_stable_is_valid_distribution(self):
        values = [50000.0, 1.0, -1.0]
        naive = kernels.naive_softmax(values, "float32")
        stable = kernels.stable_softmax(values, "float32")
        assert any(math.isnan(v) or math.isinf(v) for v in naive)
        assert all(math.isfinite(v) for v in stable)
        assert float(sum(stable)) == pytest.approx(1.0, abs=1e-5)
        # the dominant logit should take essentially all the mass
        assert float(stable[0]) == pytest.approx(1.0, abs=1e-5)


class TestCrossEntropyOverflow:
    def test_naive_nan_from_upstream_softmax_overflow(self):
        values = [50000.0, 1.0, -1.0]
        naive = kernels.naive_cross_entropy(values, 1, "float32")
        stable = kernels.stable_cross_entropy(values, 1, "float32")
        assert math.isnan(naive) or math.isinf(naive)
        assert math.isfinite(stable)
        # target index 1 is far from the dominant logit, so its true
        # cross-entropy should be roughly (50000 - 1) = 49999
        assert stable == pytest.approx(49999.0, rel=1e-6)


class TestVarianceCancellation:
    def test_naive_wrong_by_orders_of_magnitude(self):
        values = [1_000_000.0, 1_000_001.0, 1_000_002.0, 1_000_003.0]
        naive = kernels.naive_variance(values, "float32")
        stable = kernels.stable_variance(values, "float32")
        true_variance = 1.25
        assert stable == pytest.approx(true_variance, rel=1e-4)
        # the naive formula's error should be large relative to the
        # true variance -- pin a concrete threshold rather than an
        # exact brittle value, since exact float32 rounding can vary
        # subtly across numpy/BLAS builds.
        assert abs(naive - true_variance) > 100

    def test_naive_can_be_negative_impossible_for_real_variance(self):
        values = [1_000_000.0, 1_000_001.0, 1_000_002.0, 1_000_003.0]
        naive = kernels.naive_variance(values, "float32")
        # This is the headline demonstration: a real variance can never
        # be negative, but the naive one-pass formula can produce one
        # under catastrophic cancellation.
        assert naive < 0

    def test_stable_never_negative_across_offsets(self):
        for offset in (0.0, 100.0, 1_000_000.0, 3_000_000.0):
            values = [offset, offset + 1.0, offset + 2.0, offset + 3.0]
            for dtype in ("float32", "float64"):
                result = kernels.stable_variance(values, dtype)
                assert result >= 0, f"stable_variance went negative at offset={offset} dtype={dtype}"


class TestEverydayCasesAgree:
    """Baseline sanity: on ordinary inputs, naive and stable formulas
    should agree closely -- these adversarial-fixture failure modes are
    specifically about extreme magnitudes/offsets, not everyday data."""

    def test_logsumexp_small_values(self):
        values = [1.0, 2.0, 3.0]
        naive = kernels.naive_logsumexp(values, "float64")
        stable = kernels.stable_logsumexp(values, "float64")
        assert naive == pytest.approx(stable, rel=1e-9)

    def test_variance_small_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        naive = kernels.naive_variance(values, "float64")
        stable = kernels.stable_variance(values, "float64")
        assert naive == pytest.approx(2.0, rel=1e-9)
        assert stable == pytest.approx(2.0, rel=1e-9)


class TestLayerNormNegativeVarianceNaN:
    """LayerNorm built directly on the naive one-pass variance formula:
    the headline demonstration that the variance-cancellation bug is not
    merely imprecise, it can propagate all the way to NaN activations."""

    def test_naive_nans_out_on_negative_variance(self):
        values = [1_000_000.0, 1_000_001.0, 1_000_002.0, 1_000_003.0]
        naive = kernels.naive_layer_norm(values, "float32")
        stable = kernels.stable_layer_norm(values, "float32")
        # naive variance is negative here (see TestVarianceCancellation),
        # so sqrt(negative + eps) is NaN and every output element is NaN
        assert all(math.isnan(v) for v in naive)
        assert all(math.isfinite(v) for v in stable)

    def test_naive_wrong_by_three_orders_of_magnitude_when_finite(self):
        # a larger offset where naive variance is positive but still
        # wildly wrong (1048576.0 vs true 1.25) -- naive_layer_norm is
        # finite here but shrunk by ~1000x relative to the true values
        values = [3_000_000.0, 3_000_001.0, 3_000_002.0, 3_000_003.0]
        naive = kernels.naive_layer_norm(values, "float32")
        stable = kernels.stable_layer_norm(values, "float32")
        assert all(math.isfinite(v) for v in naive)
        # true normalized values are close to +-1.34/+-0.45; naive
        # values here are shrunk to ~0.001-0.0015 in magnitude
        assert all(abs(v) < 0.01 for v in naive)
        assert float(stable[0]) == pytest.approx(-1.3416407, rel=1e-4)

    def test_stable_never_nan_across_offsets(self):
        for offset in (0.0, 100.0, 1_000_000.0, 3_000_000.0):
            values = [offset, offset + 1.0, offset + 2.0, offset + 3.0]
            for dtype in ("float32", "float64"):
                result = kernels.stable_layer_norm(values, dtype)
                assert all(
                    math.isfinite(v) for v in result
                ), f"stable_layer_norm produced non-finite output at offset={offset} dtype={dtype}"

    def test_everyday_values_naive_and_stable_agree(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        naive = kernels.naive_layer_norm(values, "float64")
        stable = kernels.stable_layer_norm(values, "float64")
        for n, s in zip(naive, stable):
            assert float(n) == pytest.approx(float(s), rel=1e-6)
        # mean-centered, unit-ish variance: middle element should sit at 0
        assert float(stable[2]) == pytest.approx(0.0, abs=1e-9)


class TestRMSNormReductionOverflow:
    """RMSNorm built with the reduction (mean of squares) kept in a
    narrow dtype: the headline demonstration that this is a *distinct*
    failure shape from layer_norm's cancellation-to-NaN -- here ordinary,
    non-adversarial-looking activation values overflow mean(x^2) to inf
    purely from dtype range, and every output silently collapses to
    0.0 rather than NaN."""

    def test_naive_collapses_to_zero_on_fp16_activation_overflow(self):
        # Ordinary-looking float16 activations (not pathological): any
        # one squared (~90000) already exceeds float16's ~65504 max.
        values = [300.0, 305.0, 298.0, 310.0, 301.0]
        naive = kernels.naive_rms_norm(values, "float16")
        stable = kernels.stable_rms_norm(values, "float16")
        assert all(v == 0.0 for v in naive)
        assert all(math.isfinite(v) and v != 0.0 for v in stable)

    def test_naive_collapses_to_zero_on_float32_extreme_overflow(self):
        # Same failure shape one dtype up: squares (~4e38) exceed
        # float32's ~3.4e38 max.
        values = [2e19, 2.1e19, 1.9e19, 2.05e19, 1.95e19]
        naive = kernels.naive_rms_norm(values, "float32")
        stable = kernels.stable_rms_norm(values, "float32")
        assert all(v == 0.0 for v in naive)
        assert all(math.isfinite(v) and v != 0.0 for v in stable)

    def test_stable_never_collapses_across_scales(self):
        for values, dtype in (
            ([300.0, 305.0, 298.0, 310.0, 301.0], "float16"),
            ([2e19, 2.1e19, 1.9e19, 2.05e19, 1.95e19], "float32"),
            ([1.0, 2.0, 3.0, 4.0, 5.0], "float64"),
        ):
            result = kernels.stable_rms_norm(values, dtype)
            assert all(
                math.isfinite(v) and v != 0.0 for v in result
            ), f"stable_rms_norm collapsed at dtype={dtype}"

    def test_everyday_values_naive_and_stable_agree(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        naive = kernels.naive_rms_norm(values, "float64")
        stable = kernels.stable_rms_norm(values, "float64")
        for n, s in zip(naive, stable):
            assert float(n) == pytest.approx(float(s), rel=1e-6)

    def test_all_zero_input_is_zero_not_nan(self):
        # eps guards the 0/0 case for both variants.
        values = [0.0, 0.0, 0.0, 0.0, 0.0]
        for dtype in ("float16", "float32", "float64"):
            naive = kernels.naive_rms_norm(values, dtype)
            stable = kernels.stable_rms_norm(values, dtype)
            assert all(v == 0.0 for v in naive)
            assert all(v == 0.0 for v in stable)


class TestKLDivergenceZeroProbabilityNaN:
    """KL divergence built directly on the textbook sum(p*log(p/q))
    formula: the headline demonstration that a completely ordinary
    input (any p_i == 0, e.g. a one-hot label) hits an IEEE754 0*-inf
    indeterminate form and poisons the whole sum with NaN, even though
    the true KL divergence is an ordinary finite number by the standard
    0*log(0/q) := 0 convention."""

    def test_naive_nans_on_single_zero_probability(self):
        p = [0.5, 0.0, 0.5]
        q = [0.3, 0.3, 0.4]
        naive = kernels.naive_kl_divergence(p, q, "float64")
        stable = kernels.stable_kl_divergence(p, q, "float64")
        assert math.isnan(naive)
        assert math.isfinite(stable)
        assert stable == pytest.approx(0.36698, rel=1e-3)

    def test_naive_nans_on_one_hot_label_across_dtypes(self):
        # The single most common real input shape for this kernel:
        # a one-hot label vector against a soft prediction.
        p = [0.0, 1.0, 0.0, 0.0]
        q = [0.25, 0.25, 0.25, 0.25]
        for dtype in ("float16", "float32", "float64"):
            naive = kernels.naive_kl_divergence(p, q, dtype)
            stable = kernels.stable_kl_divergence(p, q, dtype)
            assert math.isnan(naive), f"expected naive NaN at dtype={dtype}"
            assert math.isfinite(stable), f"expected stable finite at dtype={dtype}"
            assert stable == pytest.approx(math.log(4), rel=1e-2)

    def test_stable_matches_naive_when_no_zero_probabilities(self):
        # Zero-free inputs: naive and stable should agree closely --
        # the guard logic must not change the answer when it isn't needed.
        p = [0.2, 0.3, 0.5]
        q = [0.25, 0.25, 0.5]
        naive = kernels.naive_kl_divergence(p, q, "float64")
        stable = kernels.stable_kl_divergence(p, q, "float64")
        assert math.isfinite(naive)
        assert naive == pytest.approx(stable, rel=1e-9)

    def test_identical_distributions_is_zero(self):
        p = q = [0.25, 0.25, 0.25, 0.25]
        stable = kernels.stable_kl_divergence(p, q, "float64")
        assert stable == pytest.approx(0.0, abs=1e-9)

    def test_stable_is_infinite_when_q_assigns_zero_to_possible_event(self):
        # p says an event can happen (p_i > 0) but q says it can't
        # (q_i == 0) -- the true KL divergence is +inf, and the stable
        # kernel must report that correctly rather than mask it.
        p = [0.5, 0.5]
        q = [0.5, 0.0]
        stable = kernels.stable_kl_divergence(p, q, "float64")
        assert math.isinf(stable) and stable > 0


class TestOnlineSoftmaxMissedRescale:
    """Chunked/streaming (FlashAttention-style) softmax: the headline
    demonstration that a missed incremental rescale of already-
    accumulated partial results is a distinct bug class from every
    other kernel here -- it is wrong regardless of dtype range or
    magnitude, purely because earlier chunks' contributions are never
    corrected once the running max increases."""

    def test_naive_agrees_with_stable_when_max_in_first_chunk(self):
        # Control case: the running max never increases past chunk 0,
        # so there is nothing for the missing rescale step to get
        # wrong -- naive and stable must agree closely here.
        values = [10.0, 1.0, 2.0, 1.0, 0.0, 1.0]
        naive = kernels.naive_online_softmax(values, "float32")
        stable = kernels.stable_online_softmax(values, "float32")
        for n, s in zip(naive, stable):
            assert float(n) == pytest.approx(float(s), rel=1e-4)

    def test_naive_wrong_when_max_in_middle_chunk_no_overflow_involved(self):
        # No extreme magnitude anywhere in this input -- single-digit
        # values only. Naive is still measurably wrong because it never
        # rescales the first chunk's exponentials after later chunks
        # raise the running max.
        values = [1.0, 5.0, 3.0, 8.0, 2.0, 12.0]
        naive = kernels.naive_online_softmax(values, "float32")
        stable = kernels.stable_online_softmax(values, "float32")
        # true softmax puts ~98% of the mass on the last element
        # (value 12.0, the global max); naive badly under-weights it
        # because it never rescaled away the earlier chunks' inflated
        # contributions.
        assert float(stable[-1]) == pytest.approx(0.9809567, rel=1e-3)
        assert float(naive[-1]) < 0.5, (
            "naive_online_softmax should badly under-weight the true "
            "max when it arrives in a later chunk"
        )
        assert abs(float(naive[-1]) - float(stable[-1])) > 0.4

    def test_naive_still_wrong_when_true_max_arrives_in_last_chunk(self):
        # The true global max is far larger than everything before it
        # and only appears in the final chunk -- stable must still
        # recover an (approximately) one-hot distribution; naive must
        # not.
        values = [1.0, 2.0, 3.0, 1.0, 2.0, 50000.0]
        naive = kernels.naive_online_softmax(values, "float32")
        stable = kernels.stable_online_softmax(values, "float32")
        assert float(stable[-1]) == pytest.approx(1.0, abs=1e-5)
        assert float(naive[-1]) < 0.9, (
            "naive_online_softmax should not recover the correct "
            "near-one-hot distribution when the max arrives late"
        )

    def test_stable_matches_reference_softmax_regardless_of_chunk_size(self):
        # stable_online_softmax must be algorithmically equivalent to
        # the whole-array stable softmax, independent of chunk_size.
        values = [1.0, 5.0, 3.0, 8.0, 2.0, 12.0, 4.0, 7.0]
        whole = kernels.stable_softmax(values, "float32")
        for chunk_size in (1, 2, 3, 4, 8):
            chunked = kernels.stable_online_softmax(values, "float32", chunk_size=chunk_size)
            for w, c in zip(whole, chunked):
                assert float(w) == pytest.approx(float(c), rel=1e-4), (
                    f"stable_online_softmax diverged from stable_softmax at chunk_size={chunk_size}"
                )

    def test_everyday_monotonic_case_naive_diverges_from_true_softmax(self):
        # Every chunk raises the running max -- the worst case for the
        # number of skipped rescales.
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        naive = kernels.naive_online_softmax(values, "float64", chunk_size=2)
        stable = kernels.stable_online_softmax(values, "float64", chunk_size=2)
        true_last = math.exp(6.0) / sum(math.exp(v) for v in values)
        assert float(stable[-1]) == pytest.approx(true_last, rel=1e-6)
        assert float(naive[-1]) != pytest.approx(true_last, rel=1e-2)


class TestMaskedSoftmaxAllMaskedNaN:
    """Attention padding/causal masking: the headline demonstration is
    that a fully-masked row (every position excluded -- the real shape
    of a fully-padded batch row) NaNs out the naive masked_fill(-inf)
    pattern entirely, even though no individual logit is extreme."""

    def test_naive_all_masked_row_is_nan(self):
        values = [3.0, -1.0, 4.0, 2.0]
        mask = [False, False, False, False]
        naive = kernels.naive_masked_softmax(values, mask, "float32")
        stable = kernels.stable_masked_softmax(values, mask, "float32")
        assert all(math.isnan(v) for v in naive), (
            "naive_masked_softmax should NaN out entirely when every "
            "position is masked (0.0/0.0 from an empty support)"
        )
        assert all(v == pytest.approx(0.0) for v in stable), (
            "stable_masked_softmax should return an all-zero row for "
            "an empty support instead of NaN"
        )

    def test_naive_all_masked_with_extreme_logit_still_nan(self):
        values = [50000.0, 1.0, -1.0]
        mask = [False, False, False]
        naive = kernels.naive_masked_softmax(values, mask, "float32")
        stable = kernels.stable_masked_softmax(values, mask, "float32")
        assert all(math.isnan(v) for v in naive)
        assert all(v == pytest.approx(0.0) for v in stable)

    def test_naive_single_unmasked_extreme_logit_overflows_to_nan(self):
        # A second, independent way naive_masked_softmax can fail: not
        # every failure mode here is the all-masked case -- an extreme
        # *unmasked* logit still overflows the naive exp() the same way
        # plain naive_softmax does.
        values = [50000.0, 1.0, -1.0, 2.0]
        mask = [True, False, False, False]
        naive = kernels.naive_masked_softmax(values, mask, "float32")
        stable = kernels.stable_masked_softmax(values, mask, "float32")
        assert any(math.isnan(v) or math.isinf(v) for v in naive)
        assert all(math.isfinite(v) for v in stable)
        assert float(stable[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(stable[1]) == pytest.approx(0.0, abs=1e-9)

    def test_stable_masked_positions_always_exactly_zero(self):
        values = [2.0, 5.0, 1.0, 8.0, 0.5]
        mask = [True, True, False, True, False]
        stable = kernels.stable_masked_softmax(values, mask, "float32")
        assert float(stable[2]) == 0.0
        assert float(stable[4]) == 0.0
        assert float(sum(stable)) == pytest.approx(1.0, abs=1e-5)

    def test_naive_and_stable_agree_on_ordinary_partial_masking(self):
        # Control case: with at least one unmasked position and no
        # extreme magnitudes, naive and stable should agree closely --
        # the bug is specifically conditional on the all-masked (or
        # extreme-unmasked-logit) case, not on masking itself.
        values = [2.0, 5.0, 1.0, 8.0, 0.5]
        mask = [True, True, False, True, False]
        naive = kernels.naive_masked_softmax(values, mask, "float64")
        stable = kernels.stable_masked_softmax(values, mask, "float64")
        for n, s in zip(naive, stable):
            assert float(n) == pytest.approx(float(s), rel=1e-9)

    def test_all_visible_reduces_to_ordinary_softmax(self):
        values = [1.0, 2.0, 3.0, 4.0]
        mask = [True, True, True, True]
        masked = kernels.stable_masked_softmax(values, mask, "float64")
        plain = kernels.stable_softmax(values, "float64")
        for m, p in zip(masked, plain):
            assert float(m) == pytest.approx(float(p), rel=1e-9)


class TestSumAccumulatedRoundingError:
    """Regression tests pinning summation's distinct O(n) accumulated-
    rounding-error failure -- neither overflow, cancellation, nor an
    indeterminate form, but a magnitude of error that grows with the
    number of terms because naive sequential summation lets the same
    running total absorb every addition's rounding in sequence."""

    def test_naive_wrong_on_many_small_uniform_terms(self):
        values = [1e-4] * 20000
        true_sum = 2.0
        naive = kernels.naive_sum(values, "float32")
        stable = kernels.stable_sum(values, "float32")
        assert math.isfinite(naive) and math.isfinite(stable)
        assert stable == pytest.approx(true_sum, abs=1e-6)
        # naive should be measurably off (not just last-bit rounding)
        assert abs(naive - true_sum) > 1e-4

    def test_naive_wrong_when_large_value_swamps_small_terms(self):
        values = [1e8] + [1.0] * 20000
        true_sum = 100020000.0
        naive = kernels.naive_sum(values, "float32")
        stable = kernels.stable_sum(values, "float32")
        assert stable == pytest.approx(true_sum, rel=1e-6)
        # naive drops most or all of the +1.0 increments once the
        # accumulator reaches ~1e8 (float32 has ~7 significant digits)
        assert abs(naive - true_sum) > 1000

    def test_naive_wrong_at_float16_with_far_fewer_terms(self):
        values = [0.01] * 3000
        true_sum = 30.0
        naive = kernels.naive_sum(values, "float16")
        stable = kernels.stable_sum(values, "float16")
        assert stable == pytest.approx(true_sum, rel=1e-2)
        assert abs(naive - true_sum) > 0.5

    def test_naive_and_stable_agree_on_a_handful_of_values(self):
        # Control: the bug is about scale (many terms), not about naive
        # summation being wrong in general -- a handful of ordinary
        # values should agree closely between naive and stable.
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        naive = kernels.naive_sum(values, "float64")
        stable = kernels.stable_sum(values, "float64")
        assert naive == pytest.approx(15.0, rel=1e-12)
        assert stable == pytest.approx(15.0, rel=1e-12)

    def test_stable_never_worse_than_naive_across_scales(self):
        # Property check: Kahan summation's error should never exceed
        # naive summation's error, across a range of term counts.
        true_val = 3.0
        for n in (100, 1000, 10000, 30000):
            values = [1e-4] * n
            true_sum = n * 1e-4
            naive = kernels.naive_sum(values, "float32")
            stable = kernels.stable_sum(values, "float32")
            assert abs(stable - true_sum) <= abs(naive - true_sum) + 1e-9


class TestRopeCosPositionAliasing:
    """Regression tests pinning the RoPE position-aliasing bug: when a
    rotary-embedding implementation commits position ids (and the
    per-dimension inv_freq) to a low-precision dtype *before* computing
    position*freq and its cos(), distinct nearby positions can round to
    the identical low-precision angle -- the same real-world bug
    documented for bfloat16 by Baichuan Inc.
    (zhuanlan.zhihu.com/p/651588659) and independently fixed in
    HuggingFace transformers (PR #29285, "Force float32 ... since
    bfloat16 loses precision on long contexts"), reproduced here at
    float16 (a precision this pure-numpy tool can exercise directly)."""

    def test_naive_collides_distinct_positions_stable_does_not(self):
        # Four distinct integer positions beyond float16's exact-integer
        # range (2048): naive (compute in float16 throughout) must
        # collapse at least two of them onto the identical angle, while
        # stable (compute in float32, cast only the final cos() down)
        # must keep them distinguishable.
        positions = [16384.0, 16385.0, 16392.0, 16399.0]
        freq = 0.1
        naive = kernels.naive_rope_cos(positions, freq, "float16")
        stable = kernels.stable_rope_cos(positions, freq, "float16")
        # naive: positions 16384, 16385, 16392 collide onto one angle
        assert naive[0] == naive[1] == naive[2]
        # stable: no such collision -- every value should differ
        stable_vals = [float(v) for v in stable]
        assert len(set(stable_vals)) == len(stable_vals)

    def test_naive_wrong_relative_to_true_cosine_stable_is_right(self):
        import math as _math

        positions = [16384.0, 16385.0, 16392.0, 16399.0]
        freq = 0.1
        naive = kernels.naive_rope_cos(positions, freq, "float16")
        stable = kernels.stable_rope_cos(positions, freq, "float16")
        true_vals = [_math.cos(p * freq) for p in positions]
        # naive should be far off on at least one of the colliding
        # positions (position 16384 has true cos ~0.059 but the naive
        # angle collision reports a very different value)
        naive_errs = [abs(float(n) - t) for n, t in zip(naive, true_vals)]
        stable_errs = [abs(float(s) - t) for s, t in zip(stable, true_vals)]
        assert max(naive_errs) > 0.1
        assert max(stable_errs) < 0.01

    def test_naive_and_stable_agree_on_short_context(self):
        # Control: ordinary short-sequence position ids should not
        # trigger any collision -- the bug is about scale, not the
        # formula being wrong in general.
        positions = [0.0, 1.0, 2.0, 3.0]
        freq = 0.3
        naive = kernels.naive_rope_cos(positions, freq, "float32")
        stable = kernels.stable_rope_cos(positions, freq, "float32")
        for n, s in zip(naive, stable):
            assert float(n) == pytest.approx(float(s), abs=1e-5)

    def test_stable_never_worse_than_naive_at_long_context(self):
        import math as _math

        positions = [16384.0, 16385.0, 16392.0, 16399.0]
        freq = 0.1
        naive = kernels.naive_rope_cos(positions, freq, "float16")
        stable = kernels.stable_rope_cos(positions, freq, "float16")
        true_vals = [_math.cos(p * freq) for p in positions]
        for n, s, t in zip(naive, stable, true_vals):
            assert abs(float(s) - t) <= abs(float(n) - t) + 1e-9


class TestInt8AddQuantMismatch:
    """int8_add: regression tests pinning the two documented real-world
    failure shapes (OpenVINO PR#7305/PR#1135 mismatched-scale Eltwise
    fusion; openvino#34673-style saturation-wraparound) so a future
    change that "fixes" the naive kernel or weakens a fixture value
    would be caught here even if core.py's tolerance were loosened."""

    def test_naive_reuses_operand_a_params_for_b(self):
        # a_code=60 @ (scale=0.05, zp=0) -> real_a = 3.0
        # b_code=4  @ (scale=1.0,  zp=0) -> real_b = 4.0 (true)
        # naive wrongly dequantizes b with A's params: 4 * 0.05 = 0.2
        # true sum = 7.0 -> requantized @ scale_out=0.1 -> code 70
        # naive sum = 3.2 -> requantized @ scale_out=0.1 -> code 32
        naive = kernels.naive_int8_add(60.0, 4.0, 0, 0.05, 0, 1.0, 0, 0.1)
        stable = kernels.stable_int8_add(60.0, 4.0, 0, 0.05, 0, 1.0, 0, 0.1)
        assert naive == pytest.approx(32.0)
        assert stable == pytest.approx(70.0)
        assert naive != stable

    def test_naive_wraps_on_overflow_stable_saturates(self):
        # a_code=100, b_code=100, matching (scale=1.0, zp=0): true sum
        # is 200.0, requantized code 200 -- outside int8 range.
        naive = kernels.naive_int8_add(100.0, 100.0, 0, 1.0, 0, 1.0, 0, 1.0)
        stable = kernels.stable_int8_add(100.0, 100.0, 0, 1.0, 0, 1.0, 0, 1.0)
        assert naive == pytest.approx(-56.0)  # wraps: 200 -> -56 mod 256
        assert stable == pytest.approx(127.0)  # saturates at int8 max

    def test_naive_and_stable_agree_when_params_match(self):
        # Control: when both operands already share (scale, zero_point),
        # reusing A's params for B is a no-op, so there is nothing to
        # mismatch and naive/stable must agree exactly.
        naive = kernels.naive_int8_add(50.0, 30.0, 0, 0.1, 0, 0.1, 0, 0.1)
        stable = kernels.stable_int8_add(50.0, 30.0, 0, 0.1, 0, 0.1, 0, 0.1)
        assert naive == pytest.approx(stable)
        assert naive == pytest.approx(80.0)


class TestHLLRegisterShiftOverflow:
    """hll_register: regression tests pinning the exact FLINK-39399
    failure shape (a fixed-width 32-bit int shift silently masking the
    shift distance modulo 32 for rank >= 32) so a future change that
    "fixes" the naive kernel or weakens a fixture would be caught here
    even if core.py's tolerance were loosened."""

    def test_naive_and_stable_agree_below_shift_width(self):
        # Control: rank=17 never approaches the 32-bit shift-width
        # boundary, so `rank % 32 == rank` and both formulas compute
        # the identical, correct 2**-17.
        naive = kernels.naive_hll_register_term(17)
        stable = kernels.stable_hll_register_term(17)
        assert naive == pytest.approx(stable)
        assert naive == pytest.approx(2.0 ** -17)

    def test_naive_and_stable_agree_at_boundary_rank_31(self):
        # Control: rank=31 is the last value where `rank % 32 == rank`
        # -- confirms the divergence below is specifically a >=32
        # shift-width defect, not a general off-by-one in rank handling.
        naive = kernels.naive_hll_register_term(31)
        stable = kernels.stable_hll_register_term(31)
        assert naive == pytest.approx(stable)
        assert naive == pytest.approx(2.0 ** -31)

    def test_naive_diverges_from_stable_at_flink_39399_repro_rank(self):
        # FLINK-39399's own repro: rank=35. The buggy 32-bit shift
        # computes `1 << (35 % 32)` = `1 << 3` = 8, giving a term of
        # 1/8 = 0.125 -- ten orders of magnitude too large compared to
        # the true 2**-35 (~2.9e-11). This is a real, qualitatively
        # different wrong answer, not merely an imprecise one.
        naive = kernels.naive_hll_register_term(35)
        stable = kernels.stable_hll_register_term(35)
        assert naive == pytest.approx(0.125)
        assert stable == pytest.approx(2.0 ** -35)
        assert naive != pytest.approx(stable)
        # The naive term is roughly 2**32 times too large.
        assert naive / stable == pytest.approx(2.0 ** 32, rel=1e-9)

    def test_naive_diverges_far_beyond_boundary_rank_51(self):
        # A near-maximum real register value (64-bit hash, p=14 gives
        # up to ~51): confirms the bug corrupts the entire upper half
        # of the representable range, not just values just past 32.
        naive = kernels.naive_hll_register_term(51)
        stable = kernels.stable_hll_register_term(51)
        assert naive != pytest.approx(stable)
        assert stable == pytest.approx(2.0 ** -51)


class TestFocalLossGradSaturatedZeroPower:
    """focal_loss_grad: regression tests pinning the exact sam3#575
    failure shape (an explicit (1-p_t)**(gamma-1) factor evaluating
    0**-1 == inf, then 0*inf == NaN, at a saturated-and-correct
    prediction with gamma <= 1) so a future change that 'fixes' the
    naive kernel or weakens a fixture would be caught here even if
    core.py's tolerance were loosened."""

    def test_naive_and_stable_agree_on_unsaturated_gamma_two(self):
        # Control: an ordinary unsaturated logit at the common gamma=2
        # default -- no saturation, no negative power, both formulas
        # should agree closely.
        naive = kernels.naive_focal_loss_grad(0.0, 1, 2.0, 0.25, "float64")
        stable = kernels.stable_focal_loss_grad(0.0, 1, 2.0, 0.25, "float64")
        assert naive == pytest.approx(stable, rel=1e-9)

    def test_naive_and_stable_agree_on_unsaturated_gamma_zero(self):
        # Control: gamma=0 alone, without saturation, is not sufficient
        # to trigger the bug -- isolates saturation as the second
        # necessary precondition.
        naive = kernels.naive_focal_loss_grad(0.0, 1, 0.0, 0.5, "float64")
        stable = kernels.stable_focal_loss_grad(0.0, 1, 0.0, 0.5, "float64")
        assert naive == pytest.approx(stable, rel=1e-9)
        assert naive == pytest.approx(-0.25, rel=1e-6)

    def test_naive_nans_on_saturated_correct_prediction_gamma_zero(self):
        # The sam3#575 repro shape: a confidently-correct, saturated
        # logit at gamma=0 -- naive's 0**-1 * 0 = NaN, but the true
        # gradient here is an ordinary tiny finite number.
        for dtype, logit in (("float16", 10.0), ("float32", 18.0), ("float64", 40.0)):
            naive = kernels.naive_focal_loss_grad(logit, 1, 0.0, 0.5, dtype)
            stable = kernels.stable_focal_loss_grad(logit, 1, 0.0, 0.5, dtype)
            assert math.isnan(naive), f"expected naive NaN at dtype={dtype}"
            assert math.isfinite(stable), f"expected stable finite at dtype={dtype}"

    def test_naive_nans_on_saturated_wrong_direction_gamma_zero(self):
        # The mirror case via the target=0 branch of the formula.
        naive = kernels.naive_focal_loss_grad(-18.0, 0, 0.0, 0.5, "float32")
        stable = kernels.stable_focal_loss_grad(-18.0, 0, 0.0, 0.5, "float32")
        assert math.isnan(naive)
        assert math.isfinite(stable)

    def test_naive_nans_on_saturated_fractional_gamma(self):
        # van Leeuwen et al. (TMLR 2025): the instability spans the
        # whole 0 <= gamma < 1 range, not just exactly gamma=0.
        naive = kernels.naive_focal_loss_grad(18.0, 1, 0.5, 0.25, "float32")
        stable = kernels.stable_focal_loss_grad(18.0, 1, 0.5, 0.25, "float32")
        assert math.isnan(naive)
        assert math.isfinite(stable)

    def test_naive_is_actually_fine_at_saturated_gamma_two_control(self):
        # Control: the same saturated logit at gamma=2 (a non-negative
        # exponent) -- naive is NOT buggy here, confirming the failure
        # is specific to gamma < 1, not saturation alone.
        naive = kernels.naive_focal_loss_grad(18.0, 1, 2.0, 0.25, "float32")
        stable = kernels.stable_focal_loss_grad(18.0, 1, 2.0, 0.25, "float32")
        assert math.isfinite(naive)
        assert naive == pytest.approx(stable, abs=1e-6)


class TestPearsonCorrelationCancellation:
    """Pearson's r via the textbook one-pass "sum of products" formula
    hits the same catastrophic-cancellation shape as this repo's
    `variance` kernel (E[x^2]-E[x]^2), but on two variables' cross term
    at once -- the exact failure class scipy.stats.pearsonr needed
    three separate bug reports (gh-8980, gh-9353, gh-3728) and a full
    rewrite (PR#9562) to fix."""

    def test_naive_fails_stable_survives_large_offset_float32(self):
        x = [10_001.0, 10_002.0, 10_003.0, 10_004.0, 10_005.0]
        y = [10_005.0, 10_003.0, 10_004.0, 10_002.0, 10_001.0]
        naive = kernels.naive_pearson_correlation(x, y, "float32")
        stable = kernels.stable_pearson_correlation(x, y, "float32")
        assert math.isnan(naive)
        assert math.isfinite(stable)
        assert stable == pytest.approx(-0.9, abs=1e-3)

    def test_naive_fails_worse_at_larger_offset_float32(self):
        x = [1_000_001.0, 1_000_002.0, 1_000_003.0, 1_000_004.0, 1_000_005.0]
        y = [1_000_005.0, 1_000_003.0, 1_000_004.0, 1_000_002.0, 1_000_001.0]
        naive = kernels.naive_pearson_correlation(x, y, "float32")
        stable = kernels.stable_pearson_correlation(x, y, "float32")
        assert math.isnan(naive)
        assert math.isfinite(stable)
        assert stable == pytest.approx(-0.9, abs=1e-2)

    def test_naive_fails_stable_survives_modest_offset_float16(self):
        x = [51.0, 52.0, 53.0, 54.0]
        y = [54.0, 52.0, 53.0, 51.0]
        naive = kernels.naive_pearson_correlation(x, y, "float16")
        stable = kernels.stable_pearson_correlation(x, y, "float16")
        assert math.isnan(naive)
        assert math.isfinite(stable)
        assert stable == pytest.approx(-0.8, abs=1e-2)

    def test_both_agree_on_clean_no_offset_input(self):
        # Zero-free, no-offset control: both formulas should agree
        # closely -- the bug is conditional on offset/scale, not the
        # correlation formula being wrong in general.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 5.0, 4.0, 6.0]
        naive = kernels.naive_pearson_correlation(x, y, "float64")
        stable = kernels.stable_pearson_correlation(x, y, "float64")
        assert math.isfinite(naive)
        assert naive == pytest.approx(stable, rel=1e-9)
        assert stable == pytest.approx(0.852803, rel=1e-5)

    def test_stable_returns_nan_for_exactly_constant_input(self):
        # Pearson's r is mathematically undefined when either vector
        # has zero variance -- this must return NaN explicitly
        # (scipy's PearsonRConstantInputWarning convention), not an
        # arbitrary implementation-dependent +-1/0 (the real
        # numpy#32446 / pandas#67023 / pandas#37448 failure shape).
        x = [7.0] * 5
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        stable = kernels.stable_pearson_correlation(x, y, "float64")
        assert math.isnan(stable)

    def test_perfect_correlation_is_exactly_one(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        stable = kernels.stable_pearson_correlation(x, y, "float64")
        assert stable == pytest.approx(1.0, abs=1e-9)


class TestWeightedSamplingKeyUnderflow:
    """A-Res-style weighted-reservoir-sampling comparison keys
    (Efraimidis & Spirakis 2006): computing u**(1/weight) BEFORE taking
    the log (the paper's own verbatim formula, shipped as R's wrswoR
    `sample_int_expjs` and explicitly documented there as "at the cost
    of numerical stability") underflows to exactly 0.0 for small
    weights, discarding real order-relevant magnitude behind log(0.0)
    = -inf. The log-space fix (`sample_int_expj`: compute log(u)/weight
    directly) never materializes the intermediate power at all."""

    def test_naive_underflows_stable_survives_float64(self):
        naive = kernels.naive_weighted_sampling_key(0.1, 0.001, "float64")
        stable = kernels.stable_weighted_sampling_key(0.1, 0.001, "float64")
        assert math.isinf(naive) and naive < 0
        assert math.isfinite(stable)
        assert stable == pytest.approx(-2302.585092994046, rel=1e-9)

    def test_naive_underflows_stable_survives_float32(self):
        naive = kernels.naive_weighted_sampling_key(0.5, 0.005, "float32")
        stable = kernels.stable_weighted_sampling_key(0.5, 0.005, "float32")
        assert math.isinf(naive) and naive < 0
        assert math.isfinite(stable)
        assert stable == pytest.approx(-138.6294361, rel=1e-4)

    def test_naive_underflows_stable_survives_float16(self):
        naive = kernels.naive_weighted_sampling_key(0.9, 0.005, "float16")
        stable = kernels.stable_weighted_sampling_key(0.9, 0.005, "float16")
        assert math.isinf(naive) and naive < 0
        assert math.isfinite(stable)
        assert stable == pytest.approx(-21.0937, rel=1e-2)

    def test_both_agree_on_ordinary_weight(self):
        # Control: an everyday weight (1.0) never approaches the
        # underflow boundary, so both formulas should agree closely.
        naive = kernels.naive_weighted_sampling_key(0.5, 1.0, "float64")
        stable = kernels.stable_weighted_sampling_key(0.5, 1.0, "float64")
        assert math.isfinite(naive)
        assert naive == pytest.approx(stable, rel=1e-9)
        assert stable == pytest.approx(math.log(0.5), rel=1e-9)

    def test_both_agree_just_above_the_underflow_boundary(self):
        # A weight one order of magnitude away from underflowing at
        # float64 (0.05 vs the 0.001 fixture above) -- confirms the
        # bug is conditional on the underflow boundary itself, not
        # small weights in general.
        naive = kernels.naive_weighted_sampling_key(0.5, 0.05, "float64")
        stable = kernels.stable_weighted_sampling_key(0.5, 0.05, "float64")
        assert math.isfinite(naive)
        assert naive == pytest.approx(stable, rel=1e-6)


class TestGeometricMeanOverflow:
    """naive_geometric_mean forms the raw product prod(x) before taking
    the n-th root; stable_geometric_mean works in log-space
    (exp(mean(log(x)))) instead -- mirrors this repo's other naive/
    stable pairs (logsumexp, weighted_sampling_key) that move an
    overflow/underflow-prone operation into log space."""

    def test_naive_overflows_stable_survives_float16(self):
        naive = kernels.naive_geometric_mean([10.0] * 15, "float16")
        stable = kernels.stable_geometric_mean([10.0] * 15, "float16")
        assert math.isinf(naive) and naive > 0
        assert math.isfinite(stable)
        assert stable == pytest.approx(10.0, rel=1e-2)

    def test_naive_overflows_stable_survives_float32(self):
        naive = kernels.naive_geometric_mean([1000.0] * 40, "float32")
        stable = kernels.stable_geometric_mean([1000.0] * 40, "float32")
        assert math.isinf(naive) and naive > 0
        assert math.isfinite(stable)
        assert stable == pytest.approx(1000.0, rel=1e-3)

    def test_naive_underflows_stable_survives_float32(self):
        naive = kernels.naive_geometric_mean([0.001] * 40, "float32")
        stable = kernels.stable_geometric_mean([0.001] * 40, "float32")
        assert naive == 0.0
        assert math.isfinite(stable) and stable > 0
        assert stable == pytest.approx(0.001, rel=1e-3)

    def test_naive_overflows_stable_survives_float64(self):
        # Even float64's much wider range is not immune: 400 copies of
        # an ordinary value still overflow the raw product.
        naive = kernels.naive_geometric_mean([10.0] * 400, "float64")
        stable = kernels.stable_geometric_mean([10.0] * 400, "float64")
        assert math.isinf(naive) and naive > 0
        assert math.isfinite(stable)
        assert stable == pytest.approx(10.0, rel=1e-9)

    def test_both_agree_on_everyday_values(self):
        # Control: an ordinary two-value case (true gmean = 4.0
        # exactly) never approaches the overflow/underflow boundary,
        # so both formulas should agree closely.
        naive = kernels.naive_geometric_mean([2.0, 8.0], "float64")
        stable = kernels.stable_geometric_mean([2.0, 8.0], "float64")
        assert math.isfinite(naive)
        assert naive == pytest.approx(stable, rel=1e-9)
        assert naive == pytest.approx(4.0, rel=1e-9)


class TestP2QuantileDrift:
    """naive_p2_quantile accumulates each marker's desired position via
    a per-step increment (`ns[i] += dns[i]`), reproducing the real
    rounding-drift bug reported and fixed in Andrey Akinshin's
    `perfolizer` library (GitHub issue #8); stable_p2_quantile
    recomputes each marker's desired position fresh from the running
    count every step instead -- the author's own published fix."""

    UNIFORM_STREAM = [
        46.3007, 37.3312, 13.8539, 86.6562, 0.6435, 50.2782, 89.8298,
        8.0815, 55.427, 61.665, 4.0896, 37.902, 70.348, 45.2021,
        72.5065, 15.7157, 23.8012, 11.0948, 50.6269, 92.383, 59.0428,
        77.4209, 38.3665, 74.6095, 10.1669, 29.1178, 67.4236, 72.5706,
        42.1755, 8.7712, 26.6734, 20.989, 28.1184, 80.9511, 19.9483,
        88.64, 87.9373, 5.4789, 37.8816, 49.1712, 2.3483, 42.4725,
        90.6411, 11.2046, 59.6846, 12.1232, 57.87, 89.5303, 20.3053,
        0.8253, 8.3504, 53.9769, 1.7465, 8.4837, 49.6742, 92.0926,
        42.0107, 39.8135, 63.8718, 9.3418, 57.98, 17.2555, 60.8888,
        95.8326, 5.4173, 55.5061, 60.6381, 14.9304, 26.8311, 99.4884,
        99.7964, 12.1336, 70.5468, 95.0923, 23.6786, 61.1127, 4.3031,
        36.5947, 67.4125, 59.0259, 77.4625, 8.6739, 34.7198, 86.4036,
        58.414, 45.13, 40.217, 98.6072, 57.4436, 1.8367, 79.937,
        32.8712, 43.3572, 21.3427, 44.4014, 32.471, 8.8817, 62.951,
        10.3069, 78.4099, 2.5391, 78.0721, 80.755, 49.7331, 70.9448,
        24.8261, 73.7617, 42.5007, 23.0953, 96.4075, 40.0904, 37.2969,
        85.9901, 36.936, 66.7505, 17.106, 84.3274, 25.8912, 5.0504,
        97.5259, 17.2756, 94.6519, 98.6166, 60.6558, 1.1875, 6.0911,
        20.8602, 38.8879, 61.1549, 96.6433, 35.4531, 14.0645, 56.1932,
        13.73, 8.6588, 55.596, 69.5988, 6.5597, 45.14, 70.3844,
        76.417, 38.3001, 88.7177, 16.908, 71.5466, 77.1705, 88.0256,
        49.4436, 9.9226, 4.846, 52.8838, 17.3329, 62.9684, 8.4272,
        78.0149, 22.2408, 1.3014, 17.5851, 45.6698, 55.9245, 38.8284,
        17.305, 48.2062, 94.6437, 53.543, 94.1287, 2.8773, 99.3154,
        88.8994, 54.4354, 52.3495, 53.7124, 90.9534, 6.5579, 64.281,
        54.2151, 30.0318, 72.4927, 72.0676, 10.321, 69.9521, 45.3532,
        49.0216, 63.6762, 5.2948, 60.2918, 37.3143, 87.8734, 23.1105,
        82.3122, 72.959, 62.4986, 87.5849, 3.5999, 59.6959, 61.3274,
        67.8056, 40.6676, 6.8965, 18.909, 60.8116, 18.1313, 6.4877,
        35.4794, 47.0242, 53.5569, 2.5972, 77.5574, 33.3829, 78.2821,
        0.857, 95.4039, 59.0245, 97.6365, 98.6315, 83.2784, 10.6235,
        34.8763, 23.1316, 77.9985, 19.2163, 22.1154, 11.0606, 12.0091,
        93.8131, 97.6193, 37.2428, 74.1056, 46.7509, 52.2014, 36.9488,
        63.4714, 23.6153, 25.6365, 51.7827, 19.7358, 43.9256, 94.3902,
        2.2234, 10.3254, 79.997, 5.7803, 25.1236, 85.164, 60.5253,
        22.0701, 5.8609, 27.5184, 30.0149, 80.0279, 97.5554, 78.6289,
        90.175, 93.0681, 87.6575, 91.9975, 96.7538, 9.4415, 31.7678,
        24.7183, 60.417, 49.1705, 34.6942, 83.7104, 39.9092, 20.8932,
        27.6388, 46.175, 61.0469, 56.8438, 43.9933, 18.6335, 40.2166,
        32.5024, 17.0568, 54.3882, 18.8562, 71.1602, 65.8763, 13.7693,
        62.5371, 33.7847, 44.7076, 60.6556, 76.1604, 88.3575, 73.355,
        21.1233, 54.1939, 33.537, 1.9122, 55.2461, 47.0597, 86.6907,
        32.3688, 83.357, 51.4144, 9.9094, 71.6918, 28.4658,
    ]

    def test_naive_drifts_stable_survives_float64(self):
        gold = float(reference.gold_p2_quantile(self.UNIFORM_STREAM, 0.1))
        naive = kernels.naive_p2_quantile(self.UNIFORM_STREAM, 0.1, "float64")
        stable = kernels.stable_p2_quantile(self.UNIFORM_STREAM, 0.1, "float64")
        assert abs(naive - gold) / gold > 0.03
        assert stable == pytest.approx(gold, rel=1e-9)

    def test_naive_drifts_stable_survives_float32(self):
        gold = float(reference.gold_p2_quantile(self.UNIFORM_STREAM, 0.1))
        naive = kernels.naive_p2_quantile(self.UNIFORM_STREAM, 0.1, "float32")
        stable = kernels.stable_p2_quantile(self.UNIFORM_STREAM, 0.1, "float32")
        assert abs(naive - gold) / gold > 0.03
        assert stable == pytest.approx(gold, rel=1e-4)

    def test_naive_drifts_stable_survives_float16(self):
        gold = float(reference.gold_p2_quantile(self.UNIFORM_STREAM, 0.1))
        naive = kernels.naive_p2_quantile(self.UNIFORM_STREAM, 0.1, "float16")
        stable = kernels.stable_p2_quantile(self.UNIFORM_STREAM, 0.1, "float16")
        assert abs(naive - gold) / gold > 0.05
        assert stable == pytest.approx(gold, rel=5e-2)

    def test_sine_stream_deterministic_drift_float64(self):
        # No RNG: a fixed sine-wave stream, hand-reviewable, that
        # demonstrates the same drift mechanism at p=0.35.
        values = [round(50 + 40 * math.sin(i * 0.7), 6) for i in range(200)]
        gold = float(reference.gold_p2_quantile(values, 0.35))
        naive = kernels.naive_p2_quantile(values, 0.35, "float64")
        stable = kernels.stable_p2_quantile(values, 0.35, "float64")
        assert abs(naive - gold) / gold > 0.02
        assert stable == pytest.approx(gold, rel=1e-9)

    def test_both_agree_on_tiny_stream_control(self):
        # Control: fewer than 5 marker-adjustment steps ever run, so
        # there is no opportunity for the accumulated-increment
        # bookkeeping to drift -- naive and stable agree exactly.
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        naive = kernels.naive_p2_quantile(values, 0.5, "float64")
        stable = kernels.stable_p2_quantile(values, 0.5, "float64")
        gold = float(reference.gold_p2_quantile(values, 0.5))
        assert naive == pytest.approx(gold, rel=1e-9)
        assert stable == pytest.approx(gold, rel=1e-9)


class TestRepetitionPenaltyGaugeDependence:
    """naive_repetition_penalty sign-branches on the raw logit before
    penalizing a previously-seen token -- gauge-DEPENDENT, since softmax
    (and log-probability) is exactly shift-invariant but this sign
    branch is not. stable_repetition_penalty branches on log-
    probabilities instead, which are shift-invariant by construction.
    arXiv:2607.09791 documents this in HuggingFace/vLLM/llama.cpp."""

    THETA = 1.3
    MASK = [True, False, False, False]
    BASELINE = [-2.0, -3.0, -10.0, -10.0]
    SHIFTED_PLUS10 = [8.0, 7.0, 0.0, 0.0]
    SHIFTED_MINUS10 = [-12.0, -13.0, -20.0, -20.0]

    def test_stable_is_shift_invariant_across_all_three_gauges(self):
        # This is the core correctness property: three logit vectors
        # representing the identical distribution (differing only by
        # an additive constant with no softmax-visible meaning) must
        # produce IDENTICAL post-penalty distributions under a correct
        # implementation.
        r1 = kernels.stable_repetition_penalty(self.BASELINE, self.MASK, self.THETA, "float64")
        r2 = kernels.stable_repetition_penalty(self.SHIFTED_PLUS10, self.MASK, self.THETA, "float64")
        r3 = kernels.stable_repetition_penalty(self.SHIFTED_MINUS10, self.MASK, self.THETA, "float64")
        for a, b in ((r1, r2), (r1, r3)):
            for x, y in zip(a, b):
                assert float(x) == pytest.approx(float(y), rel=1e-9, abs=1e-12)

    def test_naive_is_gauge_dependent_argmax_flips(self):
        # The headline bug: naive's sign-branch flips between the
        # baseline and the +10-shifted input (token 0's raw logit
        # crosses zero), changing which token would actually be
        # sampled greedily -- a real behavioral difference, not just a
        # numerical wobble. This assertion would FAIL against a
        # (hypothetical) gauge-invariant naive implementation.
        base = kernels.naive_repetition_penalty(self.BASELINE, self.MASK, self.THETA, "float64")
        shifted = kernels.naive_repetition_penalty(self.SHIFTED_PLUS10, self.MASK, self.THETA, "float64")
        assert int(base.argmax()) == 0
        assert int(shifted.argmax()) == 1
        assert int(base.argmax()) != int(shifted.argmax())

    def test_naive_disagrees_with_gauge_invariant_gold_at_baseline(self):
        # Even at the baseline gauge (no shift applied at all), naive's
        # probability for the leading token is already measurably
        # wrong relative to the independent Decimal gold reference --
        # this is not purely an "argmax flip" story, the intermediate
        # probabilities are distorted too.
        gold = [float(g) for g in reference.gold_repetition_penalty(self.BASELINE, self.MASK, self.THETA)]
        naive = kernels.naive_repetition_penalty(self.BASELINE, self.MASK, self.THETA, "float64").tolist()
        assert abs(naive[0] - gold[0]) / gold[0] > 0.10

    def test_stable_matches_gold_exactly_at_every_gauge(self):
        gold = [float(g) for g in reference.gold_repetition_penalty(self.BASELINE, self.MASK, self.THETA)]
        for logits in (self.BASELINE, self.SHIFTED_PLUS10, self.SHIFTED_MINUS10):
            stable = kernels.stable_repetition_penalty(logits, self.MASK, self.THETA, "float64").tolist()
            for s, g in zip(stable, gold):
                assert s == pytest.approx(g, rel=1e-9, abs=1e-12)

    def test_no_penalty_control_all_agree_and_are_shift_invariant(self):
        # Control: with an all-False mask (nothing penalized), this
        # reduces to plain softmax, which every formulation (naive,
        # stable, gold) already agrees is shift-invariant.
        no_mask = [False, False, False, False]
        for shift in (0.0, 25.0, -25.0):
            logits = [x + shift for x in self.BASELINE]
            naive = kernels.naive_repetition_penalty(logits, no_mask, self.THETA, "float64").tolist()
            stable = kernels.stable_repetition_penalty(logits, no_mask, self.THETA, "float64").tolist()
            gold = [float(g) for g in reference.gold_repetition_penalty(logits, no_mask, self.THETA)]
            for n, s, g in zip(naive, stable, gold):
                assert n == pytest.approx(g, rel=1e-9, abs=1e-12)
                assert s == pytest.approx(g, rel=1e-9, abs=1e-12)


class TestSpeculativeRejectSamplingMismatch:
    """naive_speculative_reject samples the draft token from one
    materialization of the draft probability array (bf16-rounded) but
    computes the Leviathan/Chen accept/reject math against an
    independently recomputed draft probability array at the requested
    dtype, from the SAME logits -- the exact defect class fixed in
    deepseek-ai/DeepSpec PR#30. stable_speculative_reject reuses a
    single materialized array for both roles, so r is p by
    construction and the rejection-sampling identity's guarantee
    (output distribution == target distribution q) holds exactly.
    """

    DRAFT = [0.5, 1.0, -0.3, 0.2]
    TARGET = [0.4, 1.1, -0.2, 0.3]

    def test_stable_matches_target_softmax_gold_at_float64(self):
        gold = [float(g) for g in reference.gold_speculative_reject(self.DRAFT, self.TARGET)]
        stable = kernels.stable_speculative_reject(self.DRAFT, self.TARGET, "float64").tolist()
        for s, g in zip(stable, gold):
            assert s == pytest.approx(g, rel=1e-9, abs=1e-12)

    def test_naive_diverges_from_gold_at_float64(self):
        # The headline bug: even at float64 (numguard's tightest
        # tolerance), the sampled-vs-accepted mismatch alone produces a
        # measurable, tolerance-exceeding divergence from the true
        # target distribution -- no overflow/underflow/NaN involved,
        # just a broken correctness identity.
        gold = [float(g) for g in reference.gold_speculative_reject(self.DRAFT, self.TARGET)]
        naive = kernels.naive_speculative_reject(self.DRAFT, self.TARGET, "float64").tolist()
        max_err = max(abs(n - g) for n, g in zip(naive, gold))
        assert max_err > 1e-5  # far above numguard's float64 atol+rtol floor

    def test_naive_output_is_a_valid_probability_distribution(self):
        # Confirms this is NOT an overflow/underflow/NaN bug: naive's
        # output is finite, non-negative, and sums to 1 -- a
        # well-formed but WRONG distribution, which is exactly the
        # silent-failure shape that makes this class of bug dangerous
        # in production (nothing crashes or alarms).
        naive = kernels.naive_speculative_reject(self.DRAFT, self.TARGET, "float64").tolist()
        assert all(x >= 0.0 for x in naive)
        assert sum(naive) == pytest.approx(1.0, rel=1e-9)

    def test_identical_draft_and_target_logits_stable_reproduces_gold(self):
        # When draft == target exactly, the theorem's guarantee is at
        # its simplest (r=p=q already, in exact arithmetic) -- confirms
        # stable's construction doesn't need draft != target to hold.
        logits = [1.0, 0.5, -0.5, 2.0]
        gold = [float(g) for g in reference.gold_speculative_reject(logits, logits)]
        stable = kernels.stable_speculative_reject(logits, logits, "float64").tolist()
        for s, g in zip(stable, gold):
            assert s == pytest.approx(g, rel=1e-9, abs=1e-12)


class TestWeightDecayStorageStall:
    """naive_weight_decay rounds the decayed parameter back to the
    storage dtype after EVERY step; once the per-step decay fraction
    (lr * weight_decay) is smaller than that dtype's unit-in-the-last-
    place, every update rounds away to the SAME stored value and
    weight decay silently does nothing for the entire run -- the real,
    currently-open Nerogar/OneTrainer#996 bug ("No weight decay with
    Adam, bf16 and stochastic rounding"). stable_weight_decay keeps a
    float64 master accumulator through the whole loop and rounds to
    the storage dtype only once, at read-out, so it tracks the true
    geometric decay regardless of how small the per-step fraction is
    relative to the storage dtype's precision.
    """

    W0 = 1.0
    # Typical AdamW hyperparameters: lr=1e-4, weight_decay=0.01.
    DECAY_PER_STEP = 1e-6

    def test_naive_total_stall_at_float16(self):
        # 80000 steps is enough for the true (Decimal-exact) decay to
        # reach ~7.7%, comfortably outside numguard's float16
        # atol(1e-2)+rtol(5e-2) tolerance -- but naive_weight_decay's
        # per-step float16 rounding stalls completely: the parameter
        # never moves off its initial value.
        naive = kernels.naive_weight_decay(
            [self.W0], [self.DECAY_PER_STEP], 80000, "float16"
        )
        assert naive == self.W0  # completely unchanged: the stall

    def test_stable_tracks_true_decay_at_float16(self):
        gold = reference.gold_weight_decay([self.W0], [self.DECAY_PER_STEP], 80000)
        stable = kernels.stable_weight_decay(
            [self.W0], [self.DECAY_PER_STEP], 80000, "float16"
        )
        assert stable == pytest.approx(float(gold), rel=5e-2, abs=1e-2)
        assert stable != self.W0  # actually decayed, unlike naive

    def test_naive_drifts_beyond_tolerance_at_float32(self):
        # At float32, the per-step rounding does NOT fully stall (its
        # ULP near 1.0 is smaller than the 1e-6 decay fraction), but
        # the naive per-step-rounded loop still drifts materially
        # further from the true trajectory than the stable form,
        # exceeding numguard's tight float32 tolerance (atol=1e-5,
        # rtol=1e-4).
        gold = reference.gold_weight_decay([self.W0], [self.DECAY_PER_STEP], 20000)
        naive = kernels.naive_weight_decay(
            [self.W0], [self.DECAY_PER_STEP], 20000, "float32"
        )
        err = abs(naive - float(gold))
        assert err > 1e-5 + 1e-4 * abs(float(gold))

    def test_stable_matches_gold_closely_at_float32(self):
        gold = reference.gold_weight_decay([self.W0], [self.DECAY_PER_STEP], 20000)
        stable = kernels.stable_weight_decay(
            [self.W0], [self.DECAY_PER_STEP], 20000, "float32"
        )
        assert stable == pytest.approx(float(gold), rel=1e-4, abs=1e-5)

    def test_small_step_count_control_case_both_agree(self):
        # A genuine control case (few steps, large decay fraction
        # relative to precision): both formulations should track the
        # closed-form reference closely, confirming the bug requires
        # the specific sub-ULP-decay-fraction condition, not merely
        # "any weight decay at all".
        gold = reference.gold_weight_decay([self.W0], [0.05], 10)
        naive = kernels.naive_weight_decay([self.W0], [0.05], 10, "float16")
        stable = kernels.stable_weight_decay([self.W0], [0.05], 10, "float16")
        assert naive == pytest.approx(float(gold), rel=5e-2, abs=1e-2)
        assert stable == pytest.approx(float(gold), rel=5e-2, abs=1e-2)

    def test_gold_matches_naive_geometric_decay_at_float64_no_rounding(self):
        # At float64 (negligible rounding relative to this magnitude
        # and step count), naive and stable and the Decimal gold
        # reference should all agree closely -- confirms the bug is
        # purely a storage-precision artifact, not a formula error.
        gold = reference.gold_weight_decay([self.W0], [self.DECAY_PER_STEP], 2000)
        naive = kernels.naive_weight_decay(
            [self.W0], [self.DECAY_PER_STEP], 2000, "float64"
        )
        stable = kernels.stable_weight_decay(
            [self.W0], [self.DECAY_PER_STEP], 2000, "float64"
        )
        assert naive == pytest.approx(float(gold), rel=1e-9, abs=1e-10)
        assert stable == pytest.approx(float(gold), rel=1e-9, abs=1e-10)

