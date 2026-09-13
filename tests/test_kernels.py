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

from numguard import kernels


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
