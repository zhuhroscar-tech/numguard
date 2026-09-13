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
