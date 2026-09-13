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
