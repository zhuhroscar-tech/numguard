"""Reference implementation correctness: verify the Decimal-based gold
functions against exactly-known analytic values (not just self-
consistency), so a bug shared between kernels.py and reference.py
couldn't hide by agreeing with each other."""
from __future__ import annotations

import math
from decimal import Decimal

import pytest

from numguard import reference


def test_gold_logsumexp_matches_math_log_for_small_values():
    # logsumexp([0, 0]) = log(2)
    result = reference.gold_logsumexp([0.0, 0.0])
    assert float(result) == pytest.approx(math.log(2), rel=1e-12)


def test_gold_logsumexp_single_value_is_identity():
    # logsumexp([x]) == x exactly (log(exp(x)) == x)
    result = reference.gold_logsumexp([3.5])
    assert float(result) == pytest.approx(3.5, rel=1e-12)


def test_gold_softmax_sums_to_one():
    probs = reference.gold_softmax([1.0, 2.0, 3.0, -1.0])
    total = sum(probs, Decimal(0))
    assert float(total) == pytest.approx(1.0, abs=1e-30)


def test_gold_softmax_uniform_for_equal_logits():
    probs = reference.gold_softmax([5.0, 5.0, 5.0, 5.0])
    for p in probs:
        assert float(p) == pytest.approx(0.25, rel=1e-12)


def test_gold_cross_entropy_equals_logsumexp_minus_target():
    values = [1.0, 2.0, 3.0]
    ce = reference.gold_cross_entropy(values, target_index=1)
    lse = reference.gold_logsumexp(values)
    assert float(ce) == pytest.approx(float(lse) - 2.0, rel=1e-12)


def test_gold_mean_matches_arithmetic_mean():
    assert float(reference.gold_mean([1.0, 2.0, 3.0, 4.0])) == pytest.approx(2.5)


def test_gold_variance_matches_known_value():
    # variance of [1,2,3,4,5] (population) is 2.0
    result = reference.gold_variance([1.0, 2.0, 3.0, 4.0, 5.0])
    assert float(result) == pytest.approx(2.0, rel=1e-12)


def test_gold_variance_zero_for_identical_values():
    result = reference.gold_variance([7.0, 7.0, 7.0])
    assert float(result) == pytest.approx(0.0, abs=1e-30)


def test_gold_variance_never_negative():
    # A property any correct variance must satisfy, checked across a
    # range of scales -- this is exactly the property the naive kernel
    # under test can violate.
    for offset in (0, 1, 1_000, 1_000_000, 1e9):
        values = [offset, offset + 1, offset + 2, offset + 3]
        result = reference.gold_variance(values)
        assert result >= 0


def test_gold_sum_matches_known_total():
    assert float(reference.gold_sum([1.0, 2.0, 3.0, 4.0])) == pytest.approx(10.0)


def test_gold_layer_norm_matches_known_values():
    # [1,2,3,4,5]: mean=3, variance=2, eps=1e-5 -> (x-3)/sqrt(2+1e-5)
    result = reference.gold_layer_norm([1.0, 2.0, 3.0, 4.0, 5.0], Decimal("1e-5"))
    expected = [(x - 3.0) / math.sqrt(2.0 + 1e-5) for x in [1.0, 2.0, 3.0, 4.0, 5.0]]
    for r, e in zip(result, expected):
        assert float(r) == pytest.approx(e, rel=1e-6)


def test_gold_layer_norm_middle_element_near_zero_for_symmetric_input():
    result = reference.gold_layer_norm([1.0, 2.0, 3.0, 4.0, 5.0], Decimal("1e-5"))
    assert float(result[2]) == pytest.approx(0.0, abs=1e-9)


def test_gold_layer_norm_never_nan_even_for_zero_variance_without_eps_it_would_divide_by_zero():
    # identical values -> variance is exactly 0; the eps term keeps the
    # denominator away from a literal division by zero
    result = reference.gold_layer_norm([7.0, 7.0, 7.0], Decimal("1e-5"))
    for r in result:
        assert math.isfinite(float(r))
        assert float(r) == pytest.approx(0.0, abs=1e-2)
