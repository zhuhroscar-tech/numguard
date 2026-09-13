"""Audit-engine tests (core.py): verify the CaseResult scoring logic
itself -- tolerance computation, finiteness handling, and that run_all
respects per-fixture dtype scoping -- independent of any specific
kernel's numerical behavior."""
from __future__ import annotations

import math
from decimal import Decimal

import pytest

from numguard import core


def test_within_tolerance_accepts_exact_match():
    assert core._within_tolerance(1.25, 1.25, "float32")


def test_within_tolerance_rejects_large_relative_error():
    assert not core._within_tolerance(100.0, 1.25, "float32")


def test_within_tolerance_uses_absolute_floor_near_zero():
    # A tiny absolute difference near a near-zero true value should not
    # be flagged purely because relative error looks huge.
    assert core._within_tolerance(1.00005e-4, 1.0e-4, "float32")


def test_relative_error_handles_zero_gold():
    err = core._relative_error(0.001, Decimal(0))
    assert err is not None
    assert abs(err - 0.001) < 1e-9


def test_relative_error_none_for_non_finite_computed():
    assert core._relative_error(math.inf, Decimal(1)) is None
    assert core._relative_error(math.nan, Decimal(1)) is None


def test_run_kernel_respects_dtype_scoping():
    # modest_offset_float16 is scoped to float16 only; large_offset_*
    # fixtures are scoped to float32 only. Confirm run_kernel actually
    # filters rather than running every fixture at every dtype.
    results_f16 = core.run_kernel("variance", "float16")
    results_f32 = core.run_kernel("variance", "float32")
    f16_names = {r.fixture for r in results_f16}
    f32_names = {r.fixture for r in results_f32}
    assert "modest_offset_float16" in f16_names
    assert "modest_offset_float16" not in f32_names
    assert "large_offset_small_spread" in f32_names
    assert "large_offset_small_spread" not in f16_names


def test_run_all_covers_every_kernel():
    results = core.run_all()
    kernels_seen = {r.kernel for r in results}
    assert kernels_seen == set(core.ALL_KERNELS)


def test_run_all_produces_naive_and_stable_variants_for_every_case():
    results = core.run_all(kernels_to_run=("logsumexp",), dtypes=("float64",))
    fixtures_seen = {r.fixture for r in results}
    for fixture in fixtures_seen:
        variants = {r.variant for r in results if r.fixture == fixture}
        assert variants == {"naive", "stable"}


def test_run_scalar_rejects_non_scalar_kernel():
    from numguard.fixtures import Fixture

    bogus_fixture = Fixture("bogus", [1.0, 2.0], "not a real scalar kernel case")
    with pytest.raises(ValueError, match="not a scalar kernel"):
        core._run_scalar("softmax", "naive", bogus_fixture, "float32")


def test_run_softmax_flags_out_of_tolerance_even_if_finite():
    """Directly exercises the per-element all_within=False branch: a
    softmax-shaped callable that returns finite but wrong values must be
    scored not-ok, not just checked for NaN/inf."""
    from numguard.fixtures import Fixture

    fixture = Fixture("wrong_but_finite", [1.0, 2.0, 3.0], "control case")

    def wrong_softmax(values, dtype):
        # deliberately return a finite, non-normalized, wrong result
        return [0.9, 0.9, 0.9]

    original = core.kernels.KERNELS["softmax"]
    core.kernels.KERNELS["softmax"] = (wrong_softmax, wrong_softmax)
    try:
        result = core._run_softmax("naive", fixture, "float32")
    finally:
        core.kernels.KERNELS["softmax"] = original

    assert result.is_finite is True
    assert result.ok is False
    assert result.relative_error is not None and result.relative_error > 0
