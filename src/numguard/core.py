"""Core audit engine: run naive vs stable kernels against fixtures across
dtypes, score each against the independent Decimal reference, and report
where the naive formulation breaks down (or, just as importantly, where
it turns out fine -- this tool is descriptive, not a foregone conclusion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from . import kernels, reference
from .fixtures import FIXTURES_BY_KERNEL, Fixture

ALL_KERNELS = ("logsumexp", "softmax", "cross_entropy", "variance", "layer_norm", "rms_norm", "kl_divergence")
ALL_DTYPES = ("float16", "float32", "float64")


@dataclass
class CaseResult:
    kernel: str
    variant: str  # "naive" or "stable"
    fixture: str
    dtype: str
    description: str
    ok: bool  # finite and matches reference within tolerance
    is_finite: bool
    relative_error: Optional[float]  # None if not finite / not comparable
    computed_repr: str


# Relative-error tolerance, used when the reference value is not close
# to zero. Chosen to be well above float32 eps (~1.2e-7) accumulated
# over a handful of ops, but tight enough to catch real cancellation/
# overflow damage rather than routine rounding.
DEFAULT_TOLERANCE = {
    "float16": 5e-2,
    "float32": 1e-4,
    "float64": 1e-9,
}

# Absolute-error floor added to the relative check (numpy.isclose-style
# atol + rtol*|gold|). Pure relative error is the wrong metric when the
# true value is itself tiny (e.g. cross-entropy of an already-confident
# prediction): a value of ~1e-4 computed in float32 naturally carries
# ~1e-7 absolute error from representation alone, which is a ~1e-3
# *relative* error despite the algorithm being perfectly correct. The
# absolute floor prevents that from being misreported as a kernel bug.
ABS_TOLERANCE = {
    "float16": 1e-2,
    "float32": 1e-5,
    "float64": 1e-10,
}


def _within_tolerance(computed: float, gold_f: float, dtype: str) -> bool:
    atol = ABS_TOLERANCE[dtype]
    rtol = DEFAULT_TOLERANCE[dtype]
    return abs(computed - gold_f) <= atol + rtol * abs(gold_f)


def _relative_error(computed: float, gold: Decimal) -> Optional[float]:
    if not math.isfinite(computed):
        return None
    gold_f = float(gold)
    if gold_f == 0.0:
        return abs(computed - gold_f)
    return abs((computed - gold_f) / gold_f)


def _run_scalar(kernel: str, variant: str, fixture: Fixture, dtype: str) -> CaseResult:
    naive_fn, stable_fn = kernels.KERNELS[kernel]
    fn = naive_fn if variant == "naive" else stable_fn

    if kernel == "logsumexp":
        computed = fn(fixture.values, dtype)
        gold = reference.gold_logsumexp(fixture.values)
    elif kernel == "cross_entropy":
        computed = fn(fixture.values, fixture.target_index, dtype)
        gold = reference.gold_cross_entropy(fixture.values, fixture.target_index)
    elif kernel == "variance":
        computed = fn(fixture.values, dtype)
        gold = reference.gold_variance(fixture.values)
    elif kernel == "kl_divergence":
        computed = fn(fixture.values, fixture.q_values, dtype)
        gold = reference.gold_kl_divergence(fixture.values, fixture.q_values)
    else:
        raise ValueError(f"not a scalar kernel: {kernel}")

    is_finite = math.isfinite(computed)
    rel_err = _relative_error(computed, gold) if is_finite else None
    ok = is_finite and _within_tolerance(computed, float(gold), dtype)
    return CaseResult(
        kernel=kernel,
        variant=variant,
        fixture=fixture.name,
        dtype=dtype,
        description=fixture.description,
        ok=ok,
        is_finite=is_finite,
        relative_error=rel_err,
        computed_repr=repr(computed),
    )


def _gold_softmax(fixture: Fixture) -> list:
    return reference.gold_softmax(fixture.values)


def _gold_layer_norm(fixture: Fixture) -> list:
    eps = Decimal(repr(kernels.LAYER_NORM_EPS))
    return reference.gold_layer_norm(fixture.values, eps)


def _gold_rms_norm(fixture: Fixture) -> list:
    eps = Decimal(repr(kernels.RMS_NORM_EPS))
    return reference.gold_rms_norm(fixture.values, eps)


# Kernels whose naive/stable implementations return a full array (one
# CaseResult scored across every element) rather than a single scalar.
# Adding a new array-valued kernel needs exactly one entry here (mapping
# to its gold-reference callable) instead of a bespoke ~30-line
# _run_<kernel> function -- the previous per-kernel copies (softmax,
# layer_norm, rms_norm) were flagged as growing wiring debt in the
# v0.2.0 and v0.3.0 stewardship reflections and are consolidated here.
ARRAY_VALUED_GOLD = {
    "softmax": _gold_softmax,
    "layer_norm": _gold_layer_norm,
    "rms_norm": _gold_rms_norm,
}


def _run_array_valued(kernel: str, variant: str, fixture: Fixture, dtype: str) -> CaseResult:
    naive_fn, stable_fn = kernels.KERNELS[kernel]
    fn = naive_fn if variant == "naive" else stable_fn
    computed = fn(fixture.values, dtype)
    gold = ARRAY_VALUED_GOLD[kernel](fixture)

    is_finite = bool(all(math.isfinite(float(c)) for c in computed))
    # The output must also match element-wise; a NaN-free but wrong
    # (e.g. non-normalized softmax, or collapsed-to-zero) result is
    # still a real bug, so it counts against `ok` via per-element
    # tolerance (atol + rtol*|gold|, since these outputs are often near
    # zero where pure relative error is the wrong metric).
    rel_err = None
    ok = False
    if is_finite:
        errs = []
        all_within = True
        for c, g in zip(computed, gold):
            e = _relative_error(float(c), g)
            if e is not None:
                errs.append(e)
            if not _within_tolerance(float(c), float(g), dtype):
                all_within = False
        rel_err = max(errs) if errs else None
        ok = all_within
    return CaseResult(
        kernel=kernel,
        variant=variant,
        fixture=fixture.name,
        dtype=dtype,
        description=fixture.description,
        ok=ok,
        is_finite=is_finite,
        relative_error=rel_err,
        computed_repr=repr([float(c) for c in computed]),
    )


# Thin, name-stable wrappers kept so existing call sites/tests that
# invoke _run_softmax/_run_layer_norm/_run_rms_norm directly (with their
# original (variant, fixture, dtype) signature) keep working unchanged.
def _run_softmax(variant: str, fixture: Fixture, dtype: str) -> CaseResult:
    return _run_array_valued("softmax", variant, fixture, dtype)


def _run_layer_norm(variant: str, fixture: Fixture, dtype: str) -> CaseResult:
    return _run_array_valued("layer_norm", variant, fixture, dtype)


def _run_rms_norm(variant: str, fixture: Fixture, dtype: str) -> CaseResult:
    return _run_array_valued("rms_norm", variant, fixture, dtype)


def run_kernel(kernel: str, dtype: str) -> List[CaseResult]:
    """Run every fixture for `kernel` (both naive+stable variants) at one
    dtype, skipping fixtures not meaningful at that dtype (see
    Fixture.dtypes)."""
    results: List[CaseResult] = []
    for fixture in FIXTURES_BY_KERNEL[kernel]:
        if dtype not in fixture.dtypes:
            continue
        for variant in ("naive", "stable"):
            if kernel in ARRAY_VALUED_GOLD:
                results.append(_run_array_valued(kernel, variant, fixture, dtype))
            else:
                results.append(_run_scalar(kernel, variant, fixture, dtype))
    return results


def run_all(kernels_to_run=ALL_KERNELS, dtypes=ALL_DTYPES) -> List[CaseResult]:
    results: List[CaseResult] = []
    for kernel in kernels_to_run:
        for dtype in dtypes:
            results.extend(run_kernel(kernel, dtype))
    return results
