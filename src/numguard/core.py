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

ALL_KERNELS = ("logsumexp", "softmax", "cross_entropy", "variance", "layer_norm", "rms_norm", "kl_divergence", "online_softmax", "masked_softmax", "sum", "rope_cos", "int8_add", "hll_register", "focal_loss_grad", "pearson_correlation", "explained_variance", "weighted_sampling_key", "geometric_mean", "p2_quantile", "repetition_penalty", "speculative_reject", "weight_decay", "gradient_accumulation_bias", "longrope_factor_select", "squared_euclidean_distance", "bpe_pair_count_overflow", "beam_search_length_penalty", "int32_dequant_overflow", "norm", "incremental_mean", "genlaguerre", "mannwhitney_u", "i0", "lambertw0", "sorted_search")
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


def _both_nan_match(computed: float, gold: Decimal) -> bool:
    """True when the gold reference is itself NaN (a mathematically
    undefined input, e.g. Pearson correlation of a constant vector) AND
    the kernel under test also returned NaN. Both `is_finite` and
    `_within_tolerance` are meaningless once either side is NaN --
    without this check, a kernel that correctly reports "undefined" by
    returning NaN for an undefined-input fixture was scored `ok=False`
    (via `is_finite=False` short-circuiting the AND), identical to a
    genuinely broken NaN. This made "correctly detects undefined input"
    indistinguishable from "silently produces garbage" in every report,
    self-check, and CI signal this tool has ever produced -- a false
    negative for the tool's own correctness-detection purpose. Only a
    NaN gold makes NaN an acceptable `ok=True` outcome; a finite gold
    with a NaN computed value is still exactly the failure this tool
    exists to catch, so this must never widen that case.
    """
    return math.isnan(computed) and gold.is_nan()


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
    elif kernel == "sum":
        computed = fn(fixture.values, dtype)
        gold = reference.gold_sum(fixture.values)
    elif kernel == "int8_add":
        (a_code,) = fixture.values
        (b_code,) = fixture.q_values
        computed = fn(
            a_code, b_code,
            fixture.zp_a, fixture.scale_a,
            fixture.zp_b, fixture.scale_b,
            fixture.zp_out, fixture.scale_out,
        )
        gold = reference.gold_int8_add(
            a_code, b_code,
            fixture.zp_a, fixture.scale_a,
            fixture.zp_b, fixture.scale_b,
            fixture.zp_out, fixture.scale_out,
        )
    elif kernel == "hll_register":
        computed = fn(fixture.hll_rank)
        gold = reference.gold_hll_register_term(fixture.hll_rank)
    elif kernel == "focal_loss_grad":
        (logit,) = fixture.values
        computed = fn(logit, fixture.target_index, fixture.fl_gamma, fixture.fl_alpha, dtype)
        gold = reference.gold_focal_loss_grad(
            logit, fixture.target_index, fixture.fl_gamma, fixture.fl_alpha
        )
    elif kernel == "pearson_correlation":
        computed = fn(fixture.values, fixture.q_values, dtype)
        gold = reference.gold_pearson_correlation(fixture.values, fixture.q_values)
    elif kernel == "explained_variance":
        computed = fn(fixture.values, fixture.q_values, dtype)
        gold = reference.gold_explained_variance(fixture.values, fixture.q_values)
    elif kernel == "weighted_sampling_key":
        (u,) = fixture.values
        (weight,) = fixture.q_values
        computed = fn(u, weight, dtype)
        gold = reference.gold_weighted_sampling_key(u, weight)
    elif kernel == "geometric_mean":
        computed = fn(fixture.values, dtype)
        gold = reference.gold_geometric_mean(fixture.values)
    elif kernel == "p2_quantile":
        computed = fn(fixture.values, fixture.p2_prob, dtype)
        gold = reference.gold_p2_quantile(fixture.values, fixture.p2_prob)
    elif kernel == "weight_decay":
        computed = fn(fixture.values, fixture.q_values, fixture.wd_num_steps, dtype)
        gold = reference.gold_weight_decay(fixture.values, fixture.q_values, fixture.wd_num_steps)
    elif kernel == "gradient_accumulation_bias":
        computed = fn(fixture.values, fixture.q_values, dtype)
        gold = reference.gold_gradient_accumulation_bias(fixture.values, fixture.q_values)
    elif kernel == "squared_euclidean_distance":
        computed = fn(fixture.values, fixture.q_values, dtype)
        gold = reference.gold_squared_euclidean_distance(fixture.values, fixture.q_values)
    elif kernel == "bpe_pair_count_overflow":
        computed = fn(fixture.values)
        gold = reference.gold_bpe_pair_count_overflow(fixture.values)
    elif kernel == "beam_search_length_penalty":
        computed = fn(
            fixture.values, fixture.q_values,
            fixture.bs_length_penalty, fixture.bs_ends_with_eos, dtype,
        )
        gold = reference.gold_beam_search_length_penalty(
            fixture.values, fixture.q_values,
            fixture.bs_length_penalty, fixture.bs_ends_with_eos,
        )
    elif kernel == "int32_dequant_overflow":
        (q_code,) = fixture.values
        computed = fn(q_code, fixture.dq_zero_point, fixture.dq_scale)
        gold = reference.gold_int32_dequant_overflow(
            q_code, fixture.dq_zero_point, fixture.dq_scale
        )
    elif kernel == "norm":
        computed = fn(fixture.values, dtype)
        gold = reference.gold_norm(fixture.values)
    elif kernel == "incremental_mean":
        computed = fn(fixture.values, fixture.im_num_batches, dtype)
        gold = reference.gold_incremental_mean(fixture.values, fixture.im_num_batches)
    elif kernel == "genlaguerre":
        computed = fn(fixture.values, fixture.gl_alpha, fixture.gl_x, dtype)
        (n_raw,) = fixture.values
        gold = reference.gold_genlaguerre(n_raw, fixture.gl_alpha, fixture.gl_x)
    elif kernel == "mannwhitney_u":
        computed = fn(fixture.values, fixture.q_values, dtype)
        gold = reference.gold_mannwhitney_u(fixture.values, fixture.q_values)
    elif kernel == "i0":
        computed = fn(fixture.values, dtype)
        (x_raw,) = fixture.values
        gold = reference.gold_i0(x_raw)
    elif kernel == "lambertw0":
        computed = fn(fixture.values, dtype)
        (z_raw,) = fixture.values
        gold = reference.gold_lambertw0(z_raw)
    elif kernel == "sorted_search":
        computed = fn(fixture.values, fixture.ss_needle)
        gold = reference.gold_sorted_search(fixture.values, fixture.ss_needle)
    else:
        raise ValueError(f"not a scalar kernel: {kernel}")

    is_finite = math.isfinite(computed)
    rel_err = _relative_error(computed, gold) if is_finite else None
    ok = _both_nan_match(computed, gold) or (is_finite and _within_tolerance(computed, float(gold), dtype))
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


def _gold_online_softmax(fixture: Fixture) -> list:
    return reference.gold_online_softmax(fixture.values)


def _gold_masked_softmax(fixture: Fixture) -> list:
    return reference.gold_masked_softmax(fixture.values, fixture.mask)


def _gold_rope_cos(fixture: Fixture) -> list:
    return reference.gold_rope_cos(fixture.values, fixture.freq)


def _gold_repetition_penalty(fixture: Fixture) -> list:
    return reference.gold_repetition_penalty(fixture.values, fixture.mask, fixture.rp_theta)


def _gold_speculative_reject(fixture: Fixture) -> list:
    return reference.gold_speculative_reject(fixture.values, fixture.q_values)


def _gold_longrope_factor_select(fixture: Fixture) -> list:
    return reference.gold_longrope_factor_select(
        fixture.values,
        fixture.lr_base_inv_freq,
        fixture.lr_short_factor,
        fixture.lr_long_factor,
        fixture.lr_original_max_pos,
        fixture.lr_seq_len,
    )


# Kernels whose naive/stable implementations take extra positional
# arguments beyond (values, dtype) -- masked_softmax additionally takes
# the boolean keep-mask. Every other array-valued kernel takes exactly
# (values, dtype), so this stays a short exception list rather than
# forcing every kernel through a uniform-but-awkward *args signature.
EXTRA_ARGS = {
    "masked_softmax": lambda fixture: (fixture.mask,),
    "rope_cos": lambda fixture: (fixture.freq,),
    "repetition_penalty": lambda fixture: (fixture.mask, fixture.rp_theta),
    "speculative_reject": lambda fixture: (fixture.q_values,),
    "longrope_factor_select": lambda fixture: (
        fixture.lr_base_inv_freq,
        fixture.lr_short_factor,
        fixture.lr_long_factor,
        fixture.lr_original_max_pos,
        fixture.lr_seq_len,
        fixture.lr_ctx_alloc,
    ),
}


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
    "online_softmax": _gold_online_softmax,
    "masked_softmax": _gold_masked_softmax,
    "rope_cos": _gold_rope_cos,
    "repetition_penalty": _gold_repetition_penalty,
    "speculative_reject": _gold_speculative_reject,
    "longrope_factor_select": _gold_longrope_factor_select,
}


def _run_array_valued(kernel: str, variant: str, fixture: Fixture, dtype: str) -> CaseResult:
    naive_fn, stable_fn = kernels.KERNELS[kernel]
    fn = naive_fn if variant == "naive" else stable_fn
    extra = EXTRA_ARGS[kernel](fixture) if kernel in EXTRA_ARGS else ()
    computed = fn(fixture.values, *extra, dtype)
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
