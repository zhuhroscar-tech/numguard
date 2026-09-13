"""Independent high-precision reference implementations.

These are deliberately NOT built from the same numpy/float code paths as
the naive/stable kernels under test in `kernels.py` -- they use Python's
arbitrary-precision `decimal.Decimal` at 50 significant digits, computed
straight from the mathematical definitions. This gives an independent
ground truth: a bug shared between the "naive" and "stable" numpy
formulas (e.g. both computing the wrong quantity) would not be masked by
comparing them only to each other.

Decimal's precision (50 digits) is vastly higher than float64 (~15-17
significant digits), so rounding error in these reference functions is
negligible relative to the errors we are trying to measure in float16/
float32/float64 kernels.
"""
from __future__ import annotations

from decimal import Decimal, getcontext
from typing import Sequence

_PREC = 50


def _ctx():
    ctx = getcontext().copy()
    ctx.prec = _PREC
    return ctx


def _to_decimals(values: Sequence[float]) -> list:
    ctx = _ctx()
    return [ctx.create_decimal(repr(float(v))) for v in values]


def gold_logsumexp(values: Sequence[float]) -> Decimal:
    """log(sum(exp(x))) computed at 50-digit precision, shift-free.

    Decimal has no practical overflow limit for this input scale, so no
    shift trick is needed here -- this is the textbook definition,
    evaluated exactly enough to serve as ground truth.
    """
    ctx = _ctx()
    xs = _to_decimals(values)
    total = ctx.create_decimal(0)
    for x in xs:
        total = ctx.add(total, ctx.exp(x))
    return ctx.ln(total)


def gold_softmax(values: Sequence[float]) -> list:
    """softmax(x) as a list of Decimals, from the textbook definition."""
    ctx = _ctx()
    xs = _to_decimals(values)
    exps = [ctx.exp(x) for x in xs]
    total = ctx.create_decimal(0)
    for e in exps:
        total = ctx.add(total, e)
    return [ctx.divide(e, total) for e in exps]


def gold_cross_entropy(values: Sequence[float], target_index: int) -> Decimal:
    """-log(softmax(x)[target]) == logsumexp(x) - x[target], exact-ish."""
    ctx = _ctx()
    lse = gold_logsumexp(values)
    xt = _to_decimals(values)[target_index]
    return ctx.subtract(lse, xt)


def gold_mean(values: Sequence[float]) -> Decimal:
    ctx = _ctx()
    xs = _to_decimals(values)
    total = ctx.create_decimal(0)
    for x in xs:
        total = ctx.add(total, x)
    return ctx.divide(total, ctx.create_decimal(len(xs)))


def gold_variance(values: Sequence[float]) -> Decimal:
    """Population variance mean((x - mean(x))^2), the numerically safe
    two-pass definition, at 50-digit precision."""
    ctx = _ctx()
    xs = _to_decimals(values)
    mean = gold_mean(values)
    total = ctx.create_decimal(0)
    for x in xs:
        d = ctx.subtract(x, mean)
        total = ctx.add(total, ctx.multiply(d, d))
    return ctx.divide(total, ctx.create_decimal(len(xs)))


def gold_sum(values: Sequence[float]) -> Decimal:
    ctx = _ctx()
    xs = _to_decimals(values)
    total = ctx.create_decimal(0)
    for x in xs:
        total = ctx.add(total, x)
    return total


def gold_layer_norm(values: Sequence[float], eps: Decimal) -> list:
    """(x - mean) / sqrt(variance + eps), from the textbook definition at
    50-digit precision -- variance here is always the mean-centered
    (numerically safe) population variance, since that is the
    mathematically correct quantity regardless of which formula a naive
    implementation uses to *approximate* it."""
    ctx = _ctx()
    xs = _to_decimals(values)
    mean = gold_mean(values)
    var = gold_variance(values)
    denom = (var + eps).sqrt(ctx)
    return [ctx.divide(ctx.subtract(x, mean), denom) for x in xs]


def gold_rms_norm(values: Sequence[float], eps: Decimal) -> list:
    """x / sqrt(mean(x^2) + eps), from the textbook RMSNorm definition
    at 50-digit precision -- unlike gold_layer_norm this is NOT mean-
    centered (RMSNorm deliberately omits mean-subtraction, that is the
    whole point of the "RMS" simplification versus LayerNorm), so the
    quantity under the square root is the raw mean of squares, exactly
    as both the naive and stable numpy kernels intend to approximate --
    the difference under test is only *which dtype the reduction runs
    in*, not the mathematical formula itself."""
    ctx = _ctx()
    xs = _to_decimals(values)
    total_sq = ctx.create_decimal(0)
    for x in xs:
        total_sq = ctx.add(total_sq, ctx.multiply(x, x))
    ms = ctx.divide(total_sq, ctx.create_decimal(len(xs)))
    denom = (ms + eps).sqrt(ctx)
    return [ctx.divide(x, denom) for x in xs]


def gold_kl_divergence(p_values: Sequence[float], q_values: Sequence[float]) -> Decimal:
    """sum(p_i * log(p_i / q_i)) at 50-digit precision, from the
    measure-theoretic definition with the standard 0*log(0/q) := 0
    convention applied explicitly (Decimal has no floating-point
    0*-inf indeterminate-form trap to fall into, but the *mathematical*
    convention still needs to be applied by this reference too, since
    Decimal(0) * Decimal(0).ln() would itself raise -- there is no
    "compute it and see what falls out" shortcut here; the convention
    is inherent to the definition of KL divergence, not an artifact of
    either implementation under test)."""
    ctx = _ctx()
    ps = _to_decimals(p_values)
    qs = _to_decimals(q_values)
    total = ctx.create_decimal(0)
    for p, q in zip(ps, qs):
        if p == 0:
            continue  # contributes exactly 0 by convention
        if q <= 0:
            return Decimal("Infinity")
        term = ctx.multiply(p, ctx.ln(ctx.divide(p, q)))
        total = ctx.add(total, term)
    return total


def gold_online_softmax(values: Sequence[float]) -> list:
    """The mathematical definition of softmax does not depend on how an
    implementation chunks its input -- chunking is purely an
    implementation strategy (e.g. FlashAttention-style tiling), not part
    of what the function means. So the ground truth for online_softmax
    is identical to gold_softmax: this is deliberate, and is exactly
    what lets naive_online_softmax's chunking bug be measured as a real
    error rather than an intentional difference in output.
    """
    return gold_softmax(values)


def gold_masked_softmax(values: Sequence[float], mask: Sequence[bool]) -> list:
    """softmax restricted to the unmasked positions, at 50-digit
    precision, from the textbook definition: masked positions get
    exactly 0 probability, and the remaining positions get their
    ordinary softmax renormalized over just that subset. When every
    position is masked there is no valid probability distribution over
    an empty support -- by the same convention the stable kernel
    implements, this reference returns all zeros rather than raising,
    so the masked_softmax kernels can be scored like every other
    array-valued kernel instead of needing a special case in core.py.
    """
    ctx = _ctx()
    xs = _to_decimals(values)
    keep = list(mask)
    if not any(keep):
        return [ctx.create_decimal(0) for _ in xs]
    kept_xs = [x for x, k in zip(xs, keep) if k]
    kept_exps = [ctx.exp(x) for x in kept_xs]
    total = ctx.create_decimal(0)
    for e in kept_exps:
        total = ctx.add(total, e)
    result = []
    it = iter(kept_exps)
    for k in keep:
        if k:
            result.append(ctx.divide(next(it), total))
        else:
            result.append(ctx.create_decimal(0))
    return result
