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
