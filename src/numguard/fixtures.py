"""Adversarial and everyday input fixtures for kernel auditing.

Each fixture is a small, documented case chosen to stress a specific
known failure mode (overflow, underflow, catastrophic cancellation), not
a randomly generated array. Deterministic and reviewable by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Fixture:
    name: str
    values: Sequence[float]
    description: str
    target_index: int = 0  # used by cross_entropy fixtures
    # Which dtypes this fixture is meaningful for. Some adversarial cases
    # only demonstrate their failure mode at a particular precision (e.g.
    # an offset that overflows float16 outright isn't a fair "naive vs
    # stable" comparison there -- both would be equally broken by range,
    # not by cancellation). Default: all three.
    dtypes: Sequence[str] = field(default_factory=lambda: ("float16", "float32", "float64"))
    # True for "control"/easy fixtures where the naive formula is
    # expected to also be correct (used by --check-naive-fails to know
    # which naive cases should pass vs which should demonstrate a bug).
    expect_naive_ok: bool = False


LOGSUMEXP_SOFTMAX_FIXTURES = [
    Fixture(
        "everyday_small",
        [1.0, 2.0, 3.0],
        "Ordinary small logits -- both formulas should agree closely.",
        expect_naive_ok=True,
    ),
    Fixture(
        "large_uniform_shift",
        [1000.0, 1001.0, 1002.0],
        "Large but closely-spaced logits: naive exp() overflows to inf "
        "in float32/float64 well before the shifted form does.",
    ),
    Fixture(
        "extreme_single_large",
        [50000.0, 1.0, -1.0],
        "One dominant huge logit -- classic attention-logit overflow "
        "shape; naive form should overflow in float16 and often float32.",
    ),
    Fixture(
        "all_very_negative",
        [-1000.0, -1001.0, -1002.0],
        "Very negative logits: naive exp() underflows every term to "
        "0.0, producing log(0) = -inf and a NaN/inf softmax; the "
        "shifted form keeps the largest term at exp(0)=1.",
    ),
    Fixture(
        "wide_dynamic_range",
        [0.0, 20.0, -20.0, 40.0],
        "Mixed small and large magnitudes together -- large enough to "
        "stress precision but not overflow in float32/float64; excluded "
        "from float16 where exp(40) alone overflows regardless of "
        "formula (that failure mode is already covered by the more "
        "targeted extreme/large fixtures above).",
        dtypes=("float32", "float64"),
        expect_naive_ok=True,
    ),
]

CROSS_ENTROPY_FIXTURES = [
    Fixture(
        "confident_correct",
        [10.0, 0.0, 0.0],
        "Target class already dominant -- an easy case both formulas "
        "should nail.",
        target_index=0,
        expect_naive_ok=True,
    ),
    Fixture(
        "confident_wrong",
        [0.0, 0.0, 10.0],
        "Target class is the least likely one, but the true "
        "cross-entropy here (~10.0) is still a mild value -- verified "
        "naive and stable agree closely at all three dtypes; included "
        "as a control showing that 'least-likely target' alone is not "
        "sufficient to break the naive formula without also having "
        "extreme-magnitude logits (see extreme_logits below).",
        target_index=0,
        expect_naive_ok=True,
    ),
    Fixture(
        "extreme_logits",
        [50000.0, 1.0, -1.0],
        "Large logits feeding into cross-entropy -- naive path computes "
        "softmax first (which can already be NaN) before taking -log().",
        target_index=1,
    ),
]

VARIANCE_FIXTURES = [
    Fixture(
        "everyday_spread",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "Ordinary small values -- baseline agreement case.",
        expect_naive_ok=True,
    ),
    Fixture(
        "modest_offset_float16",
        [100.0, 101.0, 102.0, 103.0],
        "float16 has only ~3-4 significant decimal digits, so even a "
        "modest offset of 100 is enough to make E[x^2]-E[x]^2 lose the "
        "true variance (1.25) to cancellation/overflow in the square, "
        "while the mean-centered form stays exact (all inputs are exact "
        "integers within float16's range).",
        dtypes=("float16",),
    ),
    Fixture(
        "large_offset_small_spread",
        [1_000_000.0, 1_000_001.0, 1_000_002.0, 1_000_003.0],
        "Classic catastrophic-cancellation shape for E[x^2]-E[x]^2: all "
        "four values are exactly representable integers in float32 (no "
        "input rounding), true variance is exactly 1.25, but naive "
        "float32 accumulation of E[x^2] (~1e12) and E[x]^2 (~1e12) "
        "loses the last ~7 significant digits and the subtraction can "
        "even come out negative -- mathematically impossible for a "
        "variance. Only demonstrates the failure at float32: float16 "
        "overflows this offset outright (unrelated failure mode), and "
        "float64's ~15-17 significant digits absorb this offset "
        "without meaningful cancellation.",
        dtypes=("float32",),
    ),
    Fixture(
        "very_large_offset",
        [3_000_000.0, 3_000_001.0, 3_000_002.0, 3_000_003.0],
        "Same integer spread (true variance 1.25) at a larger offset "
        "-- still exactly representable in float32 (well under the "
        "2^24 exact-integer limit), and naive cancellation error is "
        "worse than the previous fixture (verified: naive float32 "
        "reports 1048576.0 here vs the true 1.25). float32-only for "
        "the same reason as large_offset_small_spread.",
        dtypes=("float32",),
    ),
    Fixture(
        "zero_variance_large_offset",
        [5_000_000.0] * 6,
        "True variance is exactly 0 with identical values, an easy "
        "case for both formulas (no cancellation possible when every "
        "term is identical) -- included as a control/baseline. "
        "Excluded from float16 since 5,000,000 itself overflows "
        "float16's ~65504 max, unrelated to the variance formula.",
        dtypes=("float32", "float64"),
        expect_naive_ok=True,
    ),
]

FIXTURES_BY_KERNEL = {
    "logsumexp": LOGSUMEXP_SOFTMAX_FIXTURES,
    "softmax": LOGSUMEXP_SOFTMAX_FIXTURES,
    "cross_entropy": CROSS_ENTROPY_FIXTURES,
    "variance": VARIANCE_FIXTURES,
}


def expect_naive_ok(kernel: str, fixture_name: str) -> bool:
    """Look up whether `fixture_name` under `kernel` is a control case
    (naive expected to also pass) vs an adversarial case (naive expected
    to fail). Used by --check-naive-fails."""
    for fixture in FIXTURES_BY_KERNEL[kernel]:
        if fixture.name == fixture_name:
            return fixture.expect_naive_ok
    raise KeyError(f"no fixture named {fixture_name!r} under kernel {kernel!r}")
