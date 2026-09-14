"""Adversarial and everyday input fixtures for kernel auditing.

Each fixture is a small, documented case chosen to stress a specific
known failure mode (overflow, underflow, catastrophic cancellation), not
a randomly generated array. Deterministic and reviewable by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class Fixture:
    name: str
    values: Sequence[float]
    description: str
    target_index: int = 0  # used by cross_entropy fixtures
    # Second distribution, used only by kl_divergence fixtures (values
    # above is treated as "p", q_values as "q" in KL(p || q)). None for
    # every other kernel.
    q_values: Optional[Sequence[float]] = None
    # Boolean keep-mask, used only by masked_softmax fixtures (True =
    # position participates in the softmax, False = excluded, e.g. a
    # padding token or a position outside a causal/cross-attention
    # span). None for every other kernel.
    mask: Optional[Sequence[bool]] = None
    # Rotation frequency (inv_freq), used only by rope_cos fixtures --
    # the per-dimension-pair angular speed a real RoPE implementation
    # would derive from base**(-2i/d). None (unused) for every other
    # kernel; rope_cos fixtures always set it explicitly.
    freq: Optional[float] = None
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
    # Per-operand int8 quantization parameters, used only by int8_add
    # fixtures (values/q_values hold the two operands' raw int8 codes;
    # these six fields give each operand -- and the output, for the
    # naive kernel's reused-params bug -- its own (scale, zero_point)).
    # None for every other kernel. real_value = (code - zero_point) * scale,
    # the standard affine quantization definition (TFLite/ONNX/OpenVINO).
    zp_a: Optional[int] = None
    scale_a: Optional[float] = None
    zp_b: Optional[int] = None
    scale_b: Optional[float] = None
    zp_out: Optional[int] = None
    scale_out: Optional[float] = None
    # Register rank (leading-zero-run length + 1), used only by
    # hll_register fixtures -- the HyperLogLog register value whose
    # harmonic-sum term 2**-rank the naive/stable kernels compute.
    # None for every other kernel.
    hll_rank: Optional[int] = None
    # Focal-loss-gradient parameters, used only by focal_loss_grad
    # fixtures: `values[0]` holds the pre-sigmoid logit, `target_index`
    # (reused from the field above, 0 or 1) is the binary ground-truth
    # label, and these three give the focusing exponent (gamma),
    # class-balance weight (alpha), and precision the naive/stable
    # kernels compute d(FocalLoss)/d(logit) at. None for every other
    # kernel.
    fl_gamma: Optional[float] = None
    fl_alpha: Optional[float] = None


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

LAYER_NORM_FIXTURES = [
    Fixture(
        "everyday_small",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "Ordinary small values -- naive one-pass variance under the "
        "hood has no cancellation to lose here, so naive and stable "
        "LayerNorm should agree closely at every dtype.",
        expect_naive_ok=True,
    ),
    Fixture(
        "modest_offset_float16",
        [100.0, 101.0, 102.0, 103.0],
        "float16's naive one-pass variance collapses to 0.0 for this "
        "input (see the identical variance fixture) -- feeding that "
        "near-zero denominator into LayerNorm's sqrt(var + eps) "
        "produces normalized outputs off by roughly 350x from the "
        "true values, not a subtle rounding difference.",
        dtypes=("float16",),
    ),
    Fixture(
        "large_offset_negative_variance",
        [1_000_000.0, 1_000_001.0, 1_000_002.0, 1_000_003.0],
        "The naive one-pass variance formula (E[x^2]-E[x]^2) goes "
        "*negative* for this input at float32 (verified: -65536.0 "
        "against a true variance of 1.25) -- LayerNorm's sqrt(var + "
        "eps) then takes the square root of a negative number and "
        "every output element is NaN. This is the sharpest possible "
        "demonstration that the naive variance bug is not cosmetic: "
        "it can NaN out an entire activation tensor.",
        dtypes=("float32",),
    ),
    Fixture(
        "very_large_offset",
        [3_000_000.0, 3_000_001.0, 3_000_002.0, 3_000_003.0],
        "Same integer spread (true variance 1.25) at a larger offset "
        "where naive variance comes out positive but wildly wrong "
        "(1048576.0 instead of 1.25) -- LayerNorm output is still "
        "finite here but shrunk by roughly 1000x versus the correct "
        "normalized values, a different failure shape than the NaN "
        "case above but equally wrong.",
        dtypes=("float32",),
    ),
    Fixture(
        "zero_variance_large_offset",
        [5_000_000.0] * 6,
        "True variance is exactly 0 with identical values -- an easy "
        "control case for both formulas; LayerNorm's eps term keeps "
        "the denominator away from a literal 0/0 and every output is "
        "exactly 0.0 either way. Excluded from float16 for the same "
        "overflow reason as the variance control fixture.",
        dtypes=("float32", "float64"),
        expect_naive_ok=True,
    ),
]

RMS_NORM_FIXTURES = [
    Fixture(
        "everyday_small",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "Ordinary small values, well within every dtype's range -- "
        "mean(x^2) has nothing to overflow here, so naive (reduction "
        "kept in the input dtype) and stable (reduction upcast) "
        "RMSNorm should agree closely at every dtype.",
        expect_naive_ok=True,
    ),
    Fixture(
        "fp16_activation_overflow",
        [300.0, 305.0, 298.0, 310.0, 301.0],
        "Realistic-looking float16 activation values (not pathological "
        "-- just ordinary mid-size numbers a real Transformer layer "
        "could produce). Squaring any of them (~90000) already exceeds "
        "float16's ~65504 max, so the naive reduction (mean(x^2) kept "
        "in float16) silently overflows to inf, and x / sqrt(inf) "
        "collapses every output to 0.0 -- a *finite-looking*, "
        "silently wrong answer, not an obvious NaN/inf that would get "
        "noticed immediately. This is exactly why real RMSNorm "
        "implementations (e.g. HF Transformers' LlamaRMSNorm) upcast "
        "the reduction to float32 before squaring.",
        dtypes=("float16",),
    ),
    Fixture(
        "float32_extreme_overflow",
        [2e19, 2.1e19, 1.9e19, 2.05e19, 1.95e19],
        "Same failure shape one dtype up: values whose square "
        "(~4e38) exceeds float32's ~3.4e38 max, so the naive "
        "float32-kept reduction overflows mean(x^2) to inf and every "
        "output collapses to 0.0, while upcasting the reduction to "
        "float64 (ample range for this magnitude) recovers the "
        "correct, evenly-scaled normalized output.",
        dtypes=("float32",),
    ),
    Fixture(
        "zero_all",
        [0.0, 0.0, 0.0, 0.0, 0.0],
        "All-zero input -- an easy control case for both formulas: "
        "mean(x^2) is exactly 0, the eps term keeps the denominator "
        "away from a literal 0/0, and every output is exactly 0.0 "
        "either way. Included as a baseline showing the eps guard "
        "works correctly on its own, independent of the overflow bug "
        "this kernel targets.",
        expect_naive_ok=True,
    ),
]

KL_DIVERGENCE_FIXTURES = [
    Fixture(
        "everyday_close",
        [0.2, 0.3, 0.5],
        "Two ordinary, fully-supported distributions close to each "
        "other -- no zero probabilities anywhere, so the naive literal "
        "formula has nothing to trip on; both formulas should agree "
        "closely (true KL(p||q) ~ 0.01007).",
        q_values=[0.25, 0.25, 0.5],
        expect_naive_ok=True,
    ),
    Fixture(
        "mild_mismatch",
        [0.1, 0.4, 0.5],
        "Two ordinary distributions with a bigger but still fully-"
        "supported mismatch -- another zero-free control case (true "
        "KL(p||q) ~ 0.0458).",
        q_values=[0.2, 0.3, 0.5],
        expect_naive_ok=True,
    ),
    Fixture(
        "single_zero_probability",
        [0.5, 0.0, 0.5],
        "p has one exactly-zero entry -- an entirely ordinary case "
        "(e.g. a token or class the reference distribution assigns no "
        "mass to). By convention that term contributes exactly 0 (the "
        "limit of x*log(x) as x->0), so the true KL divergence here is "
        "an ordinary finite number (~0.36698). The naive formula "
        "computes 0 * log(0/q) = 0 * -inf, an IEEE754 indeterminate "
        "form that evaluates to NaN, poisoning the entire sum -- "
        "verified directly in this repo's test suite.",
        q_values=[0.3, 0.3, 0.4],
    ),
    Fixture(
        "one_hot_label",
        [0.0, 1.0, 0.0, 0.0],
        "A one-hot label vector against a soft prediction -- the exact "
        "shape of input KL divergence receives constantly in "
        "classification distillation/label-smoothing pipelines. Three "
        "of the four p_i are exactly 0 here (not a contrived edge "
        "case -- this is the single most common real input shape for "
        "this kernel), so the naive formula's 0*-inf NaN bug fires on "
        "the majority of terms in completely ordinary usage (true "
        "KL(p||q) ~ 1.3863, since p is a point mass and q is uniform).",
        q_values=[0.25, 0.25, 0.25, 0.25],
    ),
    Fixture(
        "identical_distributions",
        [0.25, 0.25, 0.25, 0.25],
        "p equals q exactly -- true KL divergence is exactly 0 "
        "(Gibbs' inequality's equality case). An easy control: no "
        "zero-probability terms, no mismatch, both formulas should "
        "report ~0.0.",
        q_values=[0.25, 0.25, 0.25, 0.25],
        expect_naive_ok=True,
    ),
]

ONLINE_SOFTMAX_FIXTURES = [
    Fixture(
        "max_in_first_chunk",
        [10.0, 1.0, 2.0, 1.0, 0.0, 1.0],
        "The global max is already in the first chunk, so the running "
        "max never increases after chunk 0 and no rescale is ever "
        "needed -- naive (no rescale) and stable (always rescales, but "
        "the correction factor is exp(0)=1 here) should agree closely. "
        "Included as a control showing the bug is conditional, not "
        "always visible.",
        expect_naive_ok=True,
    ),
    Fixture(
        "max_in_middle_chunk",
        [1.0, 5.0, 3.0, 8.0, 2.0, 12.0],
        "The running max increases three times across three chunks "
        "(chunk_size=2): naive forgets to rescale the exponentials "
        "already accumulated from earlier chunks each time this "
        "happens, so its output is a systematically wrong (but "
        "finite, non-NaN) probability distribution -- not a subtle "
        "rounding difference. This is the headline demonstration: no "
        "extreme magnitude or cancellation is involved, just a missed "
        "incremental-rescale step, the FlashAttention-style online-"
        "softmax correction (arXiv:2205.14135).",
    ),
    Fixture(
        "max_in_last_chunk_extreme",
        [1.0, 2.0, 3.0, 1.0, 2.0, 50000.0],
        "The true global max arrives only in the final chunk and is "
        "far larger than every earlier value -- naive's un-rescaled "
        "running sum from the first five elements remains wildly "
        "over-weighted relative to the true answer (which puts "
        "essentially all mass on the last element), while the stable "
        "incremental rescale still recovers the correct one-hot-like "
        "distribution.",
    ),
    Fixture(
        "monotonically_increasing",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Every chunk raises the running max, the worst case for the "
        "number of missed rescales -- naive accumulates compounding "
        "error from every one of the three rescale steps it skips.",
    ),
]

MASKED_SOFTMAX_FIXTURES = [
    Fixture(
        "all_visible_control",
        [1.0, 2.0, 3.0, 4.0],
        "No masking at all (every position kept) -- an easy control "
        "case where masked_softmax must reduce exactly to ordinary "
        "softmax over all four positions; both naive and stable "
        "formulas should agree closely.",
        mask=[True, True, True, True],
        expect_naive_ok=True,
    ),
    Fixture(
        "some_padding",
        [2.0, 5.0, 1.0, 8.0, 0.5],
        "Ordinary partial padding -- the realistic everyday case: two "
        "trailing padding positions in a batch masked out, three real "
        "tokens remain. No extreme magnitudes and at least one "
        "unmasked position, so naive's masked_fill(-inf)-then-softmax "
        "pattern works fine here; included as a control showing the "
        "bug is conditional on the all-masked case specifically, not "
        "masking itself.",
        mask=[True, True, False, True, False],
        expect_naive_ok=True,
    ),
    Fixture(
        "fully_padded_row",
        [3.0, -1.0, 4.0, 2.0],
        "Every position masked out -- the real shape of a fully-"
        "padded row in a batch (a short sequence padded to the "
        "batch's max length contributes trailing rows with zero real "
        "tokens). Naive's masked_fill(-inf) followed by a literal "
        "softmax computes exp(-inf)=0.0 for every position, so both "
        "the numerator and the normalizing sum are 0.0 -- the naive "
        "division 0.0/0.0 is NaN across the entire row, for every "
        "element, even though no logit here is remotely extreme. "
        "This is the headline demonstration: the bug is purely about "
        "an empty support, not about float range or cancellation "
        "(matches torchtune's documented skip_mask guard against "
        "exactly this case).",
        mask=[False, False, False, False],
    ),
    Fixture(
        "fully_padded_with_extreme_logit",
        [50000.0, 1.0, -1.0],
        "Every position masked out AND one logit is extreme -- shows "
        "the two failure modes (all-masked NaN, and overflow) are "
        "independent: even if masking were somehow skipped, this "
        "input would also overflow naive_softmax. Confirms the "
        "all-masked guard in stable_masked_softmax fires before the "
        "shift-by-max step ever looks at the (masked-away) extreme "
        "value.",
        mask=[False, False, False],
    ),
    Fixture(
        "single_unmasked_extreme",
        [50000.0, 1.0, -1.0, 2.0],
        "Only one position survives masking, and it happens to be an "
        "extreme logit -- naive's masked_fill(-inf) pattern still "
        "works here (exp(-inf)=0 for the three masked terms, "
        "exp(50000) overflows to inf, inf/inf is NaN) so this is "
        "actually a second, independent way naive_masked_softmax can "
        "fail: not every failure here is the all-masked case. Stable "
        "shifts by the max of only the unmasked logits (50000.0 "
        "itself), so exp(0)=1 and the result is exactly the one-hot "
        "distribution [1.0, 0.0, 0.0, 0.0].",
        mask=[True, False, False, False],
    ),
]

# Summation fixtures use many-element lists (unlike every other kernel's
# handful-of-values fixtures) because the naive-vs-stable summation gap
# is an O(n) accumulated-error effect that is invisible at small n --
# these are still fully deterministic (generated by a fixed, documented
# rule, not randomly sampled) and each one's true sum is exactly known
# in closed form so the independent Decimal gold reference and the
# fixture's own description can both state the exact expected value.
SUM_FIXTURES = [
    Fixture(
        "everyday_small",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "A handful of ordinary values -- naive sequential summation has "
        "essentially nothing to lose here; both formulas should agree "
        "exactly or near-exactly. Included as a control showing the "
        "summation bug is about scale (many terms), not about naive "
        "summation being wrong in general.",
        expect_naive_ok=True,
    ),
    Fixture(
        "many_small_uniform_terms",
        [1e-4] * 20000,
        "20,000 identical small terms (true sum exactly 2.0) -- each "
        "individual addition rounds the ever-growing running total to "
        "float32, and because the same accumulator absorbs all 19,999 "
        "roundings in sequence, the error grows with the number of "
        "terms (O(n*eps)) rather than staying bounded. Kahan "
        "compensated summation tracks and re-applies the lost "
        "low-order bits on every step, recovering the exact sum.",
        dtypes=("float32",),
    ),
    Fixture(
        "large_value_swamps_small_terms",
        [1e8] + [1.0] * 20000,
        "One large leading term (1e8) followed by 20,000 unit "
        "increments (true sum exactly 100020000.0) -- once the running "
        "total reaches ~1e8, float32's ~7 significant decimal digits "
        "can no longer represent an increment of 1.0 exactly relative "
        "to that magnitude, so many of the later +1.0 additions are "
        "partially or fully absorbed with no effect on the total. This "
        "is the 'gradient/loss accumulator swamped by an outlier' "
        "shape: not overflow (the total stays comfortably finite) and "
        "not cancellation (every term is positive), just order-"
        "dependent accumulated rounding.",
        dtypes=("float32",),
    ),
    Fixture(
        "float16_many_small_terms",
        [0.01] * 3000,
        "3,000 identical small terms at float16 (true sum exactly "
        "30.0) -- float16's ~3-4 significant decimal digits make this "
        "failure mode visible at a much smaller n than float32 needs; "
        "included to show the same O(n*eps) growth pattern recurs at "
        "every precision, just at a different n threshold.",
        dtypes=("float16",),
    ),
    Fixture(
        "zero_terms_control",
        [0.0, 0.0, 0.0],
        "All-zero input -- an easy control case (true sum is exactly "
        "0) where naive and stable trivially agree, independent of the "
        "accumulated-rounding bug this kernel targets.",
        expect_naive_ok=True,
    ),
]


ROPE_COS_FIXTURES = [
    Fixture(
        "everyday_short_context",
        [0.0, 1.0, 2.0, 3.0],
        "Ordinary short-sequence position ids with a typical rotary "
        "frequency -- naive (cast-before-multiply) and stable "
        "(float32-compute) formulations agree closely at every dtype; "
        "control showing the bug is about scale, not the formula being "
        "wrong in general (matches HuggingFace transformers' own "
        "'Force float32' RoPE comment, PR #29285: ordinary short "
        "contexts were never the problem).",
        freq=0.3,
        expect_naive_ok=True,
    ),
    Fixture(
        "fp16_long_context_position_aliasing",
        [16384.0, 16385.0, 16392.0, 16399.0],
        "Position ids in the range where float16 can no longer "
        "represent every integer exactly (float16 loses integer "
        "resolution above 2048) -- computing position*freq and its "
        "cos() entirely in float16 collapses several distinct nearby "
        "positions onto the same rounded angle (here 16384-16392 all "
        "round to the same float16 product), producing a materially "
        "wrong cosine for 3 of 4 positions. This is the float16 analog "
        "of the documented bfloat16 RoPE position-collision bug "
        "(Baichuan Inc., zhuanlan.zhihu.com/p/651588659; HuggingFace "
        "transformers PR #29285) -- reproduced here at a precision "
        "this tool can audit without a torch/bfloat16 dependency, "
        "since bfloat16 and float16 both have too few mantissa bits to "
        "hold large integer position ids exactly once they exceed the "
        "dtype's specific integer-exact range.",
        freq=0.1,
        dtypes=("float16",),
    ),
    Fixture(
        "everyday_medium_context",
        [100.0, 250.0, 500.0, 999.0],
        "Medium-length-sequence position ids, well within every "
        "dtype's exact-integer range -- both formulations should agree "
        "closely; a second control spanning a more realistic "
        "mid-length-sequence magnitude than the all-small-integer "
        "control above.",
        freq=0.05,
        expect_naive_ok=True,
    ),
]


INT8_ADD_FIXTURES = [
    Fixture(
        "everyday_matching_scale",
        [50.0],
        "Both operands share the same (scale, zero_point) -- the common "
        "case where a naive implementation that reuses operand A's "
        "quant params for operand B happens to be harmless because "
        "there is nothing to mismatch. Control case.",
        q_values=[30.0],
        zp_a=0, scale_a=0.1, zp_b=0, scale_b=0.1, zp_out=0, scale_out=0.1,
        dtypes=("float64",),
        expect_naive_ok=True,
    ),
    Fixture(
        "asymmetric_zero_point_everyday",
        [0.0],
        "Both operands again share identical (scale, zero_point), this "
        "time a nonzero zero-point -- confirms the control case isn't "
        "an artifact of zero_point defaulting to 0. Control case.",
        q_values=[5.0],
        zp_a=-10, scale_a=0.2, zp_b=-10, scale_b=0.2, zp_out=-10, scale_out=0.2,
        dtypes=("float64",),
        expect_naive_ok=True,
    ),
    Fixture(
        "mismatched_scale_residual_add",
        [60.0],
        "Operand A and operand B are quantized with different scales "
        "(0.05 vs 1.0) -- e.g. a fine-grained residual branch added to "
        "a coarse-grained branch, the exact 'Eltwise with very "
        "different inputs ranges' scenario openvino PR#7305 fixed and "
        "PR#1135 documented as causing 'zero accuracy'. A naive fusion "
        "that dequantizes B using A's (scale, zero_point) instead of "
        "its own produces a systematically wrong sum, not merely an "
        "imprecise one (true sum 7.0, naive computes 3.2).",
        q_values=[4.0],
        zp_a=0, scale_a=0.05, zp_b=0, scale_b=1.0, zp_out=0, scale_out=0.1,
        dtypes=("float64",),
    ),
    Fixture(
        "int8_saturation_wraparound",
        [100.0],
        "Both operands share matching quant params (isolating this "
        "fixture from the mismatched-scale bug above), but their sum "
        "overflows the int8 requantization range: the correct answer "
        "saturates at 127 (matching every real quant spec's clamp step, "
        "and the same failure family as the open, unfixed bug "
        "openvino#34673 -- INT8 residual Eltwise-Add producing "
        "catastrophic error on Apple M4 Max ARM). A naive requantizer "
        "that skips the saturating clip silently wraps modulo 256 "
        "instead, producing a negative result (-56) for a large "
        "positive sum.",
        q_values=[100.0],
        zp_a=0, scale_a=1.0, zp_b=0, scale_b=1.0, zp_out=0, scale_out=1.0,
        dtypes=("float64",),
    ),
]


HLL_REGISTER_FIXTURES = [
    Fixture(
        "everyday_small_rank",
        [],
        "A typical register value (leading-zero run of 17, well below "
        "the 32-bit shift-width boundary) -- the shift distance is never "
        "masked here, so a correct and a buggy fixed-width-32 shift "
        "agree exactly. Control case.",
        hll_rank=17,
        dtypes=("float64",),
        expect_naive_ok=True,
    ),
    Fixture(
        "boundary_rank_31",
        [],
        "One below the 32-bit shift-width boundary (rank=31, so the "
        "shift distance 31 is still representable un-masked by `% 32`) "
        "-- the last rank where a 32-bit-int shift bug and the correct "
        "answer coincide. Control case confirming the bug is specifically "
        "an off-by-boundary shift-width defect, not a general rank issue.",
        hll_rank=31,
        dtypes=("float64",),
        expect_naive_ok=True,
    ),
    Fixture(
        "flink_39399_repro_rank_35",
        [],
        "The exact repro from Apache Flink FLINK-39399: a register "
        "holding value 35 (>= 32, reachable with 64-bit hashes at "
        "billion-plus real-world cardinalities). The buggy `1 << mIdx` "
        "using a 32-bit Java int silently computes `1 << (35 % 32)` = "
        "`1 << 3` = 8 instead of the true 2**35 (~3.4e10), corrupting "
        "the register's harmonic-sum contribution by ten orders of "
        "magnitude -- reproducing the issue's own reported effect (a "
        "~95K estimate collapsing what should be a ~4e14 cardinality).",
        hll_rank=35,
        dtypes=("float64",),
    ),
    Fixture(
        "far_beyond_boundary_rank_51",
        [],
        "A near-maximum register value for a 64-bit hash with typical "
        "HLL precision (max rank is about 64 - p + 1; p=14 gives up to "
        "~51) -- confirms the bug isn't confined to values just past 32 "
        "but corrupts the entire upper half of the representable range "
        "(`1 << (51 % 32)` = `1 << 19`, wrong by a factor of ~2**32).",
        hll_rank=51,
        dtypes=("float64",),
    ),
]


FOCAL_LOSS_GRAD_FIXTURES = [
    Fixture(
        "everyday_unsaturated_gamma_two",
        [0.0],
        "An ordinary, unsaturated logit (x=0, p_t=0.5) at the default "
        "gamma=2.0 focusing exponent from Lin et al.'s original paper "
        "-- naive and stable should agree closely at every dtype; "
        "control case showing the bug is conditional on saturation, "
        "not the gradient formula being wrong in general.",
        target_index=1,
        fl_gamma=2.0,
        fl_alpha=0.25,
        expect_naive_ok=True,
    ),
    Fixture(
        "everyday_gamma_zero_unsaturated",
        [0.0],
        "gamma=0 (Focal Loss's own documented reduction to plain "
        "alpha-weighted binary cross-entropy) at an unsaturated logit "
        "-- the naive formula's (1-p_t)**(gamma-1) factor is well "
        "away from 0**-1 here, so naive and stable agree closely; a "
        "second control isolating 'gamma=0 alone' from 'saturation' "
        "as two independently-necessary preconditions for the bug.",
        target_index=1,
        fl_gamma=0.0,
        fl_alpha=0.5,
        expect_naive_ok=True,
    ),
    Fixture(
        "saturated_correct_gamma_zero_float16",
        [10.0],
        "A confidently-correct, saturated logit (p_t underflows to "
        "exactly 1.0 in float16 well before x=10) with gamma=0 -- the "
        "sam3#575 failure shape: naive's (1-p_t)**(0-1) = 0.0**-1 = "
        "inf, multiplied by a simultaneously-vanishing dp_t/dx, is an "
        "IEEE754 0*inf = NaN, even though the true gradient here is an "
        "ordinary tiny finite number (the network is simply very "
        "confident and correct).",
        target_index=1,
        fl_gamma=0.0,
        fl_alpha=0.5,
        dtypes=("float16",),
    ),
    Fixture(
        "saturated_correct_gamma_zero_float32",
        [18.0],
        "The same saturation failure one dtype up: float32 underflows "
        "1-sigmoid(18) to exactly 0.0, reproducing the identical "
        "0**-1 * 0 = NaN shape as the float16 fixture above at a "
        "logit magnitude realistic for a confidently-correct detector "
        "output late in training.",
        target_index=1,
        fl_gamma=0.0,
        fl_alpha=0.5,
        dtypes=("float32",),
    ),
    Fixture(
        "saturated_correct_gamma_zero_float64",
        [40.0],
        "Same failure shape at float64 -- confirms the bug is not an "
        "artifact of a narrow dtype's limited range, only of how far "
        "the logit must saturate before 1-p_t underflows to exactly "
        "0.0 at that dtype's precision (float64 needs a larger-"
        "magnitude logit than float16/float32 to reach the same "
        "underflow point).",
        target_index=1,
        fl_gamma=0.0,
        fl_alpha=0.5,
        dtypes=("float64",),
    ),
    Fixture(
        "saturated_wrong_direction_gamma_zero_float32",
        [-18.0],
        "The mirror case: target=0 (the true class is the negative "
        "class) and the logit is confidently, correctly very negative "
        "-- p_t = 1-sigmoid(-18) again underflows to 1.0 in float32, "
        "reproducing the identical 0**-1*0 = NaN shape via the "
        "target=0 branch of the formula rather than target=1, "
        "confirming the bug is symmetric across both classes.",
        target_index=0,
        fl_gamma=0.0,
        fl_alpha=0.5,
        dtypes=("float32",),
    ),
    Fixture(
        "saturated_fractional_gamma_float32",
        [18.0],
        "van Leeuwen et al. (TMLR 2025) show the same instability "
        "occurs for any gamma in [0, 1), not just exactly gamma=0 -- "
        "this fixture uses gamma=0.5 at the same saturated logit as "
        "the gamma=0 float32 fixture above, confirming the failure is "
        "not a special-cased 'gamma exactly equals zero' bug but a "
        "genuine (1-p_t)**(gamma-1) exponent-goes-negative issue "
        "across a whole sub-range of a real, published hyperparameter.",
        target_index=1,
        fl_gamma=0.5,
        fl_alpha=0.25,
        dtypes=("float32",),
    ),
    Fixture(
        "control_saturated_gamma_two_float32",
        [18.0],
        "Same saturated logit as the two fixtures above, but at the "
        "common gamma=2.0 (Lin et al.'s recommended default, gamma >= "
        "1) -- here (1-p_t)**(gamma-1) = (1-p_t)**1 is a NON-negative "
        "power, so it safely underflows to 0.0 instead of blowing up "
        "to inf, and the naive formula is actually fine. Included as "
        "a control proving the bug is specific to gamma < 1, not "
        "saturation alone -- exactly why real-world reports (sam3, "
        "van Leeuwen et al.) needed to specifically call out the low-"
        "gamma range rather than saturation in general.",
        target_index=1,
        fl_gamma=2.0,
        fl_alpha=0.25,
        dtypes=("float32",),
        expect_naive_ok=True,
    ),
]


FIXTURES_BY_KERNEL = {
    "logsumexp": LOGSUMEXP_SOFTMAX_FIXTURES,
    "softmax": LOGSUMEXP_SOFTMAX_FIXTURES,
    "cross_entropy": CROSS_ENTROPY_FIXTURES,
    "variance": VARIANCE_FIXTURES,
    "layer_norm": LAYER_NORM_FIXTURES,
    "rms_norm": RMS_NORM_FIXTURES,
    "kl_divergence": KL_DIVERGENCE_FIXTURES,
    "online_softmax": ONLINE_SOFTMAX_FIXTURES,
    "masked_softmax": MASKED_SOFTMAX_FIXTURES,
    "sum": SUM_FIXTURES,
    "rope_cos": ROPE_COS_FIXTURES,
    "int8_add": INT8_ADD_FIXTURES,
    "hll_register": HLL_REGISTER_FIXTURES,
    "focal_loss_grad": FOCAL_LOSS_GRAD_FIXTURES,
}


def expect_naive_ok(kernel: str, fixture_name: str) -> bool:
    """Look up whether `fixture_name` under `kernel` is a control case
    (naive expected to also pass) vs an adversarial case (naive expected
    to fail). Used by --check-naive-fails."""
    for fixture in FIXTURES_BY_KERNEL[kernel]:
        if fixture.name == fixture_name:
            return fixture.expect_naive_ok
    raise KeyError(f"no fixture named {fixture_name!r} under kernel {kernel!r}")
