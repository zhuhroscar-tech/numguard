"""Adversarial and everyday input fixtures for kernel auditing.

Each fixture is a small, documented case chosen to stress a specific
known failure mode (overflow, underflow, catastrophic cancellation), not
a randomly generated array. Deterministic and reviewable by hand.
"""
from __future__ import annotations

import math
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
    # Target quantile probability (0..1), used only by p2_quantile
    # fixtures -- `values` holds the full input stream fed to the P^2
    # estimator one observation at a time; this is the `p` the estimator
    # tracks (e.g. 0.5 for the streaming median). None for every other
    # kernel.
    p2_prob: Optional[float] = None
    # Repetition-penalty strength theta, used only by repetition_penalty
    # fixtures -- `values` holds the raw logits and the shared `mask`
    # field (reused from masked_softmax's keep-mask, but meaning
    # "previously-generated / to-be-penalized" here rather than
    # "participates in softmax") marks which positions the penalty is
    # applied to. None for every other kernel.
    rp_theta: Optional[float] = None
    # Number of AdamW decay steps to simulate, used only by
    # weight_decay fixtures -- `values` holds [w0] (initial parameter
    # value) and the shared `q_values` field (reused from
    # kl_divergence/pearson_correlation's second-array convention, but
    # meaning "[decay_per_step]" here -- lr*weight_decay, a single
    # scalar wrapped in a one-element sequence to match the field's
    # type) holds the constant per-step decay fraction. None for every
    # other kernel.
    wd_num_steps: Optional[int] = None
    # LongRoPE factor-selection parameters, used only by
    # longrope_factor_select fixtures -- `values` holds the position
    # ids to encode, and these six fields carry the real published
    # microsoft/Phi-3-mini-128k-instruct config.json shape: the base
    # (unscaled) inverse frequency for one representative head
    # dimension, the model's `short_factor`/`long_factor` scaling
    # values for that same dimension, the model's
    # `original_max_position_embeddings`, and the ACTUAL sequence
    # length of the request being encoded (as opposed to the
    # allocated context size, which the naive kernel wrongly keys off
    # instead -- see ggml-org/llama.cpp#24823). None for every other
    # kernel.
    lr_base_inv_freq: Optional[float] = None
    lr_short_factor: Optional[float] = None
    lr_long_factor: Optional[float] = None
    lr_original_max_pos: Optional[int] = None
    lr_seq_len: Optional[int] = None
    lr_ctx_alloc: Optional[int] = None


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


SQUARED_EUCLIDEAN_DISTANCE_FIXTURES = [
    Fixture(
        "everyday_clean_no_offset",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "Small values, no offset -- baseline agreement case showing "
        "naive dot-product expansion and stable difference-first "
        "formula agree closely when there is nothing to cancel "
        "(true squared distance = 2.0).",
        q_values=[1.5, 2.5, 2.5, 4.5, 4.0],
        expect_naive_ok=True,
    ),
    Fixture(
        "modest_offset_float16",
        [48.53125, 49.96875, 49.6875],
        "float16's ~3-4 significant decimal digits mean even a modest "
        "offset of ~50 is enough for the naive dot(x,x)-2*dot(x,y)+"
        "dot(y,y) expansion to lose the true squared distance (a tiny "
        "positive value from near-duplicate vectors) to catastrophic "
        "cancellation, producing a NEGATIVE squared distance (-4.0 "
        "exactly, verified in this repo's test suite) -- impossible "
        "for a real sum-of-squares -- while the difference-first "
        "stable form stays accurate throughout.",
        q_values=[48.5625, 50.03125, 49.625],
        dtypes=("float16",),
    ),
    Fixture(
        "large_offset_near_duplicate_float32",
        [100000.7578125, 99997.90625, 99996.890625, 99996.8671875, 100001.125, 100001.5703125],
        "Classic catastrophic-cancellation shape for the naive "
        "expansion (same mechanism scikit-learn's own "
        "euclidean_distances docstring documents, and which "
        "scikit-learn PR#24542 added a runtime 'negative zeros and "
        "NaNs guard' for): two near-duplicate embedding-like vectors "
        "offset far from the origin -- realistic after mean-pooling "
        "or un-normalized activations, and exactly the shape of an "
        "ANN candidate re-ranking or near-duplicate-detection "
        "comparison. dot(x,x) and dot(y,y) are both ~6e10 while the "
        "true squared distance is ~0.0067; float32's ~7 significant "
        "digits cannot represent that gap, and the naive formula "
        "verified in this repo's test suite goes NEGATIVE (-4096.0 "
        "exactly). Feeding that negative squared distance to sqrt() "
        "(to recover the actual Euclidean distance) produces NaN. "
        "float32-only: float16 overflows this offset outright (a "
        "different, unrelated failure mode) and float64's ~15-17 "
        "significant digits absorb the offset without meaningful "
        "cancellation here.",
        q_values=[100000.7578125, 99997.859375, 99996.921875, 99996.90625, 100001.171875, 100001.6015625],
        dtypes=("float32",),
    ),
    Fixture(
        "realistic_768d_embedding_offset_float32",
        [
            5002.4967141530112, 4998.617357021748, 5006.476885380727,
            5015.230298564225, 4997.658466252042, 5007.658630431311,
            5002.3172812870693, 4997.65262807024, 5015.792128155073,
            4997.674347291014,
        ],
        "10-dim slice of a realistic 768-dim-style embedding: random "
        "unit-scale values offset by ~5000 (the same order of "
        "magnitude as un-normalized transformer activations or "
        "mean-pooled embeddings before normalization) compared "
        "against a near-duplicate (small Gaussian noise ~1e-3 added) "
        "-- the exact production shape this kernel targets: ANN "
        "candidate re-ranking / near-duplicate detection over raw "
        "(non-unit-normalized) embedding vectors. True squared "
        "distance is a tiny positive number; naive dot-product "
        "expansion in float32 collapses it to a value with no "
        "correct significant digits (verified in this repo's test "
        "suite), while the stable difference-first form stays "
        "accurate.",
        q_values=[
            5002.500685567191, 4998.614696544178, 5006.478424208745,
            5015.229640089624, 4997.657624453503, 5007.657535455112,
            5002.317369520005, 4997.652351568626, 5015.791862806699,
            4997.674230068475,
        ],
        dtypes=("float32",),
    ),
    Fixture(
        "identical_vectors_large_offset_control",
        [50_000.0, 50_001.0, 50_002.0, 50_003.0, 49_999.0, 50_000.5, 49_998.5, 50_001.5],
        "Identical vectors (true squared distance exactly 0) at a "
        "large offset -- both naive and stable formulas correctly "
        "return 0 here, included as a control to confirm neither "
        "formula's guard logic corrupts the trivial exact-match case, "
        "even though this is the same offset scale where the "
        "near-duplicate fixtures above demonstrate real cancellation "
        "damage once the vectors differ even slightly. float32/"
        "float64 only: at this offset, dot(x,x) alone (~2.5e9) "
        "already overflows float16's ~65504 max range outright -- an "
        "overflow failure, not the cancellation failure this fixture "
        "is a control for, so it is excluded here rather than mixing "
        "failure modes into one fixture's expectation.",
        q_values=[50_000.0, 50_001.0, 50_002.0, 50_003.0, 49_999.0, 50_000.5, 49_998.5, 50_001.5],
        dtypes=("float32", "float64"),
        expect_naive_ok=True,
    ),
    Fixture(
        "perfect_disjoint_control",
        [0.0, 0.0, 0.0],
        "Simple orthogonal-offset control with no shared origin "
        "offset and no cancellation risk (true squared distance = "
        "3.0 exactly, a clean unit-cube diagonal) -- confirms the "
        "naive formula is also correct on ordinary, well-conditioned "
        "input, matching the tool-wide convention of pairing every "
        "adversarial fixture with at least one control.",
        q_values=[1.0, 1.0, 1.0],
        expect_naive_ok=True,
    ),
]


PEARSON_CORRELATION_FIXTURES = [
    Fixture(
        "everyday_clean_correlation",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "Ordinary small values, no offset -- baseline agreement case "
        "showing naive and stable agree closely when there is nothing "
        "to cancel (true r ~= 0.85280).",
        q_values=[2.0, 4.0, 5.0, 4.0, 6.0],
        expect_naive_ok=True,
    ),
    Fixture(
        "modest_offset_float16",
        [51.0, 52.0, 53.0, 54.0],
        "float16's ~3-4 significant decimal digits mean even a modest "
        "offset of 50 is enough for the naive one-pass sum-of-products "
        "formula to lose the true correlation (r = -0.8 exactly, an "
        "integer-ratio case) to cancellation/overflow in n*sum(xy) - "
        "sum(x)*sum(y), while the normalize-then-dot stable form stays "
        "close throughout.",
        q_values=[54.0, 52.0, 53.0, 51.0],
        dtypes=("float16",),
    ),
    Fixture(
        "large_offset_small_spread_float32",
        [10_001.0, 10_002.0, 10_003.0, 10_004.0, 10_005.0],
        "Classic catastrophic-cancellation shape for the one-pass "
        "Pearson formula (same mechanism as this repo's `variance` "
        "kernel, applied to two variables' cross term at once): true "
        "correlation is exactly -0.9, all values exactly representable "
        "in float32, but naive float32 accumulation of "
        "n*sum(xy) and sum(x)*sum(y) both grow large enough relative "
        "to their difference that the outer sqrt's argument goes "
        "negative, producing NaN (verified directly in this repo's "
        "test suite). This is the exact class of failure documented "
        "in scipy.stats.pearsonr gh-8980/gh-9353 before PR#9562's "
        "normalize-then-dot rewrite. float32-only: float16 overflows "
        "this offset outright (unrelated failure mode) and float64's "
        "~15-17 significant digits absorb it without meaningful "
        "cancellation.",
        q_values=[10_005.0, 10_003.0, 10_004.0, 10_002.0, 10_001.0],
        dtypes=("float32",),
    ),
    Fixture(
        "very_large_offset_float32",
        [1_000_001.0, 1_000_002.0, 1_000_003.0, 1_000_004.0, 1_000_005.0],
        "Same integer spread (true correlation exactly -0.9) at a "
        "much larger offset -- still exactly representable in "
        "float32 (well under the 2^24 exact-integer limit), and the "
        "naive one-pass formula's cancellation is identically total: "
        "verified naive float32 reports NaN here just as at the "
        "smaller offset above, confirming the failure persists (does "
        "not self-correct) as the offset grows further. float32-only "
        "for the same reason as the fixture above.",
        q_values=[1_000_005.0, 1_000_003.0, 1_000_004.0, 1_000_002.0, 1_000_001.0],
        dtypes=("float32",),
    ),
    Fixture(
        "perfect_positive_correlation_control",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        "y is an exact positive linear function of x (true r = "
        "+1.0 exactly) -- an easy control with no offset and no "
        "cancellation, included to confirm neither formula's guard "
        "logic corrupts the ordinary/well-conditioned case. "
        "float16-excluded: y's own magnitude (up to 50) makes the "
        "naive formula's squared cross-denominator term "
        "(n*sxx-sx^2)*(n*syy-sy^2) itself exceed float16's ~65504 "
        "range even with zero cancellation error -- an unrelated "
        "range limitation, not a formula defect, so not a fair "
        "naive-vs-stable comparison at that dtype (the same "
        "exclusion rationale already used by this repo's `variance` "
        "kernel's offset fixtures).",
        q_values=[10.0, 20.0, 30.0, 40.0, 50.0],
        dtypes=("float32", "float64"),
        expect_naive_ok=True,
    ),
]


WEIGHTED_SAMPLING_KEY_FIXTURES = [
    Fixture(
        "everyday_moderate_weight",
        [0.5],
        "An ordinary uniform draw against a moderate weight (u=0.5, "
        "weight=1.0) -- u**(1/weight) stays an ordinary O(1) number "
        "here, so naive and stable agree closely. Control case showing "
        "the bug is conditional on weight magnitude, not the key "
        "formula being wrong in general.",
        q_values=[1.0],
        expect_naive_ok=True,
    ),
    Fixture(
        "small_weight_still_representable",
        [0.5],
        "A small-but-not-extreme weight (0.05, e.g. a long-tail item "
        "whose importance/score is two orders of magnitude below a "
        "typical item in a recommendation-ranking or load-balancing "
        "weighted sample) -- 1/weight=20 is large enough that "
        "u**(1/weight) is already a small number, but still just above "
        "the point where float32/float64 underflow it to exactly 0.0, "
        "so naive and stable still agree (to within normal float "
        "rounding). Confirms the bug is specifically about the "
        "underflow boundary, not small weights in general.",
        q_values=[0.05],
        dtypes=("float32", "float64"),
        expect_naive_ok=True,
    ),
    Fixture(
        "small_weight_underflow_float64",
        [0.1],
        "A realistic small weight (0.001 -- a three-orders-of-magnitude "
        "long-tail item, the kind wrswoR's own docs warn `sample_int_"
        "expjs` is numerically unsafe for) drives 1/weight=1000, so "
        "u**(1/weight) = 0.1**1000 underflows to exactly 0.0 even in "
        "float64. The naive kernel's log(0.0) then reports -inf, "
        "discarding real magnitude information and colliding every "
        "similarly-tiny-weight item onto the same -inf key -- exactly "
        "the failure this kernel exists to catch, per wrswoR's own "
        "documented distinction between `sample_int_expjs` (verbatim, "
        "numerically unsafe) and `sample_int_expj` (log-space, safe).",
        q_values=[0.001],
        expect_naive_ok=False,
    ),
    Fixture(
        "small_weight_underflow_float32",
        [0.5],
        "The same underflow shape one dtype down: float32's narrower "
        "range means a less extreme weight (0.005, giving 1/weight=200) "
        "is already enough to underflow 0.5**200 to exactly 0.0, "
        "confirming the bug's severity scales with a dtype's dynamic "
        "range rather than being a float64-only edge case.",
        q_values=[0.005],
        dtypes=("float32",),
        expect_naive_ok=False,
    ),
    Fixture(
        "small_weight_underflow_float16",
        [0.9],
        "float16's much narrower range means even a fairly ordinary "
        "small weight (0.005, giving 1/weight=200) is enough to "
        "underflow 0.9**200 to exactly 0.0 -- the same failure shape "
        "reproducing at every dtype this repo audits, each needing "
        "progressively less extreme inputs to trigger as precision "
        "narrows.",
        q_values=[0.005],
        dtypes=("float16",),
        expect_naive_ok=False,
    ),
    Fixture(
        "u_near_one_small_weight",
        [0.999],
        "Even a uniform draw very close to 1 (u=0.999, the case an "
        "engineer might assume is 'safe' since u**(1/weight) should "
        "stay close to 1 for reasonable weights) still underflows once "
        "the weight is small enough (0.0005, 1/weight=2000): "
        "0.999**2000 ~= 0.135, which is NOT what actually underflows -- "
        "included as a near-miss control confirming this specific u is "
        "still finite at this weight (naive and stable still agree), "
        "isolating that the failure genuinely requires BOTH a small "
        "weight AND a u bounded away from 1, not small weight alone.",
        q_values=[0.0005],
        dtypes=("float64",),
        expect_naive_ok=True,
    ),
]


GEOMETRIC_MEAN_FIXTURES = [
    Fixture(
        "everyday_two_values",
        [2.0, 8.0],
        "Textbook example (true geometric mean = 4.0 exactly) -- "
        "baseline agreement case with nothing to overflow or "
        "underflow.",
        expect_naive_ok=True,
    ),
    Fixture(
        "everyday_mixed_scale",
        [4.0, 1.0, 1.0 / 32],
        "Ordinary small values spanning a couple of orders of "
        "magnitude (true geometric mean = 0.5 exactly) -- still a "
        "control case, included to confirm neither formula's guard "
        "logic corrupts an easy, non-adversarial input.",
        expect_naive_ok=True,
    ),
    Fixture(
        "float16_overflow",
        [10.0] * 15,
        "float16's max representable value is ~65504. The true "
        "geometric mean here is exactly 10.0, but the naive formula's "
        "intermediate raw product (10**15) overflows float16 to +inf "
        "long before the 15th root is ever taken -- inf**(1/15) is "
        "still +inf, silently reporting an infinite average of ordinary "
        "finite values. This is exactly the shape scipy.stats.gmean's "
        "own issue tracker (gh-1053/Trac#526, \"gmean cannot handle "
        "large numbers\") describes as motivating its log-space fix.",
        dtypes=("float16",),
        expect_naive_ok=False,
    ),
    Fixture(
        "float32_overflow",
        [1000.0] * 40,
        "float32's max representable value is ~3.4e38. The true "
        "geometric mean is exactly 1000.0, but the naive raw product "
        "(1000**40 = 1e120) overflows float32 to +inf long before the "
        "40th root is taken, while the log-space stable form (working "
        "with log(1000)*40 ~= 276, an ordinary float32 magnitude) "
        "recovers the correct answer to within float32 precision.",
        dtypes=("float32",),
        expect_naive_ok=False,
    ),
    Fixture(
        "float32_underflow",
        [0.001] * 40,
        "The symmetric underflow case: true geometric mean is exactly "
        "0.001, but the naive raw product (0.001**40 = 1e-120) "
        "underflows float32's smallest representable positive value "
        "(~1.2e-38) to exactly 0.0 long before the 40th root is taken, "
        "silently reporting zero for a collection of ordinary nonzero "
        "values. The log-space stable form never forms this "
        "intermediate at all.",
        dtypes=("float32",),
        expect_naive_ok=False,
    ),
    Fixture(
        "float64_overflow",
        [10.0] * 400,
        "Even float64's much wider range (~1.8e308) is not immune: "
        "400 copies of an ordinary value (true geometric mean exactly "
        "10.0) still overflow the raw product (10**400) to +inf, "
        "confirming this is a fundamental evaluation-order defect of "
        "the naive formula rather than a narrow low-precision-dtype "
        "edge case.",
        dtypes=("float64",),
        expect_naive_ok=False,
    ),
]


P2_QUANTILE_FIXTURES = [
    Fixture(
        "everyday_small_stream",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "A tiny 7-value stream at the median (p=0.5): fewer than 5 "
        "marker-adjustment steps ever run, so there is no opportunity "
        "for the naive accumulate-a-per-step-increment bookkeeping to "
        "drift -- naive and stable agree exactly. Control case showing "
        "the bug needs many observations to accumulate, not that the "
        "estimator is wrong in general.",
        p2_prob=0.5,
        expect_naive_ok=True,
    ),
    Fixture(
        "uniform_stream_p10_drift",
        [
            46.3007, 37.3312, 13.8539, 86.6562, 0.6435, 50.2782, 89.8298,
            8.0815, 55.427, 61.665, 4.0896, 37.902, 70.348, 45.2021,
            72.5065, 15.7157, 23.8012, 11.0948, 50.6269, 92.383, 59.0428,
            77.4209, 38.3665, 74.6095, 10.1669, 29.1178, 67.4236, 72.5706,
            42.1755, 8.7712, 26.6734, 20.989, 28.1184, 80.9511, 19.9483,
            88.64, 87.9373, 5.4789, 37.8816, 49.1712, 2.3483, 42.4725,
            90.6411, 11.2046, 59.6846, 12.1232, 57.87, 89.5303, 20.3053,
            0.8253, 8.3504, 53.9769, 1.7465, 8.4837, 49.6742, 92.0926,
            42.0107, 39.8135, 63.8718, 9.3418, 57.98, 17.2555, 60.8888,
            95.8326, 5.4173, 55.5061, 60.6381, 14.9304, 26.8311, 99.4884,
            99.7964, 12.1336, 70.5468, 95.0923, 23.6786, 61.1127, 4.3031,
            36.5947, 67.4125, 59.0259, 77.4625, 8.6739, 34.7198, 86.4036,
            58.414, 45.13, 40.217, 98.6072, 57.4436, 1.8367, 79.937,
            32.8712, 43.3572, 21.3427, 44.4014, 32.471, 8.8817, 62.951,
            10.3069, 78.4099, 2.5391, 78.0721, 80.755, 49.7331, 70.9448,
            24.8261, 73.7617, 42.5007, 23.0953, 96.4075, 40.0904, 37.2969,
            85.9901, 36.936, 66.7505, 17.106, 84.3274, 25.8912, 5.0504,
            97.5259, 17.2756, 94.6519, 98.6166, 60.6558, 1.1875, 6.0911,
            20.8602, 38.8879, 61.1549, 96.6433, 35.4531, 14.0645, 56.1932,
            13.73, 8.6588, 55.596, 69.5988, 6.5597, 45.14, 70.3844,
            76.417, 38.3001, 88.7177, 16.908, 71.5466, 77.1705, 88.0256,
            49.4436, 9.9226, 4.846, 52.8838, 17.3329, 62.9684, 8.4272,
            78.0149, 22.2408, 1.3014, 17.5851, 45.6698, 55.9245, 38.8284,
            17.305, 48.2062, 94.6437, 53.543, 94.1287, 2.8773, 99.3154,
            88.8994, 54.4354, 52.3495, 53.7124, 90.9534, 6.5579, 64.281,
            54.2151, 30.0318, 72.4927, 72.0676, 10.321, 69.9521, 45.3532,
            49.0216, 63.6762, 5.2948, 60.2918, 37.3143, 87.8734, 23.1105,
            82.3122, 72.959, 62.4986, 87.5849, 3.5999, 59.6959, 61.3274,
            67.8056, 40.6676, 6.8965, 18.909, 60.8116, 18.1313, 6.4877,
            35.4794, 47.0242, 53.5569, 2.5972, 77.5574, 33.3829, 78.2821,
            0.857, 95.4039, 59.0245, 97.6365, 98.6315, 83.2784, 10.6235,
            34.8763, 23.1316, 77.9985, 19.2163, 22.1154, 11.0606, 12.0091,
            93.8131, 97.6193, 37.2428, 74.1056, 46.7509, 52.2014, 36.9488,
            63.4714, 23.6153, 25.6365, 51.7827, 19.7358, 43.9256, 94.3902,
            2.2234, 10.3254, 79.997, 5.7803, 25.1236, 85.164, 60.5253,
            22.0701, 5.8609, 27.5184, 30.0149, 80.0279, 97.5554, 78.6289,
            90.175, 93.0681, 87.6575, 91.9975, 96.7538, 9.4415, 31.7678,
            24.7183, 60.417, 49.1705, 34.6942, 83.7104, 39.9092, 20.8932,
            27.6388, 46.175, 61.0469, 56.8438, 43.9933, 18.6335, 40.2166,
            32.5024, 17.0568, 54.3882, 18.8562, 71.1602, 65.8763, 13.7693,
            62.5371, 33.7847, 44.7076, 60.6556, 76.1604, 88.3575, 73.355,
            21.1233, 54.1939, 33.537, 1.9122, 55.2461, 47.0597, 86.6907,
            32.3688, 83.357, 51.4144, 9.9094, 71.6918, 28.4658,
        ],
        "300 independently-drawn uniform(0,100) observations tracking "
        "the p=0.1 quantile -- a realistic streaming-telemetry shape "
        "(e.g. tracking p10 latency without storing every sample). "
        "Reproduces AndreyAkinshin/perfolizer#8 across all three "
        "dtypes: naive_p2_quantile's `ns[i] += dns[i]` bookkeeping "
        "drifts by ~4-7% relative error versus the 50-digit Decimal "
        "reference, while stable_p2_quantile's recompute-from-count "
        "form matches the reference to within ordinary float rounding "
        "at every precision.",
        p2_prob=0.1,
        expect_naive_ok=False,
    ),
    Fixture(
        "sine_stream_p35_drift",
        [round(50 + 40 * math.sin(i * 0.7), 6) for i in range(200)],
        "A deterministic, hand-reviewable sine-wave stream (no RNG) "
        "tracking the p=0.35 quantile -- picked because this "
        "probability's marker lands close enough to an integer "
        "boundary that the naive accumulated `dns[i]` bookkeeping's "
        "rounding drift (versus the recompute-fresh-each-step fix) "
        "changes which marker-adjustment branch fires partway through "
        "the stream, producing a ~2.9% relative error at float64 that "
        "compounds for the rest of the stream -- exactly the class of "
        "bug perfolizer's own issue #8 describes (a deferred marker "
        "adjustment corrupting subsequent state, not a one-off "
        "rounding blip).",
        p2_prob=0.35,
        dtypes=("float64",),
        expect_naive_ok=False,
    ),
]

REPETITION_PENALTY_FIXTURES = [
    Fixture(
        "no_penalty_control",
        [3.0, 1.0, -2.0, -2.0],
        "Control case: mask marks no token as previously-seen, so the "
        "penalty branch never fires for any position and this reduces "
        "to ordinary softmax -- shift-invariant by construction "
        "regardless of naive vs stable, since neither ever touches the "
        "sign-branch logic here. Confirms the bug is specific to the "
        "penalty step itself, not to this kernel's softmax plumbing.",
        mask=[False, False, False, False],
        rp_theta=1.3,
        expect_naive_ok=True,
    ),
    Fixture(
        "gauge_baseline",
        [-2.0, -3.0, -10.0, -10.0],
        "Ordinary logits with token 0 (the current leader) marked as "
        "previously-generated (mask[0]=True) and penalized at theta="
        "1.3. Both naive and stable agree on the argmax here (token 0 "
        "still wins even after penalty), but naive's post-penalty "
        "probability for token 0 (0.598) already diverges materially "
        "from the gauge-invariant gold answer (0.712) -- a real "
        "distortion, just not yet large enough to flip the argmax at "
        "this particular additive offset.",
        mask=[True, False, False, False],
        rp_theta=1.3,
        expect_naive_ok=False,
    ),
    Fixture(
        "gauge_shifted_plus10",
        [8.0, 7.0, 0.0, 0.0],
        "The EXACT same underlying distribution as gauge_baseline -- "
        "every logit shifted by the same +10.0 constant, which "
        "softmax alone would treat as entirely meaningless (softmax "
        "is shift-invariant: softmax(x) == softmax(x+c) for any c). "
        "The headline demonstration: naive_repetition_penalty's "
        "sign-branch flips (token 0's raw logit crosses from negative "
        "to positive once shifted), so it divides instead of "
        "multiplies -- the penalized argmax changes from token 0 to "
        "token 1, a different next token would be sampled, purely "
        "because of an arbitrary additive constant with no semantic "
        "meaning. stable_repetition_penalty (branching on shift-"
        "invariant log-probabilities instead) produces the identical "
        "gauge-invariant answer as gauge_baseline, matching the "
        "gold reference exactly.",
        mask=[True, False, False, False],
        rp_theta=1.3,
        expect_naive_ok=False,
    ),
    Fixture(
        "gauge_shifted_minus10",
        [-12.0, -13.0, -20.0, -20.0],
        "The same logical distribution again, this time shifted by "
        "-10.0 instead of +10.0 -- confirms the gauge-dependence bug "
        "is not a one-directional artifact of positive shifts only. "
        "naive_repetition_penalty produces yet a THIRD different "
        "probability vector (all three of gauge_baseline, this "
        "fixture, and gauge_shifted_plus10 represent the identical "
        "underlying distribution before penalization), while "
        "stable_repetition_penalty again reproduces the exact same "
        "gauge-invariant gold answer as the other two.",
        mask=[True, False, False, False],
        rp_theta=1.3,
        expect_naive_ok=False,
    ),
]


SPECULATIVE_REJECT_FIXTURES = [
    Fixture(
        "near_degenerate_draft_control",
        [10.0, -30.0, -30.0, -30.0],
        "Control case: draft is nearly a point mass on token 0, and the "
        "target distribution is close to it (token 1 gets meaningful "
        "mass, tokens 2-3 negligible). The escaped/residual probability "
        "mass needed to correct r toward q is tiny here, so even a "
        "sampled-vs-accepted-distribution mismatch (naive's bf16-vs-"
        "dtype recompute) stays within tolerance at every dtype -- "
        "confirms the bug's magnitude scales with how much correction "
        "speculative rejection sampling actually needs to do, not a "
        "blanket always-broken claim.",
        q_values=[10.0, 9.0, -30.0, -30.0],
        expect_naive_ok=True,
    ),
    Fixture(
        "float16_tolerance_masks_mismatch",
        [0.5, 1.0, -0.3, 0.2],
        "The SAME draft/target pair as everyday_close_draft_target "
        "below, restricted to float16 only: float16's coarse tolerance "
        "(atol=1e-2) is wide enough to hide the sampled-vs-accepted "
        "distribution mismatch -- this fixture documents that the bug "
        "is real but invisible at float16 precision, not that float16 "
        "somehow fixes it (naive_speculative_reject performs the "
        "identical wrong computation at every dtype; only the scoring "
        "tolerance differs).",
        q_values=[0.4, 1.1, -0.2, 0.3],
        dtypes=("float16",),
        expect_naive_ok=True,
    ),
    Fixture(
        "everyday_close_draft_target",
        [0.5, 1.0, -0.3, 0.2],
        "Ordinary, closely-matched draft and target logit vectors -- "
        "not an extreme/adversarial input by any other kernel's "
        "standard. naive_speculative_reject samples the draft token "
        "from a bfloat16-rounded probability array (r) but computes "
        "the Leviathan/Chen accept/reject math against an independently "
        "recomputed probability array at the requested dtype (p) from "
        "the SAME logits -- the exact defect class fixed in "
        "deepseek-ai/DeepSpec PR#30. Both r and p are individually "
        "finite, valid distributions; the bug is purely that r != p, "
        "which breaks the rejection-sampling identity's guarantee that "
        "the output distribution equals the target q. At float32/"
        "float64 this produces a real, tolerance-exceeding divergence "
        "from q -- speculative decoding's core 'lossless' promise "
        "silently fails to hold, with no crash, NaN, or inf anywhere.",
        q_values=[0.4, 1.1, -0.2, 0.3],
        dtypes=("float32", "float64"),
        expect_naive_ok=False,
    ),
    Fixture(
        "dominant_low_prob_channel",
        [3.0, -25.0, 2.0, 1.0],
        "The draft model is highly confident in token 0 and gives "
        "token 1 a vanishingly small probability, but the target model "
        "assigns token 1 real, non-negligible mass -- exactly the "
        "'draft disagrees on a specific token' shape rejection sampling "
        "exists to correct. The naive sampled-vs-accepted mismatch "
        "compounds with this large per-token correction, producing a "
        "materially larger tolerance-exceeding divergence than the "
        "everyday_close fixture above at the same dtypes.",
        q_values=[3.0, 4.0, 2.0, 1.0],
        dtypes=("float32", "float64"),
        expect_naive_ok=False,
    ),
    Fixture(
        "tiny_residual_mass",
        [1.0, 2.0, 0.5, -1.0],
        "Draft and target logits differ by only a small perturbation "
        "(~0.02-0.05 per position) -- deliberately the OPPOSITE extreme "
        "from dominant_low_prob_channel, to confirm the bug is not an "
        "artifact of large draft/target disagreement: even when almost "
        "no rejection-sampling correction is mathematically needed, "
        "naive's r!=p sampling/accept-math mismatch alone is enough to "
        "push the output outside tolerance at float32/float64.",
        q_values=[1.02, 1.98, 0.55, -0.95],
        dtypes=("float32", "float64"),
        expect_naive_ok=False,
    ),
]

WEIGHT_DECAY_FIXTURES = [
    Fixture(
        "small_step_count_no_stall",
        [1.0],
        "Only 10 decay steps at a decay fraction well above the "
        "float16 ULP at magnitude ~1.0 (~1e-2 near the value 1.0 in "
        "IEEE754 half precision) -- the naive per-step-rounded loop "
        "and the float64-master stable loop have not yet diverged "
        "enough to exceed this kernel's tolerance at any dtype. A "
        "genuine control case, not a constructed pass: with so few "
        "steps and this large a decay fraction relative to float16's "
        "own precision floor, both formulations still track the "
        "closed-form Decimal reference closely.",
        q_values=[0.05],
        dtypes=("float16", "float32", "float64"),
        wd_num_steps=10,
        expect_naive_ok=True,
    ),
    Fixture(
        "float16_total_stall_typical_adamw_hparams",
        [1.0],
        "The exact defect class in Nerogar/OneTrainer#996 (open bug, "
        "filed 2025-09-13): typical AdamW hyperparameters (lr=1e-4, "
        "weight_decay=0.01, decay_per_step = lr*wd = 1e-6) applied for "
        "80000 steps to a parameter stored at bf16/float16 precision. "
        "float16's ULP near 1.0 is ~1e-3, far larger than the "
        "per-step 1e-6 decay fraction -- naive_weight_decay's per-step "
        "round-to-storage-dtype rounds every single update back to "
        "the SAME stored value, so weight decay silently does "
        "NOTHING for the entire run (naive_final stays exactly 1.0, "
        "the untouched initial value) while the float64-master stable "
        "form correctly tracks the ~7.7% total decay the configured "
        "hyperparameters actually specify.",
        q_values=[1e-6],
        dtypes=("float16",),
        wd_num_steps=80000,
        expect_naive_ok=False,
    ),
    Fixture(
        "float32_partial_drift_same_hparams",
        [1.0],
        "The identical typical-AdamW-hyperparameter scenario as the "
        "float16 fixture above (lr=1e-4, wd=0.01), but at float32 "
        "storage precision over 20000 steps. float32's ULP near 1.0 "
        "(~1.2e-7) is smaller than the 1e-6 per-step decay fraction, "
        "so this is NOT a total stall -- some updates do register -- "
        "but the naive per-step rounding still accumulates materially "
        "more drift from the true geometric-decay trajectory (final "
        "value ~0.9797 vs the correct ~0.9802) than keeping a "
        "float64 master and rounding only once at read-out, "
        "exceeding this kernel's tight float32 tolerance.",
        q_values=[1e-6],
        dtypes=("float32",),
        wd_num_steps=20000,
        expect_naive_ok=False,
    ),
]

GRADIENT_ACCUMULATION_BIAS_FIXTURES = [
    Fixture(
        "equal_token_counts_control",
        [10.0, 12.0, 11.0],
        "Every micro-batch happens to have the SAME non-padding token "
        "count (no length variation this accumulation window) -- a "
        "genuine control case, not a constructed pass: when n_i is "
        "constant across all k steps, the naive mean-of-means formula "
        "and the correct token-weighted global mean are algebraically "
        "identical (both reduce to sum(g_i) / (k*n)), so naive is "
        "expected to also match the reference here.",
        q_values=[8.0, 8.0, 8.0],
        dtypes=("float16", "float32", "float64"),
        expect_naive_ok=True,
    ),
    Fixture(
        "moderate_length_imbalance",
        [10.0, 25.0, 4.0],
        "The real huggingface.co/blog/gradient_accumulation / "
        "PyTorch-Lightning#20350 bug pattern: 3 accumulated "
        "micro-batches with moderately different non-padding token "
        "counts (7, 20, 3), representative of ordinary variable-length "
        "sequence batching. naive_gradient_accumulation_bias averages "
        "the k per-microbatch MEANS (1.333/step) instead of dividing "
        "the summed loss by the summed token count (1.300 true), a "
        "~2.9% relative bias -- small-looking per step but, as the "
        "Unsloth/HF write-ups document, systematic across an entire "
        "training run and large enough to exceed this kernel's float32 "
        "tolerance.",
        q_values=[7.0, 20.0, 3.0],
        dtypes=("float32", "float64"),
        expect_naive_ok=False,
    ),
    Fixture(
        "severe_length_imbalance",
        [50.0, 48.0, 2.0],
        "One much shorter sequence (2 non-padding tokens) accumulated "
        "alongside two long ones (40, 38 tokens) -- the case that "
        "actually matters in practice: a short outlier sequence's "
        "per-token mean loss gets weighted EQUALLY with the long "
        "sequences' means under the naive 1/k averaging, wildly "
        "overweighting the short sequence's per-token loss relative to "
        "its true share of the accumulated batch. Produces a larger "
        "(~6.3%) relative divergence than the moderate-imbalance "
        "fixture above at the same dtypes, confirming the bug's "
        "magnitude scales with how uneven the token-count distribution "
        "is, not merely whether it is uneven at all.",
        q_values=[40.0, 38.0, 2.0],
        dtypes=("float32", "float64"),
        expect_naive_ok=False,
    ),
    Fixture(
        "many_steps_typical_llm_finetune",
        [120.0, 95.0, 200.0, 60.0, 150.0, 40.0, 180.0, 75.0],
        "A more realistic 8-step accumulation window (gradient_"
        "accumulation_steps=8, a common LoRA/QLoRA fine-tuning "
        "setting) with token counts spanning a 4x range (32-512), "
        "modeling a batch mixing short instructions with long "
        "multi-turn conversations. Confirms the bias is not an "
        "artifact of the toy 3-microbatch fixtures above -- it "
        "persists (and stays outside float32 tolerance) at a step "
        "count matching real training configurations.",
        q_values=[64.0, 48.0, 512.0, 32.0, 300.0, 40.0, 256.0, 80.0],
        dtypes=("float32", "float64"),
        expect_naive_ok=False,
    ),
]


LONGROPE_FACTOR_SELECT_FIXTURES = [
    Fixture(
        "short_doc_under_large_allocated_context",
        [0.0, 100.0, 320.0, 639.0],
        "The exact ggml-org/llama.cpp#24823 repro shape: a real "
        "microsoft/Phi-3-mini-128k-instruct dimension (head_dim=96, "
        "index 30 of 48 -- one of the lowest-frequency, largest "
        "long_factor/short_factor-ratio dimensions in the model's "
        "published rope_scaling table) encoding positions from a "
        "~640-token document (matching the issue's own repro size) "
        "while the serving session has allocated an 8192-token "
        "context window. The correct HF transformers reference "
        "(modeling_rope_utils._compute_longrope_parameters) selects "
        "the factor from the ACTUAL sequence length (640 <= "
        "original_max_position_embeddings=4096, so short_factor "
        "applies). naive_longrope_factor_select reproduces llama.cpp's "
        "documented bug of selecting from the ALLOCATED context size "
        "instead (8192 > 4096, so it wrongly picks long_factor), "
        "silently re-scaling inv_freq by short_factor/long_factor "
        "=2.05/60.89 (~30x) on this dimension -- the same magnitude "
        "the issue reports causing catastrophic degradation on "
        "factor-sensitive fine-tunes.",
        q_values=None,
        lr_base_inv_freq=0.003162277660168379,
        lr_short_factor=2.0500000000000007,
        lr_long_factor=60.887386322021484,
        lr_original_max_pos=4096,
        lr_seq_len=640,
        lr_ctx_alloc=8192,
        expect_naive_ok=False,
    ),
    Fixture(
        "short_doc_moderate_dimension",
        [0.0, 100.0, 320.0, 639.0],
        "Same short-document-under-large-allocated-context scenario, "
        "but at a moderate-ratio dimension (index 17 of 48, "
        "long_factor/short_factor ~9.9x rather than dim 30's ~30x) -- "
        "confirms the bug's magnitude scales with how extreme the "
        "model's own published per-dimension factor ratio is, not "
        "merely whether the wrong factor set was picked at all.",
        q_values=None,
        lr_base_inv_freq=0.038311868495572866,
        lr_short_factor=2.000000000000001,
        lr_long_factor=19.809999465942383,
        lr_original_max_pos=4096,
        lr_seq_len=640,
        lr_ctx_alloc=8192,
        expect_naive_ok=False,
    ),
    Fixture(
        "both_short_context_control",
        [0.0, 100.0, 320.0, 639.0],
        "Control: allocated context (2048) is also below "
        "original_max_position_embeddings (4096), so both the "
        "actual-seq_len selection (correct) and the allocated-ctx "
        "selection (buggy) agree on short_factor -- no divergence. "
        "Shows the bug is specifically about the short/long BOUNDARY "
        "being crossed by allocation-but-not-content, not about the "
        "formula disagreeing in general.",
        q_values=None,
        lr_base_inv_freq=0.003162277660168379,
        lr_short_factor=2.0500000000000007,
        lr_long_factor=60.887386322021484,
        lr_original_max_pos=4096,
        lr_seq_len=640,
        lr_ctx_alloc=2048,
        expect_naive_ok=True,
    ),
    Fixture(
        "both_long_context_control",
        [0.0, 1000.0, 3000.0, 5000.0],
        "Control: the document itself is genuinely long (5000 "
        "positions, seq_len > original_max_position_embeddings) AND "
        "the allocated context (8192) is also long, so both "
        "selections agree on long_factor -- no divergence. This is "
        "the ordinary, intended long-context case the LongRoPE factor "
        "was designed for; the bug only appears when allocation and "
        "actual content length disagree about which side of the "
        "boundary they're on.",
        q_values=None,
        lr_base_inv_freq=0.003162277660168379,
        lr_short_factor=2.0500000000000007,
        lr_long_factor=60.887386322021484,
        lr_original_max_pos=4096,
        lr_seq_len=5000,
        lr_ctx_alloc=8192,
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
    "pearson_correlation": PEARSON_CORRELATION_FIXTURES,
    "weighted_sampling_key": WEIGHTED_SAMPLING_KEY_FIXTURES,
    "geometric_mean": GEOMETRIC_MEAN_FIXTURES,
    "p2_quantile": P2_QUANTILE_FIXTURES,
    "repetition_penalty": REPETITION_PENALTY_FIXTURES,
    "speculative_reject": SPECULATIVE_REJECT_FIXTURES,
    "weight_decay": WEIGHT_DECAY_FIXTURES,
    "gradient_accumulation_bias": GRADIENT_ACCUMULATION_BIAS_FIXTURES,
    "longrope_factor_select": LONGROPE_FACTOR_SELECT_FIXTURES,
    "squared_euclidean_distance": SQUARED_EUCLIDEAN_DISTANCE_FIXTURES,
}


def expect_naive_ok(kernel: str, fixture_name: str) -> bool:
    """Look up whether `fixture_name` under `kernel` is a control case
    (naive expected to also pass) vs an adversarial case (naive expected
    to fail). Used by --check-naive-fails."""
    for fixture in FIXTURES_BY_KERNEL[kernel]:
        if fixture.name == fixture_name:
            return fixture.expect_naive_ok
    raise KeyError(f"no fixture named {fixture_name!r} under kernel {kernel!r}")
