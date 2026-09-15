"""Naive and numerically-stable kernel implementations, parameterized by
numpy dtype (float16 / float32 / float64), operating purely with numpy so
no ML framework (torch/tensorflow) is required to run or test this tool.

Every "naive" function here is the literal textbook formula. Every
"stable" function applies the standard mitigation (shift-by-max for
exp/log, two-pass mean-centering for variance) -- see
docs/numerical-stability.md for the derivation of each.
"""
from __future__ import annotations

import numpy as np

DTYPES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def _arr(values, dtype: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(values, dtype=DTYPES[dtype])


# --- log-sum-exp -----------------------------------------------------

def naive_logsumexp(values, dtype: str) -> float:
    x = _arr(values, dtype)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.log(np.sum(np.exp(x))))


def stable_logsumexp(values, dtype: str) -> float:
    x = _arr(values, dtype)
    m = np.max(x)
    with np.errstate(over="ignore", invalid="ignore"):
        result = m + np.log(np.sum(np.exp(x - m)))
    return float(result)


# --- softmax -----------------------------------------------------------

def naive_softmax(values, dtype: str) -> np.ndarray:
    x = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x)
        return (e / np.sum(e)).astype(dtype)


def stable_softmax(values, dtype: str) -> np.ndarray:
    x = _arr(values, dtype)
    m = np.max(x)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(x - m)
        return (e / np.sum(e)).astype(dtype)


# --- cross-entropy (single target index) --------------------------------

def naive_cross_entropy(values, target_index: int, dtype: str) -> float:
    """-log(softmax(x)[target]), computed via the naive softmax above --
    this is the "compute probabilities then take log" pattern flagged as
    unstable because log() of a near-zero probability loses precision,
    and the naive softmax itself can already be NaN from overflow."""
    probs = naive_softmax(values, dtype)
    p = float(probs[target_index])
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.log(np.asarray(p, dtype=dtype)))


def stable_cross_entropy(values, target_index: int, dtype: str) -> float:
    """logsumexp(x) - x[target], the direct log-domain formula that never
    materializes an explicit probability."""
    x = _arr(values, dtype)
    lse = stable_logsumexp(values, dtype)
    return float(np.asarray(lse, dtype=dtype) - x[target_index])


# --- variance ------------------------------------------------------------

def naive_variance(values, dtype: str) -> float:
    """E[x^2] - E[x]^2 -- the textbook one-pass formula, prone to
    catastrophic cancellation when values cluster far from zero relative
    to their spread."""
    x = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        mean_sq = np.mean(x.astype(dtype) ** 2)
        sq_mean = np.mean(x) ** 2
        return float(mean_sq - sq_mean)


def stable_variance(values, dtype: str) -> float:
    """mean((x - mean(x))^2) -- two-pass, mean-centered first."""
    x = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(x)
        centered = (x - mean).astype(dtype)
        return float(np.mean(centered ** 2))


# --- layer normalization ---------------------------------------------------

# Standard LayerNorm epsilon (matches torch.nn.LayerNorm / TF's default),
# added inside the sqrt purely to avoid a literal division by zero when
# variance is exactly 0 -- NOT large enough to rescue a naive variance
# that has gone catastrophically wrong (see large_offset_negative_variance
# fixture, where the naive one-pass variance formula produces a *negative*
# number and sqrt(negative + eps) is still NaN).
LAYER_NORM_EPS = 1e-5


def naive_layer_norm(values, dtype) -> np.ndarray:
    """(x - mean) / sqrt(E[x^2] - E[x]^2 + eps) -- LayerNorm built directly
    on top of the naive one-pass variance formula. This is the shape of
    bug that reaches production when someone implements LayerNorm from
    the textbook variance formula instead of a mean-centered one: it is
    not just "less precise", it can hand sqrt() a negative number and
    silently produce NaN/inf activations for every element."""
    x = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(x)
        mean_sq = np.mean(x.astype(dtype) ** 2)
        sq_mean = np.mean(x) ** 2
        var = mean_sq - sq_mean
        denom = np.sqrt(np.asarray(var + DTYPES[dtype](LAYER_NORM_EPS), dtype=dtype))
        return ((x - mean) / denom).astype(dtype)


def stable_layer_norm(values, dtype) -> np.ndarray:
    """(x - mean) / sqrt(mean((x - mean)^2) + eps) -- mean-centered
    (two-pass) variance first, so the value under the square root can
    never go negative from cancellation."""
    x = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(x)
        centered = (x - mean).astype(dtype)
        var = np.mean(centered ** 2)
        denom = np.sqrt(np.asarray(var + DTYPES[dtype](LAYER_NORM_EPS), dtype=dtype))
        return (centered / denom).astype(dtype)


# --- RMS normalization ---------------------------------------------------

# Matches the epsilon used by real RMSNorm implementations (e.g. HF
# Transformers' LlamaRMSNorm/T5LayerNorm default of 1e-6, PyTorch's
# nn.RMSNorm default of 1e-5-ish depending on version) -- picked to match
# LAYER_NORM_EPS above so the two normalization kernels are directly
# comparable, not because the exact value matters to the bug being shown.
RMS_NORM_EPS = 1e-5

# Which dtype to upcast the reduction (mean of squares) into before
# taking the sqrt, for each input dtype -- this mirrors the standard
# mitigation used by real frameworks (e.g. LlamaRMSNorm/T5LayerNorm in
# HF Transformers call `hidden_states.to(torch.float32).pow(2).mean(...)`
# specifically because bf16/fp16 activations are kept in low precision
# for memory/speed but the *reduction* is done in fp32 to avoid this
# exact overflow bug). float64 has nowhere higher to upcast to, so it
# upcasts to itself -- the stable path is then identical to the naive
# one at float64, which is expected: this kernel's bug is about the
# reduction dtype's *range*, not catastrophic cancellation, so float64's
# already-ample range means there is nothing to fix at that precision.
RMS_NORM_UPCAST = {"float16": "float32", "float32": "float64", "float64": "float64"}


def naive_rms_norm(values, dtype) -> np.ndarray:
    """x / sqrt(mean(x^2) + eps), with the reduction (mean of squares)
    computed in the *same* narrow dtype as the input activations -- the
    shape of bug that reaches production when a low-precision (fp16)
    activation tensor is RMS-normalized without upcasting the reduction
    first. Unlike layer_norm's naive formula (which can cancel to a
    negative variance), this failure mode is pure dtype-range overflow:
    squaring an ordinary-looking float16 activation (e.g. ~300) already
    exceeds float16's ~65504 max, so mean(x^2) silently overflows to inf,
    and inf/inf (or x/inf) collapses every output to 0.0 or NaN --
    the opposite failure shape from layer_norm's NaN-from-cancellation,
    but just as wrong."""
    x = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        ms = np.mean(x.astype(DTYPES[dtype]) ** 2)
        denom = np.sqrt(np.asarray(ms + DTYPES[dtype](RMS_NORM_EPS), dtype=dtype))
        return (x / denom).astype(dtype)


def stable_rms_norm(values, dtype) -> np.ndarray:
    """x / sqrt(mean(x^2) + eps), with the reduction upcast to a wider
    dtype first (RMS_NORM_UPCAST) -- the standard real-world mitigation:
    keep the activation tensor itself in its original (possibly narrow)
    dtype for memory/speed, but compute the sum-of-squares reduction at
    higher precision so it cannot silently overflow the way the naive
    path does."""
    x = _arr(values, dtype)
    up = DTYPES[RMS_NORM_UPCAST[dtype]]
    with np.errstate(over="ignore", invalid="ignore"):
        x_up = x.astype(up)
        ms = np.mean(x_up ** 2)
        denom = np.sqrt(x_up.dtype.type(ms) + up(RMS_NORM_EPS))
        return (x_up / denom).astype(dtype)


# --- KL divergence (discrete, two distributions) --------------------------

def naive_kl_divergence(p_values, q_values, dtype: str) -> float:
    """sum(p_i * log(p_i / q_i)) -- the textbook discrete KL-divergence
    formula, evaluated literally. Whenever p_i is exactly 0 (an entirely
    ordinary case -- e.g. a one-hot label vector in distillation/label
    smoothing, or any sparse target distribution), the mathematically
    correct convention is that the term contributes exactly 0 (the
    limit of x*log(x) as x->0 is 0). But IEEE754 float arithmetic
    computes 0 * log(0/q) as 0 * -inf = NaN, poisoning the entire sum
    even though the true KL divergence is an ordinary finite number.
    This is a distinct failure shape from this repo's other kernels:
    not overflow, not cancellation, but a 0-times-infinity
    indeterminate form from a literal (non-guarded) formula
    transcription."""
    p = _arr(p_values, dtype)
    q = _arr(q_values, dtype)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        terms = p * np.log(p / q)
        return float(np.sum(terms))


def stable_kl_divergence(p_values, q_values, dtype: str) -> float:
    """Same quantity, guarding the two indeterminate/undefined cases
    explicitly instead of letting IEEE754 arithmetic decide:
    - p_i == 0: contributes exactly 0 regardless of q_i (the standard
      0*log(0/q) := 0 convention, matching scipy.special.rel_entr/
      kl_div and the measure-theoretic definition of KL divergence).
    - p_i > 0 and q_i <= 0: the true KL divergence is +inf (q assigns
      zero probability to an event p considers possible) -- this is
      the mathematically CORRECT answer, not a bug to hide, so it is
      passed through rather than masked.
    Algebraically this is the identical quantity as the naive formula
    everywhere p_i > 0 and q_i > 0; it only changes behavior at the
    two edge cases above, where the naive formula's literal transcription
    hits an IEEE754 indeterminate form instead of the intended
    mathematical limit."""
    p = _arr(p_values, dtype)
    q = _arr(q_values, dtype)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        safe_q = np.where(q > 0, q, DTYPES[dtype](1.0))
        terms = p * np.log(p / safe_q)
        terms = np.where((q <= 0) & (p > 0), np.array(np.inf, dtype=dtype), terms)
        terms = np.where(p <= 0, DTYPES[dtype](0.0), terms)
        return float(np.sum(terms.astype(dtype)))


# --- online (chunked/streaming) softmax -----------------------------------

# The FlashAttention-style incremental-softmax pattern: process the input
# in fixed-size chunks, maintaining a running max `m` and running
# normalizer `l` instead of materializing the whole array at once. This
# is algorithmically distinct from the other kernels in this module --
# it is not about float range/cancellation at a single reduction, but
# about whether *previously accumulated* partial results get correctly
# rescaled when a later chunk raises the running max. FlashAttention
# (Dao et al., arXiv:2205.14135) derives this rescale-by-exp(m_old -
# m_new) step explicitly; forgetting it is a real, documented class of
# bug distinct from every other kernel in this repo, because the naive
# version here is not "less precise" -- it computes a different,
# systematically wrong quantity whenever the running max increases
# after the first chunk, independent of any overflow/underflow.
def naive_online_softmax(values, dtype, chunk_size: int = 2):
    """Chunked softmax that updates the running max but forgets to
    rescale the exponentials (and running sum) already accumulated from
    earlier chunks. Matches the true softmax only when the global max
    happens to already be in the first chunk (no rescale is ever
    needed); otherwise every earlier element's relative weight is
    wrong, not merely imprecise."""
    x = _arr(values, dtype)
    n = len(x)
    m = -np.inf
    l = np.zeros((), dtype=DTYPES[dtype])
    exps = np.zeros_like(x)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(0, n, chunk_size):
            chunk = x[i : i + chunk_size]
            new_m = max(m, float(np.max(chunk)))
            chunk_exp = np.exp((chunk - new_m).astype(DTYPES[dtype]))
            exps[i : i + chunk_size] = chunk_exp
            l = l + np.sum(chunk_exp)
            m = new_m
        return (exps / l).astype(dtype)


def stable_online_softmax(values, dtype, chunk_size: int = 2):
    """Chunked softmax with the correct incremental rescale: whenever a
    new chunk raises the running max from m_old to m_new, every
    previously accumulated exponential (and the running sum) is
    multiplied by exp(m_old - m_new) before the new chunk is folded in
    -- the standard FlashAttention-style online-softmax correction, so
    the result is exact regardless of which chunk contains the true
    max."""
    x = _arr(values, dtype)
    n = len(x)
    m = -np.inf
    l = np.zeros((), dtype=DTYPES[dtype])
    exps = np.zeros_like(x)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(0, n, chunk_size):
            chunk = x[i : i + chunk_size]
            new_m = max(m, float(np.max(chunk)))
            if np.isfinite(m):
                scale = np.exp(DTYPES[dtype](m - new_m))
                l = l * scale
                exps[:i] = exps[:i] * scale
            chunk_exp = np.exp((chunk - new_m).astype(DTYPES[dtype]))
            exps[i : i + chunk_size] = chunk_exp
            l = l + np.sum(chunk_exp)
            m = new_m
        return (exps / l).astype(dtype)


# --- masked softmax (attention padding/causal masking) --------------------

# The single most common real-world attention pattern: some positions
# (padding tokens, future positions under a causal mask, cross-attention
# keys outside a valid span) must receive exactly zero probability mass.
# This is algorithmically distinct from every other kernel in this
# module -- it is not about a single reduction's float range or
# cancellation, but about what happens when *every* position for a given
# query is excluded. That is not a contrived edge case: it happens for
# real on fully-padded rows in a batch (a sequence padded to the batch's
# max length has trailing rows with no valid key at all) and is a
# documented, recurring bug class (e.g. torchtune's transformer.py
# explicitly guards it with a `skip_mask`; a HF-Transformers-adjacent
# forum diagnosis titled "NaN in NSA _compress_branch -> fully-masked
# rows softmax to NaN" independently describes the identical failure).
def naive_masked_softmax(values, mask, dtype):
    """Fill masked positions with -inf and take a literal softmax --
    the textbook `masked_fill(mask, -inf)` pattern used throughout
    attention implementations. Each individual masked position is fine
    (`exp(-inf) == 0.0`), but if *every* position for this row is masked,
    every term underflows to `0.0`, the normalizer sum is `0.0`, and the
    division `0.0 / 0.0` is `NaN` for the entire row -- not because any
    input was extreme, but purely because there was nothing valid left
    to normalize over."""
    x = _arr(values, dtype)
    keep = np.asarray(mask, dtype=bool)
    neg_inf = DTYPES[dtype](-np.inf)
    masked_logits = np.where(keep, x, neg_inf)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.exp(masked_logits)
        total = np.sum(e)
        return (e / total).astype(dtype)


def stable_masked_softmax(values, mask, dtype):
    """Guard the all-masked case explicitly (return an all-zero row --
    the same convention real fixes use: there is no valid probability
    distribution over an empty support, so the defined answer is 'no
    mass anywhere', not NaN) and otherwise shift by the max of only the
    *unmasked* logits before exponentiating, so a masked position's
    `-inf` can never corrupt the shift even when combined with an
    otherwise-extreme unmasked logit."""
    x = _arr(values, dtype)
    keep = np.asarray(mask, dtype=bool)
    if not keep.any():
        return np.zeros_like(x, dtype=DTYPES[dtype])
    neg_inf = DTYPES[dtype](-np.inf)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = np.max(x[keep])
        masked_logits = np.where(keep, x, neg_inf)
        e = np.exp((masked_logits - m).astype(DTYPES[dtype]))
        total = np.sum(e)
        return (e / total).astype(dtype)


# --- summation (running total over a sequence of terms) -------------------

# Every kernel above is about a single reduction's *range* (overflow/
# underflow) or a *cancellation*/indeterminate-form/streaming-rescale
# bug. Summation is algorithmically distinct again: naive sequential
# summation is neither too large nor cancelling anything -- it is
# simply the wrong *order* of additions, and its worst-case rounding
# error grows as O(n*eps) in the number of terms n, a pattern that is
# completely invisible for the small handful-of-elements fixtures used
# by every other kernel in this repo but becomes a real, measurable
# error once n reaches a few thousand -- exactly the scale of a loss
# aggregated over a batch/dataset, a gradient accumulated over many
# micro-batches, or numpy's own np.sum switching to pairwise summation
# specifically to avoid this (numpy docs: "directly adding each number
# individually to the result causing rounding errors in every step").
# --- RoPE (rotary position embedding) angle/cos -------------------------
#
# Real-world bug, not a constructed one: production RoPE implementations
# compute a per-position rotation angle as position_id * inv_freq and
# then take cos()/sin() of that angle. If position_id and inv_freq are
# committed to a low-precision dtype (float16, or bfloat16 in torch)
# *before* the multiply, distinct integer positions can round to the
# same low-precision value once they exceed that dtype's integer
# resolution (float16 represents integers exactly only up to 2048; the
# equivalent limit for bfloat16 is 256) -- so two different tokens get
# an identical rotation angle ("position aliasing"), and the model
# cannot tell them apart via RoPE alone. This is exactly the bug
# documented in HuggingFace transformers PR #29285 ("Force float32
# since bfloat16 loses precision on long contexts", now load-bearing
# boilerplate in every RoPE implementation in that codebase) and
# independently reported by Baichuan Inc. for both RoPE and ALiBi.  See
# docs/numerical-stability.md for the citations. The mitigation is not
# a new algorithm, just precision discipline: always compute the
# position * inv_freq product (and the cos/sin of it) in at least
# float32, and only cast the final result down to the model's storage
# dtype -- never carry the position id itself in low precision.
_ROPE_COMPUTE_DTYPE = {"float16": "float32", "float32": "float32", "float64": "float64"}


def naive_rope_cos(values, freq, dtype):
    """Bug pattern: position ids (and the frequency) are cast down to
    the model's storage dtype *before* the angle is computed, so two
    distinct positions can collide onto the same low-precision angle
    well before the trig function is even applied."""
    positions = _arr(values, dtype)
    freq_val = _arr([freq], dtype)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        angle = (positions * freq_val).astype(DTYPES[dtype])
        return np.cos(angle).astype(DTYPES[dtype])


def stable_rope_cos(values, freq, dtype):
    """Fix pattern (matches HF transformers' 'Force float32' RoPE
    boilerplate): always multiply positions by the frequency, and take
    cos() of the result, in at least float32 -- regardless of the
    model's storage dtype -- then cast only the final trig output down
    to the requested dtype."""
    compute_dtype = _ROPE_COMPUTE_DTYPE[dtype]
    positions = _arr(values, compute_dtype)
    freq_val = _arr([freq], compute_dtype)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        angle = (positions * freq_val).astype(DTYPES[compute_dtype])
        result = np.cos(angle)
        return result.astype(DTYPES[dtype])


def naive_sum(values, dtype) -> float:
    """Sequential (left-to-right) running-total summation -- the
    textbook `for x in xs: total += x` loop. Each addition rounds to
    the working dtype, and because every one of the n-1 additions
    accumulates into the same ever-growing running total, the
    worst-case rounding error grows as O(n * eps) in the number of
    terms: correct for a handful of values (this is not an overflow or
    cancellation bug), but measurably wrong once summing many
    similarly-scaled small terms, or once one large term is present
    that "swamps" the accumulator so later small terms are partially
    or fully lost to rounding at the accumulator's own magnitude."""
    dt = DTYPES[dtype]
    total = dt(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for v in values:
            total = dt(total + dt(v))
    return float(total)


def stable_sum(values, dtype) -> float:
    """Kahan compensated summation: track a running compensation term
    `c` for the low-order bits lost on each addition, and fold it back
    in before the next term is added. This reduces the worst-case
    rounding-error bound from O(n * eps) to O(eps) (independent of n),
    the standard mitigation for exactly the accumulator-drift failure
    naive_sum demonstrates -- see Kahan (1965) / the Kahan-Babuska
    variant, and numpy's own switch to pairwise summation in np.sum
    for the same underlying reason."""
    dt = DTYPES[dtype]
    total = dt(0.0)
    compensation = dt(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for v in values:
            y = dt(dt(v) - compensation)
            t = dt(total + y)
            compensation = dt(dt(t - total) - y)
            total = t
    return float(total)


# --- int8 element-wise add (cross-scale/zero-point requantization) --------
#
# Real-world bug, not a constructed one: affine ("zero-point") int8
# quantization represents a real value as (code - zero_point) * scale
# (TFLite's own 8-bit quantization spec, ONNX QuantizeLinear/
# DequantizeLinear, OpenVINO's LPT). When two int8 tensors with
# *different* (scale, zero_point) pairs are added element-wise -- the
# ordinary shape of a residual connection, where one branch has been
# quantized more coarsely than the other -- each operand MUST be
# dequantized with its OWN params before the add, then the float sum
# requantized (with saturation) into the output's params. Getting this
# wrong is a documented, recurring bug class across multiple real
# toolkits: OpenVINO PR#7305 ("Fixed scale factors propagation for
# Eltwise with very different inputs ranges"), OpenVINO PR#1135 (a
# per-channel quant mismatch causing "zero accuracy"), an open/unfixed
# bug openvino#34673 (INT8 residual Eltwise-Add producing catastrophic
# ~10% accuracy on Apple M4 Max ARM, isolated by the reporter to
# exactly the residual Add nodes), and onnxruntime#25823 (zero-point
# miscalculated for uint8 symmetric quantization). This is categorically
# distinct from every float-precision kernel above: the bug is in
# *requantization bookkeeping* (wrong scale/zero-point applied, or a
# missing saturating clamp), not in float range or cancellation.
INT8_QMIN, INT8_QMAX = -128, 127


def naive_int8_add(a_code, b_code, zp_a, scale_a, zp_b, scale_b, zp_out, scale_out):
    """Dequantize BOTH operands using operand A's (scale, zero_point) --
    the bug pattern real fusion/kernel-selection code hits when it
    assumes (or caches) a single shared quant descriptor for an
    Eltwise-Add instead of tracking each input's own params (the exact
    OpenVINO PR#7305/PR#1135 failure class) -- then requantize the sum
    into the output's int8 space WITHOUT a saturating clamp, so a sum
    that overflows the representable range silently wraps modulo 256
    instead of clamping (the same failure shape as openvino#34673's
    catastrophic residual-Add error)."""
    real_a = (a_code - zp_a) * scale_a
    real_b_wrong = (b_code - zp_a) * scale_a  # bug: reuses A's params for B
    real_sum = real_a + real_b_wrong
    raw_code = round(real_sum / scale_out) + zp_out
    # No saturating clamp: wrap into signed-int8 range via two's-complement
    # modulo arithmetic, exactly what happens if the result is stored into
    # an actual int8 buffer without a clip/saturate step.
    wrapped = ((raw_code - INT8_QMIN) % 256) + INT8_QMIN
    return float(wrapped)


def stable_int8_add(a_code, b_code, zp_a, scale_a, zp_b, scale_b, zp_out, scale_out):
    """Dequantize each operand with its OWN (scale, zero_point), sum in
    float, then requantize into the output's int8 space WITH a
    saturating clamp to [-128, 127] -- the standard, spec-correct
    affine-quantization add (matches TFLite/ONNX/OpenVINO's documented
    per-tensor Eltwise-Add semantics)."""
    real_a = (a_code - zp_a) * scale_a
    real_b = (b_code - zp_b) * scale_b
    real_sum = real_a + real_b
    raw_code = round(real_sum / scale_out) + zp_out
    clamped = max(INT8_QMIN, min(INT8_QMAX, raw_code))
    return float(clamped)


# --- HyperLogLog register term (fixed-width integer shift overflow) --

# A real, documented, currently-relevant bug class, categorically
# distinct from every kernel above (all of which are either float-
# range/cancellation bugs or int8-requantization bookkeeping bugs):
# a *fixed-width integer left-shift* that silently wraps instead of
# producing the intended value. Apache Flink FLINK-39399 (filed and
# fixed in 2026): HyperLogLogPlusPlus.query() computes each register's
# harmonic-sum contribution via `1 << mIdx`, where the literal `1` is
# an ordinary Java `int` and `mIdx` is the register's stored value. Per
# JLS 15.19, shifting a 32-bit int by a distance >= 32 uses only the
# distance's low 5 bits (i.e. `distance % 32`), so once a register's
# value reaches 32 or more the shift silently computes `1 << (mIdx %
# 32)` -- a small, wrong integer -- instead of overflowing loudly or
# computing the intended (arbitrarily large) power of two. The Flink
# issue's own repro: a register holding 35 with the bug reports an
# estimate of ~95K; the fix (change `1` to `1L`, i.e. widen the shift
# to 64 bits) reports the correct ~4e14. This register-value range is
# not exotic: HyperLogLog implementations deliberately use 64-bit
# hashes specifically so cardinalities beyond 2^32 can be represented
# (see the ClickHouse uniqHLL12 large-cardinality bug report, and the
# HyperLogLog++ paper's own justification for 64-bit hashing) -- and
# once you hash with 64 bits, leading-zero-run lengths of 32+ occur
# routinely at billion-plus cardinalities, exactly where a cardinality
# estimator is most needed and least excusable to silently corrupt.
HLL_SHIFT_BITS = 32


def naive_hll_register_term(rank: int) -> float:
    """The buggy FLINK-39399 shape: compute a register's harmonic-sum
    contribution 2**-rank via a fixed-width 32-bit integer left shift
    of 1, reproducing the JLS int-shift distance masking (mod 32) that
    silently corrupts the result for rank >= 32 instead of computing
    the true (much smaller) value."""
    masked_shift = rank % HLL_SHIFT_BITS  # the bug: shift distance is masked
    shifted = int(np.int32(1)) << masked_shift
    return 1.0 / float(shifted)


def stable_hll_register_term(rank: int) -> float:
    """Spec-correct: no fixed-width shift at all. A register's harmonic-
    sum contribution is exactly 2**-rank; computing it via Python's
    arbitrary-precision integers/floats (or, in the real Flink fix, a
    64-bit `long` shift) never wraps for any register value a 64-bit
    hash can actually produce (max ~64-p, far below 64)."""
    return 2.0 ** (-rank)


# --- sigmoid focal loss gradient (binary classification) -----------------
#
# Real-world bug, not a constructed one: Lin et al.'s Focal Loss (arXiv:
# 1708.02002) reshapes binary cross-entropy as
#   FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
# where p_t is the model's predicted probability of the TRUE class and
# gamma is the "focusing" exponent that down-weights easy examples. This
# kernel audits d(FL)/dx (x = the pre-sigmoid logit), the quantity that
# actually drives training via backpropagation.
#
# The naive gradient is the literal, un-simplified product/chain rule
# applied straight to the formula above:
#   d(FL)/dx = -alpha_t * [ -gamma*(1-p_t)**(gamma-1)*dp_t/dx*log(p_t)
#                           + (1-p_t)**gamma * (1/p_t) * dp_t/dx ]
# This is mathematically correct wherever it is well-defined, but it
# contains an explicit (1 - p_t)**(gamma - 1) factor. When gamma == 0
# (a real, supported, non-exotic configuration -- gamma=0 is Focal
# Loss's own documented reduction to plain alpha-weighted binary
# cross-entropy) and the prediction has saturated in the correct
# direction (an ordinary, even desirable, training outcome: p_t very
# close to 1, e.g. a confidently-correct high-magnitude logit), then
# `1 - p_t` underflows to exactly 0.0 in floating point, and
# `0.0 ** (0 - 1) == 0.0 ** -1` is `inf` (a literal division by zero
# inside the power). That `inf` is then multiplied by `dp_t/dx`, which
# has ALSO underflowed to 0.0 at the same saturation point -- an
# IEEE754 `0 * inf` indeterminate form that evaluates to `NaN` and
# poisons the entire gradient, even though the true gradient at that
# point is an ordinary, tiny, finite number (the network is simply
# very confident and correct, exactly the case Focal Loss should
# handle gracefully by driving the gradient toward zero, not NaN).
#
# This is documented and independently confirmed in at least three
# distinct real codebases: facebookresearch/sam3#575 ("Reduced Triton
# sigmoid focal loss returns NaN gradients for gamma=0 when logits
# saturate" -- filed and fixed via PR#576, June 2026, root-caused to
# exactly this `(1 - p_t) ** (gamma - 1)` term evaluating `0 ** -1`),
# kornia/kornia#918/PR#924 ("NaN gradients on backward pass with focal
# loss", fixed by adding an epsilon), and van Leeuwen et al., "A Note
# on the Stability of the Focal Loss" (TMLR 2025, arXiv/OpenReview
# eCYActnGbu), which independently derives and empirically demonstrates
# the identical failure for the broader 0 <= gamma < 1 range (not just
# exactly gamma=0) across real CNN/ViT/U-Net training runs on CIFAR-10
# and MNIST, and proposes the same "add an epsilon inside the
# modulating factor" mitigation used in production fixes -- this repo's
# stable variant instead uses an exact algebraic rewrite (see
# stable_focal_loss_grad below) that needs no epsilon at all.


def naive_focal_loss_grad(logit, target: int, gamma: float, alpha: float, dtype):
    """d(FL)/dx via the literal, un-simplified product/chain rule --
    reproduces the sam3#575 failure shape: an explicit
    `(1 - p_t) ** (gamma - 1)` factor that is `0 ** -1 == inf` whenever
    p_t saturates to exactly 1.0 and gamma <= 1, multiplied against a
    simultaneously-vanishing `dp_t/dx` term for an IEEE754 `0 * inf`
    indeterminate form (NaN), even though the true gradient at that
    point is an ordinary small finite number."""
    dt = DTYPES[dtype]
    x = dt(logit)
    with np.errstate(over="ignore"):
        p = dt(1) / (dt(1) + np.exp(-x, dtype=dt))
    if target == 1:
        pt = p
        alpha_t = dt(alpha)
        dpt_dx = p * (dt(1) - p)
    else:
        pt = dt(1) - p
        alpha_t = dt(1) - dt(alpha)
        dpt_dx = -(p * (dt(1) - p))
    one_minus_pt = dt(1) - pt
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        term1 = -dt(gamma) * (one_minus_pt ** dt(gamma - 1)) * dpt_dx * np.log(pt, dtype=dt)
        term2 = (one_minus_pt ** dt(gamma)) * (dt(1) / pt) * dpt_dx
        result = -alpha_t * (term1 + term2)
    return float(result)


def stable_focal_loss_grad(logit, target: int, gamma: float, alpha: float, dtype):
    """d(FL)/dx via a closed form that never materializes a negative
    power of `(1 - p_t)`: substituting `dp_t/dx = p_t*(1-p_t)` (target=1)
    or `-p_t*(1-p_t)` (target=0) into the naive expression above and
    cancelling the shared `(1-p_t)` (target=1) or `p_t` (target=0)
    factor algebraically -- on paper, identical to the naive formula
    everywhere both are well-defined -- leaves only NON-negative
    integer/real powers of probabilities, which safely underflow to
    0.0 (a correct, finite answer) instead of blowing up to inf/NaN.
    This mirrors the real-world fixes (sam3 PR#576 routes gamma=0
    through a numerically stable BCE-equivalent path; kornia PR#924
    adds an epsilon) without needing an ad hoc epsilon: the algebraic
    rewrite removes the negative-power term entirely."""
    dt = DTYPES[dtype]
    x = dt(logit)
    with np.errstate(over="ignore"):
        p = dt(1) / (dt(1) + np.exp(-x, dtype=dt))
    with np.errstate(invalid="ignore", divide="ignore"):
        if target == 1:
            one_minus_p = dt(1) - p
            log_p = np.log(p, dtype=dt)
            bracket = dt(gamma) * p * log_p - one_minus_p
            result = dt(alpha) * (one_minus_p ** dt(gamma)) * bracket
        else:
            log_one_minus_p = np.log(dt(1) - p, dtype=dt)
            bracket = p - dt(gamma) * (dt(1) - p) * log_one_minus_p
            result = (dt(1) - dt(alpha)) * (p ** dt(gamma)) * bracket
    return float(result)


# --- Pearson correlation coefficient -------------------------------------
#
# Real, actively-reported bug class, not a constructed one: the textbook
# "sum of products" one-pass formula for Pearson's r,
#   r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2) - sum(x)^2) *
#                                           (n*sum(y^2) - sum(y)^2))
# computes the SAME catastrophic-cancellation shape as this repo's
# existing `variance` kernel (E[x^2] - E[x]^2), but independently and
# on two variables at once, and is what a huge fraction of from-scratch
# / textbook / interview-style implementations actually write (see
# st-hakky.hatenablog.com JA blog above using exactly this "scratch"
# form, and countless "pearson correlation from scratch numpy" EN
# tutorials/StackOverflow answers). Confirmed real-world failure shapes
# from major libraries built around exactly this cancellation:
#  - scipy.stats.pearsonr required THREE separate overflow/underflow
#    rewrites (gh-8980 "overflows with high values of x and y", gh-9353
#    "returns r=1 if r_num/r_den = inf" from tiny values, gh-3728
#    constant-input NaN-vs-1.0 inconsistency) culminating in a full
#    rewrite in PR#9562 that switched to a normalize-then-dot form
#    specifically to avoid materializing sum-of-squares before dividing.
#  - numpy/numpy#32446 (filed 2026, open) and pandas-dev/pandas#67023 /
#    #37448 / #45640: mean-centered corrcoef/corr can still return
#    exactly +1/-1/NaN inconsistently for exactly-constant columns
#    purely from float mean-reduction residue, not a defined
#    correlation at all.
# This kernel's "naive" variant reproduces the pre-PR#9562 one-pass
# formula (the actual documented failure mode); "stable" reproduces the
# post-rewrite normalize-then-dot approach (divide each centered vector
# by its own norm before the dot product, so no cross term ever grows
# past O(1) regardless of input scale) plus an explicit zero-variance
# guard returning NaN (matching scipy's current PearsonRConstantInput
# convention and gh-32446/gh-67023's "correct" answer) instead of an
# arbitrary clipped +-1.


def naive_squared_euclidean_distance(x_values, y_values, dtype: str) -> float:
    """dist^2(x, y) = dot(x,x) - 2*dot(x,y) + dot(y,y) -- the exact
    expansion documented in scikit-learn's own
    sklearn.metrics.pairwise.euclidean_distances docstring as "the most
    precise way of doing this computation" is explicitly NOT this
    formula: "this is not the most precise way of doing this
    computation, because this equation potentially suffers from
    catastrophic cancellation." Real ANN/vector-search libraries use
    this expansion because dot(x,x) and dot(y,y) can be precomputed
    once per vector and reused across every query -- the same
    efficiency tradeoff sklearn's docstring describes -- so it appears
    in production embedding-similarity/KNN candidate re-ranking code,
    not just textbooks. When x and y are nearly identical (near-
    duplicate detection, ANN candidate re-ranking, embedding drift
    checks) and both are offset far from the origin (realistic after
    mean-pooling, positional-embedding bias, or un-normalized
    activations), dot(x,x) and dot(y,y) are both huge and nearly equal
    to 2*dot(x,y); subtracting them cancels almost all significant
    digits, and rounding noise in the surviving digits can push the
    float result negative -- an impossible value for a real squared
    distance, and one that immediately produces NaN if fed to sqrt() to
    recover the actual (non-squared) Euclidean distance. scikit-learn's
    own PR#24542 added a runtime "negative zeros and NaNs guard" to its
    Cython pairwise-distance reduction kernels specifically to paper
    over this at the top of pairwise.py's fast path -- confirming this
    is a real, previously-patched failure mode, not a hypothetical one.
    """
    x = _arr(x_values, dtype)
    y = _arr(y_values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.dot(x, x) - 2.0 * np.dot(x, y) + np.dot(y, y))


def stable_squared_euclidean_distance(x_values, y_values, dtype: str) -> float:
    """Compute the difference vector first, then sum its squares --
    dist^2(x, y) = sum((x_i - y_i)^2). Each term is bounded by the
    actual per-coordinate difference rather than by the vectors'
    absolute scale, so there is nothing large to cancel regardless of
    how far x and y sit from the origin: this is the same
    "difference-before-reduction" principle already used by this
    repo's `stable_variance` and `stable_pearson_correlation` kernels,
    applied to the squared-distance formula scikit-learn's own docs
    recommend as more precise than the dot-product expansion above.
    """
    x = _arr(x_values, dtype)
    y = _arr(y_values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = x - y
        return float(np.dot(diff, diff))


# --- BPE trainer pair-count accumulator (huggingface/tokenizers #2058) ---
#
# Real, currently-open bug (issue #2058, still open; the three linked
# fix PRs #2059/#2087/#2105 are all still unmerged as of this writing --
# verified live via the GitHub API before writing this kernel's
# acceptance card, per this repo's own reproduce-before-accept
# discipline). huggingface/tokenizers' Rust BpeTrainer accumulates each
# candidate merge pair's corpus-wide occurrence count in
# `AHashMap<Pair, i32>` (`tokenizers/src/models/bpe/trainer.rs`),
# incremented with plain `+=` and no overflow check:
#   *pair_counts.entry(cur_pair).or_default() += counts[i] as i32;
# A pair whose true count exceeds i32::MAX (2,147,483,647) -- reached in
# practice by common pairs (e.g. two-space indentation) in large code
# corpora -- silently wraps via two's-complement instead of raising, and
# can even go negative. Since BpeTrainer's own merge-selection step
# picks the highest-count pair at each step, a wrapped-negative count
# makes the trainer skip what is actually the single most frequent pair
# in the corpus, silently corrupting the learned merge order (and every
# downstream tokenization) with no error or warning at training time.
def naive_bpe_pair_count_overflow(increments) -> float:
    """Reproduce Rust's i32 `+=` wrapping behavior for a pair's running
    occurrence count, one corpus-scan increment at a time (numpy int32
    arithmetic wraps identically to Rust's release-mode integer
    overflow, which is what the compiled BpeTrainer actually ships)."""
    total = np.int32(0)
    with np.errstate(over="ignore"):
        for inc in increments:
            total = np.add(total, np.int32(inc), dtype=np.int32)
    return float(total)


def stable_bpe_pair_count_overflow(increments) -> float:
    """Spec-correct: accumulate in arbitrary precision (Python int --
    matching the actual merged-but-not-yet-released fix in PR#2059/
    #2087/#2105, which widens the map's value type from i32 to i64;
    i64::MAX is far beyond any realistic corpus's pair count), so the
    running total for a pair never wraps for any real-world input."""
    total = 0
    for inc in increments:
        total += int(inc)
    return float(total)


def naive_pearson_correlation(x_values, y_values, dtype: str) -> float:
    """(n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2)-sum(x)^2) *
    (n*sum(y^2)-sum(y)^2)) -- the literal "sum of products" formula for
    Pearson's r, taught in many textbooks/tutorials and the exact shape
    scipy.stats.pearsonr used before PR#9562. Squares and cross-products
    of the raw (uncentered) values can overflow or lose all precision
    to cancellation long before the mean-centered form would, for
    ordinary data offset far from zero (sensor timestamps, prices,
    large IDs)."""
    x = _arr(x_values, dtype)
    y = _arr(y_values, dtype)
    dt = DTYPES[dtype]
    n = dt(len(x))
    with np.errstate(over="ignore", invalid="ignore"):
        sx = np.sum(x)
        sy = np.sum(y)
        sxy = np.sum(x * y)
        sxx = np.sum(x * x)
        syy = np.sum(y * y)
        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        return float(num / den)


def stable_pearson_correlation(x_values, y_values, dtype: str) -> float:
    """Mean-center both vectors, then normalize each by its own norm
    BEFORE the dot product (scipy PR#9562's fix), so no intermediate
    term ever exceeds O(1) regardless of the input's absolute scale --
    the same rescale-before-combine principle already used by this
    repo's `stable_rms_norm`/`stable_online_softmax` kernels. An exact
    zero-variance input (either vector exactly constant) is undefined
    for Pearson's r and returns NaN explicitly, matching scipy's
    PearsonRConstantInputWarning convention, rather than letting a
    0/0 division fall through to an arbitrary implementation-dependent
    value (the numpy#32446 / pandas#67023 / pandas#37448 failure this
    kernel's fixtures reproduce)."""
    x = _arr(x_values, dtype)
    y = _arr(y_values, dtype)
    dt = DTYPES[dtype]
    with np.errstate(over="ignore", invalid="ignore"):
        xm = (x - np.mean(x)).astype(dt)
        ym = (y - np.mean(y)).astype(dt)
        norm_x = np.sqrt(np.sum(xm * xm))
        norm_y = np.sqrt(np.sum(ym * ym))
        if norm_x == 0 or norm_y == 0:
            return float("nan")
        r = float(np.sum((xm / norm_x) * (ym / norm_y)))
    return max(min(r, 1.0), -1.0)


# --- weighted-reservoir-sampling comparison key -------------------------

def naive_weighted_sampling_key(u: float, weight: float, dtype: str) -> float:
    """Literal Efraimidis-Spirakis A-Res key, log(u**(1/weight)), computed
    by taking the power FIRST (as the algorithm's original paper and a
    verbatim implementation do -- R's wrswoR package ships this as
    `sample_int_expjs`, explicitly documented as \"at the cost of
    numerical stability\") and only then taking the log to make it
    comparable on a single scale across items with wildly different
    weights. For any weight small enough that 1/weight is large, u**(1/
    weight) underflows to exactly 0.0 in IEEE754 long before the true
    value is actually zero -- log(0.0) is then -inf, discarding the
    real (still order-relevant) magnitude entirely and making every
    sufficiently-low-weight item's key collide at the same -inf,
    silently destroying the proportional-selection guarantee among
    them (a low-weight item that should still occasionally win against
    an even-lower-weight one no longer can, because both keys are now
    identical -inf).
    """
    dt = DTYPES[dtype]
    u_v = dt(u)
    w_v = dt(weight)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        key = u_v ** (dt(1.0) / w_v)
        return float(np.log(key))


def stable_weighted_sampling_key(u: float, weight: float, dtype: str) -> float:
    """The A-ExpJ / wrswoR `sample_int_expj` fix: compute log(u)/weight
    directly, never materializing u**(1/weight) as an intermediate at
    all. This is exactly the same log-space trick as this repo's other
    kernels (logsumexp, kl_divergence) -- do the operation prone to
    overflow/underflow (here, a power with a huge exponent) in log
    space via a plain division instead, so the result stays a finite,
    order-preserving real number all the way down to the smallest
    weights actually used in practice."""
    dt = DTYPES[dtype]
    u_v = dt(u)
    w_v = dt(weight)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        return float(np.log(u_v) / w_v)


# --- P^2 streaming quantile estimator (marker-position bookkeeping) ------
#
# Real, author-documented bug, not a constructed one: the P^2 (Piecewise-
# Parabolic) algorithm (Jain & Chlamtac, CACM 1985) is the classic O(1)-
# memory streaming/online quantile estimator -- it never stores the input
# stream, only five running "marker" positions/heights. The original
# paper's own pseudocode suggests maintaining each marker's *desired*
# position n'_i via a running increment (`dns[i]`, a per-marker constant
# added once per observation) "to reduce CPU overhead" rather than
# recomputing n'_i = count * p_i from scratch every step. Andrey Akinshin
# (maintainer of the perfolizer benchmarking-statistics library) received
# and confirmed a real bug report on exactly this (GitHub issue
# AndreyAkinshin/perfolizer#8, "P2QuantileEstimator rounding issue"): the
# repeated `ns[i] += dns[i]` accumulation drifts under floating-point
# rounding (e.g. a quantity that should land exactly on an integer marker
# boundary like 5.999999994 instead of 6.0), silently deferring a marker
# adjustment that should have fired -- corrupting the estimator's internal
# state for the rest of the stream, not just one output value. The fix
# (recompute each `ns[i]` fresh from `count` every observation, never
# accumulating) is documented by the same author to be simultaneously
# *more* accurate and *faster* (no `dns` array to maintain). This is a
# distinct mechanism from every other kernel in this file: not an
# overflow/underflow in a single expression (geometric_mean,
# weighted_sampling_key) and not catastrophic cancellation in a sum
# (variance, pearson_correlation, sum) -- it is accumulator drift in a
# streaming algorithm's *internal bookkeeping state*, which then silently
# skips a marker-adjustment decision for the remainder of the stream.


def _p2_quantile(values, prob, dtype: str, *, recompute_ns: bool) -> float:
    dt = DTYPES[dtype]
    with np.errstate(over="ignore", invalid="ignore"):
        p = dt(prob)
        q = [dt(0.0)] * 5
        n = [0, 0, 0, 0, 0]
        ns = [dt(0.0)] * 5
        dns = [dt(0.0)] * 5
        count = 0

        def parabolic(i: int, d: int) -> float:
            d_v = dt(d)
            n_ip1_ip1 = dt(float(n[i + 1] - n[i - 1]))
            term_a = dt(dt(float(n[i] - n[i - 1] + d)) * dt(q[i + 1] - q[i]) / dt(float(n[i + 1] - n[i])))
            term_b = dt(dt(float(n[i + 1] - n[i] - d)) * dt(q[i] - q[i - 1]) / dt(float(n[i] - n[i - 1])))
            return float(dt(q[i] + dt(d_v / n_ip1_ip1) * dt(term_a + term_b)))

        def linear(i: int, d: int) -> float:
            return float(dt(q[i] + dt(dt(float(d)) * dt(q[i + d] - q[i]) / dt(float(n[i + d] - n[i])))))

        for raw in values:
            x = dt(raw)
            if count < 5:
                q[count] = x
                count += 1
                if count == 5:
                    q = sorted(q)
                    n = [0, 1, 2, 3, 4]
                    two, four, one = dt(2.0), dt(4.0), dt(1.0)
                    ns[0] = dt(0.0)
                    ns[1] = dt(two * p)
                    ns[2] = dt(four * p)
                    ns[3] = dt(two + dt(two * p))
                    ns[4] = dt(4.0)
                    if not recompute_ns:
                        dns[0] = dt(0.0)
                        dns[1] = dt(p / two)
                        dns[2] = dt(p)
                        dns[3] = dt(dt(one + p) / two)
                        dns[4] = one
                continue
            if x < q[0]:
                q[0] = x
                k = 0
            elif x < q[1]:
                k = 0
            elif x < q[2]:
                k = 1
            elif x < q[3]:
                k = 2
            elif x < q[4]:
                k = 3
            else:
                q[4] = x
                k = 3
            for i in range(k + 1, 5):
                n[i] += 1
            if recompute_ns:
                cnt = dt(float(count))
                two = dt(2.0)
                ns[1] = dt(dt(cnt * p) / two)
                ns[2] = dt(cnt * p)
                ns[3] = dt(dt(cnt * dt(dt(1.0) + p)) / two)
                ns[4] = cnt
            else:
                for i in range(5):
                    ns[i] = dt(ns[i] + dns[i])
            for i in range(1, 4):
                d = dt(ns[i] - dt(float(n[i])))
                if (d >= dt(1.0) and (n[i + 1] - n[i]) > 1) or (
                    d <= dt(-1.0) and (n[i - 1] - n[i]) < -1
                ):
                    d_int = 1 if d > dt(0.0) else -1
                    qs = parabolic(i, d_int)
                    qs_dt = dt(qs)
                    if q[i - 1] < qs_dt < q[i + 1]:
                        q[i] = qs_dt
                    else:
                        q[i] = dt(linear(i, d_int))
                    n[i] += d_int
            count += 1

        if count <= 5:
            ordered = sorted(q[:count])
            idx = round((count - 1) * float(p))
            return float(ordered[idx])
        return float(q[2])


def naive_p2_quantile(values, prob: float, dtype: str) -> float:
    """P^2 streaming quantile estimator using the original paper's
    running-increment (`ns[i] += dns[i]`) marker-position bookkeeping --
    see perfolizer issue #8 for the real-world bug report this
    reproduces."""
    return _p2_quantile(values, prob, dtype, recompute_ns=False)


def stable_p2_quantile(values, prob: float, dtype: str) -> float:
    """P^2 streaming quantile estimator recomputing each marker's desired
    position fresh from `count` every observation (`ns[i] = count * p_i`)
    instead of accumulating -- the author-published fix, which is both
    more accurate (no accumulated rounding drift) and cheaper (no `dns`
    array to maintain)."""
    return _p2_quantile(values, prob, dtype, recompute_ns=True)


# --- geometric mean ------------------------------------------------------

def naive_geometric_mean(values, dtype: str) -> float:
    """prod(x)**(1/n) -- the literal textbook geometric-mean formula:
    multiply every value together first, then take the n-th root. This
    is exactly the shape flagged in scipy.stats.gmean's own history
    (scipy gh-1053 / Trac#526, "gmean cannot handle large numbers"):
    multiplying even a modest number of values whose magnitude is above
    1 grows the intermediate product past a dtype's max representable
    value long before the true (much smaller, order-1) n-th root is
    reached, overflowing to +inf; symmetrically, multiplying many
    values below 1 underflows the intermediate product to exactly 0.0
    long before the true root is computed, silently returning 0
    instead of a small-but-nonzero geometric mean. Both failures are
    pure intermediate-representation artifacts of the evaluation
    order, not of the mathematical quantity itself (which is always
    finite and of the same order of magnitude as a typical input)."""
    x = _arr(values, dtype)
    n = len(x)
    dt = DTYPES[dtype]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prod = np.prod(x)
        return float(prod ** (dt(1.0) / dt(n)))


def stable_geometric_mean(values, dtype: str) -> float:
    """exp(mean(log(x))) -- the standard log-space fix (the same one
    scipy adopted in the gh-1053 fix and that R's own geometric-mean
    recipes use): take the log of each value first (bringing every
    term down to an O(1)-scale exponent regardless of the input's raw
    magnitude), average those logs, and only then exponentiate once at
    the very end. No intermediate ever approaches a dtype's overflow
    or underflow boundary the way the raw running product does, so the
    result stays finite and accurate across the same input range where
    the naive formula silently returns +inf or exactly 0."""
    x = _arr(values, dtype)
    dt = DTYPES[dtype]
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_mean = np.mean(np.log(x).astype(dt))
        return float(np.exp(log_mean))


# --- repetition penalty (sign-branched gauge dependence) ----------------
#
# Real-world bug, not a constructed one: arXiv:2607.09791 ("Gauge
# dependence and structured-output corruption in sign-branched
# repetition penalties") documents that the multiplicative repetition
# penalty shipped across the LLM inference ecosystem -- HuggingFace
# transformers, vLLM, llama.cpp, and "a dozen further engines" per the
# paper -- branches on the *sign* of each raw logit before penalizing a
# previously-seen token: divide by theta if the logit is positive,
# multiply by theta if it is non-positive. This makes the penalized
# result depend on which arbitrary additive constant the model's logits
# happen to sit at, even though softmax itself is exactly shift-
# invariant (softmax(x) == softmax(x + c) for any constant c, since the
# added exp(c) factor cancels between numerator and denominator). Two
# logit vectors that represent the *identical* underlying probability
# distribution before penalization (one is just the other shifted by a
# constant -- a difference every downstream softmax treats as
# meaningless) can select different next tokens after an ordinary
# repetition penalty is applied, purely as an artifact of where the
# unpenalized logits happened to sit relative to zero. This is not a
# hypothetical: gpt2's own per-position logits are bimodal (the paper
# reports deciles spanning roughly -230 to +150), so ordinary generation
# already produces exactly the straddling-zero condition this bug
# depends on, and the paper's own StarCoder2-7B HumanEval experiment
# demonstrates real greedy-decode token flips from this mechanism.
#
# The naive formulation below reproduces the documented shape exactly:
# sign-branch on the raw logit. The stable formulation applies the
# identical branch/divide-multiply logic to log-probabilities (logit
# minus the log-partition-function logsumexp(logits)) instead of raw
# logits -- log-probabilities are themselves shift-invariant by
# construction (subtracting logsumexp cancels any additive shift to the
# input before the branch ever runs), so the penalized distribution no
# longer depends on the arbitrary additive gauge of the input logits.
# This is the same fix direction the paper itself notes already exists
# inside HuggingFace: "beam search has applied its entire logits-
# processor chain, repetition penalty included, to log-probabilities
# since at least transformers v4.0.0", while greedy/sampled decoding
# apply the raw (gauge-dependent) form -- i.e. the same library already
# ships both the buggy and the fixed behavior, selected only by which
# decoding strategy happens to be in use.


def naive_repetition_penalty(values, seen_mask, theta: float, dtype):
    """Sign-branch on the raw logit -- the documented HF/vLLM/llama.cpp
    pattern: `logit/theta` if positive, `logit*theta` otherwise, applied
    only to positions in `seen_mask` (previously-generated tokens),
    followed by an ordinary softmax. Gauge-dependent: shifting every
    input logit by the same additive constant (which leaves the
    pre-penalty distribution identical) can change which branch fires
    for a penalized position near zero, and therefore change the
    resulting (post-penalty, post-softmax) distribution and argmax.
    """
    x = _arr(values, dtype).astype(np.float64)
    keep = np.asarray(seen_mask, dtype=bool)
    out = x.copy()
    for i in range(len(out)):
        if keep[i]:
            out[i] = out[i] / theta if out[i] > 0 else out[i] * theta
    with np.errstate(over="ignore", invalid="ignore"):
        m = np.max(out)
        e = np.exp(out - m)
        return (e / np.sum(e)).astype(DTYPES[dtype])


def stable_repetition_penalty(values, seen_mask, theta: float, dtype):
    """Apply the identical sign-branch/divide-multiply penalty to
    log-probabilities (logit - logsumexp(logits)) instead of raw
    logits. Subtracting the log-partition-function makes the quantity
    being branched on shift-invariant by construction (an additive
    shift to every input logit shifts logsumexp by the same constant,
    which cancels exactly), so the resulting penalized distribution no
    longer depends on the arbitrary additive gauge of the unpenalized
    logits -- only on the actual (gauge-invariant) probabilities.
    """
    x = _arr(values, dtype).astype(np.float64)
    keep = np.asarray(seen_mask, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        m = np.max(x)
        logz = m + np.log(np.sum(np.exp(x - m)))
        logp = x - logz
        out = logp.copy()
        for i in range(len(out)):
            if keep[i]:
                out[i] = out[i] / theta if out[i] > 0 else out[i] * theta
        m2 = np.max(out)
        e = np.exp(out - m2)
        return (e / np.sum(e)).astype(DTYPES[dtype])


def _bf16_round(x64: np.ndarray) -> np.ndarray:
    """Round a float64 array to bfloat16 precision (7 explicit mantissa
    bits, round-to-nearest-even at the truncation boundary), returned
    widened back to float64. Used only to emulate the fixed hardware
    compute dtype a real inference stack's draft model would sample
    from, independent of the audit `dtype` sweep this tool otherwise
    parameterizes every other kernel by.
    """
    x32 = x64.astype(np.float32)
    as_uint = x32.view(np.uint32)
    rounded = (as_uint + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def _softmax64(x: np.ndarray, dtype: str) -> np.ndarray:
    m = np.max(x)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp((x - m).astype(DTYPES[dtype]))
        return (e / np.sum(e)).astype(DTYPES[dtype]).astype(np.float64)


def _rejection_sample_mix(r: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The Leviathan/Chen (2023) speculative-decoding rejection-sampling
    output-distribution identity: given the token was actually sampled
    from `r`, accepted with probability min(1, q/p) computed against
    `p`, and on rejection resampled from normalize(max(0, q-p)), the
    output distribution is r*a + (escaped mass)*resid. This equals `q`
    exactly (up to floating rounding of the shared array) precisely
    when r == p -- the entire correctness property this kernel audits.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, np.minimum(1.0, q / np.where(p > 0, p, 1.0)), 1.0)
    resid_raw = np.clip(q - p, 0, None)
    s = resid_raw.sum()
    resid = resid_raw / s if s > 0 else np.zeros_like(resid_raw)
    escape = np.sum(r * (1 - a))
    return r * a + escape * resid


def naive_speculative_reject(values, q_values, dtype):
    """Speculative-decoding rejection sampling with a real-world
    precision hazard: the draft token is SAMPLED from the model's
    native hardware compute-dtype probabilities (bfloat16, the
    deployed-model storage dtype in practice, modeled here independent
    of the requested audit `dtype`), but the accept/reject math
    RECOMPUTES the draft probability at a different (the requested)
    dtype from the same logits. This is the exact defect class fixed in
    deepseek-ai/DeepSpec PR#30 ("Draft samples were drawn from
    native-dtype probabilities while rejection used float32
    probabilities") and discussed in vllm-project/vllm PR#48641/#53630
    (every downstream consumer of a rewritten/re-derived logits/probs
    array inherits an extra, mismatched rounding). `values` holds the
    draft logits, `q_values` holds the target logits (the naming
    matches every other two-distribution kernel in this file, e.g.
    kl_divergence).

    Every individual probability array here is finite and a valid
    (sums-to-1, non-negative) distribution -- this is NOT an
    overflow/underflow bug. The bug is purely that the distribution
    SAMPLED from (r) differs from the distribution the accept/reject
    identity assumes (p), which silently breaks speculative decoding's
    core lossless guarantee: the output no longer matches the target
    model's true distribution q, even though nothing crashes or
    produces NaN/inf.
    """
    xd = _arr(values, dtype).astype(np.float64)
    xt = _arr(q_values, dtype).astype(np.float64)
    r_fp64 = _softmax64(xd, "float64")
    r = _bf16_round(r_fp64)
    r = r / r.sum()
    p = _softmax64(xd, dtype)
    q = _softmax64(xt, dtype)
    return _rejection_sample_mix(r, p, q).astype(DTYPES[dtype])


def stable_speculative_reject(values, q_values, dtype):
    """The DeepSpec PR#30 fix pattern: materialize the draft-probability
    array ONCE and reuse the identical array for both the sampling step
    and the accept/reject math, so r is p by construction (not merely
    by numerical accident) -- the mismatch naive_speculative_reject
    introduces cannot occur here regardless of which compute dtype a
    real deployment happens to sample from.
    """
    xd = _arr(values, dtype).astype(np.float64)
    xt = _arr(q_values, dtype).astype(np.float64)
    rp = _softmax64(xd, dtype)
    q = _softmax64(xt, dtype)
    return _rejection_sample_mix(rp, rp, q).astype(DTYPES[dtype])


def naive_weight_decay(values, q_values, num_steps: int, dtype: str):
    """Real, currently-open production bug: Nerogar/OneTrainer#996
    ("No weight decay with Adam, bf16 and stochastic rounding" -- filed
    2025-09-13, still open). AdamW's decoupled weight decay multiplies
    the parameter by (1 - lr*wd) and writes the result straight back
    into the parameter's storage dtype every step. When that storage
    dtype is bf16 (or, as modeled here via the audited `dtype` sweep,
    any format whose unit-in-the-last-place exceeds the per-step decay
    fraction lr*wd), the update rounds away to the SAME stored value
    every single step -- not a one-off rounding error, a total, silent
    stall: weight decay is configured, runs without error or warning,
    and does precisely nothing for the entire run. This is the same
    "quantized-EMA-state stalling" mechanism formalized in
    arXiv:2603.16731 (state-update stalling once |update| < half a ULP)
    and arXiv:2607.09800 (The Silent Freeze), and the fix already
    shipped for bf16 training master weights by imoneoi/bf16_fused_adam
    and (for Adafactor specifically) OneTrainer's own adafactor_extensions.py.

    `values` holds [w0] (the initial parameter value), `q_values` holds
    [decay_per_step] (lr*weight_decay, the fractional shrink AdamW
    applies each step), matching this file's established two-array
    naming convention for two-argument kernels (e.g.
    weighted_sampling_key's [u], [weight]).
    """
    (w0,) = values
    (decay_per_step,) = q_values
    dt = DTYPES[dtype]
    w = dt(w0)
    factor = dt(1.0) - dt(decay_per_step)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for _ in range(num_steps):
            w = dt(w * factor)
    return float(w)


def stable_weight_decay(values, q_values, num_steps: int, dtype: str):
    """The OneTrainer#996 fix pattern (mirrors what Adafactor already
    does in the same codebase, and what imoneoi/bf16_fused_adam and
    apex's FusedAdam(master_weights=True) do generally): keep the
    running parameter value in float64 precision for the ENTIRE decay
    loop, and round to the requested storage dtype only once, at the
    point the value is actually read back out for use -- never
    mid-loop. Every sub-ULP-at-storage-precision update still shrinks
    the float64 master correctly, so decay accumulates exactly as
    configured; only the final read-out loses precision, not the
    100+ intermediate steps.
    """
    (w0,) = values
    (decay_per_step,) = q_values
    w = np.float64(w0)
    factor = 1.0 - np.float64(decay_per_step)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for _ in range(num_steps):
            w = w * factor
    return float(DTYPES[dtype](w))


def naive_gradient_accumulation_bias(values, q_values, dtype: str) -> float:
    """The real, widely-reported HuggingFace Trainer / PyTorch-Lightning
    bug (huggingface.co/blog/gradient_accumulation, reported by Benjamin
    Marie and independently by Unsloth in 2024, matching PyTorch-
    Lightning#20350): gradient accumulation is supposed to be
    mathematically equivalent to training on one large batch, but when
    the k accumulated micro-batches have different non-padding TOKEN
    COUNTS (the normal case for variable-length sequences), naively
    computing each micro-batch's own MEAN per-token loss and then
    averaging those k means is NOT the same as the true large-batch
    mean. `values` holds the per-microbatch SUMMED per-token loss g_i
    for each of the k accumulation steps; `q_values` holds the matching
    per-microbatch non-padding token count n_i. This is the exact
    "loss = loss / self.args.gradient_accumulation_steps" shape the
    pre-fix HF Trainer used, applied here to a pre-summed per-step loss
    for auditability without needing a real model or GPU.
    """
    g = _arr(values, dtype)
    n = _arr(q_values, dtype)
    dt = DTYPES[dtype]
    k = dt(len(g))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        per_step_mean = g / n
        return float(dt(np.sum(per_step_mean) / k))


def stable_gradient_accumulation_bias(values, q_values, dtype: str) -> float:
    """The fix HuggingFace actually shipped (huggingface.co/blog/gradient_accumulation
    PR#34198, and the equivalent fix in Unsloth and PyTorch-Lightning#20350):
    sum every micro-batch's summed per-token loss FIRST across the whole
    accumulation window, then divide once by the TOTAL non-padding token
    count across all micro-batches -- the true large-batch-equivalent
    mean regardless of how unevenly tokens are distributed across steps.
    """
    g = _arr(values, dtype)
    n = _arr(q_values, dtype)
    dt = DTYPES[dtype]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return float(dt(np.sum(g)) / dt(np.sum(n)))


def naive_longrope_factor_select(
    values, base_inv_freq, short_factor, long_factor,
    original_max_pos, seq_len, ctx_alloc, dtype,
):
    """Real, currently-unfixed bug: ggml-org/llama.cpp#24823 ("Phi-3 /
    Phi-4 LongRoPE: short sequences silently use long-context RoPE
    factors when the allocated context exceeds
    original_max_position_embeddings"). Phi-3/Phi-4's 'longrope'
    rope_type keeps two per-dimension scaling factors -- `short_factor`
    for sequences at or below `original_max_position_embeddings`,
    `long_factor` above it (real values from the published
    microsoft/Phi-3-mini-128k-instruct config.json: original_max_
    position_embeddings=4096, head_dim=96). The reference HF
    transformers implementation (modeling_rope_utils.
    _compute_longrope_parameters) selects the factor from the ACTUAL
    sequence length being encoded. llama.cpp instead selects it from
    the ALLOCATED context size (n_ctx_seq) -- so loading a model with
    a large context window and then serving a short document (the
    issue's own repro: a ~640-token document under `-c 8192`) silently
    encodes every position with the long-context factor the model was
    never trained to use at that length, changing inv_freq by up to
    ~30x on the lowest-frequency dimensions. The bug is entirely about
    *which* factor is selected, not float rounding, so it reproduces
    identically at every dtype swept here; still parameterized by
    dtype like every other kernel for interface consistency and so a
    future dtype-dependent regression would still be caught.
    """
    factor = long_factor if ctx_alloc > original_max_pos else short_factor
    dt = DTYPES[dtype]
    inv_freq = dt(base_inv_freq) / dt(factor)
    positions = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        angle = (positions * inv_freq).astype(dt)
        return np.cos(angle).astype(dt)


def stable_longrope_factor_select(
    values, base_inv_freq, short_factor, long_factor,
    original_max_pos, seq_len, ctx_alloc, dtype,
):
    """The fix already correct in HF transformers
    (modeling_rope_utils._compute_longrope_parameters) and the mitigation
    llama.cpp#24823 itself proposes: select the scaling factor from the
    ACTUAL sequence length being encoded, never from the allocated
    context size -- a short document stays on the short-context factor
    no matter how large a KV-cache/context window the session allocated
    for it.
    """
    factor = long_factor if seq_len > original_max_pos else short_factor
    dt = DTYPES[dtype]
    inv_freq = dt(base_inv_freq) / dt(factor)
    positions = _arr(values, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        angle = (positions * inv_freq).astype(dt)
        return np.cos(angle).astype(dt)


KERNELS = {
    "logsumexp": (naive_logsumexp, stable_logsumexp),
    "softmax": (naive_softmax, stable_softmax),
    "cross_entropy": (naive_cross_entropy, stable_cross_entropy),
    "variance": (naive_variance, stable_variance),
    "layer_norm": (naive_layer_norm, stable_layer_norm),
    "rms_norm": (naive_rms_norm, stable_rms_norm),
    "kl_divergence": (naive_kl_divergence, stable_kl_divergence),
    "online_softmax": (naive_online_softmax, stable_online_softmax),
    "masked_softmax": (naive_masked_softmax, stable_masked_softmax),
    "sum": (naive_sum, stable_sum),
    "rope_cos": (naive_rope_cos, stable_rope_cos),
    "int8_add": (naive_int8_add, stable_int8_add),
    "hll_register": (naive_hll_register_term, stable_hll_register_term),
    "focal_loss_grad": (naive_focal_loss_grad, stable_focal_loss_grad),
    "pearson_correlation": (naive_pearson_correlation, stable_pearson_correlation),
    "weighted_sampling_key": (naive_weighted_sampling_key, stable_weighted_sampling_key),
    "geometric_mean": (naive_geometric_mean, stable_geometric_mean),
    "p2_quantile": (naive_p2_quantile, stable_p2_quantile),
    "repetition_penalty": (naive_repetition_penalty, stable_repetition_penalty),
    "speculative_reject": (naive_speculative_reject, stable_speculative_reject),
    "weight_decay": (naive_weight_decay, stable_weight_decay),
    "gradient_accumulation_bias": (naive_gradient_accumulation_bias, stable_gradient_accumulation_bias),
    "longrope_factor_select": (naive_longrope_factor_select, stable_longrope_factor_select),
    "squared_euclidean_distance": (naive_squared_euclidean_distance, stable_squared_euclidean_distance),
    "bpe_pair_count_overflow": (naive_bpe_pair_count_overflow, stable_bpe_pair_count_overflow),
}
