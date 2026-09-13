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


KERNELS = {
    "logsumexp": (naive_logsumexp, stable_logsumexp),
    "softmax": (naive_softmax, stable_softmax),
    "cross_entropy": (naive_cross_entropy, stable_cross_entropy),
    "variance": (naive_variance, stable_variance),
    "layer_norm": (naive_layer_norm, stable_layer_norm),
    "rms_norm": (naive_rms_norm, stable_rms_norm),
    "kl_divergence": (naive_kl_divergence, stable_kl_divergence),
    "online_softmax": (naive_online_softmax, stable_online_softmax),
}
