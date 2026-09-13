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


KERNELS = {
    "logsumexp": (naive_logsumexp, stable_logsumexp),
    "softmax": (naive_softmax, stable_softmax),
    "cross_entropy": (naive_cross_entropy, stable_cross_entropy),
    "variance": (naive_variance, stable_variance),
    "layer_norm": (naive_layer_norm, stable_layer_norm),
}
