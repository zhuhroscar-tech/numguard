# Numerical stability derivations

This document is the derivation reference cited by `src/numguard/kernels.py`'s
module docstring ("see docs/numerical-stability.md for the derivation of
each"). It explains, for each of numguard's six kernels, why the naive
(textbook) formula is mathematically equivalent to the stable formula on
paper but behaves differently in floating-point arithmetic, and exactly
what mitigation the stable variant applies. Every claim below matches the
actual implementation in `kernels.py` and is exercised by `reference.py`'s
independent `Decimal`-based ground truth and the fixtures in `fixtures.py`.

## log-sum-exp

**Textbook definition:** `logsumexp(x) = log(sum(exp(x_i)))`

**Naive formula** (`naive_logsumexp`): evaluate the definition directly.
`exp(x_i)` overflows to `inf` in float32 for `x_i > ~88.7` (and in float16
for `x_i > ~11.1`), and every term underflows to `0.0` for very negative
`x_i`, at which point `log(0)` is `-inf` even when the true value is
finite.

**Stable formula** (`stable_logsumexp`): subtract the max element first,
`m + log(sum(exp(x_i - m)))` where `m = max(x)`. Every shifted exponent
`x_i - m` is `<= 0`, so `exp(x_i - m) <= 1` and can never overflow; the
largest term is always exactly `1`, so the sum can never underflow to
`0`. This is the standard "log-sum-exp trick" and is mathematically
identical to the naive formula: `log(sum(exp(x_i))) = m + log(sum(exp(x_i
- m)))` follows directly from factoring `exp(m)` out of the sum.

## softmax

**Textbook definition:** `softmax(x)_i = exp(x_i) / sum_j(exp(x_j))`

**Naive formula** (`naive_softmax`): evaluate directly -- same overflow
failure mode as naive log-sum-exp (`exp` overflows for large `x_i`,
producing `nan` from `inf / inf`), plus underflow-to-zero for very
negative inputs (which is actually often *correct*, since the true
probability really is negligible -- the bug case is specifically the
overflow one).

**Stable formula** (`stable_softmax`): shift by the max, `exp(x_i - m) /
sum_j(exp(x_j - m))`. Multiplying numerator and denominator of the
textbook definition by `exp(-m)` leaves the ratio unchanged, so this is
exact, and by the same argument as log-sum-exp, no term can overflow.

## cross-entropy (single target index)

**Textbook definition:** `cross_entropy(x, t) = -log(softmax(x)[t])`

**Naive formula** (`naive_cross_entropy`): materialize the full
probability vector via naive softmax, index it, then take `-log`. This
compounds two problems: naive softmax's overflow bug, and a second
precision loss even when softmax itself doesn't overflow -- if the true
probability at the target index underflows to exactly `0.0` (correct
softmax, incorrect log input), `-log(0.0)` is `inf` even though the true
cross-entropy is an ordinary finite number in the tens.

**Stable formula** (`stable_cross_entropy`): use the log-domain identity
`-log(softmax(x)[t]) = logsumexp(x) - x[t]` directly -- this never
computes an explicit probability, so there is no intermediate underflow
to lose precision on. It is algebraically the same quantity (substitute
the softmax definition and simplify the log of a quotient), computed via
`stable_logsumexp`, so it inherits that kernel's overflow safety too.

## variance

**Textbook (one-pass) definition:** `Var(x) = E[x^2] - E[x]^2`

**Naive formula** (`naive_variance`): evaluate the one-pass formula
directly. When every `x_i` is offset far from zero relative to the
spread of the data (e.g. `[1000000, 1000001, 1000002, 1000003]`, true
variance `1.25`), `E[x^2]` and `E[x]^2` are both huge, nearly-equal
numbers (~1e12). Subtracting two nearly-equal floating point numbers of
that magnitude is catastrophic cancellation: the result can lose most or
all of its significant digits, and in float32 can even come out
*negative* -- mathematically impossible for a true variance, since it is
a sum of squares.

**Stable formula** (`stable_variance`): the two-pass, mean-centered
formula `mean((x - mean(x))^2)`. Centering the data around its own mean
first means every squared term is bounded by the *spread* of the data,
not its absolute magnitude, so there is no large-minus-large
cancellation to lose precision to. This is algebraically identical to
the one-pass formula (expand `(x - mean(x))^2` and the cross terms
telescope back to `E[x^2] - E[x]^2`), just computed in an order that
avoids the numerically dangerous subtraction.

## LayerNorm

**Textbook definition:** `LayerNorm(x) = (x - mean(x)) / sqrt(Var(x) +
eps)`

**Naive formula** (`naive_layer_norm`): builds directly on the naive
one-pass variance formula above. This is not merely "less precise" --
when the naive variance formula cancels to a *negative* number (see
`large_offset_negative_variance` fixture), `sqrt(negative + eps)` is
`NaN`, and the entire normalized output becomes `NaN` for every element,
silently, for perfectly ordinary-looking activations. Since LayerNorm
sits inside essentially every Transformer block, this is a realistic
production failure mode, not a contrived edge case.

**Stable formula** (`stable_layer_norm`): builds on the two-pass
mean-centered variance, which can never go negative (it's a mean of
squares), so `sqrt(var + eps)` is always well-defined.

## RMSNorm

**Textbook definition:** `RMSNorm(x) = x / sqrt(mean(x^2) + eps)` -- note
there is deliberately no mean-subtraction (unlike LayerNorm); that
omission is the entire point of the "RMS" simplification used by LLaMA,
T5, and most modern LLMs in place of LayerNorm.

**Naive formula** (`naive_rms_norm`): computes the `mean(x^2)` reduction
in the *same* narrow dtype as the input activations. Unlike LayerNorm's
naive formula, there is no cancellation here -- the failure mode is pure
dtype-range overflow. An ordinary-looking float16 activation value like
`~300` squares to `~90000`, already past float16's `~65504` max, so
`mean(x^2)` silently overflows to `inf`, and `x / sqrt(inf)` collapses
every output to a finite-looking `0.0` (or `NaN`, depending on the exact
path) -- a failure that looks like *nothing happened*, arguably more
dangerous than an obvious `NaN`/`inf` because it doesn't trigger the
usual "something exploded" alarms.

**Stable formula** (`stable_rms_norm`): upcast the input to a wider dtype
(`RMS_NORM_UPCAST`: float16->float32, float32->float64, float64->float64)
*before* squaring and reducing, matching what real implementations do
(e.g. HF Transformers' `LlamaRMSNorm` explicitly upcasts to float32
before `pow(2).mean(...)`). The activation tensor itself can stay in its
original narrow dtype for memory/speed -- only the reduction needs the
wider range. At float64 there is nowhere higher to upcast to, so the
stable path is identical to the naive path at that precision; this is
expected, since the bug is about the reduction dtype's *range*, not
cancellation, and float64's range is already ample.

## KL divergence

**Textbook definition:** `KL(p || q) = sum_i(p_i * log(p_i / q_i))`,
the discrete Kullback-Leibler divergence between two probability
distributions `p` and `q`.

**Naive formula** (`naive_kl_divergence`): evaluate the sum literally.
This has a distinct failure shape from every other kernel in this repo
-- not overflow, not cancellation, but an IEEE754 *indeterminate form*.
Whenever `p_i` is exactly `0` (an entirely ordinary case: a one-hot
label vector in classification distillation, label smoothing, or any
sparse target distribution), the mathematically correct convention is
that the term contributes exactly `0` -- this is the standard
`x*log(x) -> 0` limit as `x -> 0`, used by every serious KL-divergence
implementation (e.g. `scipy.special.rel_entr`/`kl_div`). But naive
floating-point arithmetic computes `0 * log(0 / q_i)` as `0 * -inf`,
which IEEE754 defines as `NaN`, poisoning the entire sum even though
the true KL divergence is an ordinary finite number. This bug fires on
the *majority* of terms in one of the single most common real inputs
to this kernel (a one-hot label vector), not an obscure edge case.

**Stable formula** (`stable_kl_divergence`): guard the two
mathematically meaningful edge cases explicitly instead of letting
IEEE754 arithmetic decide by accident:
- `p_i == 0`: contributes exactly `0`, matching the standard
  convention, regardless of what `q_i` is.
- `p_i > 0` and `q_i <= 0`: the true KL divergence is `+inf` (`q`
  assigns zero probability to an event `p` considers possible) -- this
  is the mathematically *correct* answer and is passed through rather
  than masked.

Algebraically this is the identical quantity as the naive formula
everywhere both `p_i > 0` and `q_i > 0`; it only changes behavior at
the two edge cases above, where the naive formula's literal
transcription hits an IEEE754 indeterminate form instead of applying
the intended mathematical convention/limit.

## Independent ground truth

All seven derivations above are cross-checked in this repository against
an independent implementation (`reference.py`) that uses Python's
arbitrary-precision `decimal.Decimal` (50 significant digits) evaluated
directly from the mathematical definitions -- not derived from the same
numpy code paths being tested. This means a bug shared between the naive
and stable numpy formulas (e.g. both computing the wrong quantity) would
not be masked by comparing them only to each other. `numguard
--check-naive-fails` (run in CI on every push) asserts that every naive
fixture documented above still actually fails, and every stable
counterpart still actually passes -- proving these are live regression
tests, not decorative claims.

## References

- Blanchard, P., Higham, D.J., Higham, N.J. (2019), "Accurately computing
  the log-sum-exp and softmax functions" -- rounding-error analysis of
  the log-sum-exp trick used above.
- Welford's algorithm and the standard two-pass mean-centered variance
  formula, as covered in most numerical-analysis textbooks (e.g. Higham,
  "Accuracy and Stability of Numerical Algorithms").
- HuggingFace Transformers' `LlamaRMSNorm` / `T5LayerNorm` source, which
  upcasts the RMSNorm reduction to float32 before squaring -- the exact
  real-world mitigation this repo's `stable_rms_norm` demonstrates.
