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

## Online (chunked/streaming) softmax

**Textbook definition:** same softmax as above, but computed
incrementally over fixed-size chunks of the input instead of all at
once -- the pattern used by FlashAttention-style kernels that tile a
long sequence through limited on-chip memory (Dao et al., "FlashAttention:
Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135).
This is a genuinely distinct algorithmic axis from every other kernel in
this repo: it is not about a single reduction's float range or
cancellation, but about whether *previously accumulated partial
results* get correctly corrected when a later chunk reveals a new
running maximum.

**Naive formula** (`naive_online_softmax`): track a running max `m` and
running sum of exponentials `l` as chunks arrive, but forget to rescale
the exponentials (and running sum) already accumulated from earlier
chunks whenever a later chunk raises `m`. This produces a finite,
non-NaN output that is nonetheless a *different, systematically wrong*
probability distribution whenever the true global max is not already in
the first chunk -- the earlier elements keep the relative weight they
were assigned under the old (too-low) max, permanently overweighting
them relative to the true softmax.

**Stable formula** (`stable_online_softmax`): the standard FlashAttention
incremental correction -- whenever the running max increases from `m_old`
to `m_new`, multiply every already-accumulated exponential and the
running sum by `exp(m_old - m_new)` before folding in the new chunk.
This keeps the running quantities algebraically equivalent to computing
the shifted-softmax numerator/denominator over the whole array seen so
far, at every step, regardless of which chunk contains the true maximum.

The bug is orthogonal to dtype range or magnitude: `max_in_middle_chunk`
below uses only single-digit inputs, no overflow or cancellation is
involved anywhere, and naive is still measurably wrong -- this
demonstrates that "no extreme values" is not sufficient evidence that a
chunked/streaming reduction is correct.

## Masked softmax

**Textbook definition:** given a boolean keep-mask over positions,
`masked_softmax(x)_i = exp(x_i) / sum_j(exp(x_j))` restricted to the
positions where the mask is `True`, and exactly `0` at every masked
position. This is the standard attention-padding/causal-masking pattern:
real implementations set masked logits to `-inf` before a softmax so
`exp(-inf) == 0.0` drops them out of both the numerator (for masked
positions) and the normalizing sum.

**Naive formula** (`naive_masked_softmax`): apply the `-inf` fill and
take a literal softmax, with no other guard. This is correct for every
individual masked position on its own -- `exp(-inf) == 0.0` is exactly
right. The bug is a *global* property of the row, not a per-element
one: if *every* position in a row is masked (a fully-padded row -- a
completely ordinary occurrence any time a batch contains sequences
shorter than the batch's max length), every term underflows to `0.0`,
the normalizing sum is `0.0`, and the division `0.0 / 0.0` is `NaN` for
every element in the row, even though no logit involved was remotely
extreme. This is algorithmically distinct from every overflow/
cancellation bug elsewhere in this module: the failure is about an
*empty support*, not about float range. It is also a real, previously
documented bug class, not a contrived scenario invented for this
tool -- torchtune's `modules/transformer.py` carries an explicit
`skip_mask` guard specifically to work around "a full row of the
attention matrix being masked out ... causes a NaN", and independent
production-debugging notes (e.g. an "NSA `_compress_branch`" incident
writeup) describe the identical "fully-masked rows softmax to NaN"
failure and the same guard as the fix.

A second, independent way this naive formula can fail: if the *only*
surviving (unmasked) logit is itself extreme (e.g. `50000.0`), the
literal `exp()` overflows to `inf` regardless of masking, and `inf/inf`
is `NaN` -- the ordinary softmax-overflow bug from earlier in this
document, simply co-occurring with masking rather than caused by it
(see `single_unmasked_extreme` in `fixtures.py`).

**Stable formula** (`stable_masked_softmax`): guard the all-masked case
explicitly -- when no position survives, there is no valid probability
distribution over an empty support, so the defined answer is "no mass
anywhere" (an all-zero row), matching the convention real fixes (like
torchtune's `skip_mask`) use, rather than propagating `NaN`. Otherwise,
shift by the max of only the *unmasked* logits before exponentiating
(the ordinary log-sum-exp trick, but computed over the surviving subset
so a masked position's `-inf` can never influence the shift), then
apply the `-inf` fill and normalize as usual. This is mathematically
identical to the naive formula everywhere at least one position is
unmasked and no unmasked logit is extreme; it only changes behavior at
the two edge cases above.

## Independent ground truth

All nine derivations above are cross-checked in this repository against
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
- Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. (2022), "FlashAttention:
  Fast and Memory-Efficient Exact Attention with IO-Awareness"
  (arXiv:2205.14135) -- derives the incremental rescale-by-exp(m_old -
  m_new) correction that `stable_online_softmax` implements, and that
  `naive_online_softmax` demonstrates the effect of omitting.
- `torchtune/modules/transformer.py` (PyTorch's `torchtune` library) --
  carries an explicit `skip_mask` guard against a fully-masked attention
  row producing `NaN`, the exact real-world instance of the failure mode
  `masked_softmax`'s `fully_padded_row` fixture demonstrates.
