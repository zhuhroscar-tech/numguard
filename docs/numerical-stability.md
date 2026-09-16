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

## Summation

**Textbook definition:** `sum(x) = x_1 + x_2 + ... + x_n`, computed as
a running total over the sequence.

**Naive formula** (`naive_sum`): the ordinary sequential loop --
`total = 0; for x in xs: total += x`. Each individual addition rounds
correctly to the working dtype (this is not an overflow or
cancellation bug at any single step), but because the *same* running
total absorbs every one of the `n - 1` roundings in sequence, the
worst-case accumulated rounding error grows as `O(n * eps)` in the
number of terms `n`. This is algorithmically distinct from every other
kernel in this module: it is not a single reduction's range or
cancellation, and not a streaming-rescale omission -- it is purely
about the *order* in which additions happen, and the error is
invisible at small `n` (a handful of terms) and only becomes
measurable once `n` reaches the thousands, exactly the scale of a
batch-aggregated loss or a gradient accumulated over many
micro-batches.

**Stable formula** (`stable_sum`): Kahan compensated summation. A
second running variable `c` (the "compensation") tracks the low-order
bits lost on the previous addition; before adding the next term, `c` is
subtracted from it first, and after the addition, the newly-lost bits
are recovered algebraically and stored back into `c` for the next
step. This reduces the worst-case rounding-error bound from `O(n *
eps)` to `O(eps)`, independent of `n` -- the standard mitigation
(Kahan, 1965) for exactly the accumulator-drift failure `naive_sum`
demonstrates. NumPy itself switched `np.sum`'s default implementation
from naive sequential summation to pairwise summation (a different,
also-effective mitigation, reducing the bound to `O(eps * log n)`)
specifically to address this same class of error, independent
corroboration that this is a real, previously-hit issue rather than a
contrived demonstration.

The bug is visible even with every term identical and unremarkable in
magnitude (`many_small_uniform_terms`: 20,000 copies of `1e-4` at
float32 sum to a measurably wrong total), and separately when one
large term "swamps" the accumulator so later small increments are
partially or fully lost (`large_value_swamps_small_terms`) -- two
independent ways the same O(n) accumulation error manifests.

## RoPE (rotary position embedding) angle/cos

**Textbook definition:** RoPE assigns each token position `p` a
rotation angle `theta = p * inv_freq` (per attention-head dimension
pair, `inv_freq` derived from `base**(-2i/d)`), then rotates the query
and key vectors by `cos(theta)`/`sin(theta)` -- the mechanism that lets
attention scores depend only on relative position (Su et al., 2021,
"RoFormer"). `rope_cos` audits just the `cos(position * freq)` step,
the piece of the computation actually vulnerable to precision loss.

**Naive formula** (`naive_rope_cos`): both the position id and
`inv_freq` are cast to the model's storage dtype *before* the
multiply, and the multiply and `cos()` are also computed in that
dtype. This is not a range or cancellation bug -- it is a *resolution*
bug: every floating-point dtype can only represent integers exactly up
to a limit set by its mantissa width (256 for bfloat16's 7-bit
mantissa, 2048 for float16's 10-bit mantissa, 2**24 for float32's
23-bit mantissa). Beyond that limit, distinct integer positions
literally round to the *same* stored value before the angle is even
computed, so two different tokens receive an identical rotation angle
("position aliasing") -- RoPE's entire ability to distinguish those
positions is lost, silently, with no NaN/inf to signal it.

**Stable formula** (`stable_rope_cos`): the position*freq product and
its `cos()` are computed in at least float32 regardless of the model's
storage dtype, and only the final trig output is cast down. This is
the exact fix HuggingFace's `transformers` library applies in every
RoPE implementation in that codebase (PR #29285: "Force float32 ...
since bfloat16 loses precision on long contexts") -- not a new
algorithm, just precision discipline about *where* the cast happens.

This repo's fixtures reproduce the aliasing at float16 (position ids
16384-16392, beyond float16's 2048 exact-integer limit collapse onto
one angle) rather than bfloat16, since bfloat16 requires a torch
dependency this pure-numpy tool deliberately avoids; float16 exhibits
the identical resolution-limit mechanism the Baichuan Inc. report and
HuggingFace's fix both document for bfloat16, just at a different
(larger) position-id threshold.

## int8 element-wise add

**Textbook definition:** affine ("zero-point") int8 quantization
represents a real value as `real = (code - zero_point) * scale` --
the definition shared by TFLite's own 8-bit quantization spec, ONNX's
`QuantizeLinear`/`DequantizeLinear` operators, and OpenVINO's Low
Precision Transformations (LPT). Adding two int8 tensors element-wise
(the ordinary shape of a residual/skip connection) is only correct if
each operand is dequantized with *its own* `(scale, zero_point)`
before the sum, and the float result is then requantized -- with
saturation -- into the output's own quant parameters. Unlike every
other kernel in this repo, `int8_add`'s bug is not about float range
or cancellation at all: it is about *quantization-parameter
bookkeeping*, a categorically distinct failure family.

This is a documented, recurring real-world bug class, not a
constructed one:
- OpenVINO PR #7305, "Fixed scale factors propagation for Eltwise with
  very different inputs ranges" -- the exact mismatched-scale scenario
  `mismatched_scale_residual_add` reproduces.
- OpenVINO PR #1135, "[LPT] Eltwise Prod transformation fix", filed
  against "Quantized model with per-channel quantization for
  element-wise operations has zero accuracy".
- OpenVINO issue #34673 (open, unfixed as of this writing): INT8
  inference produces ~10% (random-guess) accuracy on Apple M4 Max ARM;
  independent reproduction in the issue thread isolates the failure to
  exactly the residual `Add` nodes, with the reporter's own analysis
  pointing at "an int32 accumulator overflow, an incorrect broadcast
  of the scale/zero-point vectors, or a broken ... instruction dispatch
  ... for the addition of tensors with different scales" -- the same
  requantization-bookkeeping failure family this kernel targets,
  though not a bit-for-bit reproduction of that specific ARM kernel bug.
- ONNX Runtime issue #25823: "Incorrect calculation of zero point for
  uint8 symmetric quantized case" (zero_point computed as 127 instead
  of 128) -- a distinct but adjacent zero-point-bookkeeping defect in
  the same problem family.

**Naive formula** (`naive_int8_add`): dequantizes *both* operands using
operand A's `(scale, zero_point)` -- the bug pattern a fusion or
kernel-selection path hits when it assumes (or caches) a single shared
quantization descriptor for an Eltwise-Add instead of tracking each
input's own parameters -- and then requantizes the sum into the
output's int8 space *without* a saturating clamp, so an overflowing
sum silently wraps modulo 256 (two's-complement wraparound) instead of
saturating.

**Stable formula** (`stable_int8_add`): dequantizes each operand with
its own `(scale, zero_point)`, sums in float, and requantizes into the
output's int8 space *with* a saturating clamp to `[-128, 127]` -- the
standard, spec-correct affine-quantization add matching TFLite/ONNX/
OpenVINO's documented per-tensor Eltwise-Add semantics.

Unlike this repo's other kernels, `int8_add`'s fixtures are integer
quantization codes rather than a dtype-swept float array, so it is
scored at float64 only (there is no float16/float32/float64 variant of
"which dtype is an int8 code stored in").

## HyperLogLog register term

**Textbook definition:** HyperLogLog (Flajolet et al., 2007) estimates
set cardinality from `m` registers, each storing the length of the
longest leading-zero run seen in a hashed element's bits (its "rank").
The cardinality estimate is a harmonic mean built from each register's
contribution `2**-rank` summed across all `m` registers. This kernel
isolates exactly that one per-register term -- the smallest unit where
a real, documented implementation bug lives.

Unlike every other kernel in this repo (float range/cancellation bugs,
or `int8_add`'s quantization-bookkeeping bug), this bug is a **fixed-
width integer left-shift** silently wrapping instead of computing the
intended value:

- Apache Flink, FLINK-39399 ("Integer overflow in
  HyperLogLogPlusPlus.query() causes APPROX_COUNT_DISTINCT undercount
  at high cardinality"): `HyperLogLogPlusPlus.query()` computes
  `1 << mIdx` where the literal `1` is an ordinary 32-bit Java `int`.
  Per the Java Language Specification section 15.19, shifting a 32-bit
  int by a distance >= 32 uses only the distance's low 5 bits (i.e.
  `distance % 32`), so once a register's stored rank reaches 32 or
  more, the shift silently computes a small, wrong power of two
  instead of overflowing loudly or computing the intended (much
  larger) value. The issue's own reported repro: a register holding
  35 produces an estimate of ~95,000 with the bug; the fix (widening
  the shift to a 64-bit `long`, i.e. changing the literal `1` to `1L`)
  produces the correct ~4e14.
- This register-value range is not exotic or contrived: real HyperLogLog
  implementations deliberately hash to 64 bits specifically so
  cardinalities beyond 2^32 can be represented (see Heule, Nunkesser,
  Hall, "HyperLogLog in Practice", 2013, and the ClickHouse `uniqHLL12`
  large-cardinality bug report below) -- and once hashing uses 64 bits,
  leading-zero-run lengths of 32+ occur routinely at billion-plus real
  cardinalities, exactly where a cardinality estimator is most needed
  and least excusable to silently corrupt.
- ClickHouse PR #1844 ("Fix uniqHLL12 and uniqCombined for cardinalities
  100M+"), filed by Cloudflare: reports `uniqHLL12` returning "rubbish
  -- either 0 or a very inaccurate result" above ~200M elements, with
  the fix's own analysis pointing at exactly this class of large-range/
  large-register-value correction defect.
- Apache Druid issue #19649 ("HyperLogLogCollector estimates 0
  cardinality when single element overflows into sparse mode"): a
  related but distinct register-overflow defect in Druid's sparse
  representation, corroborating that fixed-width register/overflow
  bookkeeping is a recurring, cross-implementation HyperLogLog failure
  family, not a one-off bug in a single codebase.

**Naive formula** (`naive_hll_register_term`): computes a register's
harmonic-sum term via `int(np.int32(1)) << (rank % 32)` -- reproducing
the JLS 32-bit-int shift-distance masking that silently corrupts the
result for `rank >= 32` instead of computing the true (much larger)
power of two.

**Stable formula** (`stable_hll_register_term`): computes `2.0 **
(-rank)` directly with no fixed-width shift at all -- matching the
real Flink fix's approach of widening to a shift width that can never
be exceeded by any register value a 64-bit hash can actually produce.

Like `int8_add`, `hll_register`'s input is a single integer register
value rather than a dtype-swept float array, so it is scored at
float64 only.

## Sigmoid focal loss gradient

**Textbook definition:** Lin et al.'s Focal Loss (arXiv:1708.02002)
reshapes binary cross-entropy as `FL(p_t) = -alpha_t * (1 - p_t)**gamma
* log(p_t)`, where `p_t` is the model's predicted probability of the
TRUE class (`p` if the label is 1, `1-p` if the label is 0) and `gamma`
is the "focusing" exponent that down-weights easy, already-correctly-
classified examples. This kernel audits `d(FL)/dx` (`x` the pre-sigmoid
logit) -- the quantity that actually drives training via
backpropagation, not the loss value itself.

**Naive formula** (`naive_focal_loss_grad`): the literal, un-simplified
product/chain rule applied straight to the formula above, including an
explicit `(1 - p_t)**(gamma - 1)` factor. This is mathematically
correct wherever it is well-defined, but when `gamma <= 1` (which
includes `gamma=0`, Focal Loss's own documented reduction to plain
alpha-weighted binary cross-entropy) and the prediction has saturated
in the correct direction (an ordinary, even desirable, training
outcome -- `p_t` very close to `1`), `1 - p_t` underflows to exactly
`0.0` in floating point, and `0.0 ** (gamma - 1)` with `gamma <= 1` is
`0.0 ** (a non-positive exponent)`, i.e. a literal division by zero
inside the power -- `inf` (or, for `gamma` strictly between 0 and 1,
still blows up via the same negative-exponent mechanism). That `inf`
is then multiplied by `dp_t/dx`, which has ALSO underflowed to `0.0` at
the same saturation point -- an IEEE754 `0 * inf` indeterminate form
that evaluates to `NaN` and poisons the entire gradient, even though
the true gradient at that point is an ordinary, tiny, finite number.

This is documented and independently confirmed in at least three
distinct real codebases:
- `facebookresearch/sam3#575` ("Reduced Triton sigmoid focal loss
  returns NaN gradients for gamma=0 when logits saturate", filed and
  fixed via PR#576) -- root-caused to exactly this
  `(1 - p_t) ** (gamma - 1)` term evaluating `0 ** -1`, with the
  reporter's own minimal repro (`x=18.0, y=1.0, gamma=0.0` ->
  `x.grad = nan`) matching this kernel's `saturated_correct_gamma_zero_float32`
  fixture almost exactly.
- `kornia/kornia#918` / PR#924 ("NaN gradients on backward pass with
  focal loss"), fixed by adding an epsilon inside the modulating
  factor.
- van Leeuwen et al., "A Note on the Stability of the Focal Loss" (TMLR
  2025, OpenReview `eCYActnGbu`), which independently derives and
  empirically demonstrates the identical failure across the broader
  `0 <= gamma < 1` range (not just exactly `gamma=0`) in real CNN/ViT/
  U-Net training runs on CIFAR-10 and MNIST, and proposes the same
  "add an epsilon inside the modulating factor" mitigation used in the
  production fixes above.

**Stable formula** (`stable_focal_loss_grad`): an algebraic rewrite
that substitutes `dp_t/dx = p_t*(1-p_t)` (target=1) or `-p_t*(1-p_t)`
(target=0) into the naive expression and cancels the shared `(1-p_t)`
(target=1) or `p_t` (target=0) factor by hand before ever evaluating a
power -- on paper, algebraically identical to the naive formula
everywhere both are well-defined, but the result contains only
NON-negative powers of probabilities (`(1-p_t)**gamma` and `p_t**gamma`
with `gamma >= 0`), which safely underflow to `0.0` (a correct, finite
answer) instead of blowing up to `inf`/`NaN`. This mirrors the intent
of the real-world fixes (sam3 PR#576 routes `gamma=0` through a
numerically stable BCE-equivalent path; kornia PR#924 adds an epsilon)
without needing an ad hoc epsilon constant: the algebraic rewrite
removes the negative-power term entirely, for any `gamma >= 0`, not
just the `gamma=0` special case.

A control fixture (`control_saturated_gamma_two_float32`) uses the
same saturated logit at the common `gamma=2.0` default (Lin et al.'s
recommendation, `gamma >= 1`): here `(1-p_t)**(gamma-1) = (1-p_t)**1`
is a non-negative power, so the naive formula is actually fine --
confirming the bug is specific to `gamma < 1`, not to saturation alone.

## Pearson correlation coefficient

**Textbook definition:** Pearson's r measures the linear correlation
between two variables, `r = cov(x, y) / (std(x) * std(y))`, ranging
from -1 (perfect negative) to +1 (perfect positive), undefined (0/0)
when either variable has zero variance.

**Naive formula** (`naive_pearson_correlation`): the "sum of products"
one-pass formula taught in many textbooks and used by countless
from-scratch implementations,
`r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2)-sum(x)^2) *
(n*sum(y^2)-sum(y)^2))`. This computes the same catastrophic-
cancellation shape as this repo's `variance` kernel's `E[x^2]-E[x]^2`,
independently on two variables at once: `n*sum(xy)` and `sum(x)*sum(y)`
both grow with the absolute scale of the data (not just its spread),
so an ordinary offset (timestamps, prices, large IDs) makes their
difference lose precision to cancellation, or the intermediate squares
overflow outright, long before the true correlation's magnitude would
suggest any instability.

This is a real, actively-reported bug class, not a constructed one.
`scipy.stats.pearsonr` needed three separate bug reports before a full
rewrite fixed it:
- `scipy/scipy#8980` ("pearsonr overflows with high values of x and
  y") -- squaring large offset values before combining overflowed to
  `inf`/`NaN` even for perfectly ordinary, well-correlated data.
- `scipy/scipy#9353` ("pearsonr returns r=1 if r_num/r_den = inf") --
  the opposite failure, tiny values underflowing the denominator to
  exactly `0.0`, producing a spurious `r=1` via `inf` incorrectly
  clamped by a "just in case" `min(max(r,-1),1)` guard.
- `scipy/scipy#3728` -- inconsistent `NaN` vs `+-1` for exactly
  constant input, depending on incidental floating-point residue.
- `scipy/scipy#9562` (the fix) rewrote the whole calculation to
  normalize each mean-centered vector by its own norm *before* the dot
  product, specifically to avoid ever materializing a sum-of-squares
  term whose magnitude depends on the input's absolute scale.
- `numpy/numpy#32446` (2026, open) and `pandas-dev/pandas#67023` /
  `#37448` / `#45640`: even the *mean-centered* form (already one step
  more stable than the naive one-pass formula) can still return
  `+1`, `-1`, or `NaN` inconsistently for exactly-constant columns,
  purely from float mean-reduction residue -- confirming that
  guarding the zero-variance case explicitly (not just centering
  first) is necessary for a correct implementation.
- `st-hakky.hatenablog.com`'s Japanese "Pearson correlation from
  scratch" tutorial (2018) uses exactly the mean-centered-but-
  undivided-by-norm "scratch" form as its first, most basic example,
  demonstrating this formula shape's wide real-world prevalence in
  from-scratch/textbook/interview-style code outside scipy itself.

**Stable formula** (`stable_pearson_correlation`): mean-centers both
vectors, then divides each by its own norm *before* the dot product
(the same fix as `scipy` PR#9562, and the same "normalize/rescale
before combining" principle already used by this repo's
`stable_rms_norm` and `stable_online_softmax` kernels) -- no
intermediate term ever exceeds `O(1)` regardless of the input's
absolute scale. An exactly-zero-variance input (either vector
constant) returns `NaN` explicitly, matching `scipy`'s current
`PearsonRConstantInputWarning` convention and the "correct" answer
identified by `numpy#32446`/`pandas#67023`, rather than an arbitrary
clipped `+-1`.

## Weighted reservoir-sampling comparison key

**Textbook definition:** Efraimidis & Spirakis's A-Res algorithm
(2006) does single-pass weighted random sampling without replacement
from a stream: assign each item a key `u**(1/weight)` (`u` a fresh
uniform(0,1) draw, `weight` the item's importance/score), keep the `k`
items with the largest keys seen so far. This is the same algorithm
grpc's proposal A113 cites for weighted pick-first shuffling and the
basis for R's `wrswoR` package and Java's LinkedIn/DataSketches-style
weighted reservoirs.

**Naive formula** (`naive_weighted_sampling_key`): the paper's own
literal formula, `log(u**(1/weight))`, computed by taking the power
first and only then a log to put every item's key on one common,
directly-comparable scale. R's `wrswoR` ships this exact verbatim form
as `sample_int_expjs`, its own documentation explicitly noting it does
so "at the cost of numerical stability." For any weight small enough
that `1/weight` is large -- a realistic long-tail item whose
score/importance is orders of magnitude below a typical item, exactly
the regime weighted sampling is meant to still handle fairly --
`u**(1/weight)` underflows to exactly `0.0` in IEEE754 long before the
true value is actually zero. `log(0.0)` is then `-inf`, discarding the
real (still order-relevant) magnitude entirely: every sufficiently
low-weight item's key collapses onto the identical `-inf`, silently
destroying the proportional-selection guarantee among them (an item
that should still occasionally out-rank an even-lower-weight one no
longer can, because both keys are now indistinguishable).

**Stable formula** (`stable_weighted_sampling_key`): the wrswoR
package's own `sample_int_expj` fix -- compute `log(u)/weight`
directly, never materializing `u**(1/weight)` as an intermediate at
all. This is the exact same log-space substitution already used by
this repo's `logsumexp`/`kl_divergence` kernels: replace an operation
prone to overflow/underflow (a power with a huge exponent) with a plain
division in log space, so the result stays a finite, order-preserving
real number all the way down to the smallest weights actually used in
practice. `wrswoR`'s own documentation states this algebraic identity
directly: "It can be shown that the order statistic of `U^(1/w_i)` has
the same distribution as random sampling without replacement... To
increase numerical stability, `log(U)/w_i` is computed instead; the
log transform does not change the order statistic."

## Geometric mean

**Textbook definition:** the geometric mean of `n` positive values is
`(prod(x_i))**(1/n)` -- the appropriate average for ratios, growth
rates, and normalized scores (CAGR, index numbers, the UN Human
Development Index since 2010), where the arithmetic mean systematically
over-weights outliers on a multiplicative scale.

**Naive formula** (`naive_geometric_mean`): the literal textbook
formula -- multiply every value together first (`prod(x)`), then take
the `n`-th root of that single running product. `scipy.stats.gmean`'s
own issue tracker documents exactly this failure under its original
name (`scipy/scipy#1053` / Trac#526, "gmean cannot handle large
numbers"): for values whose magnitude is above 1, the raw running
product grows past a dtype's max representable value long before the
much smaller (order-1-scale) `n`-th root is ever taken, overflowing to
`+inf` -- silently reporting an infinite average of entirely ordinary,
finite inputs. Symmetrically, for values below 1, the running product
underflows to exactly `0.0` long before the true root is computed,
silently reporting zero instead of a small but definitely nonzero
geometric mean. Neither failure depends on any single input being
unusually large or small -- accumulating enough ordinary-magnitude
terms is sufficient, as this repo's `very_large_offset`-style fixtures
demonstrate at all three dtypes (`float16`, `float32`, and even
`float64` given enough terms).

**Stable formula** (`stable_geometric_mean`): the standard log-space
fix, the same one `scipy.stats.gmean`'s bug report above settled on and
that every "geometric mean from scratch" reference implementation
teaches: take the log of each value first (collapsing every term to an
`O(1)`-scale exponent regardless of the input's raw magnitude), average
those logs, and exponentiate only once at the very end --
`exp(mean(log(x)))`. No intermediate ever approaches a dtype's overflow
or underflow boundary the way the raw running product does, because
the log transform turns a product-then-root computation (whose
intermediate can grow or shrink combinatorially with `n`) into a
sum-then-divide-then-exponentiate one (whose intermediate is always
just the *average* magnitude of the inputs' logs, an `O(1)`-scale
quantity independent of `n`). This is the identical log-space
substitution this repo's `logsumexp` and `weighted_sampling_key`
kernels already use for the same underlying reason: move an
overflow/underflow-prone power or product operation into log space,
where it becomes an ordinary, boundedly-scaled sum.

## P^2 (Piecewise-Parabolic) streaming quantile estimator

**Textbook definition:** the P^2 algorithm (Jain & Chlamtac, *The P^2
algorithm for dynamic calculation of quantiles and histograms without
storing observations*, CACM 28(10), 1985) estimates a target quantile
(e.g. the streaming median, or a tail latency percentile) from a data
stream using only five running "marker" heights/positions -- O(1)
memory regardless of stream length, the classic choice for telemetry
and monitoring systems that cannot afford to store every observation.
After the first five values initialize the markers, each new
observation updates one marker's *desired* position `n'_i` and, when it
has drifted far enough from the marker's *actual* integer position
`n_i`, triggers a piecewise-parabolic (or linear, as a fallback)
interpolation step that moves the marker.

**Naive formula** (`naive_p2_quantile`): the original paper's own
suggested optimization -- maintain each marker's desired position via a
per-observation accumulated increment, `ns[i] += dns[i]` where `dns[i]`
is a fixed per-marker constant computed once at initialization ("to
reduce CPU overhead" versus recomputing from scratch). This is a real,
author-documented bug, not a constructed one: GitHub issue
`AndreyAkinshin/perfolizer#8` ("P2QuantileEstimator rounding issue"),
filed against Andrey Akinshin's `perfolizer` benchmarking-statistics
library, reports and the library's own maintainer confirms that this
accumulation drifts under ordinary floating-point rounding -- a
quantity that should land exactly on an integer marker boundary (e.g.
`6.0` after enough observations at `p=0.6`) instead lands a few ULPs
off (`5.999999994`), so the `d >= 1` / `d <= -1` marker-adjustment
trigger silently fails to fire when it should have. Because the
marker's *position* bookkeeping (not just one output value) is now
wrong, every subsequent observation's parabolic/linear interpolation
computes from a stale marker position -- the error compounds for the
rest of the stream rather than resetting each step. This repository's
own `uniform_stream_p10_drift` fixture reproduces a 4-7% relative error
across float16/float32/float64 from exactly this mechanism, and
`sine_stream_p35_drift` reproduces a deterministic, hand-reviewable
~2.9% relative error at float64 with a fixed (no-RNG) input.

**Stable formula** (`stable_p2_quantile`): the author's own published
fix -- recompute each marker's desired position fresh from the running
observation count every step (`ns[i] = count * p_i`, no `dns` array
ever materialized), documented by the same `perfolizer` maintainer to
be simultaneously *more* accurate (no accumulated rounding drift) and
*faster* (one fewer array to maintain). This is a different mechanism
from every other kernel in this repository: not an overflow/underflow
in a single expression (`geometric_mean`, `weighted_sampling_key`) and
not catastrophic cancellation in a sum (`variance`,
`pearson_correlation`, `sum`), but drift in a *stateful streaming
algorithm's internal bookkeeping* that silently skips a state-transition
decision and then compounds for the remainder of the stream.

## Repetition penalty

**Textbook definition:** to discourage a language model from repeating
itself, a repetition penalty multiplicatively rescales the logit of
each previously-generated token before softmax: if the logit is
positive, divide it by a penalty strength `theta > 1`; if it is
non-positive, multiply it by `theta`. This exact form -- branching on
the *sign of the raw logit* -- is the pattern shipped across the LLM
inference ecosystem (HuggingFace `transformers`, vLLM, llama.cpp, and,
per the paper below, "a dozen further engines").

**Naive formula** (`naive_repetition_penalty`): apply the sign-branch
penalty directly to the raw logits, then softmax. This is a real,
documented bug, not a constructed one: Hollows, P. (2026), "Gauge
dependence and structured-output corruption in sign-branched
repetition penalties: measurements across models, inference stacks,
and alternative repetition controls" (arXiv:2607.09791) proves that
this pattern is **gauge-dependent** -- it depends on an arbitrary
additive constant added to every logit, even though softmax itself is
mathematically shift-invariant (`softmax(x) == softmax(x + c)` for any
constant `c`, since the added `exp(c)` factor cancels between numerator
and denominator). Two logit vectors representing the *identical*
pre-penalty distribution (one is the other plus a constant shift) can
select different next tokens after the penalty is applied, purely
because of where the unpenalized logits happen to sit relative to
zero. The paper reports this is not a corner case: gpt2's own
per-position logits are bimodal (deciles spanning roughly -230 to
+150), so ordinary generation already produces the sign-straddling
condition the bug depends on, and its own StarCoder2-7B HumanEval
experiment demonstrates real greedy-decode token flips from exactly
this mechanism. This repository's own `gauge_shifted_plus10` and
`gauge_shifted_minus10` fixtures reproduce the identical shape: three
logit vectors representing one underlying distribution, shifted by
`0`, `+10`, and `-10`, that `naive_repetition_penalty` maps to three
*different* output distributions (and, in the `+10`/`-10` cases, a
different argmax than the unshifted baseline).

**Stable formula** (`stable_repetition_penalty`): apply the identical
sign-branch/divide-multiply penalty to **log-probabilities**
(`logit - logsumexp(logits)`) instead of raw logits. Subtracting the
log-partition-function makes the quantity being branched on
shift-invariant by construction -- an additive shift to every input
logit shifts `logsumexp` by the same constant, which cancels exactly
before the branch ever runs -- so the resulting penalized distribution
depends only on the actual (gauge-invariant) probabilities, never on
the arbitrary additive gauge of the unpenalized logits. This mirrors a
fix direction the paper itself notes already exists inside the same
library that ships the bug: "HuggingFace's beam search has applied its
entire logits-processor chain, repetition penalty included, to
log-probabilities since at least transformers v4.0.0", while greedy
and sampled decoding apply the raw (gauge-dependent) form -- the same
codebase already contains both the buggy and the fixed behavior,
selected only by which decoding strategy happens to be in use.

## Speculative-decoding rejection sampling

**Textbook definition:** speculative decoding (Leviathan et al. 2023,
"Fast Inference from Transformers via Speculative Decoding"; Chen et
al. 2023) accelerates autoregressive LLM decoding by having a small
draft model propose a token, then verifying it against the (larger)
target model's distribution with a rejection-sampling rule: sample the
draft token from distribution `p` (the draft model's probability for
that token), accept it with probability `a = min(1, q/p)` where `q` is
the target model's probability for the same token; on rejection,
resample from the normalized residual `max(0, q - p)`. The theorem's
entire guarantee -- the property that makes this "lossless" and safe
to deploy in production inference stacks (vLLM, TensorRT-LLM,
llama.cpp) -- is that the **output distribution equals the target
distribution `q` exactly**, provided the draft token is sampled from
**the same `p`** used in the accept/reject math.

**Naive formula** (`naive_speculative_reject`): sample the draft token
from one materialization of the draft probability array (`r`), but
compute the accept/reject math against an **independently recomputed**
draft probability array (`p`) from the same logits, at a different
precision. This models a real, documented defect class: the
speculative-decoding rejection sampler in deepseek-ai/DeepSpec PR#30
was fixed for exactly this shape -- "Draft samples were drawn from
native-dtype probabilities while rejection used float32 probabilities
... Cast logits to float32 before temperature scaling and softmax in
sample_tokens. Removes a speculative-decoding distribution mismatch."
The same class of hazard is discussed at length in vLLM's own
rejection-sampler evolution (PR#48641, reverted, then reintroduced with
care in PR#53630): removing or reintroducing an extra materialization
of the logits/probabilities changes which array downstream consumers
(sampling, verification, and reporting) actually read, and any
resulting rounding difference between the *sampled-from* and
*verified-against* arrays reopens this exact correctness gap. This
repository's kernel emulates the concrete case of a draft model whose
weights (and therefore its natural sampling distribution) live in
bfloat16 -- the standard deployed storage dtype -- while the
accept/reject math recomputes the draft probability at a different
precision from the same logits: `r != p` even though both are
individually finite, valid (non-negative, sums-to-1) probability
vectors. There is no NaN, no inf, no crash anywhere; the entire failure
is that the rejection-sampling identity's guarantee silently stops
holding, and the sampled output token distribution measurably diverges
from the target model's true distribution `q`.

**Stable formula** (`stable_speculative_reject`): materialize the draft
probability array **once**, and reuse the identical array for both the
sampling step and the accept/reject math, so `r` is `p` by
construction rather than merely by numerical accident (the DeepSpec
PR#30 fix pattern: "That makes the sampled draft distribution match
the `p` used by rejection sampling."). When `r == p` exactly, the
output-distribution identity `out = r*a + (escaped mass)*resid`
collapses to exactly `q` (up to ordinary floating-point rounding of the
one shared array), regardless of which precision or code path produced
it.

## AdamW decoupled weight decay storage stall

**Textbook definition:** AdamW's decoupled weight decay (Loshchilov &
Hutter, 2019, "Decoupled Weight Decay Regularization") multiplies the
parameter by `(1 - lr * weight_decay)` at every optimizer step,
independent of the gradient-based Adam update. Over `N` steps this is
mathematically equivalent to the closed-form geometric decay
`w0 * (1 - lr*weight_decay)**N`.

**Naive formula** (`naive_weight_decay`): apply the per-step multiply
and **write the result straight back into the parameter's storage
dtype every step** -- exactly what a real training loop does when the
parameter tensor itself (not a separate float32 master copy) is bf16
or float16. This is the real, currently-open production bug in
Nerogar/OneTrainer#996 ("No weight decay with Adam, bf16 and
stochastic rounding", filed 2025-09-13): "In Adam and AdamW, weight
decay is applied in bf16, which is ineffective in bf16 ... With weight
decay usually set at 0.01, multiplied by the learning rate, this is
always smaller than what bf16 can represent and no weight decay is
applied." The same mechanism is formalized generally as "state-update
stalling" in arXiv:2603.16731 (an update smaller than half a
unit-in-the-last-place rounds back to the same stored value under
round-to-nearest) and arXiv:2607.09800 ("The Silent Freeze" -- a
gradient-descent update below half a ULP freezes a coordinate
deterministically, predictable a priori from the mantissa length
alone). Once the per-step decay fraction `lr * weight_decay` is
smaller than the storage dtype's ULP at the parameter's magnitude,
**every single step's update rounds away to the identical stored
value** -- not a one-off rounding error but a total, silent stall:
weight decay is configured, the optimizer runs without error or
warning, and it does precisely nothing for the entire training run.

**Stable formula** (`stable_weight_decay`): keep the running parameter
value in float64 precision for the entire decay loop -- exactly the
fix already shipped for bf16 training master weights by
imoneoi/bf16_fused_adam ("A mixed-precision optimizer to solve the
stale weights problem of bfloat16 training") and by NVIDIA/apex's
`FusedAdam(master_weights=True)` generally, and the same pattern
OneTrainer's own `adafactor_extensions.py` already uses for Adafactor
specifically (only AdamW was left applying decay directly in the
storage dtype, per the linked issue) -- and round to the requested
storage dtype only **once**, at the point the value is actually read
back out for use. Every sub-ULP-at-storage-precision update still
shrinks the float64 master correctly, so decay accumulates exactly as
configured; only the final read-out loses precision, not the
thousands of intermediate steps.

## Gradient-accumulation loss bias

**Textbook definition:** gradient accumulation is intended to be
mathematically equivalent to training on one large batch: split a
target effective batch into `k` smaller micro-batches, run each
forward/backward pass separately, and combine the `k` micro-batch
losses into the same large-batch loss the single big batch would have
produced. For token-level losses (e.g. causal-LM cross-entropy), the
true large-batch mean is `sum(all per-token losses) / (total number of
non-padding tokens across all k micro-batches)`.

**Naive formula** (`naive_gradient_accumulation_bias`): compute each
micro-batch's own mean per-token loss `g_i / n_i` (its summed loss
divided by its own token count), then average those `k` per-microbatch
means: `(1/k) * sum(g_i / n_i)`. This is exactly the
`loss = loss / self.args.gradient_accumulation_steps` shape the
pre-fix HuggingFace `Trainer` used (huggingface.co/blog/gradient_accumulation,
originally reported by Benjamin Marie and independently rediscovered by
Unsloth in 2024, and the identical failure documented in
Lightning-AI/pytorch-lightning#20350: "Gradient accumulation
calculation may be incorrect"). Whenever the `k` micro-batches have
*different* non-padding token counts `n_i` -- the normal case for
variable-length sequences padded per micro-batch rather than globally
-- this mean-of-means double-counts short micro-batches and
under-counts long ones relative to their true share of the accumulated
batch: `naive - true = sum_i g_i * (1/(k*n_i) - 1/N)` where `N = sum(n_i)`,
which is non-zero exactly when the `n_i` are unequal and vanishes only
in the degenerate case where every micro-batch has the same token
count.

**Stable formula** (`stable_gradient_accumulation_bias`): the fix
HuggingFace actually shipped
(huggingface.co/blog/gradient_accumulation, PR#34198) and the
equivalent correction independently arrived at for PyTorch-Lightning
(discuss.huggingface.co/t/bug-in-gradient-accumulation-training-step...):
sum every micro-batch's *summed* per-token loss across the whole
accumulation window first, then divide **once** by the *total*
non-padding token count across all `k` micro-batches:
`sum(g_i) / sum(n_i)`. This is the true large-batch-equivalent mean
regardless of how unevenly tokens are distributed across the
accumulated steps.

All twenty-two derivations above are cross-checked in this repository
against an independent implementation (`reference.py`) that uses
Python's arbitrary-precision `decimal.Decimal` (50 significant digits)
evaluated directly from the mathematical definitions -- not derived
from the same numpy code paths being tested (this includes
`rope_cos`'s reference `cos()`, computed via a from-scratch Decimal
Taylor series, deliberately not `math.cos`, so that a bug shared with a
float64 trig implementation could not hide behind comparing against
itself; `focal_loss_grad`'s reference, computed via a symmetric
central-difference numerical derivative of the focal loss formula
itself at 50-digit precision, deliberately not via either kernel's
analytic chain-rule derivation, so a shared algebra mistake in the
naive/stable gradient formulas could not hide behind comparing them
only to each other; `weight_decay`'s reference, computed via
Decimal exponentiation of the exact closed-form geometric-decay
identity, deliberately not by looping the kernel's own step-by-step
multiply; and `gradient_accumulation_bias`'s reference, computed
directly from the `sum(g_i)/sum(n_i)` definition in Decimal, which is
mathematically identical to both the naive and stable formulas'
intended target -- they differ only in evaluation order, not in which
quantity is being computed, so this single reference is valid ground
truth for scoring both). This means a bug shared between the naive and
stable numpy formulas (e.g. both computing the wrong quantity) would
not be masked by comparing them only to each other. `numguard
--check-naive-fails` (run in CI on every push) asserts that every naive
fixture documented above still actually fails, and every stable
counterpart still actually passes -- proving these are live regression
tests, not decorative claims.

## LongRoPE short/long scaling-factor selection

**Textbook/reference definition:** Phi-3 and Phi-4's `"longrope"` RoPE
variant (HF transformers `modeling_rope_utils._compute_longrope_
parameters`) keeps two published per-dimension scaling factor tables,
`short_factor` and `long_factor` (real values, from
`microsoft/Phi-3-mini-128k-instruct`'s own `config.json`:
`original_max_position_embeddings=4096`, `head_dim=96`, 48 per-
dimension factors each). The reference implementation selects
`long_factor` when the **actual sequence length being encoded**
exceeds `original_max_position_embeddings`, and `short_factor`
otherwise -- the model was fine-tuned expecting exactly this rule, and
mixing up the two factor sets changes the effective RoPE inverse
frequency (`base_inv_freq / factor`) by up to ~30x on the
lowest-frequency dimensions of this real model's own published table.

**Naive formula** (`naive_longrope_factor_select`): select the factor
set from the **allocated context size** (`n_ctx_alloc`, i.e. how large
a KV-cache/context window the serving session was configured with)
instead of the actual sequence length -- the real, currently-open bug
documented in `ggml-org/llama.cpp#24823` ("Phi-3 / Phi-4 LongRoPE:
short sequences silently use long-context RoPE factors when the
allocated context exceeds original_max_position_embeddings"). The
issue's own repro is exactly this shape: a llama.cpp server loaded with
`-c 8192` (context > 4096) silently encodes every request with
`long_factor`, even a short ~640-token document that HF transformers
(and llama.cpp itself, when loaded with `-c 4096`) correctly encodes
with `short_factor`. The divergence is silent -- no error, no warning,
and (as the issue documents) model-dependent in severity, which is
exactly what makes it easy to misattribute to the model or fine-tune
rather than to the serving path.

**Stable formula** (`stable_longrope_factor_select`): select the
factor set from the actual sequence length being encoded, matching the
correct HF transformers reference and the mitigation the issue itself
proposes -- a short document stays on `short_factor` no matter how
large a context window the session happens to have allocated for it.

All twenty-three derivations above are cross-checked in this
repository against an independent implementation (`reference.py`) ...
(see prior paragraph); `longrope_factor_select`'s reference
(`gold_longrope_factor_select`) is built the same way as `rope_cos`'s
-- the from-scratch Decimal `_decimal_cos` Taylor series, not
`math.cos` -- with the added independent factor-selection rule applied
directly from the actual-sequence-length definition, so a bug shared
between the naive and stable kernels in *which* factor gets selected
cannot hide behind comparing them only to each other. `numguard
--check-naive-fails` (run in CI on every push) asserts that every naive
fixture documented above still actually fails, and every stable
counterpart still actually passes -- proving these are live regression
tests, not decorative claims.

## Squared Euclidean distance via dot-product expansion

**Textbook/reference definition:** the squared Euclidean distance
between two vectors, `dist^2(x, y) = sum((x_i - y_i)^2)`, computed
directly from the elementwise-difference definition.

**Naive formula** (`naive_squared_euclidean_distance`): the algebraic
expansion `dist^2(x, y) = dot(x,x) - 2*dot(x,y) + dot(y,y)`. This is
not a strawman -- it is the exact formula documented in
`sklearn.metrics.pairwise.euclidean_distances`'s own docstring, which
states plainly: "this is not the most precise way of doing this
computation, because this equation potentially suffers from
catastrophic cancellation." The expansion remains common in production
ANN/vector-search code because `dot(x,x)` and `dot(y,y)` can each be
precomputed once per vector and reused across every query against it --
exactly the efficiency tradeoff scikit-learn's own docs describe. When
`x` and `y` are nearly identical (near-duplicate detection, ANN
candidate re-ranking, embedding-drift checks) and both sit far from the
origin (realistic after mean-pooling, positional-embedding bias, or
un-normalized activations), `dot(x,x)` and `dot(y,y)` are both large
and nearly equal to `2*dot(x,y)`; subtracting them cancels almost every
significant digit of the true (small) squared distance. Verified
directly in this repository's fixtures and test suite, the naive
formula's rounding noise in the surviving digits can push the float
result NEGATIVE -- an impossible value for a real sum-of-squares --
at float16 (`-4.0` exactly, at an offset of only ~50) and at float32
(`-4096.0` exactly, at an offset of ~1e5). Feeding a negative squared
distance to `sqrt()` (the natural next step to recover the actual
Euclidean distance) silently produces NaN with no error or warning.
scikit-learn's own PR#24542 ("Add a negative zeros and NaNs guard for
the Euclidean specialisation") added a runtime clamp to its Cython
pairwise-distance reduction kernels specifically to paper over this --
independent confirmation from the library's own maintainers that this
is a real, previously-encountered failure mode, not a hypothetical one.

**Stable formula** (`stable_squared_euclidean_distance`): compute the
difference vector first, then sum its squares --
`dist^2(x, y) = sum((x_i - y_i)^2)`, the same "difference-before-
reduction" principle this repository's `stable_variance` and
`stable_pearson_correlation` kernels already use. Each term is bounded
by the actual per-coordinate difference rather than by the vectors'
absolute scale, so there is nothing large to cancel regardless of how
far `x` and `y` sit from the origin.

All twenty-four derivations above are cross-checked in this repository
against an independent implementation (`reference.py`) built entirely
from Python's arbitrary-precision `decimal.Decimal` type at 50
significant digits, evaluated directly from each kernel's mathematical
definition rather than derived from the same numpy code path under
test (see prior paragraph); `squared_euclidean_distance`'s reference
(`gold_squared_euclidean_distance`) sums `(x_i - y_i)^2` per-coordinate
in Decimal arithmetic, matching the stable kernel's formula shape but
at 50-digit precision rather than the dtype under test, so a bug shared
between the naive and stable float kernels cannot hide behind comparing
them only to each other. `numguard --check-naive-fails` (run in CI on
every push) asserts that every naive fixture documented above still
actually fails, and every stable counterpart still actually passes --
proving these are live regression tests, not decorative claims.

## BPE trainer pair-count accumulator overflow

**Textbook/reference definition:** the total number of times a
candidate merge pair occurs across an entire training corpus -- an
unbounded, non-negative integer count, computed simply as the sum of
every per-chunk occurrence increment observed while scanning the
corpus.

**Naive formula** (`naive_bpe_pair_count_overflow`): accumulate each
increment into a fixed-width 32-bit signed integer with plain `+=` and
no overflow check -- the exact shape of huggingface/tokenizers'
`BpeTrainer` (`tokenizers/src/models/bpe/trainer.rs`), which stores
`AHashMap<Pair, i32>` and increments it via
`*pair_counts.entry(cur_pair).or_default() += counts[i] as i32;` with
no guard. This is a real, currently-open bug: GitHub issue
huggingface/tokenizers#2058, still open at time of writing, with three
linked fix PRs (#2059, #2087, #2105) all still unmerged -- verified
live via the GitHub API immediately before this kernel was accepted,
per this repository's reproduce-before-accept discipline (a GitHub
issue is evidence a bug EXISTED at some version, not that it still does
now; the fix here is confirmed NOT yet released). Two-space indentation
is an extremely common pair in code corpora and can exceed
`i32::MAX` (2,147,483,647) occurrences in a sufficiently large training
set (the issue's own reproduction: ~2.37 billion occurrences from a
30-million-line synthetic corpus). Once a pair's true count crosses
that boundary, the fixed-width accumulator silently wraps via two's-
complement -- verified in this repository's test suite going strictly
NEGATIVE (`-1,924,967,296` at the issue's own ~2.37B repro shape,
`1,705,032,704` at a further ~6B shape, confirming genuine modular
wraparound rather than a one-off sign flip). Since `BpeTrainer`'s own
merge-selection step picks the single highest-count pair at every
training step, a wrapped-negative (or merely wrapped-wrong) count
silently removes what is actually the corpus's most frequent pair from
contention, corrupting the entire downstream merge order and every
tokenization built from it -- with no error, warning, or crash at
training time.

**Stable formula** (`stable_bpe_pair_count_overflow`): accumulate in
arbitrary precision (Python's native unbounded `int`), matching the
real, already-drafted-but-unmerged fix in PR#2059/#2087/#2105, which
widens the map's value type from `i32` to `i64` (`i64::MAX` is
far beyond any realistic corpus's pair count, so this never wraps for
real-world input).

All twenty-six derivations above are cross-checked in this repository
against an independent implementation (`reference.py`) built entirely
from Python's arbitrary-precision `decimal.Decimal` type at 50
significant digits, evaluated directly from each kernel's mathematical
definition rather than derived from the same numpy/int-width code path
under test (see prior paragraphs); `bpe_pair_count_overflow`'s
reference (`gold_bpe_pair_count_overflow`) sums every increment in
Decimal arithmetic with no fixed-width integer type at all, so a bug
shared between the naive and stable accumulators cannot hide behind
comparing them only to each other. `numguard --check-naive-fails` (run
in CI on every push) asserts that every naive fixture documented above
still actually fails, and every stable counterpart still actually
passes -- proving these are live regression tests, not decorative
claims.

## Beam-search length-penalty prompt-length leak

**Textbook/reference definition:** the standard length-penalty formula
for scoring a completed beam (Wu et al. 2016, Sec. 7; the same
definition implemented by HuggingFace transformers'
`BeamHypotheses.add`): `score = cum_logprob / (output_len ** length_
penalty)`, where `output_len` is the number of tokens the model
GENERATED for that beam -- never the prompt/context length that
preceded generation.

**Naive formula** (`naive_beam_search_length_penalty`): divide by
`(prompt_len + output_len) ** length_penalty` instead. This is the
exact, currently-present shape of vLLM's own beam-search scorer
(`get_beam_search_score` in
`vllm/entrypoints/generate/beam_search/utils.py`): the function's
`seq_len` argument is computed as `len(tokens)`, and vLLM's
`BeamSearchSequence.tokens` list is seeded with the prompt's token IDs
and then grown one generated token at a time (`tokens=prompt_token_ids`
at beam-start, then `tokens=current_beam.tokens + [token_id]` every
subsequent step, in both the offline and online beam-search code
paths) -- so `len(tokens)` silently includes the entire prompt length
in every score. Confirmed still present by reading vLLM's current
mainline source directly (commit `c6fa1f0`, 2026-09-15): there is no
prompt-length subtraction anywhere in `get_beam_search_score`. This is
a real, previously-reported defect: GitHub issue
vllm-project/vllm#2606 ("Beam Search Length Normalization Wrong"),
independently rediscovered and reported again in linked PR #7007 ("Use
correct length in beam search scoring") by two further users who
noticed vLLM-served beam-search output diverging from the same model
served via HuggingFace `generate`. Issue #2606 was auto-closed by a
stale-activity bot, not because it was fixed, and PR #7007 was closed
unmerged -- verified live via the GitHub API immediately before this
kernel was accepted, per this repository's reproduce-before-accept
discipline. In production terms: whenever a request's prompt is much
longer than its generated completion (summarization, RAG, long-context
chat with a short reply -- an extremely common shape), the shared
prompt length dominates every beam's score denominator almost equally,
so `length_penalty` -- a parameter users set specifically to bias
ranking toward shorter or longer completions -- ends up having almost
no effect on which beam actually wins, exactly as both the issue's
original reporter and PR #7007's authors independently observed.

**Stable formula** (`stable_beam_search_length_penalty`): divide by
`output_len ** length_penalty` only, exactly PR #7007's proposed fix
(swap `get_len()` for `get_output_len()`).

`beam_search_length_penalty`'s reference
(`gold_beam_search_length_penalty`) computes `cum_logprob / (Decimal(
output_len) ** Decimal(length_penalty))` directly from the textbook
definition in 50-digit Decimal arithmetic, independent of how either
kernel under test tracks `seq_len` -- so a bug shared between the naive
and stable kernels (e.g. both mishandling the EOS-token adjustment)
could not hide behind comparing them only to each other. `numguard
--check-naive-fails` (run in CI on every push) asserts that every
naive fixture documented above still actually fails, and every stable
counterpart still actually passes -- proving these are live regression
tests, not decorative claims.

## int32 dequantization subtraction overflow

**Textbook/reference definition:** affine (zero-point) dequantization
is `real = (code - zero_point) * scale`, an exact algebraic
subtraction and multiply -- the standard definition shared by
PyTorch's own quantized-tensor semantics, TFLite, and ONNX
QuantizeLinear/DequantizeLinear. The subtraction itself has no inherent
width limit; it is only limited by whatever integer type the
implementation happens to perform it in.

**Naive formula** (`naive_int32_dequant_overflow`): perform
`(code - zero_point)` using ordinary 32-bit signed integer arithmetic
-- exactly what CPU `torch.dequantize` does internally for a `qint32`
tensor. When `code` and `zero_point` are far enough apart that the true
difference exceeds the representable range of a signed 32-bit integer
(`[-2**31, 2**31-1]`), the subtraction silently wraps via
two's-complement instead of raising an error or producing `inf`/`NaN`.
This is a real, currently-open bug: pytorch/pytorch#153358 ("torch.
dequantize result inconsistent on CPU and GPU"), whose own reported
repro is `code=2147483647` (`INT32_MAX`), `zero_point=-2147483648`
(`INT32_MIN`), `scale=1e-10` -- the true difference is `2**32-1`
(~4.29e9), which wraps to `-1`, so CPU `torch.dequantize` returns
`-1e-10` instead of the correct `+0.4294967295` -- a WRONG-SIGN result,
not merely a precision loss. GPU dequantize does not share this bug
(the issue's own reporter confirmed `a - b.to("cpu")` is nonzero,
i.e. CPU and GPU disagree), so a model whose qint32 activations hit
this range dequantizes correctly under CUDA but silently wrong under
CPU, with no error, warning, or NaN to signal the problem. Reproduced
from scratch against this repository's installed `torch==2.14.0`
before this kernel was accepted (`torch.quantize_per_tensor` +
`torch.dequantize` reproduce the issue's own exact `-1e-10` result for
its own exact inputs). A PyTorch maintainer (Xia-Weiwen) confirmed the
root cause in the issue thread and stated the team is moving to new
dequantize ops in `torchao` and "probably won't fix this issue" --
verified the issue is still `state=open` via the GitHub API immediately
before acceptance, per this repository's reproduce-before-accept
discipline.

**Stable formula** (`stable_int32_dequant_overflow`): perform
`(code - zero_point)` using 64-bit integer arithmetic instead. Since
`code` and `zero_point` are each drawn from the 32-bit signed range,
their difference is bounded by `[-(2**32-1), 2**32-1]`, which fits
comfortably within int64's much larger range -- so no width-driven
wraparound is possible for any pair of int32 inputs. This matches
`torch.dequantize`'s own (bug-free) GPU-kernel behavior and the
textbook affine-quantization definition.

`int32_dequant_overflow`'s reference (`gold_int32_dequant_overflow`)
computes `(Decimal(code) - Decimal(zero_point)) * Decimal(scale)`
directly in 50-digit arbitrary-precision arithmetic, with NO fixed-width
integer type anywhere -- independent of whether either kernel under
test happens to perform the subtraction in a 32-bit or 64-bit register.
`numguard --check-naive-fails` (run in CI on every push) asserts that
every naive fixture documented above still actually fails -- including
two boundary controls (a code/zero_point pair whose difference is
exactly `INT32_MAX`, the largest value that still fits without
wrapping, and an ordinary small-value control) that confirm naive and
stable agree exactly right up to the wraparound boundary and diverge
only once it is crossed -- and every stable counterpart still actually
passes, proving these are live regression tests targeting the specific
overflow boundary, not decorative claims.

## Vector 2-norm

**Textbook/reference definition:** the Euclidean 2-norm of a vector `x`
is `sqrt(sum(x_i^2))` -- an exact algebraic sum-of-squares followed by a
single square root, with no inherent bound on the intermediate
magnitude other than whatever numeric type performs the sum.

**Naive formula** (`naive_norm`): compute `sqrt(dot(x, x))`, exactly
NumPy's own vector-norm code path for `ord=None`/`ord=2`
(`numpy/linalg/_linalg.py`, `norm()`). Every element is squared BEFORE
the sum, so the intermediate `dot(x, x)` can overflow to `+inf` (or
every squared term can underflow to exactly `0.0`) at roughly the
SQUARE of the vector's true safe magnitude -- far below where the
actual norm (the square root of that sum, hence the same order of
magnitude as the inputs themselves) would ever need to overflow. This
This is a real, currently-open bug: numpy/numpy#32372 ("`numpy.linalg.norm`
overflows for intermediate values due to naive sum-of-squares
implementation"), whose own reported repro is three `float16` values of
`200.0` each -- the true norm is `~346.4`, comfortably within
`float16`'s `~65504` max representable value, but `dot(x,x) =
3*200**2 = 120000` already exceeds it, so `numpy.linalg.norm` returns
`inf` for a perfectly representable answer. Independently
re-reproduced on this repository's installed `numpy==2.5.2` before this
kernel was accepted (`np.linalg.norm([200,200,200], dtype=np.float16)`
reproduces the issue's own exact `inf` result), per this repository's
reproduce-before-accept discipline. A fix PR (numpy/numpy#31927, "fix
np.linalg.norm overflow for representable results") was confirmed
still open and unmerged via the GitHub API immediately before this
kernel's acceptance card was written. The issue's own author also notes
NumPy's `norm` is treated as the cross-library ground-truth reference
by both JAX's and PyTorch's own test suites, so this failure mode does
not stay contained to one library. The same intermediate-squaring
mechanism also produces a symmetric UNDERFLOW failure (every squared
term rounds to exactly `0.0` for a vector of sufficiently small but
still safely-representable values), directly demonstrated in this
repository's `test_kernels.py::TestNormOverflow` unit tests -- though
this repo's own fixture set marks that specific case as a CLI/audit
control rather than an adversarial finding, since the true norm there
is itself far below the audit tool's tolerance floor for that dtype
(see the `float32_underflow_tiny_uniform` fixture's own docstring in
`fixtures.py` for the full reasoning).

**Stable formula** (`stable_norm`): the LAPACK `dnrm2`-style fix
proposed in numpy/numpy#31927 -- divide by the largest-magnitude
element BEFORE squaring and summing (accumulating in `float64` for
extra headroom, matching LAPACK's own higher-precision reduction), then
multiply the result back by that same max magnitude at the very end.
Every scaled term is now bounded by `1.0` regardless of the vector's
raw scale, so nothing overflows or underflows before the sum -- the
same "move the scale-sensitive step out of the danger zone, restore
the scale afterward" principle already used by this repository's
`stable_geometric_mean` (log-space) and `stable_pearson_correlation`
(normalize-then-dot) kernels.

`norm`'s reference (`gold_norm`) computes `sqrt(sum(x_i^2))` directly
in 50-digit arbitrary-precision `Decimal` arithmetic, with no
fixed-width intermediate anywhere -- independent of whether either
kernel under test squares-then-sums directly or scales first.
`numguard --check-naive-fails` (run in CI on every push) asserts that
every naive fixture documented above marked adversarial (float16
overflow, float32 overflow, float64 overflow) still actually fails at
the CLI/audit tolerance level, that both control cases (the everyday
3-4-5 case, and the float32 underflow case -- adversarial at the direct
kernel level per `test_kernels.py`, but a control at the CLI/audit
level since the true norm there sits far below this tool's float32
tolerance floor) still agree between naive and stable within tolerance,
and that every stable counterpart still actually passes -- proving
these are live regression tests targeting the specific overflow/
underflow boundaries, not decorative claims.

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
- Kahan, W. (1965), "Further remarks on reducing truncation errors" --
  the compensated-summation algorithm `stable_sum` implements, reducing
  the worst-case rounding-error bound from O(n*eps) to O(eps)
  independent of the number of terms n.
- NumPy's own `numpy.sum` documentation and PR #3685 ("ENH: implement
  pairwise summation") -- NumPy itself switched its default reduction
  from naive sequential summation to pairwise summation specifically to
  reduce this O(n) accumulated-rounding-error growth, independent
  corroboration that `sum`'s naive/stable gap is a real, previously-hit
  production issue rather than a contrived demonstration.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021),
  "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  (arXiv:2104.09864) -- introduces RoPE and the `cos(position * freq)`
  rotation `rope_cos` audits.
- Baichuan Inc., "混合精度下位置编码竟有大坑，LLaMA等主流开源模型纷纷中招"
  (zhuanlan.zhihu.com/p/651588659) -- documents the RoPE/ALiBi
  position-encoding collision bug under low-precision (bfloat16/float16)
  position ids that `rope_cos`'s `fp16_long_context_position_aliasing`
  fixture reproduces.
- HuggingFace `transformers`, PR #29285 ("Force float32 ... since
  bfloat16 loses precision on long contexts") -- the real-world fix
  `stable_rope_cos` implements, now load-bearing boilerplate in every
  RoPE implementation in that codebase.
- Lin, T-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017), "Focal
  Loss for Dense Object Detection" (arXiv:1708.02002) -- introduces the
  Focal Loss formula `focal_loss_grad` audits the gradient of.
- `facebookresearch/sam3#575` / PR#576 ("Reduced Triton sigmoid focal
  loss returns NaN gradients for gamma=0 when logits saturate") -- the
  real-world bug report and fix `naive_focal_loss_grad`/
  `stable_focal_loss_grad` reproduce and resolve.
- `kornia/kornia#918` / PR#924 ("NaN gradients on backward pass with
  focal loss") -- an independent real-world instance of the same
  failure family in a different codebase.
- van Leeuwen, M.P., Haak, K.V., Saygili, G., Postma, E.O., Ong, L.L.S.
  (2025), "A Note on the Stability of the Focal Loss", Transactions on
  Machine Learning Research (OpenReview `eCYActnGbu`) -- independently
  derives and empirically demonstrates the same instability across the
  broader `0 <= gamma < 1` range in real CNN/ViT/U-Net training runs.
- `scipy/scipy` issues #8980, #9353, #3728 and PR #9562 -- the
  overflow, underflow, and constant-input bug reports that drove
  `scipy.stats.pearsonr`'s rewrite to the normalize-then-dot form
  `stable_pearson_correlation` reproduces.
- `numpy/numpy#32446` and `pandas-dev/pandas#67023` / `#37448` /
  `#45640` -- real, currently open/recently-fixed reports that even
  the mean-centered (but not norm-divided) form of Pearson's r returns
  inconsistent `NaN`/`+-1` for exactly-constant columns, motivating
  this kernel's explicit zero-variance guard.
- `scipy/scipy#1053` (Trac#526), "scipy.stats.gmean cannot handle large
  numbers" -- the real-world bug report that drove `scipy.stats.gmean`
  to switch from the literal `prod(x)**(1/n)` formula to the log-space
  `exp(mean(log(x)))` form `stable_geometric_mean` reproduces; also
  independently corroborated in `numpy/numpy#14985` ("using the
  formulation `prod(x)**(1/count)` can overflow unnecessarily... that
  can be avoided by working with logarithms").
- Hollows, P. (2026), "Gauge dependence and structured-output
  corruption in sign-branched repetition penalties: measurements
  across models, inference stacks, and alternative repetition
  controls" (arXiv:2607.09791) -- proves the sign-branched
  multiplicative repetition penalty shipped in HuggingFace
  `transformers`, vLLM, and llama.cpp is gauge-dependent (violates
  softmax's shift-invariance), with a StarCoder2-7B HumanEval
  measurement of real greedy-decode token flips from this mechanism,
  the bug `naive_repetition_penalty` reproduces and
  `stable_repetition_penalty` (branching on log-probabilities instead
  of raw logits) resolves.
