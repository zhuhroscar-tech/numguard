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

## Independent ground truth

All sixteen derivations above are cross-checked in this repository
against an independent implementation (`reference.py`) that uses
Python's arbitrary-precision `decimal.Decimal` (50 significant digits)
evaluated directly from the mathematical definitions -- not derived
from the same numpy code paths being tested (this includes
`rope_cos`'s reference `cos()`, computed via a from-scratch Decimal
Taylor series, deliberately not `math.cos`, so that a bug shared with a
float64 trig implementation could not hide behind comparing against
itself; and `focal_loss_grad`'s reference, computed via a symmetric
central-difference numerical derivative of the focal loss formula
itself at 50-digit precision, deliberately not via either kernel's
analytic chain-rule derivation, so a shared algebra mistake in the
naive/stable gradient formulas could not hide behind comparing them
only to each other). This means a bug shared between the naive and
stable numpy formulas (e.g. both computing the wrong quantity) would
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
