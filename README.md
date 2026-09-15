# numguard

Audit naive vs numerically-stable ML math kernels against an
independent high-precision reference -- catch the classic
overflow/underflow/cancellation bugs before they reach production.

## Simple explanation

Some of the most common formulas in machine learning -- softmax,
cross-entropy, log-sum-exp, variance -- have a "textbook" version and a
"numerically stable" version that are mathematically identical on
paper but behave very differently on real floating-point hardware. The
textbook version can silently produce `inf`, `NaN`, or even a
*negative variance* (impossible for a real variance) on ordinary-looking
inputs. numguard runs both versions against a set of documented
adversarial fixtures and an independent arbitrary-precision reference,
and tells you exactly where and how the naive formula breaks.

## The problem

- `exp(x)` overflows in float32 for `x > ~88`, and underflows every
  term to `0.0` for very negative `x` -- both break a naive softmax or
  log-sum-exp.
- `E[x^2] - E[x]^2` (the "one-pass" variance formula taught in most
  intro stats courses) subtracts two large, nearly-equal numbers when
  the data is offset far from zero -- classic catastrophic
  cancellation, and it can even go negative.
- **That negative-variance bug is not just cosmetic**: LayerNorm --
  used in essentially every Transformer block -- divides by
  `sqrt(variance + eps)`. Build LayerNorm on top of the naive one-pass
  variance formula and a negative variance means `sqrt()` of a negative
  number: every element of the normalized output becomes `NaN`, silently,
  for perfectly ordinary-looking input activations (verified in this
  repo: `layer_norm`'s `large_offset_negative_variance` fixture).
- **RMSNorm has a distinct, arguably sneakier failure mode**: unlike
  LayerNorm, RMSNorm (used in LLaMA, T5, and most modern LLMs in place
  of LayerNorm) has nothing to cancel -- there is no mean-subtraction --
  but its `mean(x^2)` reduction still needs to fit inside whatever
  dtype it runs in. Keep that reduction in the activation's own narrow
  dtype (float16) and *ordinary*, non-adversarial-looking activation
  values (e.g. ~300) square to ~90000, already past float16's ~65504
  max: `mean(x^2)` silently overflows to `inf`, and `x / sqrt(inf)`
  collapses every output to a *finite-looking* `0.0` -- not an obvious
  `NaN`/`inf` that would get noticed immediately (verified in this
  repo: `rms_norm`'s `fp16_activation_overflow` fixture). This is
  exactly why real implementations (e.g. HF Transformers'
  `LlamaRMSNorm`) upcast the reduction to float32 before squaring.
- `-log(softmax(x)[target])` computes an explicit probability before
  taking a log; if that probability underflows to `0.0`, the result is
  `inf`/`NaN` even though the true cross-entropy is a perfectly
  ordinary finite number.
- **KL divergence has a different failure shape again**: `sum(p_i *
  log(p_i / q_i))` hits an IEEE754 `0 * -inf` indeterminate form
  whenever `p_i` is exactly `0` -- an entirely ordinary case (a
  one-hot label vector in distillation/label-smoothing is *the* most
  common real input to this kernel). The naive formula evaluates to
  `NaN` for the whole sum; the correct answer, by the standard
  `x*log(x) -> 0` convention, is that the term simply contributes `0`
  (verified in this repo: `kl_divergence`'s `one_hot_label` fixture).
- **Online (chunked/streaming) softmax has a distinct, non-overflow
  failure mode**: FlashAttention-style kernels process a long sequence
  in tiles, tracking a running max and running sum instead of holding
  the whole array in memory at once. A naive incremental implementation
  that forgets to rescale its already-accumulated partial results when
  a later tile raises the running max produces a *finite, non-NaN, but
  systematically wrong* probability distribution -- not a subtle
  rounding difference, and not caused by any extreme magnitude
  (verified in this repo: `online_softmax`'s `max_in_middle_chunk`
  fixture uses only single-digit inputs).
- **Masked softmax has yet another distinct failure shape**: attention
  padding/causal masking sets excluded positions' logits to `-inf`
  before a softmax -- correct for each masked position on its own
  (`exp(-inf) == 0.0`), but if *every* position in a row is masked (a
  fully-padded row -- an entirely ordinary occurrence any time a batch
  contains sequences shorter than the batch's max length), the
  normalizing sum is also `0.0`, and `0.0 / 0.0` is `NaN` across the
  *entire row*, even though no individual logit was extreme. This is a
  real, previously documented bug class -- PyTorch's `torchtune`
  library carries an explicit `skip_mask` guard against exactly this
  failure (verified in this repo: `masked_softmax`'s
  `fully_padded_row` fixture).
- **Summation has a distinct, scale-dependent failure mode**: naive
  sequential summation (`total += x` in a loop) rounds the running
  total on every addition, and because the same accumulator absorbs
  every one of those roundings in sequence, the worst-case error grows
  as `O(n * eps)` in the number of terms `n` -- invisible for a
  handful of values, but measurably wrong once summing thousands of
  similarly-scaled terms (e.g. a loss or gradient accumulated across a
  batch/dataset), or once one large term "swamps" the accumulator so
  later small terms are partially lost. Kahan compensated summation
  tracks the lost low-order bits and re-applies them, reducing the
  error bound to `O(eps)` independent of `n` (verified in this repo:
  `sum`'s `many_small_uniform_terms` and
  `large_value_swamps_small_terms` fixtures; NumPy's own `np.sum`
  switched to pairwise summation for the same underlying reason).
- **RoPE (rotary position embedding) has a resolution-limit failure,
  not an overflow one**: the rotation angle `position * inv_freq` and
  its `cos()` need enough mantissa bits to keep distinct integer
  position ids distinguishable -- compute them in the model's storage
  dtype (float16, or bfloat16 in torch) instead of at least float32,
  and positions beyond that dtype's exact-integer range (2048 for
  float16, 256 for bfloat16) silently collide onto the *same* rotation
  angle, so the model loses its ability to tell those token positions
  apart, with no NaN/inf to signal it. This is a real, previously
  documented production bug (Baichuan Inc.'s report on RoPE/ALiBi under
  bfloat16; HuggingFace `transformers` PR #29285, "Force float32...
  since bfloat16 loses precision on long contexts", now load-bearing
  boilerplate in every RoPE implementation in that codebase) --
  reproduced here at float16 to keep this tool torch-free (verified in
  this repo: `rope_cos`'s `fp16_long_context_position_aliasing`
  fixture).

These are not edge cases invented for this tool -- they are exactly the
failure modes documented in the numerical-stability literature (Blanchard
et al. 2019 on log-sum-exp rounding error; the standard "subtract the
max before exp" and "Welford/two-pass" mitigations used throughout
deep-learning frameworks). What was missing was a small, dependency-light
CLI that demonstrates and regression-tests the gap directly, rather than
a blog post or a framework-internal function you have to trust blindly.

- **int8 element-wise add mismatches quantization parameters, not
  float precision**: affine int8 quantization stores `real = (code -
  zero_point) * scale`; adding two int8 tensors (an ordinary
  residual/skip connection) is only correct if each operand is
  dequantized with *its own* `(scale, zero_point)`, then the sum is
  requantized with saturation. A naive fusion path that reuses one
  operand's quant params for the other -- exactly the bug OpenVINO
  PR #7305 fixed ("Eltwise with very different inputs ranges") and
  PR #1135 documented as causing "zero accuracy" -- or that skips the
  saturating clamp on requantization (the same failure family as the
  open, unfixed bug openvino#34673 on Apple M4 Max ARM) produces a
  systematically wrong result, not merely an imprecise one. This is a
  different bug *class* from every kernel above: bookkeeping, not
  float range/cancellation (verified in this repo: `int8_add`'s
  `mismatched_scale_residual_add` and `int8_saturation_wraparound`
  fixtures).
- **HyperLogLog register term** (`hll_register`): a register's harmonic-
  sum contribution is `2**-rank`. A naive kernel computes this via a
  fixed-width 32-bit integer left shift (`1 << rank`), silently masking
  the shift distance modulo 32 once `rank >= 32` -- reproducing Apache
  Flink's FLINK-39399 bug (a register holding 35 wrongly estimates
  ~95,000 instead of ~4e14), corroborated by ClickHouse's `uniqHLL12`
  large-cardinality bug report and Druid's sparse-mode register-overflow
  issue. This is a different bug class again: fixed-width integer
  overflow, not float precision or quantization bookkeeping.
- **P^2 streaming quantile estimator has a stateful-drift failure mode**
  (`p2_quantile`): the P^2 algorithm (Jain & Chlamtac, CACM 1985) tracks
  a running median/quantile in O(1) memory without ever storing the
  input stream. Its own paper suggests maintaining each marker's desired
  position via a per-step accumulated increment (`ns[i] += dns[i]`) "to
  reduce CPU overhead" -- but that accumulation drifts under floating-
  point rounding, exactly the real bug reported against and fixed in
  Andrey Akinshin's `perfolizer` library (GitHub issue
  AndreyAkinshin/perfolizer#8): a value that should land exactly on an
  integer marker boundary lands a few ULPs off instead, silently
  deferring a marker adjustment that should have fired and corrupting
  the estimator's internal state for the rest of the stream. This is a
  different bug class from every kernel above: drift in a *stateful*
  algorithm's internal bookkeeping across many steps, not a single
  expression's overflow/underflow/cancellation.
- **Repetition penalty is gauge-dependent, not overflow-prone**
  (`repetition_penalty`): the multiplicative repetition penalty shipped
  across HuggingFace `transformers`, vLLM, and llama.cpp branches on
  the *sign of the raw logit* before penalizing a previously-generated
  token -- but arXiv:2607.09791 proves this makes the result depend on
  an arbitrary additive shift to the logits, even though softmax itself
  is exactly shift-invariant. Two logit vectors representing the
  identical pre-penalty distribution can select different next tokens
  after an ordinary repetition penalty, purely from where the raw
  logits happen to sit relative to zero -- not an overflow, underflow,
  or cancellation bug at all, a distinct "wrong invariant" failure
  shape shared only with `hll_register`'s fixed-width-shift bug above.
- **Speculative-decoding rejection sampling breaks when the sampled-from
  and accepted-against distributions differ** (`speculative_reject`):
  the Leviathan/Chen speculative-decoding theorem guarantees the output
  token distribution equals the target model's true distribution
  *provided the draft token is sampled from exactly the same
  distribution used in the accept/reject math*. deepseek-ai/DeepSpec
  PR#30 fixed a real production bug of exactly this shape ("Draft
  samples were drawn from native-dtype probabilities while rejection
  used float32 probabilities"), and vLLM's own rejection-sampler
  history (PR#48641/#53630) shows how easily an extra materialization
  of the logits/probabilities array reintroduces this mismatch. Every
  individual probability array involved is finite and well-formed --
  there is no NaN, inf, or crash -- yet the sampled output silently
  stops matching the target model's true distribution, defeating
  speculative decoding's entire "lossless" guarantee. A different bug
  class from every kernel above: a correctness identity across TWO
  probability distributions (draft and target), not a single
  expression's numerical behavior.

- **AdamW decoupled weight decay** (`weight_decay`): the naive
  formula multiplies the parameter by `(1 - lr*weight_decay)` and
  writes the result straight back into the parameter's own storage
  dtype at *every* optimizer step -- exactly what happens when the
  trained parameter tensor itself (not a separate float32 master copy)
  is bf16/float16. This is the real, currently-open production bug in
  Nerogar/OneTrainer#996 ("No weight decay with Adam, bf16 and
  stochastic rounding"): once the per-step decay fraction
  `lr*weight_decay` is smaller than the storage dtype's precision at
  that magnitude, every update rounds away to the identical stored
  value and weight decay silently does *nothing* for the entire
  training run, with no error, warning, or crash. A different bug
  class again: a total silent stall from repeated intermediate
  rounding, not a formula or cross-distribution mismatch.

## What this does

For each of twenty-one kernels (`logsumexp`, `softmax`, `cross_entropy`,
`variance`, `layer_norm`, `rms_norm`, `kl_divergence`, `online_softmax`,
`masked_softmax`, `sum`, `rope_cos`, `int8_add`, `hll_register`,
`focal_loss_grad`, `pearson_correlation`, `weighted_sampling_key`,
`geometric_mean`, `p2_quantile`, `repetition_penalty`,
`speculative_reject`, `weight_decay`),
across three dtypes (`float16`, `float32`, `float64` -- `int8_add` and
`hll_register` are scored at `float64` only, since they audit integer
codes/register values rather than a dtype-swept float array), on a
curated set of adversarial and everyday fixtures, numguard:

1. Runs the naive (textbook) formula and the stable (standard
   mitigation) formula, both implemented in plain numpy.
2. Computes an independent ground truth using Python's arbitrary
   precision `decimal.Decimal` (50 significant digits) evaluated
   directly from the mathematical definition -- not derived from the
   same numpy code path being tested.
3. Reports pass/fail per case (finite + within tolerance), and a
   `--check-naive-fails` self-check mode that asserts every documented
   naive failure mode still actually fails and every stable
   counterpart still actually passes -- this is what proves the
   fixtures are real, not decorative, and it is exactly what the CI
   pipeline runs on every push.

This tool does **not** require PyTorch, TensorFlow, or a GPU -- it is
pure numpy, so the same failure modes that show up in production ML
frameworks (which use the same underlying formulas) can be demonstrated
and regression-tested anywhere numpy runs, on macOS or Linux, CPU-only.

## Install

**Not published to PyPI, and the name `numguard` is already taken on PyPI by
an unrelated project** (a different "agent verification" tool by a different
author — confirmed via `pypi.org/pypi/numguard/json`). This repo cannot use
`pip install numguard` even after publishing; a rename or a different
distribution name (e.g. `numguard-kernels`) would be needed first.

Install the latest GitHub Release wheel directly (checksum-verified,
CI-built):

```bash
pip install https://github.com/zhuhroscar-tech/numguard/releases/latest/download/numguard-0.10.0-py3-none-any.whl
```

Or from source:

```bash
git clone https://github.com/zhuhroscar-tech/numguard.git
cd numguard && pip install -e ".[dev]"
```

Note: unlike this account's other small CLIs, numguard does not ship a
standalone `.pyz` zipapp -- numpy ships compiled C extension modules
(`.so` files), which cannot execute from inside a zip archive the way
pure-Python zipapps can (verified: this was attempted and fails with
`NotADirectoryError` on the numpy `_core` package). A wheel/sdist
(installed via pip into a venv) is the correct distribution format for
a numpy-dependent tool; CI builds and smoke-tests both.

## Usage

```bash
# Full report across all kernels and dtypes
numguard

# Just one kernel, one dtype
numguard --kernel variance --dtype float32

# Only the stable formulas (exits 0 if they all check out)
numguard --variant stable

# Machine-readable output
numguard --json

# CI self-check: verify every fixture still demonstrates a real bug
numguard --check-naive-fails
```

Exit codes: `0` all requested cases passed; `1` at least one case that
should have passed did not (a real finding, or a self-check
regression); `2` usage error.

## Example finding

```
$ numguard --kernel variance --dtype float32 --variant naive
  [X] naive / float32 / large_offset_small_spread  non-finite (inf/nan)
      naive float32 accumulation of E[x^2] (~1e12) and E[x]^2 (~1e12)
      loses the last ~7 significant digits ...
```

True variance of `[1000000, 1000001, 1000002, 1000003]` is exactly
`1.25`. The naive one-pass formula in float32 can report a negative
number here -- verified directly in this repo's test suite
(`tests/test_kernels.py::TestVarianceCancellation`).

## Reproducible build & test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v --cov=numguard --cov-report=term-missing
numguard --check-naive-fails
```

CI runs this on both `ubuntu-latest` and `macos-latest` across Python
3.9 and 3.12, then builds a wheel and sdist, computes
`SHA256SUMS.txt`, and smoke-tests the built wheel (installed into a
clean venv) on the Linux runner before publishing a release.

## Limitations

- The fixture set is curated and documented, not exhaustive -- it
  demonstrates known, well-understood failure modes, not every
  possible numerical bug.
- Tolerances (`DEFAULT_TOLERANCE` / `ABS_TOLERANCE` in `core.py`) are
  deliberately generous for float16 and tight for float64; they are a
  reasonable default, not a formal error bound.
- This targets scalar/array numpy kernels, not full framework
  operators (e.g. it does not exercise PyTorch's fused
  `log_softmax`/`cross_entropy` C++ kernels directly) -- it demonstrates
  the *underlying formula-level* problem that those fused ops exist to
  solve.
- No GPU-specific behavior (MPS/CUDA numerical differences) is tested
  here; this is a CPU-only, framework-independent tool by design.

## License

MIT
