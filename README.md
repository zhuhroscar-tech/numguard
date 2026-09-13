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
- `-log(softmax(x)[target])` computes an explicit probability before
  taking a log; if that probability underflows to `0.0`, the result is
  `inf`/`NaN` even though the true cross-entropy is a perfectly
  ordinary finite number.

These are not edge cases invented for this tool -- they are exactly the
failure modes documented in the numerical-stability literature (Blanchard
et al. 2019 on log-sum-exp rounding error; the standard "subtract the
max before exp" and "Welford/two-pass" mitigations used throughout
deep-learning frameworks). What was missing was a small, dependency-light
CLI that demonstrates and regression-tests the gap directly, rather than
a blog post or a framework-internal function you have to trust blindly.

## What this does

For each of five kernels (`logsumexp`, `softmax`, `cross_entropy`,
`variance`, `layer_norm`), across three dtypes (`float16`, `float32`,
`float64`), on a curated set of adversarial and everyday fixtures,
numguard:

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

```bash
pip install numguard  # if/when published to PyPI
# or from source:
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
