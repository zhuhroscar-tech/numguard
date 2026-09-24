# numguard

[![CI](https://github.com/zhuhroscar-tech/numguard/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuhroscar-tech/numguard/actions/workflows/ci.yml) [![Latest release](https://img.shields.io/github/v/release/zhuhroscar-tech/numguard?display_name=tag&sort=semver)](https://github.com/zhuhroscar-tech/numguard/releases/latest)


[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-555555?style=flat)](README.zh-CN.md)

Compare naive and stable numerical kernels against independent high-precision references. numguard makes overflow, cancellation, integer-overflow and algorithmic bookkeeping errors reproducible with small, documented fixtures—not a full model or GPU benchmark.

## What it covers

- NumPy implementations of softmax, log-sum-exp, normalization, reductions, quantization and other ML/statistical operations.
- `float16`, `float32` and `float64` comparisons where applicable; integer-focused cases use their own scoring rules.
- Per-case finiteness and tolerance checks, readable reports and JSON output.
- A regression self-check that verifies expected naive failures, designated control cases and stable results.

## Install

Requires Python 3.9+ and NumPy 1.24+; no PyTorch, TensorFlow or GPU is needed. **Do not use `pip install numguard`: the PyPI name belongs to an unrelated project.** Install this repository in an isolated environment:

```bash
git clone https://github.com/zhuhroscar-tech/numguard.git
cd numguard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Release wheels are also available on the [releases page](https://github.com/zhuhroscar-tech/numguard/releases); choose an actual asset and check its release checksum. See [CHANGELOG.md](CHANGELOG.md) for repository release history. There is no standalone zipapp because NumPy needs native extensions.

## Quick start

```bash
numguard --kernel variance --dtype float32
numguard --variant stable
numguard --json
numguard --check-naive-fails
```

With no arguments, the CLI runs all configured cases and both variants. Failures in the naive report are intentional demonstrations, so that report can exit nonzero. Exit `0` means the requested checks passed, `1` means at least one failed, and `2` indicates invalid usage. For CI, use `--check-naive-fails` rather than expecting the ordinary mixed report to pass.

## Scope and development

Fixtures are curated, not exhaustive. Tolerances are practical thresholds, not formal error bounds. Results concern the implementations here, not current fused framework operators or CUDA/MPS behavior; passing them does not certify a model or production library.

[Derivations](docs/numerical-stability.md), [fixtures](src/numguard/fixtures.py) and [kernel implementations](src/numguard/kernels.py) provide the detailed reference.

```bash
python -m pytest -q
numguard --check-naive-fails
```

[MIT license](LICENSE)
