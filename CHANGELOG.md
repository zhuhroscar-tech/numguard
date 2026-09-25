# Changelog

All notable source-quality changes are recorded here. Release artifacts live on the [GitHub releases page](https://github.com/zhuhroscar-tech/numguard/releases); this project intentionally ships wheels/sdists rather than a zipapp because NumPy depends on native extensions.

## v0.32.3 - Maintenance metadata and tag CI

- Added package metadata links for issues and this changelog.
- Made GitHub Actions CI run explicitly on `v*` release tags.
- Added repository-contract coverage so release maintenance metadata and tag CI wiring do not drift.

## v0.32.2 - Changelog and release-history contract

- Added this changelog so release history is visible in the repository, not only in GitHub's releases UI.
- Linked release history from both English and Simplified Chinese READMEs.
- Added repository-contract coverage requiring the changelog and the current version entry to stay in sync.

## v0.32.1 - Packaging license metadata

- Modernized packaging metadata to the current SPDX license format.
- Added regression coverage for license-file metadata and package/runtime version parity.

## v0.32.0 - sorted_search kernel

- Added the `sorted_search` guard for NumPy's large-integer searchsorted promotion issue (`numpy/numpy#29727`).

## v0.31.0 - explained_variance kernel and scoring fix

- Added explained-variance coverage.
- Fixed scoring behavior for NaN gold/reference cases.

## v0.30.0 - lambertw0 kernel

- Added the `lambertw0` guard for SciPy's branch-point NaN issue (`scipy/scipy#24770`).

## v0.29.0 - i0 kernel

- Added the `i0` guard for shared NumPy/SciPy/JAX premature-overflow behavior.

## v0.28.0 - mannwhitney_u kernel

- Added the `mannwhitney_u` guard for catastrophic-cancellation risk in rank-sum arithmetic.

## v0.27.0 - genlaguerre kernel

- Added the `genlaguerre` guard for generalized Laguerre polynomial evaluation drift.

## v0.26.1 - LayerNorm float16 fixture coverage

- Added a float16 squaring-overflow fixture for `layer_norm`.

## v0.26.0 - incremental_mean kernel

- Added incremental/streaming mean stability coverage.

## v0.25.0 - norm kernel

- Added vector 2-norm stability coverage.

## v0.24.0 - int32_dequant_overflow kernel

- Added int32 dequantization overflow coverage.

## Earlier releases

- v0.23.0 through v0.1.0 built out the kernel catalog from the initial softmax/log-sum-exp/cross-entropy/variance guards through beam search, BPE overflow, squared Euclidean distance, LongRoPE factor selection, gradient accumulation, weight decay, speculative rejection, repetition penalty, P² quantiles, geometric mean, weighted sampling, Pearson correlation, focal loss gradient, HyperLogLog registers, int8 add, RoPE cosine, summation, masked/online softmax, KL divergence, LayerNorm, and RMSNorm.
