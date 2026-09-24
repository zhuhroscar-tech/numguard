"""Packaging/metadata drift guards.

The CLI's --help description already had a regression test
(test_cli.py::test_help_description_lists_every_kernel) after it was
caught listing only 4 of 6 kernels post layer_norm/rms_norm. The same
hand-written-prose drift existed in two more places nobody was
checking: the package's own module docstring (src/numguard/__init__.py)
and the PyPI-facing description in pyproject.toml. Both are prose, not
generated from core.ALL_KERNELS, so they can silently drift again the
next time a kernel is added -- these tests make that impossible to miss.

Deliberately avoids the `tomllib` stdlib module (Python 3.11+ only) and
any third-party TOML parser dependency, since CI runs this suite on
Python 3.9 too -- a couple of targeted regexes on two known-format
single-line fields are simpler and dependency-free.
"""
from __future__ import annotations

import re
from pathlib import Path

from numguard import core

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROSE_NAME = {"logsumexp": "log-sum-exp", "cross_entropy": "cross-entropy"}


def _expected_names():
    return [_PROSE_NAME.get(k, k) for k in core.ALL_KERNELS]


def _pyproject_field(field: str) -> str:
    text = (_REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(rf'^{field}\s*=\s*"([^"]*)"', text, re.MULTILINE)
    assert match, f"could not find a {field!r} field in pyproject.toml"
    return match.group(1)


def test_init_docstring_lists_every_kernel():
    import numguard

    doc = numguard.__doc__ or ""
    for expected in _expected_names():
        assert expected in doc, (
            f"kernel {expected!r} missing from numguard/__init__.py module "
            f"docstring -- update it when adding/removing a kernel"
        )


def test_pyproject_description_lists_every_kernel():
    description = _pyproject_field("description")
    for expected in _expected_names():
        assert expected in description, (
            f"kernel {expected!r} missing from pyproject.toml's "
            f"[project].description -- update it when adding/removing a kernel"
        )


def test_init_version_matches_pyproject_version():
    import numguard

    assert numguard.__version__ == _pyproject_field("version"), (
        "src/numguard/__init__.py __version__ and pyproject.toml "
        "[project].version have drifted apart"
    )


def test_pyproject_uses_current_license_metadata():
    text = (_REPO_ROOT / "pyproject.toml").read_text()

    assert 'license = "MIT"' in text
    assert 'license-files = ["LICENSE"]' in text
    assert "license = {" not in text, (
        "setuptools now warns about table-style project.license metadata; "
        "use a SPDX license string instead"
    )
    assert "License :: OSI Approved :: MIT License" not in text, (
        "setuptools warns that license classifiers are deprecated when "
        "SPDX license metadata is present"
    )


def test_numerical_stability_doc_exists_and_covers_every_kernel():
    """kernels.py's own module docstring tells readers: 'see
    docs/numerical-stability.md for the derivation of each' -- that file
    did not exist anywhere in this repo's history until this test and
    the doc it guards were added together. This is the same
    documented-but-missing class of bug as the --help/docstring/
    pyproject drift above, just one level up: a citation to a file
    instead of a citation to a kernel name. This test fails loudly if
    the file is ever deleted, or if a kernel is added without an
    accompanying '## <Kernel Name>' section explaining its naive/stable
    derivation."""
    doc_path = _REPO_ROOT / "docs" / "numerical-stability.md"
    assert doc_path.is_file(), (
        "src/numguard/kernels.py's module docstring cites "
        "docs/numerical-stability.md as the derivation reference, but "
        "that file does not exist"
    )
    text = doc_path.read_text()
    # Each kernel needs its own explanatory section; check by the
    # underlying dispatch-table key names as well as their prose forms,
    # since the doc's section headers use human-readable names
    # (e.g. "## log-sum-exp", "## LayerNorm") rather than the raw
    # identifiers ("logsumexp", "layer_norm").
    section_aliases = {
        "logsumexp": "log-sum-exp",
        "softmax": "softmax",
        "cross_entropy": "cross-entropy",
        "variance": "variance",
        "layer_norm": "LayerNorm",
        "rms_norm": "RMSNorm",
        "kl_divergence": "KL divergence",
        "online_softmax": "Online (chunked/streaming) softmax",
        "masked_softmax": "Masked softmax",
        "sum": "Summation",
        "rope_cos": "RoPE (rotary position embedding) angle/cos",
        "int8_add": "int8 element-wise add",
        "hll_register": "HyperLogLog register term",
        "focal_loss_grad": "Sigmoid focal loss gradient",
        "pearson_correlation": "Pearson correlation coefficient",
        "explained_variance": "Explained variance regression score",
        "weighted_sampling_key": "Weighted reservoir-sampling comparison key",
        "geometric_mean": "Geometric mean",
        "p2_quantile": "P^2 (Piecewise-Parabolic) streaming quantile estimator",
        "repetition_penalty": "Repetition penalty",
        "speculative_reject": "Speculative-decoding rejection sampling",
        "weight_decay": "AdamW decoupled weight decay storage stall",
        "gradient_accumulation_bias": "Gradient-accumulation loss bias",
        "longrope_factor_select": "LongRoPE short/long scaling-factor selection",
        "squared_euclidean_distance": "Squared Euclidean distance via dot-product expansion",
        "bpe_pair_count_overflow": "BPE trainer pair-count accumulator overflow",
        "beam_search_length_penalty": "Beam-search length-penalty prompt-length leak",
        "int32_dequant_overflow": "int32 dequantization subtraction overflow",
        "norm": "Vector 2-norm",
        "incremental_mean": "Incremental (streaming) mean",
        "genlaguerre": "Generalized Laguerre polynomial evaluation",
        "mannwhitney_u": "Mann-Whitney U statistic",
        "i0": "Modified Bessel function of the first kind, order 0",
        "lambertw0": "Lambert W function, principal branch (W_0)",
        "sorted_search": "sorted-array integer search",
        }
    for kernel in core.ALL_KERNELS:
        assert kernel in section_aliases, (
            f"kernel {kernel!r} has no known doc-section alias mapping in "
            f"this test -- add one when adding a kernel, then add a "
            f"matching '## ...' section to docs/numerical-stability.md"
        )
        alias = section_aliases[kernel]
        assert re.search(rf"^##\s+.*{re.escape(alias)}", text, re.MULTILINE), (
            f"docs/numerical-stability.md is missing a '## {alias}' "
            f"section for kernel {kernel!r}"
        )
    # And the reverse: the doc shouldn't silently list a stale kernel
    # that no longer exists in core.ALL_KERNELS (or the alias map above
    # would need updating too).
    assert len(section_aliases) == len(core.ALL_KERNELS), (
        "section_aliases in this test has drifted from core.ALL_KERNELS "
        "-- update both together when a kernel is added or removed"
    )
