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
