"""Fixture-registry tests: the small helper that looks up whether a
fixture is a control case, plus a structural sanity check that every
declared fixture has a valid dtype scope."""
from __future__ import annotations

import pytest

from numguard import fixtures
from numguard.core import ALL_DTYPES


def test_expect_naive_ok_true_for_known_control_case():
    assert fixtures.expect_naive_ok("variance", "everyday_spread") is True


def test_expect_naive_ok_false_for_known_adversarial_case():
    assert fixtures.expect_naive_ok("variance", "large_offset_small_spread") is False


def test_expect_naive_ok_raises_for_unknown_fixture():
    with pytest.raises(KeyError):
        fixtures.expect_naive_ok("variance", "not-a-real-fixture")


def test_every_fixture_dtype_is_a_valid_dtype_name():
    for kernel, fixture_list in fixtures.FIXTURES_BY_KERNEL.items():
        for fixture in fixture_list:
            for dtype in fixture.dtypes:
                assert dtype in ALL_DTYPES, (
                    f"{kernel}/{fixture.name} declares unknown dtype {dtype!r}"
                )


def test_every_fixture_has_nonempty_dtype_scope():
    for kernel, fixture_list in fixtures.FIXTURES_BY_KERNEL.items():
        for fixture in fixture_list:
            assert len(fixture.dtypes) > 0, f"{kernel}/{fixture.name} has empty dtype scope"
