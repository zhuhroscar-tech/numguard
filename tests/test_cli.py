"""CLI-level tests: exercise numguard.cli.main() via subprocess-free
direct calls, checking exit codes, JSON structure, and the
--check-naive-fails self-check mode end to end."""
from __future__ import annotations

import json

import pytest

from numguard import cli


def test_help_description_lists_every_kernel():
    """Regression guard: the --help description text is hand-written
    prose, not generated from core.ALL_KERNELS, so it drifted out of
    sync when layer_norm/rms_norm were added (description still said
    only 'softmax, log-sum-exp, cross-entropy, variance'). Assert every
    real kernel name appears in the parser description so this can't
    silently drift again when a 7th kernel is added."""
    from numguard import core

    parser = cli._build_parser()
    description = parser.description
    # logsumexp is written in prose as "log-sum-exp"; every other kernel
    # name matches its identifier literally.
    prose_name = {"logsumexp": "log-sum-exp", "cross_entropy": "cross-entropy"}
    for kernel in core.ALL_KERNELS:
        expected = prose_name.get(kernel, kernel)
        assert expected in description, (
            f"kernel {kernel!r} (expected {expected!r}) missing from "
            f"--help description text"
        )


def test_check_naive_fails_passes_on_current_fixtures(capsys):
    """This is the load-bearing regression test: it proves the fixture
    set genuinely demonstrates the naive/stable gap right now, not just
    at authoring time. If a numpy upgrade or a kernel edit quietly
    'fixes' a naive formula (or breaks a stable one), this test fails."""
    exit_code = cli.main(["--check-naive-fails", "--json", "--no-color"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0, f"self-check found problems: {payload['problems']}"
    assert payload["problems"] == []


def test_default_run_exits_nonzero_because_naive_cases_are_included():
    # Default --variant=both includes the naive cases, which are
    # expected to fail on adversarial fixtures -- so exit code 1 here
    # is the CORRECT behavior, not a bug in the tool.
    exit_code = cli.main(["--json", "--no-color"])
    assert exit_code == 1


def test_stable_only_run_exits_zero():
    exit_code = cli.main(["--variant", "stable", "--json", "--no-color"])
    assert exit_code == 0


def test_json_output_is_well_formed(capsys):
    cli.main(["--kernel", "variance", "--dtype", "float64", "--json", "--no-color"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert len(payload) > 0
    for row in payload:
        assert set(row) >= {
            "kernel", "variant", "fixture", "dtype", "description",
            "ok", "is_finite", "relative_error", "computed",
        }
        assert row["kernel"] == "variance"
        assert row["dtype"] == "float64"


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0


def test_invalid_kernel_choice_is_a_usage_error():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--kernel", "not-a-real-kernel"])
    assert exc_info.value.code == 2


def test_human_readable_report_runs_without_error(capsys):
    exit_code = cli.main(["--kernel", "logsumexp", "--no-color"])
    captured = capsys.readouterr()
    assert "numguard" in captured.out
    assert "logsumexp" in captured.out
    assert exit_code in (0, 1)


def test_self_check_flags_control_case_that_unexpectedly_fails(capsys):
    """Exercises the 'naive control case regressed' problem branch: if a
    fixture marked expect_naive_ok=True stops passing, --check-naive-fails
    must report it, not silently ignore it."""
    from numguard import core

    fake = core.CaseResult(
        kernel="variance",
        variant="naive",
        fixture="everyday_spread",  # a real control-case fixture name
        dtype="float64",
        description="test",
        ok=False,
        is_finite=True,
        relative_error=0.5,
        computed_repr="1.0",
    )
    exit_code = cli._run_self_check([fake], cli.resolve_style(no_color_flag=True), as_json=True)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert len(payload["problems"]) == 1
    assert "control case expected to be fine" in payload["problems"][0]


def test_self_check_flags_stable_regression(capsys):
    from numguard import core

    fake = core.CaseResult(
        kernel="variance",
        variant="stable",
        fixture="everyday_spread",
        dtype="float64",
        description="test",
        ok=False,
        is_finite=True,
        relative_error=0.5,
        computed_repr="1.0",
    )
    exit_code = cli._run_self_check([fake], cli.resolve_style(no_color_flag=True), as_json=False)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "real regression in the stable kernel" in captured.out


def test_self_check_flags_naive_adversarial_case_unexpectedly_passing(capsys):
    """Exercises the other regression direction: a fixture that is
    supposed to demonstrate a naive-formula bug (expect_naive_ok=False)
    but the naive result now passes -- meaning the fixture stopped being
    adversarial (e.g. a numpy behavior change) or the naive kernel was
    silently 'fixed'. Either way this must be reported, not ignored."""
    from numguard import core

    fake = core.CaseResult(
        kernel="variance",
        variant="naive",
        fixture="large_offset_small_spread",  # a real adversarial fixture
        dtype="float32",
        description="test",
        ok=True,
        is_finite=True,
        relative_error=0.0,
        computed_repr="1.25",
    )
    exit_code = cli._run_self_check([fake], cli.resolve_style(no_color_flag=True), as_json=True)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert len(payload["problems"]) == 1
    assert "unexpectedly PASSED" in payload["problems"][0]


def test_self_check_no_problems_prints_ok_headline_in_human_mode(capsys):
    """Covers the human-readable (non-JSON) success path with an empty
    problem list, distinct from the JSON success path already covered by
    test_check_naive_fails_passes_on_current_fixtures."""
    exit_code = cli._run_self_check([], cli.resolve_style(no_color_flag=True), as_json=False)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "all naive cases failed as expected" in captured.out
