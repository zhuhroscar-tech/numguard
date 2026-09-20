"""numguard CLI: audit naive vs stable ML math kernels for numerical bugs.

Exit codes:
  0 -- all requested kernel/variant/dtype cases matched the reference
       within tolerance (or --check-naive-fails and every naive case did
       fail, and every stable case passed -- see that flag's help).
  1 -- at least one case that should have been correct was not (a real
       finding, or --check-naive-fails found a naive case that
       unexpectedly passed / a stable case that unexpectedly failed).
  2 -- usage error (bad kernel/dtype name, no JSON fixtures found, etc).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import __version__, core
from .fixtures import expect_naive_ok as fixture_expect_naive_ok
from .style import resolve_style, section, status_headline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="numguard",
        description=(
            "Compare naive vs numerically-stable ML math kernels "
            "(softmax, log-sum-exp, cross-entropy, variance, layer_norm, "
            "rms_norm, kl_divergence, online_softmax, masked_softmax, "
            "sum, rope_cos, int8_add, hll_register, focal_loss_grad, "
            "pearson_correlation, explained_variance, weighted_sampling_key, "
            "geometric_mean, p2_quantile, repetition_penalty, "
            "speculative_reject, weight_decay, "
            "gradient_accumulation_bias, longrope_factor_select, "
            "squared_euclidean_distance, bpe_pair_count_overflow, "
            "beam_search_length_penalty, int32_dequant_overflow, norm, "
            "incremental_mean, genlaguerre, mannwhitney_u, i0, "
            "lambertw0, sorted_search) against "
            "an independent "
            "high-precision reference, across "
            "float16/float32/float64, on adversarial fixtures."
        ),
    )
    p.add_argument("--version", action="version", version=f"numguard {__version__}")
    p.add_argument(
        "--kernel",
        action="append",
        choices=core.ALL_KERNELS,
        dest="kernels",
        help="Limit to one kernel (repeatable). Default: all.",
    )
    p.add_argument(
        "--dtype",
        action="append",
        choices=core.ALL_DTYPES,
        dest="dtypes",
        help="Limit to one dtype (repeatable). Default: all.",
    )
    p.add_argument(
        "--variant",
        choices=("naive", "stable", "both"),
        default="both",
        help="Which formulation(s) to report. Default: both.",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color.")
    p.add_argument(
        "--check-naive-fails",
        action="store_true",
        help=(
            "Self-check mode for CI: assert that every 'naive' case "
            "that has a documented failure mode actually fails, and "
            "every 'stable' case actually passes. Exits 1 if the naive "
            "kernels have silently gotten 'fixed' (fixture drift) or "
            "the stable kernels have regressed. This is what proves "
            "the fixtures are real adversarial cases, not decorative."
        ),
    )
    return p


def _filter_variant(results: List[core.CaseResult], variant: str) -> List[core.CaseResult]:
    if variant == "both":
        return results
    return [r for r in results if r.variant == variant]


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    style = resolve_style(no_color_flag=args.no_color)

    kernels_to_run = tuple(args.kernels) if args.kernels else core.ALL_KERNELS
    dtypes_to_run = tuple(args.dtypes) if args.dtypes else core.ALL_DTYPES

    results = core.run_all(kernels_to_run, dtypes_to_run)

    if args.check_naive_fails:
        return _run_self_check(results, style, args.json)

    display_results = _filter_variant(results, args.variant)

    if args.json:
        payload = [
            {
                "kernel": r.kernel,
                "variant": r.variant,
                "fixture": r.fixture,
                "dtype": r.dtype,
                "description": r.description,
                "ok": r.ok,
                "is_finite": r.is_finite,
                "relative_error": r.relative_error,
                "computed": r.computed_repr,
            }
            for r in display_results
        ]
        print(json.dumps(payload, indent=2))
        return 0 if all(r.ok for r in display_results) else 1

    _print_report(display_results, style)
    return 0 if all(r.ok for r in display_results) else 1


def _print_report(results: List[core.CaseResult], style) -> None:
    print(style.bold(f"numguard {__version__} -- numerical stability audit"))
    by_kernel = {}
    for r in results:
        by_kernel.setdefault(r.kernel, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.ok)

    for kernel, rows in by_kernel.items():
        section(style.bold(kernel))
        for r in rows:
            level = "ok" if r.ok else ("fail" if not r.is_finite else "warn")
            err_str = (
                "non-finite (inf/nan)"
                if not r.is_finite
                else (f"rel_err={r.relative_error:.3e}" if r.relative_error is not None else "n/a")
            )
            headline = (
                f"{r.variant:>6} / {r.dtype:<7} / {r.fixture:<26} {err_str}"
            )
            print(f"  {status_headline(style, level, headline)}")
            print(f"      {style.dim(r.description)}")

    section(style.bold("Summary"))
    level = "ok" if passed == total else "warn"
    print(f"  {status_headline(style, level, f'{passed}/{total} cases within tolerance')}")


def _run_self_check(results: List[core.CaseResult], style, as_json: bool) -> int:
    """Every fixture in fixtures.py claims a specific naive failure mode.
    This mode verifies that claim mechanically: naive cases are expected
    to fail (non-finite or over tolerance) and stable cases are expected
    to pass. If a naive case unexpectedly passes, either the fixture no
    longer demonstrates a real problem (numpy/hardware behavior drift)
    or the naive implementation was accidentally fixed -- either way the
    fixture set needs review, so this must not be silently green."""
    problems = []
    for r in results:
        naive_ok_expected = fixture_expect_naive_ok(r.kernel, r.fixture)
        if r.variant == "naive" and naive_ok_expected and not r.ok:
            problems.append(
                f"naive {r.kernel}/{r.fixture}/{r.dtype} FAILED but is a "
                f"control case expected to be fine (rel_err="
                f"{r.relative_error}, finite={r.is_finite}) -- either the "
                f"naive kernel regressed or this fixture needs re-review"
            )
        if r.variant == "naive" and not naive_ok_expected and r.ok:
            problems.append(
                f"naive {r.kernel}/{r.fixture}/{r.dtype} unexpectedly PASSED "
                f"(expected to demonstrate a failure mode) -- fixture no "
                f"longer adversarial, or naive kernel was fixed; update "
                f"fixtures.py or remove the stale claim"
            )
        if r.variant == "stable" and not r.ok:
            problems.append(
                f"stable {r.kernel}/{r.fixture}/{r.dtype} FAILED "
                f"(rel_err={r.relative_error}, finite={r.is_finite}) -- "
                f"this is a real regression in the stable kernel"
            )

    if as_json:
        print(json.dumps({"problems": problems}, indent=2))
    else:
        print(style.bold("numguard self-check: naive-fails / stable-passes"))
        if problems:
            for p in problems:
                print(f"  {status_headline(style, 'fail', p)}")
        else:
            print(f"  {status_headline(style, 'ok', 'all naive cases failed as expected; all stable cases passed')}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
