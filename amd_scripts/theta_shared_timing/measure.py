#!/usr/bin/env python3
"""Run two validated regional granule timing comparisons on one MI300A node."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

from analyze_dycore_causal import contrast_uncertainty, order_review
from dycore_causal_core import THETA, paired_summary, require
from run_dycore_causal import (
    TARGET,
    code_audits,
    configure_compilation,
    dump,
    provenance,
    run,
    timing_report,
    validate_report,
)
from theta_compiler import check_graphs, comparisons

SCRIPT_ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_overlay(source, out, bundle=SCRIPT_ROOT):
    manifest = json.loads((bundle / "PATCH_MANIFEST.json").read_text())
    module = Path(manifest["module"])
    before = digest(source / module)
    require(before in manifest["accepted_installed_sha256"], "GT4Py patch base differs; aborting.")
    require(
        digest(source / "next/iterator/ir_utils/domain_utils.py")
        == manifest["domain_utils_sha256"],
        "Required domain_utils float-conversion fix differs or is missing.",
    )
    require(
        digest(bundle / "map_fusion_extended.py") == manifest["patched_sha256"],
        "Compiler patch checksum mismatch.",
    )
    destination = out / "compiler_overlay" / "gt4py"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(bundle / "map_fusion_extended.py", destination / module)
    dump(
        out / "compiler_overlay.json",
        dict(
            original=str(source),
            overlay=str(destination),
            module=str(module),
            original_sha256=before,
            patched_sha256=manifest["patched_sha256"],
        ),
    )
    return destination.parent


def source_hashes(root, overlay):
    result = provenance(root)
    for prefix, directory in (("timing_bundle", SCRIPT_ROOT), ("compiler_overlay", overlay)):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".sh", ".json"):
                result[prefix + "/" + str(path.relative_to(directory))] = digest(path)
    return result


def summarize(report):
    result = dict(validation=report["validation"], hostname=report["hostname"], metrics={})
    for metric in ("granule_device", "theta_device", "granule_wall"):

        def value(block, metric=metric):
            if metric == "theta_device":
                return block["programs"][THETA]
            return block["wall_ms"] if metric == "granule_wall" else block["device_ms"]

        def paired(blocks):
            return paired_summary([dict(b, device_ms=value(b)) for b in blocks])

        timing = paired(report["blocks"])
        placebo = paired(report["placebo"]["blocks"])
        samples = timing["quartet_saved_ms"]
        item = dict(timing=timing, placebo=dict(timing=placebo), interleaved_controls=True)
        review = order_review(item)
        result["metrics"][metric] = dict(
            baseline_ms=statistics.mean(timing["quartet_arm_ms"]["A"]),
            compiler_ms=statistics.mean(timing["quartet_arm_ms"]["B"]),
            saved_ms=timing["saved_ms"],
            raw_interval=contrast_uncertainty(samples, [0.0] * len(samples)),
            timing=timing,
            placebo=placebo,
            order_review=review,
        )
    return result


def write_summary(out, results):
    dump(out / "TIMING_SUMMARY.json", results)
    lines = [
        "# MI300A regional/120 compiler-fusion timing",
        "",
        "Positive saving means compiler fusion is faster. These are same-node granule measurements.",
        "Raw intervals do not automatically establish a benefit: review the interleaved A/A controls and order diagnostics in JSON.",
        "",
        "| Comparison | Metric | A ms | Compiler ms | Saved ms | Raw 95% CI ms |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, result in results.items():
        for metric, item in result["metrics"].items():
            lo, hi = item["raw_interval"]["ci95_ms"]
            lines.append(
                f"| {name} | {metric} | {item['baseline_ms']:.6f} | "
                f"{item['compiler_ms']:.6f} | {item['saved_ms']:.6f} | [{lo:.6f}, {hi:.6f}] |"
            )
    (out / "TIMING_SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, out = Path.cwd().resolve(), args.output.resolve()
    require(not out.is_relative_to(SCRIPT_ROOT), "Results must be outside the source bundle.")
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="running", phase="compiler overlay")
    dump(out / "STATUS.json", status)
    results = {}
    try:
        package = importlib.util.find_spec("gt4py")
        require(package and package.submodule_search_locations, "Cannot locate installed GT4Py.")
        overlay = build_overlay(Path(package.submodule_search_locations[0]), out)
        initial = source_hashes(root, overlay)
        dump(out / "source_hashes.json", initial)
        shutil.copytree(
            SCRIPT_ROOT, out / "experiment_sources", ignore=shutil.ignore_patterns("__pycache__")
        )
        status["phase"] = "compiler preflight"
        dump(out / "STATUS.json", status)
        configure_compilation(root, out)
        for index, spec in enumerate(comparisons()):
            status["phase"] = spec["name"]
            dump(out / "STATUS.json", status)
            directory = out / spec["name"]
            directory.mkdir()
            dump(directory / "case.json", spec)
            env = dict(os.environ)
            env.update(
                PYTHONPATH=os.pathsep.join(
                    [str(overlay), str(SCRIPT_ROOT), env.get("PYTHONPATH", "")]
                ),
                ICON4PY_CAUSAL_CASE=json.dumps(spec),
                ICON4PY_CAUSAL_PLATFORM="amd",
                ICON4PY_CAUSAL_OUTPUT=str(directory),
                ICON4PY_CAUSAL_MODE="timing",
                GT4PY_BUILD_CACHE_DIR=str(out / "build"),
                ICON4PY_PROFILE_WARMUP="5",
                ICON4PY_PROFILE_ROUNDS="10",
                ICON4PY_PROFILE_SEED="20260910",
                ICON4PY_PROFILE_MODE="timing",
                ICON_GRID="icon_benchmark_regional",
                CAUSAL_QUARTETS="12",
                CAUSAL_ORDER_SEED=str(20260915 + index),
                ICON4PY_PROFILE_REPORT=str(directory / "timing.{pid}.json"),
                GT4PY_METRICS_OUTPUT_PATH=str(directory / "timers.timing.{pid}.json"),
                THETA_COMPILER_OVERLAY=str(overlay / "gt4py"),
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-sv",
                    "-p",
                    "dycore_causal_plugin",
                    "-p",
                    "no:tach",
                    "-m",
                    "continuous_benchmarking",
                    "--benchmark-disable",
                    "--backend=dace_gpu",
                    "--grid=icon_benchmark_regional:120",
                    TARGET,
                ],
                directory / "timing.log",
                env,
            )
            report = validate_report(timing_report(directory), spec, "regional", "timing")
            require(report["validation"]["status"] == "passed", "Granule validation did not pass.")
            fingerprints = {
                key: report[key]
                for key in (
                    "initial_state_sha256",
                    "initial_scalar_state_sha256",
                    "grid_dimensions",
                )
            }
            if index == 0:
                initial_inputs = fingerprints
            else:
                require(fingerprints == initial_inputs, "Inputs differ between comparisons.")
            dump(directory / "input_fingerprints.json", fingerprints)
            audits, _ = code_audits(out / "build")
            dump(directory / "code_audit.json", check_graphs(spec, audits))
            results[spec["name"]] = summarize(report)
            write_summary(out, results)
        final = source_hashes(root, overlay)
        dump(out / "source_hashes.final.json", final)
        require(initial == final, "Experiment/compiler sources changed during timing.")
        status.update(status="complete", phase="complete")
        (out / "COMPLETE").touch()
        print(f"Complete: {out / 'TIMING_SUMMARY.md'}", flush=True)
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        dump(out / "STATUS.json", status)


if __name__ == "__main__":
    main()
