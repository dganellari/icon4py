#!/usr/bin/env python3
"""Run one preregistered case on both grids; no SSH or scheduler operations."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import amd_cache_counters as amd
import analyze_cache_comparison as nvidia
from dycore_causal_core import CASES, THETA, original_name, require, sha256


EXPERIMENT_SCRIPTS = (
    "causal_cmake.py",
    "dycore_causal_core.py",
    "dycore_causal_plugin.py",
    "run_dycore_causal.py",
    "analyze_dycore_causal.py",
    "amd_cache_counters.py",
    "analyze_cache_comparison.py",
    "dycore_profile_window.py",
    "collect_dycore_profile_metadata.py",
)

AMD_GROUPS = dict(amd.GROUPS) | {
    "occupancy": ["MeanOccupancyPerCU"],
    "occupancy_active": ["MeanOccupancyPerActiveCU"],
    "vmem_latency": ["VmemLatency"],
    "issue": ["ValuPipeIssueUtil", "MemUnitStalled"],
    "l2_busy": ["TCC_BUSY_sum", "TCC_CYCLE_sum"],
}
AMD_GROUPS["l1_stall"] = amd.GROUPS["l1_stall"] + ["TCP_GATE_EN2_sum"]
NV_METRICS = list(nvidia.COUNTERS) + [
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__inst_executed.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
]
TARGET = "model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]"


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def save_domain_warnings(log):
    """Retain emitted compiler warnings, including when the child fails."""
    log = Path(log)
    text = log.read_text(errors="replace")
    matches = list(
        re.finditer(
            r"UserWarning: Translating .*?Please consider reordering your mesh\.",
            text,
            flags=re.DOTALL,
        )
    )
    if matches:
        output = log.with_suffix(".domain_warnings.json")
        dump(
            output,
            dict(
                source_log=str(log),
                warnings=[
                    dict(line=text.count("\n", 0, match.start()) + 1, text=match.group())
                    for match in matches
                ],
            ),
        )
        print(f"Domain-contiguity warnings: {len(matches)}; see {output}.", flush=True)


def run(command, log, env=None):
    print(
        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        " ".join(map(str, command)),
        flush=True,
    )
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("w") as stream:
        process = subprocess.run(
            list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, env=env
        )
    save_domain_warnings(log)
    require(process.returncode == 0, f"Command failed ({process.returncode}); see {log}.")


def inventory_check(text, metrics):
    missing = [
        m for m in metrics if not re.search(r"(?<![\w.])" + re.escape(m) + r"(?![\w.])", text)
    ]
    require(not missing, "Required metrics unavailable: " + ", ".join(missing))


def configure_compilation(root, out):
    # Runtime code-generation hooks must execute in this pytest process;
    # spawned compilation workers do not inherit its monkeypatches.
    os.environ["GT4PY_BUILD_JOBS"] = "0"
    os.environ["GT4PY_BUILD_JOBS_MODE"] = "serial"
    os.environ["DACE_debugprint"] = "1"
    real_cmake = shutil.which("cmake")
    require(real_cmake, "CMake is unavailable.")
    directory = out / "compiler_tools"
    directory.mkdir()
    wrapper = directory / "cmake"
    source = (root / "amd_scripts/causal_cmake.py").read_text()
    wrapper.write_text("#!" + sys.executable + "\n" + source.split("\n", 1)[1])
    wrapper.chmod(0o755)
    os.environ["ICON4PY_CAUSAL_REAL_CMAKE"] = real_cmake
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ["PATH"]
    dump(
        out / "compiler_settings.json",
        dict(
            real_cmake=real_cmake,
            mode="serial",
            build_jobs=0,
            cmake_timeout_seconds=os.environ.get("CAUSAL_CMAKE_TIMEOUT_SECONDS", "1200"),
        ),
    )
    # Exercise CMake's uname/compiler detection before expensive fixture setup.
    probe = out / "compiler_probe"
    probe.mkdir()
    (probe / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(causal_probe LANGUAGES CXX)\n"
        "add_executable(causal_probe main.cpp)\n"
    )
    (probe / "main.cpp").write_text("int main() { return 0; }\n")
    env = dict(os.environ, CAUSAL_CMAKE_TIMEOUT_SECONDS="120")
    run(
        ["cmake", "-S", probe, "-B", probe / "build", "-G", "Ninja"],
        out / "compiler_configure.log",
        env,
    )
    run(["cmake", "--build", probe / "build", "--parallel", "1"], out / "compiler_build.log", env)
    (out / "COMPILER_PREFLIGHT_COMPLETE").touch()


def captured_amd(directory, metrics, rounds, pattern=r"dc_.*fieldop"):
    """ROCTx trace membership selects counters by same-process dispatch identity.

    SDK 1.1.0 may also count outside the selected region. Keep the complete raw
    CSV, explicitly exclude those rows, and require every captured dispatch.
    No shared per-kernel iteration assumption is made.
    """
    traces = amd.trace_dispatches(directory, pattern)
    counters = amd.counter_dispatches(directory, metrics, pattern)
    keys = {(r["Agent_Id"], r["Dispatch_Id"]) for r in traces}
    captured = [c for c in counters if (c["agent"], c["dispatch_id"]) in keys]
    amd.match_counter_trace(captured, traces, str(directory))
    require(len(traces) % rounds == 0, "Incomplete captured granule calls.")
    size = len(traces) // rounds
    signatures = [amd.full_trace_identity(t) for t in traces]
    require(
        all(signatures[i : i + size] == signatures[:size] for i in range(0, len(traces), size)),
        "Captured launch sequences differ between calls.",
    )
    return captured, traces, len(counters) - len(captured)


def provenance(root):
    files = {
        str((root / "amd_scripts" / name).relative_to(root.parent)): sha256(
            (root / "amd_scripts" / name).read_text(errors="replace")
        )
        for name in EXPERIMENT_SCRIPTS
    }
    for directory in (
        root / "model",
        root.parent / "gt4py" / "src",
        root.parent / "dace" / "dace",
    ):
        for file in directory.rglob("*.py"):
            if any(
                part in ("cache_runs", "causal_runs", ".gt4py_cache", "__pycache__")
                for part in file.parts
            ):
                continue
            files[str(file.relative_to(root.parent))] = sha256(file.read_text(errors="replace"))
    return files


def code_audits(cache):
    audits = [json.loads(p.read_text()) for p in cache.rglob("causal_code.json")]
    require(audits, "No generated-code audit found; compiler hook did not run.")
    owners = {}
    for audit in audits:
        for kernel in audit["kernels"].values():
            require(
                kernel not in owners or owners[kernel] == audit["program"],
                "Generated kernel owner collision.",
            )
            owners[kernel] = audit["program"]
    return audits, owners


def check_intervention(spec, audits):
    if spec.get("baseline"):
        check_intervention(spec["baseline"], audits)
        for arm, arm_case, factor in (
            ("A", spec["baseline"], 4),
            ("B", spec, spec["vlb"]),
        ):
            evidence = [
                a for a in audits if a["program"] == THETA + "__causal_" + arm_case["name"]
            ]
            require(
                evidence and all(a.get("vertical_block_factors") == [factor] for a in evidence),
                f"Arm {arm} did not produce the expected vertical blocking factor {factor}.",
            )
            require(
                all(k["block"] == [256, 1, 1] for a in evidence for k in a["launches"])
                and all(a["launches"] for a in evidence),
                f"Arm {arm} changed the AMD thread-block shape or lacks launch evidence.",
            )
    if spec.get("order_width"):
        changed = [
            a
            for a in audits
            if "__causal_" + spec["name"] in a["program"]
            and any(t["changed"] for t in a["traversal"])
        ]
        require(
            any(a["original_program"] == THETA for a in changed),
            "Block-order intervention did not reach theta-rho.",
        )
        if spec.get("scope") == "all":
            require(
                len({a["original_program"] for a in changed}) >= 3,
                "Full-granule block-order intervention changed fewer than three programs.",
            )
    if spec.get("block_2d"):
        target = spec["target"]
        native = [k for a in audits if a["program"] == target for k in a.get("launches", [])]
        modified = [
            k
            for a in audits
            if a["program"] == target + "__causal_" + spec["name"]
            for k in a.get("launches", [])
        ]
        require(
            native and len(native) == len(modified),
            "Launch-shape trial changed kernel count or lacks native/modified launch audits.",
        )
        require(
            all(k["block"] is not None for k in native + modified),
            "Unrecognised launch syntax; cannot verify block shapes.",
        )
        for role in ("vertex_gather", "edge_gather"):
            a = [k for k in native if k[role]]
            b = [k for k in modified if k[role]]
            require(
                len(a) == len(b) == 1
                and b[0]["block"] == spec["block_2d"]
                and a[0]["block"] != b[0]["block"],
                f"Launch-shape intervention did not change the {role} kernel as requested.",
            )
        require(
            all(k["block"] == spec["block_2d"] for k in modified if k["vertical_threads"]),
            "A two-dimensional kernel retained an unexpected block shape.",
        )
        require(
            sorted(k["block"] for k in native if not k["vertical_threads"])
            == sorted(k["block"] for k in modified if not k["vertical_threads"]),
            "Launch-shape trial also changed a one-dimensional kernel.",
        )
    if spec.get("fusion"):
        require(
            any(
                a["program"] == THETA + "__causal_" + spec["name"]
                and a.get("fused_theta_pressure_kernels")
                for a in audits
            ),
            "Compiler did not produce a kernel combining rho/theta, pressure gradient and wind outputs; the proposed fusion was not achieved.",
        )


def parse_amd(directory, groups, rounds, owners):
    merged, signature, excluded = [], None, {}
    for group, metrics in groups.items():
        counters, traces, dropped = captured_amd(directory / group, metrics, rounds)
        actual = [amd.full_trace_identity(t) for t in traces]
        if signature is None:
            signature = actual
            merged = [
                dict(kernel=t["Kernel_Name"], trace=t, raw={}, group_duration_ns={}) for t in traces
            ]
        require(actual == signature, f"Launch sequence changed in counter group {group}.")
        excluded[group] = dropped
        for output, values, trace in zip(merged, counters, traces):
            output["raw"][group] = values["counters"]
            output["group_duration_ns"][group] = int(trace["End_Timestamp"]) - int(
                trace["Start_Timestamp"]
            )
    for item in merged:
        bare = item["kernel"].split("(", 1)[0].strip()
        require(bare in owners, f"Unmapped captured kernel {bare}.")
        item["program"] = original_name(owners[bare])
        item["derived"] = amd.derived(item["raw"])
    require(THETA in {k["program"] for k in merged}, "Theta-rho absent from counters.")
    return dict(kernels=merged, rounds=rounds, excluded_outside_window=excluded)


def parse_nvidia(path, metrics, rounds, owners):
    # Extend the existing wide/long CSV parser's metric inventory explicitly.
    previous = nvidia.COUNTERS
    nvidia.COUNTERS = metrics
    try:
        rows, issues = nvidia._csv_rows(path)
    finally:
        nvidia.COUNTERS = previous
    require(not issues, f"NCU CSV errors: {issues}")
    kernels = {}
    for row in rows:
        name = row["Kernel Name"]
        if not re.search(r"dc_.*fieldop", name):
            continue
        key = tuple(row.get(k, "") for k in ("ID", "Process ID", "Context", "Stream", "Device"))
        item = kernels.setdefault(
            key,
            dict(
                kernel=name,
                raw={},
                units={},
                identity=key,
                launch={k: row.get(k) for k in ("Block Size", "Grid Size")},
            ),
        )
        metric, unit = row["Metric Name"], row.get("Metric Unit", "")
        value, error = nvidia._canonical(metric, row["Metric Value"], unit)
        require(not error and value is not None, f"Invalid NCU metric {metric}: {row}")
        require(metric not in item["raw"], f"Duplicate NCU metric {metric} for {key}.")
        item["raw"][metric], item["units"][metric] = value, unit
    result = list(kernels.values())
    require(result and len(result) % rounds == 0, "Incomplete NCU capture.")
    for item in result:
        require(set(item["raw"]) == set(metrics), f"Missing NCU metrics for {item['kernel']}.")
        match = re.search(r"\b(dc_[A-Za-z0-9_]+)", item["kernel"])
        require(match and match[1] in owners, "Unknown NCU kernel owner.")
        item["program"] = original_name(owners[match[1]])
        item["derived"] = nvidia._derived(item["raw"])
    size = len(result) // rounds
    sig = [(k["kernel"], k["launch"]) for k in result]
    require(
        all(sig[i : i + size] == sig[:size] for i in range(0, len(sig), size)),
        "NCU launch sequences differ between calls.",
    )
    return dict(kernels=result, rounds=rounds)


def validate_report(path, spec, grid, mode):
    report = json.loads(path.read_text())
    require(
        report["status"] == "complete" and report["case"] == spec and report["mode"] == mode,
        f"Invalid window report {path}.",
    )
    require(
        report["grid"] == f"icon_benchmark_{grid}:{spec['levels']}", "Actual grid/levels mismatch."
    )
    require(
        report["grid_dimensions"]["levels"] == spec["levels"],
        "Fixture did not construct requested levels.",
    )
    return report


def validate_counter_report(path, spec, grid, mode, arm):
    report = validate_report(path, spec, grid, mode)
    require(report.get("counter_arm") == arm, "Profiler captured the wrong arm.")
    return report


def collect_counters(platform, spec, grid, directory, env, pytest_args, owners, arm):
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(env, ICON4PY_CAUSAL_COUNTER_ARM=arm)
    mode = "hip" if platform == "amd" else "cuda"
    if platform == "amd":
        for group, metrics in AMD_GROUPS.items():
            dest = directory / "counters" / group
            dest.mkdir(parents=True)
            env.update(
                ICON4PY_PROFILE_REPORT=str(dest / "window.{pid}.json"),
                GT4PY_METRICS_OUTPUT_PATH=str(dest / "timers.{pid}.json"),
            )
            run(
                [
                    "rocprofv3",
                    "--selected-regions",
                    "--pmc",
                    *metrics,
                    "--kernel-trace",
                    "--marker-trace",
                    "--kernel-include-regex",
                    ".*dc_.*fieldop.*",
                    "--output-format",
                    "csv",
                    "-d",
                    dest,
                    "-o",
                    group,
                    "--",
                    *pytest_args,
                ],
                dest / "profile.log",
                env,
            )
            validate_counter_report(amd.only_file(dest, "window.*.json"), spec, grid, mode, arm)
            captured_amd(dest, metrics, 3)
        counters = parse_amd(directory / "counters", AMD_GROUPS, 3, owners)
    else:
        env.update(
            ICON4PY_PROFILE_REPORT=str(directory / "counter-window.{pid}.json"),
            GT4PY_METRICS_OUTPUT_PATH=str(directory / "timers.counter.{pid}.json"),
        )
        run(
            [
                "ncu",
                "--replay-mode",
                "application",
                "--app-replay-match",
                "all",
                "--app-replay-mode",
                "strict",
                "--cache-control",
                "none",
                "--clock-control",
                "none",
                "--profile-from-start",
                "off",
                "--target-processes",
                "all",
                "--kernel-name",
                "regex:.*dc_.*fieldop.*",
                "--metrics",
                ",".join(NV_METRICS),
                "--page",
                "raw",
                "--csv",
                "--print-units",
                "base",
                "--export",
                directory / "counter-report",
                "--log-file",
                directory / "counters.csv",
                *pytest_args,
            ],
            directory / "profile.log",
            env,
        )
        for file in directory.glob("counter-window.*.json"):
            validate_counter_report(file, spec, grid, mode, arm)
        require(list(directory.glob("counter-window.*.json")), "NCU window reports missing.")
        counters = parse_nvidia(directory / "counters.csv", NV_METRICS, 3, owners)
    counters["arm"] = arm
    dump(directory / "COUNTERS.json", counters)


def timing_report(directory):
    """Select a process report without treating warning sidecars as results."""
    files = [
        path
        for path in Path(directory).glob("timing.*.json")
        if re.fullmatch(r"timing\.[0-9]+\.json", path.name)
    ]
    require(len(files) == 1, f"Expected one PID timing report in {directory}, found {len(files)}.")
    return files[0]


def execute(args):
    root = Path.cwd().resolve()
    spec = CASES[args.case]
    require(
        spec.get("platform", args.platform) == args.platform,
        "This case is an AMD-only incremental comparison against fused theta-rho.",
    )
    array = os.environ.get("CAUSAL_RUN_ID") or os.environ.get(
        "SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "manual")
    )
    out = root / "amd_scripts" / "causal_runs" / f"{args.platform}_{array}" / spec["name"]
    require(not out.exists(), f"Output already exists: {out}; refusing to mix attempts.")
    out.mkdir(parents=True)
    dump(out / "case.json", spec)
    dump(out / "source_hashes.json", provenance(root))
    for filename in EXPERIMENT_SCRIPTS:
        shutil.copy2(root / "amd_scripts" / filename, out / filename)
    configure_compilation(root, out)
    inventory = out / "available_metrics.txt"
    if args.platform == "amd":
        run(["rocprofv3", "--list-avail"], inventory)
        inventory_check(inventory.read_text(), {m for v in AMD_GROUPS.values() for m in v})
        dump(out / "requested_metrics.json", AMD_GROUPS)
        for group, metrics in AMD_GROUPS.items():
            dest = out / "preflight" / group
            dest.mkdir(parents=True)
            run(
                [
                    "rocprofv3",
                    "--selected-regions",
                    "--pmc",
                    *metrics,
                    "--kernel-trace",
                    "--marker-trace",
                    "--kernel-include-regex",
                    ".*dycore_cache_probe.*",
                    "--output-format",
                    "csv",
                    "-d",
                    dest,
                    "-o",
                    "probe",
                    "--",
                    sys.executable,
                    "amd_scripts/amd_cache_counters.py",
                    "probe",
                ],
                dest / "probe.log",
            )
            captured_amd(dest, metrics, 1, r"dycore_cache_probe")
    else:
        run(["ncu", "--query-metrics", "--query-metrics-mode", "all"], inventory)
        # NCU query-metrics omits software launch attributes. The actual probe
        # below must return all four launch attributes before model setup.
        inventory_check(
            inventory.read_text(), [m for m in NV_METRICS if not m.startswith("launch__")]
        )
        dump(out / "requested_metrics.json", NV_METRICS)
        # An actual CUDA probe catches permission, replay and counter issues.
        probe = out / "probe.py"
        probe.write_text(
            "import cupy as cp\nx=cp.zeros(65536,dtype=cp.float64)\nk=cp.RawKernel('extern \\\"C\\\" __global__ void dc_probe_fieldop(double*x){int i=blockIdx.x*blockDim.x+threadIdx.x;x[i]+=1;}', 'dc_probe_fieldop')\nk((256,),(256,),(x,))\ncp.cuda.runtime.deviceSynchronize()\ncp.cuda.runtime.profilerStart()\nk((256,),(256,),(x,))\ncp.cuda.runtime.deviceSynchronize()\ncp.cuda.runtime.profilerStop()\n"
        )
        run(
            [
                "ncu",
                "--replay-mode",
                "application",
                "--cache-control",
                "none",
                "--clock-control",
                "none",
                "--profile-from-start",
                "off",
                "--metrics",
                ",".join(NV_METRICS),
                "--page",
                "raw",
                "--csv",
                "--print-units",
                "base",
                "--log-file",
                out / "preflight.csv",
                sys.executable,
                probe,
            ],
            out / "preflight.log",
        )
        parse_nvidia(out / "preflight.csv", NV_METRICS, 1, {"dc_probe_fieldop": "probe"})
    (out / "PREFLIGHT_COMPLETE").touch()
    grids = {}
    grid_order = ["global", "regional"] if args.case % 2 == 0 else ["regional", "global"]
    if os.environ.get("CAUSAL_GRIDS"):
        grid_order = os.environ["CAUSAL_GRIDS"].split(",")
        require(
            len(set(grid_order)) == len(grid_order) and set(grid_order) <= {"global", "regional"},
            "Invalid selected grids.",
        )
    if os.environ.get("CAUSAL_REVERSE_GRIDS") == "1":
        grid_order.reverse()
    for grid in grid_order:
        directory = out / grid
        directory.mkdir()
        env = dict(os.environ)
        env.update(
            ICON4PY_CAUSAL_CASE=json.dumps(spec),
            ICON4PY_CAUSAL_PLATFORM=args.platform,
            ICON4PY_CAUSAL_OUTPUT=str(directory),
            GT4PY_BUILD_CACHE_DIR=str(directory / "build"),
            ICON4PY_PROFILE_WARMUP="5",
            ICON4PY_PROFILE_SEED="20260910",
            ICON_GRID=f"icon_benchmark_{grid}",
        )
        pytest_args = [
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
            f"--grid=icon_benchmark_{grid}:{spec['levels']}",
            TARGET,
        ]
        env.update(
            ICON4PY_CAUSAL_MODE="timing",
            ICON4PY_PROFILE_MODE="timing",
            ICON4PY_PROFILE_ROUNDS="10",
            ICON4PY_PROFILE_REPORT=str(directory / "timing.{pid}.json"),
            GT4PY_METRICS_OUTPUT_PATH=str(directory / "timers.timing.{pid}.json"),
        )
        run(pytest_args, directory / "timing.log", env)
        timing_file = timing_report(directory)
        validate_report(timing_file, spec, grid, "timing")
        audits, owners = code_audits(directory / "build")
        check_intervention(spec, audits)
        dump(directory / "code_audit.json", audits)
        dump(directory / "kernel_owners.json", owners)
        run(
            [
                sys.executable,
                "amd_scripts/collect_dycore_profile_metadata.py",
                "--cache-dir",
                directory / "build",
                "--output",
                directory / "metadata.json",
                "--label",
                grid + str(spec["levels"]),
            ],
            directory / "metadata.log",
            env,
        )
        mode = "hip" if args.platform == "amd" else "cuda"
        env.update(
            ICON4PY_CAUSAL_MODE=mode,
            ICON4PY_PROFILE_MODE=mode,
            ICON4PY_CAUSAL_PROOF=str(timing_file),
            ICON4PY_PROFILE_ROUNDS="3",
        )
        arms = ("A", "B") if os.environ.get("CAUSAL_PAIRED_COUNTERS") == "1" else ("B",)
        if int(os.environ.get("CAUSAL_ORDER_SEED", "20260912")) % 2:
            arms = tuple(reversed(arms))
        counter_paths = {}
        for arm in arms:
            destination = directory if arm == "B" else directory / "native_counters"
            collect_counters(args.platform, spec, grid, destination, env, pytest_args, owners, arm)
            counter_paths[arm] = str((destination / "COUNTERS.json").relative_to(out))
        grids[grid] = dict(
            timing=str(timing_file.relative_to(out)),
            counters=counter_paths["B"],
            counter_arms=counter_paths,
        )
    final_sources = provenance(root)
    dump(out / "source_hashes.final.json", final_sources)
    initial_sources = json.loads((out / "source_hashes.json").read_text())
    changed = sorted(
        name
        for name in initial_sources.keys() | final_sources.keys()
        if initial_sources.get(name) != final_sources.get(name)
    )
    require(not changed, "Experiment sources changed: " + ", ".join(changed))
    dump(
        out / "manifest.json",
        dict(case=spec, platform=args.platform, grids=grids, status="collection_complete"),
    )
    (out / "COLLECTION_COMPLETE").touch()
    run([sys.executable, "amd_scripts/analyze_dycore_causal.py", out], out / "analysis.log")
    (out / "COMPLETE").touch()
    print(f"Complete: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("amd", "nvidia"), required=True)
    parser.add_argument("--case", type=int, choices=range(len(CASES)), required=True)
    execute(parser.parse_args())
