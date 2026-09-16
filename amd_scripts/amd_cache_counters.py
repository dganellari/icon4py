#!/usr/bin/env python3
"""MI300A counter collection support, validation and per-program reporting.

Byte formulas describe the TCC/EA interface, not physical HBM transactions.
Definitions: https://instinct.docs.amd.com/develop/gpu-arch/mi300-mi200-performance-counters.html
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from collections import defaultdict


GROUPS = {
    "fabric_read": ["TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum", "TCC_BUBBLE_sum"],
    "fabric_write": ["TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum"],
    "l2": ["TCC_HIT_sum", "TCC_MISS_sum", "TCC_REQ_sum"],
    "l2_types": ["TCC_READ_sum", "TCC_WRITE_sum", "TCC_ATOMIC_sum"],
    "l1": ["TCP_TA_TCP_STATE_READ_sum", "TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TCC_READ_REQ_sum"],
    "l1_stall": ["TCP_PENDING_STALL_CYCLES_sum", "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum"],
    "fabric_destination": ["TCC_EA0_RDREQ_DRAM_sum", "TCC_EA0_WRREQ_DRAM_sum"],
    "fabric_latency": ["TCC_EA0_RDREQ_LEVEL_sum", "TCC_EA0_RDREQ_sum"],
    "fabric_stall": ["TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum", "TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum"],
}
TARGETS = {"compute_horizontal_velocity_quantities_and_fluxes", "compute_rho_theta_pgrad_and_update_vn"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ProfilerControl:
    """SDK ROCTx trace control; requires rocprofv3 --selected-regions."""

    def __init__(self):
        override = os.environ.get("ICON4PY_ROCTX_LIBRARY")
        directories = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
        for name in ("ROCM_PATH", "ROCM_HOME"):
            if os.environ.get(name):
                directories.extend(str(Path(os.environ[name]) / sub) for sub in ("lib", "lib64"))
        candidates = [override] if override else [ctypes.util.find_library("rocprofiler-sdk-roctx")]
        if not override:
            for directory in directories:
                candidates.extend(str(p) for p in sorted(Path(directory).glob("librocprofiler-sdk-roctx.so*")))
            candidates.append("librocprofiler-sdk-roctx.so")
        errors = []
        for path in dict.fromkeys(p for p in candidates if p):
            try:
                lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                for name in ("roctxProfilerPause", "roctxProfilerResume"):
                    function = getattr(lib, name)
                    function.argtypes = [ctypes.c_uint64]
                    function.restype = ctypes.c_int
                self.lib, self.path = lib, path
                self.active = False
                break
            except (OSError, AttributeError) as error:
                errors.append(f"{path}: {error}")
        else:
            raise RuntimeError("Cannot load SDK ROCTx profiler control: " + "; ".join(errors))

    def pause(self):
        # SDK 1.1.0 aborts the process when stopping an inactive context.
        require(self.active, "Cannot pause an inactive ROCTx trace context.")
        require(self.lib.roctxProfilerPause(0) == 0, "ROCTx profiler pause failed.")
        self.active = False

    def resume(self):
        require(not self.active, "ROCTx trace context is already active.")
        require(self.lib.roctxProfilerResume(0) == 0, "ROCTx profiler resume failed.")
        self.active = True


def configure(available, destination):
    text = Path(available).read_text()
    missing = [m for m in sorted({m for group in GROUPS.values() for m in group})
               if not re.search(r"(?<!\w)" + re.escape(m) + r"(?!\w)", text)]
    require(not missing, "Required MI300A counters unavailable: " + ", ".join(missing))
    Path(destination).write_text("".join(f"{name}\t{' '.join(metrics)}\n" for name, metrics in GROUPS.items()))


def probe():
    # Same probe kernel runs twice outside each side of the measurement window.
    # Exactly one dispatch must survive in every counter group, proving gating.
    control = ProfilerControl()
    import cupy as cp
    require(cp.cuda.runtime.is_hip, "Expected HIP CuPy on MI300A.")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    arch = props.get("gcnArchName", b"")
    arch = arch.decode() if isinstance(arch, bytes) else str(arch)
    require(arch.startswith("gfx942"), f"Expected gfx942, got {arch!r}.")
    array = cp.zeros(65536, dtype=cp.float64)
    kernel = cp.RawKernel('extern "C" __global__ void dycore_cache_probe(double* x) '
                         '{ int i=blockIdx.x*blockDim.x+threadIdx.x; x[i]=x[i]+1.0; }',
                         "dycore_cache_probe")
    def launch():
        kernel((256,), (256,), (array,))
        cp.cuda.runtime.deviceSynchronize()
    launch()
    launch()
    control.resume()
    try:
        launch()
    finally:
        control.pause()
    launch()
    launch()
    print(json.dumps({"probe": "complete", "roctx_library": control.path, "arch": arch}))


def only_file(directory, pattern):
    files = list(Path(directory).rglob(pattern))
    require(len(files) == 1, f"Expected one {pattern} in {directory}, found {len(files)}.")
    return files[0]


def counter_dispatches(directory, metrics, name_pattern):
    path = only_file(directory, "*counter_collection.csv")
    dispatches = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            require(re.search(name_pattern, row["Kernel_Name"]), f"Unexpected counter kernel in {path}.")
            key = (row["Process_Id"], row["Agent_Id"], row["Dispatch_Id"])
            identity = (row["Kernel_Name"], int(row["Grid_Size"]), int(row["Workgroup_Size"]))
            entry = dispatches.setdefault(key, {"identity": identity, "counters": {},
                                                "dispatch_id": row["Dispatch_Id"], "agent": row["Agent_Id"]})
            require(entry["identity"] == identity, "Conflicting dispatch metadata.")
            name = row["Counter_Name"]
            require(name not in entry["counters"], f"Duplicate counter {name} for dispatch {key}.")
            value = float(row["Counter_Value"])
            require(math.isfinite(value) and value >= 0, f"Invalid counter value: {row}.")
            entry["counters"][name] = value
    require(dispatches, f"No counters collected in {path}.")
    require(len({key[:2] for key in dispatches}) == 1, "Multiple processes/devices in counter pass.")
    result = sorted(dispatches.values(), key=lambda d: int(d["dispatch_id"]))
    for entry in result:
        require(set(entry["counters"]) == set(metrics), f"Missing/unexpected counters: {entry}.")
    return result


def trace_dispatches(directory, name_pattern):
    path = only_file(directory, "*kernel_trace.csv")
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if re.search(name_pattern, r["Kernel_Name"])]
    require(rows, f"No matching trace dispatches in {path}.")
    require(len({(r["Agent_Id"], r["Queue_Id"]) for r in rows}) == 1, "Multiple devices/queues in trace.")
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    return rows


def trace_identity(row):
    return (row["Kernel_Name"], math.prod(int(row[f"Grid_Size_{x}"]) for x in "XYZ"),
            math.prod(int(row[f"Workgroup_Size_{x}"]) for x in "XYZ"))


def full_trace_identity(row):
    return (row["Kernel_Name"], *(int(row[f"{kind}_Size_{x}"])
            for kind in ("Grid", "Workgroup") for x in "XYZ"))


def match_counter_trace(counters, traces, label):
    # Compare IDs only within the same process/pass. Identical warmup kernels
    # have the same names and launch shapes, so those alone cannot prove scope.
    actual = [(c["agent"], c["dispatch_id"], c["identity"]) for c in counters]
    expected = [(r["Agent_Id"], r["Dispatch_Id"], trace_identity(r)) for r in traces]
    require(actual == expected, f"Counter/trace dispatch mismatch: {label}.")


def validate_trace(directory, rounds):
    require(rounds > 0, "Capture rounds must be positive.")
    traces = trace_dispatches(directory, r"fieldop")
    signatures = [full_trace_identity(r) for r in traces]
    require(len(traces) % rounds == 0, "Trace does not contain complete repeated calls.")
    per_call = len(traces) // rounds
    require(all(signatures[i * per_call:(i + 1) * per_call] == signatures[:per_call]
                for i in range(rounds)), "Per-call trace sequences differ.")
    # A common per-kernel iteration range is valid only with one launch of each
    # kernel per call. Same-pass ID matching also detects setup-induced offsets.
    names = [r["Kernel_Name"] for r in traces[:per_call]]
    require(len(set(names)) == per_call,
            "Repeated kernel within a granule: a common kernel iteration range is unsafe.")
    return traces


def validate_counter_pass(directory, group, reference, rounds):
    traces = validate_trace(directory, rounds)
    expected = validate_trace(reference, rounds)
    require([full_trace_identity(r) for r in traces] == [full_trace_identity(r) for r in expected],
            f"Launch sequence mismatch: {directory}.")
    counters = counter_dispatches(directory, GROUPS[group], r"fieldop")
    match_counter_trace(counters, traces, directory)
    return counters


def validate_probe(directory, group):
    counters = counter_dispatches(directory, GROUPS[group], r"dycore_cache_probe")
    traces = trace_dispatches(directory, r"dycore_cache_probe")
    require(len(counters) == len(traces) == 1,
            f"Capture gating failed for {group}: expected one captured probe, got {len(counters)}/{len(traces)}.")
    match_counter_trace(counters, traces, group)
    print(f"Validated {group}: one measured probe, four excluded probes.")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def gigabytes_text(value):
    return "n/a" if value is None else f"{value / 1e9:.6f}"


def derived(groups):
    read, write, l2, l1 = (groups[g] for g in ("fabric_read", "fabric_write", "l2", "l1"))
    nr, nr32 = read["TCC_EA0_RDREQ_sum"], read["TCC_EA0_RDREQ_32B_sum"]
    nr128 = read["TCC_BUBBLE_sum"]
    nw, nw64 = write["TCC_EA0_WRREQ_sum"], write["TCC_EA0_WRREQ_64B_sum"]
    # gfx942 FETCH_SIZE includes 128-byte reads, counted by TCC_BUBBLE.
    # ROCm documents inconsistent subsets on MI300; retain raw values and
    # withhold affected byte estimates instead of clamping or losing the run.
    issues = []
    read_bytes = None
    write_bytes = None
    if nr32 + nr128 > nr:
        issues.append("L2-to-fabric read bytes withheld: 32-byte plus 128-byte request counts "
                      "exceed total read requests (known MI300 counter inconsistency).")
    else:
        read_bytes = 128 * nr128 + 64 * (nr - nr128 - nr32) + 32 * nr32
    if nw64 > nw:
        issues.append("L2-to-fabric write bytes withheld: 64-byte request count "
                      "exceeds total write requests.")
    else:
        write_bytes = 32 * (nw - nw64) + 64 * nw64
    return {
        "l2_fabric_read_bytes": read_bytes,
        "l2_fabric_write_bytes": write_bytes,
        "byte_counter_issues": issues,
        "l2_all_request_hit_fraction": ratio(l2["TCC_HIT_sum"], l2["TCC_HIT_sum"] + l2["TCC_MISS_sum"]),
        "l1_accesses_per_read_request": ratio(l1["TCP_TOTAL_CACHE_ACCESSES_sum"], l1["TCP_TA_TCP_STATE_READ_sum"]),
        "l1_tcc_read_requests_per_access": ratio(l1["TCP_TCC_READ_REQ_sum"], l1["TCP_TOTAL_CACHE_ACCESSES_sum"]),
        "physical_hbm_read_bytes": None,
        "physical_hbm_write_bytes": None,
    }


def read_window(root, label, rounds, warmup, mode):
    file = only_file(root / "windows", label + ".*.json")
    window = json.loads(file.read_text())
    require(window["status"] == "complete" and window["rounds_completed"] == rounds
            and window["warmup_completed"] == warmup and window["mode"] == mode,
            f"Incomplete/wrong measurement window: {file}.")
    require(window["gpu_api"] == "hip", "Expected HIP measurement window.")
    return window


def resolve_trace_owner(row, candidates, directory):
    """Resolve cache-name collisions using an enclosing same-thread SDFG range."""
    if len(candidates) == 1:
        return candidates[0], "unique-cache-owner"
    with only_file(directory, "*marker_api_trace.csv").open() as stream:
        markers = list(csv.DictReader(stream))
    owners = {m["Function"][5:] for m in markers
              if m["Function"].startswith("sdfg_")
              and m["Thread_Id"] == row["Thread_Id"]
              and int(m["Start_Timestamp"]) <= int(row["Start_Timestamp"])
              and int(row["End_Timestamp"]) <= int(m["End_Timestamp"])}
    require(len(owners) == 1 and owners <= set(candidates),
            f"Missing/ambiguous program ownership: {row['Kernel_Name']}: {candidates}; markers: {owners}.")
    return owners.pop(), "enclosing-sdfg-marker"


def analyze(root, levels, warmup, rounds, timing_rounds):
    root = Path(root)
    report = {"schema_version": 1, "vendor": "AMD", "grids": {}, "scope": {
        "traffic": "TCC/EA interface bytes; not physical HBM bytes (MALL service is not distinguished).",
        "cache_control": "rocprofv3 default; no equivalent of NCU all/none is asserted.",
        "timing": "Separate unprofiled fixed-call run. Counter-pass durations are not production timings.",
        "normalization": "Counters per full granule call, with per-program and per-kernel attribution. Grid domains differ.",
    }}
    device_keys = set()
    for grid in ("global", "regional"):
        prefix = f"{grid}{levels}"
        meta = json.loads((root / f"{prefix}_metadata.json").read_text())
        mapping = meta["cache"]["kernel_to_programs"]
        timing = read_window(root, prefix + "_timing", timing_rounds, warmup, "timing")
        trace_window = read_window(root, prefix + "_trace", rounds, warmup, "hip")
        trace = validate_trace(root / "traces" / prefix, rounds)
        per_call = len(trace) // rounds
        metrics_by_dispatch = [dict() for _ in trace]
        for group, metrics in GROUPS.items():
            label = f"{prefix}_{group}"
            window = read_window(root, label, rounds, warmup, "hip")
            require(window["input_rng_at_entry"] == timing["input_rng_at_entry"], "Input seeds/creation differ.")
            require(window["grid"] == timing["grid"], "Grid mismatch between passes.")
            device_keys.add((window["hostname"], window["device_id"], window["device_name"]))
            folder = root / "counters" / label
            counters = validate_counter_pass(folder, group, root / "traces" / prefix, rounds)
            for index, entry in enumerate(counters):
                metrics_by_dispatch[index][group] = entry["counters"]
        for w in (timing, trace_window):
            device_keys.add((w["hostname"], w["device_id"], w["device_name"]))
            require(w["input_rng_at_entry"] == timing["input_rng_at_entry"], "Trace input mismatch.")
        timer_file = only_file(root / "timers", prefix + "_timing.*.json")
        timers = json.loads(timer_file.read_text())
        times = defaultdict(float)
        for entry in timers.values():
            values = entry.get("metrics", {}).get("compute", [])
            if len(values) == warmup + timing_rounds:
                times[entry["metadata"]["name"]] += statistics.median(values[warmup:]) * 1000
        require(TARGETS <= times.keys(), "Target ordinary device timers missing.")
        programs = {}
        kernels = []
        for row, groups in zip(trace, metrics_by_dispatch):
            name = row["Kernel_Name"].split("(", 1)[0].strip()
            owners = mapping.get(name, [])
            owner, attribution = resolve_trace_owner(row, owners, root / "traces" / prefix)
            require(owner in times, f"No ordinary device timer for {owner}.")
            program = programs.setdefault(owner, {"raw_counters_per_call": {
                group: dict.fromkeys(metrics, 0.0) for group, metrics in GROUPS.items()},
                "device_median_ms": times[owner], "launches_per_call": 0.0})
            program["launches_per_call"] += 1 / rounds
            for group, metrics in groups.items():
                for metric, value in metrics.items():
                    program["raw_counters_per_call"][group][metric] += value / rounds
            kernels.append({"kernel": name, "program": owner, "trace": row,
                            "program_candidates": owners, "attribution": attribution,
                            "raw_counters": groups, "derived": derived(groups)})
        for owner, program in programs.items():
            program["derived"] = derived(program["raw_counters_per_call"])
            for kernel in (k for k in kernels if k["program"] == owner):
                for direction in ("read", "write"):
                    key = f"l2_fabric_{direction}_bytes"
                    if kernel["derived"][key] is None:
                        program["derived"][key] = None
                program["derived"]["byte_counter_issues"].extend(kernel["derived"]["byte_counter_issues"])
            program["derived"]["byte_counter_issues"] = sorted(set(program["derived"]["byte_counter_issues"]))
        report["grids"][grid] = {"launches_per_call": per_call, "programs": programs, "kernels": kernels,
            "wall_median_ms": timing["median_wall_ms"], "sum_program_device_medians_ms": sum(times.values()),
            "ordinary_program_device_medians_ms": dict(times), "input_rng": timing["input_rng_at_entry"]}
    require(len(device_keys) == 1, "Grid/pass measurements used different nodes or devices.")
    report["device"] = list(device_keys.pop())
    lines = ["# MI300A dycore cache collection", "", *report["scope"].values(), "",
             "| Grid / program | Device ms | L2 hit (all requests) | L2→fabric read GB/call | L2→fabric write GB/call |",
             "| --- | ---: | ---: | ---: | ---: |"]
    byte_notes = []
    for grid, result in report["grids"].items():
        for name, program in result["programs"].items():
            d = program["derived"]
            hit = d["l2_all_request_hit_fraction"]
            hit_text = "n/a" if hit is None else f"{100 * hit:.2f}%"
            lines.append(f"| {grid} / {name} | {program['device_median_ms']:.6f} | {hit_text} | "
                         f"{gigabytes_text(d['l2_fabric_read_bytes'])} | {gigabytes_text(d['l2_fabric_write_bytes'])} |")
            for issue in d["byte_counter_issues"]:
                byte_notes.append(f"{grid}/{name}: {issue}")
    if byte_notes:
        lines.extend(["", *byte_notes])
    (root / "analysis_amd.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "ANALYSIS_AMD.md").write_text("\n".join(lines) + "\n")
    print(f"Validated both grids; wrote {root / 'ANALYSIS_AMD.md'}")


def compare(root, nvidia_root, levels):
    root, nvidia_root = Path(root), Path(nvidia_root)
    amd = json.loads((root / "analysis_amd.json").read_text())
    source = nvidia_root / "analysis_recovered.json"
    nvidia = json.loads(source.read_text())
    output = {"schema_version": 1, "amd_run": str(root), "nvidia_run": str(nvidia_root),
              "nvidia_counter_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": amd["scope"], "grids": {}}
    lines = ["# MI300A / GH200 dycore comparison", "",
             "Device times come from ordinary runs. AMD fabric bytes and GH200 HBM bytes are different measurement boundaries.",
             "AMD profiler cache policy is not asserted equivalent to either NCU arm; both GH200 arms are retained.",
             "No traffic ratio or cross-vendor cache-hit subtraction is computed. Raw totals across grids require domain normalization.", "",
             "| Grid / program | AMD device ms | GH200 device ms | AMD/GH200 time | AMD L2→fabric reads GB | GH200 L2 read misses GB, flushed | GH200 HBM reads GB, flushed / preserved |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for grid in ("global", "regional"):
        prefix = f"{grid}{levels}"
        window = json.loads(only_file(nvidia_root / "windows", prefix + "_timing.*.json").read_text())
        require(window["status"] == "complete", "Incomplete GH200 timing reference.")
        require(window["input_rng_at_entry"] == amd["grids"][grid]["input_rng"], "Cross-vendor input seed/setup mismatch.")
        warmup, count = window["warmup_completed"], window["rounds_completed"]
        timing_json = json.loads(only_file(nvidia_root / "timers", prefix + "_timing.*.json").read_text())
        times = defaultdict(float)
        for entry in timing_json.values():
            values = entry.get("metrics", {}).get("compute", [])
            if len(values) == warmup + count:
                times[entry["metadata"]["name"]] += statistics.median(values[warmup:]) * 1000
        revisions = {}
        for vendor, path in (("amd", root), ("nvidia", nvidia_root)):
            metadata = json.loads((path / f"{prefix}_metadata.json").read_text())
            revisions[vendor] = {name: value.get("commit") for name, value in metadata["repositories"].items()}
        programs = {}
        for name, program in amd["grids"][grid]["programs"].items():
            require(name in times, f"Missing GH200 ordinary timer for {grid}/{name}.")
            entry = {"amd_device_ms": program["device_median_ms"], "gh200_device_ms": times[name],
                     "amd_over_gh200_device_time": program["device_median_ms"] / times[name],
                     "amd": program, "gh200": {}}
            for arm in ("all", "none"):
                runs = [r for r in nvidia["runs"] if r["grid"] == grid and r["cache_control"] == arm]
                require(len(runs) == 1, f"Expected one GH200 {grid}/{arm} reference.")
                run = runs[0]
                require(not run["issues"], f"Invalid GH200 read-counter reference: {run['issues']}.")
                windows = [json.loads(p.read_text()) for p in (nvidia_root / "windows").glob(f"{prefix}_{arm}_r1.*.json")]
                require(windows and all(w["status"] == "complete" and w["warmup_completed"] == warmup
                        and w["input_rng_at_entry"] == window["input_rng_at_entry"] for w in windows), "GH200 replay mismatch.")
                counts = {w["rounds_completed"] for w in windows}
                require(len(counts) == 1, "GH200 replay call counts differ.")
                calls = counts.pop()
                # The capture excludes setup; retain the original candidates and
                # explicitly record resolution against programs timed every call.
                kernels = [k for k in run["kernels"]
                           if set(k.get("program_candidates", [])) & times.keys() == {name}]
                require(kernels, f"No uniquely mapped GH200 kernels for {name}.")
                def total(metric):
                    return sum(k["metrics"][metric]["value"] for k in kernels) / calls
                read = total("lts__t_sectors_srcunit_tex_op_read.sum")
                hits = total("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum")
                require(0 <= hits <= read, "Invalid GH200 L2 read ratio.")
                entry["gh200"][arm] = {
                    "attribution": "unique timed hot-loop owner among saved cache candidates",
                    "kernel_candidates": [{"kernel": k["kernel_name"],
                                            "candidates": k["program_candidates"]} for k in kernels],
                    "l2_tex_read_hit_fraction": ratio(hits, read),
                    "l2_tex_read_miss_sector_bytes": 32 * (read - hits),
                    "physical_hbm_read_bytes": total("dram__bytes_read.sum"),
                    "physical_hbm_write_bytes": total("dram__bytes_write.sum"),
                }
            programs[name] = entry
            cold, warm = entry["gh200"]["all"], entry["gh200"]["none"]
            lines.append(f"| {grid} / {name} | {entry['amd_device_ms']:.6f} | {times[name]:.6f} | "
                         f"{entry['amd_over_gh200_device_time']:.3f} | {gigabytes_text(program['derived']['l2_fabric_read_bytes'])} | "
                         f"{cold['l2_tex_read_miss_sector_bytes']/1e9:.6f} | "
                         f"{cold['physical_hbm_read_bytes']/1e9:.6f} / {warm['physical_hbm_read_bytes']/1e9:.6f} |")
        output["grids"][grid] = {"programs": programs, "revisions": revisions,
            "commits_match": revisions["amd"] == revisions["nvidia"],
            "amd_wall_ms": amd["grids"][grid]["wall_median_ms"], "gh200_wall_ms": window["median_wall_ms"]}
        if revisions["amd"] != revisions["nvidia"]:
            lines.append(f"\nRevision mismatch on {grid}: inspect saved metadata and patches before attributing differences to hardware.\n")
    (root / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure")
    setup.add_argument("available")
    setup.add_argument("destination")
    commands.add_parser("probe")
    check = commands.add_parser("validate-probe")
    check.add_argument("directory")
    check.add_argument("group", choices=GROUPS)
    trace_check = commands.add_parser("validate-trace")
    trace_check.add_argument("directory")
    trace_check.add_argument("--rounds", type=int, default=3)
    pass_check = commands.add_parser("validate-pass")
    pass_check.add_argument("directory")
    pass_check.add_argument("group", choices=GROUPS)
    pass_check.add_argument("reference")
    pass_check.add_argument("--rounds", type=int, default=3)
    result = commands.add_parser("analyze")
    result.add_argument("directory")
    result.add_argument("--levels", type=int, default=120)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--rounds", type=int, default=3)
    result.add_argument("--timing-rounds", type=int, default=50)
    comparison = commands.add_parser("compare")
    comparison.add_argument("directory")
    comparison.add_argument("nvidia_directory")
    comparison.add_argument("--levels", type=int, default=120)
    args = parser.parse_args()
    if args.command == "configure":
        configure(args.available, args.destination)
    elif args.command == "probe":
        probe()
    elif args.command == "validate-probe":
        validate_probe(args.directory, args.group)
    elif args.command == "validate-trace":
        validate_trace(args.directory, args.rounds)
    elif args.command == "validate-pass":
        validate_counter_pass(args.directory, args.group, args.reference, args.rounds)
    elif args.command == "analyze":
        analyze(args.directory, args.levels, args.warmup, args.rounds, args.timing_rounds)
    else:
        compare(args.directory, args.nvidia_directory, args.levels)


if __name__ == "__main__":
    main()
