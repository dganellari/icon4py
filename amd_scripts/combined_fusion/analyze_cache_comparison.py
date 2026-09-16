#!/usr/bin/env python3
"""Summarize Nsight Compute --page raw CSVs without merging distinct launches.

Usage:
    python analyze_cache_comparison.py /path/to/experiment
    python analyze_cache_comparison.py global120_none.csv global120_all.csv

Writes cache_comparison.json and cache_comparison.md next to the input by default.
Counter timings are diagnostic; compare production timings from unprofiled runs.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pathlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any

CONFIG_PATTERN = re.compile(r"(?P<grid>global|regional)(?P<levels>\d+)_(?P<cache>none|all)(?=$|[_.-])")
L1 = "l1tex__t_"
L2 = "lts__t_sectors_srcunit_tex_op_"
COUNTERS = [
    *(f"{L1}requests_pipe_lsu_mem_global_op_{op}.sum" for op in ("ld", "st")),
    *(f"{L1}sectors_pipe_lsu_mem_global_op_{op}{hit}.sum"
      for op in ("ld", "st") for hit in ("", "_lookup_hit")),
    *(f"{L2}{op}{hit}.sum" for op in ("read", "write") for hit in ("", "_lookup_hit")),
    "dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
]
WARNINGS = [
    "Cross-grid raw totals are not comparable without matching program/domain work; "
    "regional and global execute different numbers of points and kernel branches.",
    "Counter-pass durations are diagnostic, not unprofiled production timings.",
    "Cache-control all flushes before each profiled kernel, removing reuse from both "
    "preceding programs and earlier kernels within the same program.",
    "DRAM bytes / L1 sector bytes is a traffic ratio, not a probability or an exact "
    "fraction of requests reaching HBM; the counter scopes and write behavior differ.",
]


def _number(value: str) -> int | float | None:
    try:
        number = Decimal(value.replace(",", "").strip())
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    return int(number) if number == number.to_integral() else float(number)


def _canonical(metric: str, value: str, unit: str) -> tuple[int | float | None, str]:
    number = _number(value)
    if number is None:
        return None, "Unavailable or nonfinite metric value."
    if number < 0:
        return None, "Negative metric value."
    unit = unit.strip()
    if metric.startswith("dram__bytes"):
        factors = {"byte": 1, "bytes": 1, "B": 1, "Kbyte": 1e3, "Mbyte": 1e6,
                   "Gbyte": 1e9, "Tbyte": 1e12, "KiB": 1024, "MiB": 1024**2,
                   "GiB": 1024**3, "TiB": 1024**4}
    elif metric == "gpu__time_duration.sum":
        factors = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3,
                   "second": 1, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}
    elif metric in COUNTERS:
        base = "request" if "requests_" in metric else "sector"
        factors = {base: 1, base + "s": 1, "K" + base: 1e3,
                   "M" + base: 1e6, "G" + base: 1e9, "T" + base: 1e12}
    else:
        # Preserve other metrics for inspection; no cross-unit aggregation is performed.
        return number, ""
    # NCU raw-page exports leave request-count units blank.
    if unit == "" and metric in COUNTERS and "requests_" in metric:
        unit = "request"
    if unit not in factors:
        return None, f"Unsupported unit '{unit}' for '{metric}'."
    converted = number * factors[unit]
    if isinstance(converted, float) and not math.isfinite(converted):
        return None, "Nonfinite converted metric value."
    return converted, ""


def _csv_rows(path: pathlib.Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read long metric rows or raw-page wide rows with a separate units row."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines(keepends=True)
    header = None
    issues = []
    wide = False
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if {"ID", "Metric Name", "Metric Unit", "Metric Value"} <= set(fields):
            header = index
            break
        if {"ID", "Kernel Name"} <= set(fields) and set(COUNTERS).intersection(fields):
            header, wide = index, True
            break
        if "ERROR" in line.upper() or "ERR_NVGPUCTRPERM" in line:
            issues.append(line.strip())
    if header is None:
        return [], issues + ["No Nsight Compute metric CSV header found."]
    reader = csv.DictReader(io.StringIO("".join(lines[header:])))
    rows = []
    if wide:
        units = next(reader, None)
        if units is None or units.get("ID", "") != "":
            return [], issues + ["Wide Nsight CSV is missing its units row."]
        if None in units or any(value is None for value in units.values()):
            return [], issues + ["Malformed wide Nsight CSV units row."]
        identity_columns = ("ID", "Process ID", "Context", "Stream", "Device",
                            "Kernel Name", "Block Size", "Grid Size", "CC",
                            "Host Name", "Process Name")
        metrics = [name for name in COUNTERS if name in (reader.fieldnames or [])]
        for row in reader:
            if not row.get("ID", "").isdigit():
                if "ERROR" in str(row).upper():
                    issues.append(str(row))
                continue
            if None in row or any(value is None for value in row.values()):
                issues.append(f"Malformed wide CSV row for launch {row.get('ID')}.")
                continue
            identity = {name: row.get(name, "") for name in identity_columns}
            for metric in metrics:
                rows.append({**identity, "Metric Name": metric,
                             "Metric Unit": units[metric], "Metric Value": row[metric]})
        return rows, issues
    for row in reader:
        if row.get("Metric Name") == "Metric Name":
            continue
        if not row.get("Metric Name") or row.get("Metric Value") is None:
            if "ERROR" in str(row).upper():
                issues.append(str(row))
            continue
        rows.append(row)
    return rows, issues


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator else None


def _derived(totals: dict[str, int | float | None]) -> dict[str, int | float | None]:
    out = {}
    for op, label in (("ld", "load"), ("st", "store")):
        sectors = totals.get(f"{L1}sectors_pipe_lsu_mem_global_op_{op}.sum")
        hits = totals.get(f"{L1}sectors_pipe_lsu_mem_global_op_{op}_lookup_hit.sum")
        out[f"l1_{label}_hit_fraction"] = _ratio(hits, sectors)
        out[f"l1_{label}_sectors_per_request"] = _ratio(
            sectors, totals.get(f"{L1}requests_pipe_lsu_mem_global_op_{op}.sum"))
    for op in ("read", "write"):
        sectors = totals.get(f"{L2}{op}.sum")
        hits = totals.get(f"{L2}{op}_lookup_hit.sum")
        out[f"l2_{op}_hit_fraction"] = _ratio(hits, sectors)
        out[f"l2_{op}_miss_sectors"] = sectors - hits if sectors is not None and hits is not None else None
    reads, writes = totals.get("dram__bytes_read.sum"), totals.get("dram__bytes_write.sum")
    out["dram_total_bytes"] = reads + writes if reads is not None and writes is not None else None
    loads = totals.get(f"{L1}sectors_pipe_lsu_mem_global_op_ld.sum")
    out["dram_read_bytes_per_l1_load_sector_byte"] = _ratio(reads, 32 * loads if loads is not None else None)
    out["counter_duration_seconds"] = totals.get("gpu__time_duration.sum")
    return out


def _attach_program_metadata(path: pathlib.Path, config: re.Match | None,
                             kernels: list[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        return {"path": None, "note": "Grid/levels could not be determined."}
    basename = f"{config['grid']}{config['levels']}_metadata.json"
    candidates = [parent / basename for parent in list(path.resolve().parents)[:3]]
    metadata_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if metadata_path is None:
        return {"path": None, "note": "No optional per-grid cache metadata found."}
    try:
        metadata = json.loads(metadata_path.read_text())
        mapping = metadata["cache"]["kernel_to_programs"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"path": str(metadata_path), "note": f"Metadata could not be read: {error}"}
    for kernel in kernels:
        raw_name = kernel["kernel_name"]
        bare_name = raw_name.split("(", 1)[0].strip()
        match = re.search(r"(?:^|\s)([A-Za-z_]\w*)(?:<.*>)?$", bare_name)
        owner_key = raw_name if raw_name in mapping else match.group(1) if match else bare_name
        owners = sorted(set(mapping.get(owner_key, [])))
        kernel["program_candidates"] = owners
        kernel["attribution"] = (
            "unique-cache-owner" if len(owners) == 1 else
            "ambiguous-cache-owners" if owners else "unmapped"
        )
    return {"path": str(metadata_path), "note":
            "Cache ownership is a candidate mapping, not proof of executed program identity; "
            "multiple owners remain ambiguous."}


def load_run(path: pathlib.Path) -> dict[str, Any]:
    config = CONFIG_PATTERN.search(path.name)
    rows, issues = _csv_rows(path)
    kernels: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        identity = {name: row.get(name, "") for name in ("ID", "Process ID", "Context", "Stream", "Device")}
        if not identity["ID"]:
            issues.append("Skipped metric row without launch ID.")
            continue
        key = tuple(identity.values())
        launch = kernels.setdefault(key, {
            "identity": identity,
            "kernel_name": row.get("Kernel Name", ""),
            "launch_metadata": {name: row.get(name, "") for name in ("Block Size", "Grid Size", "CC", "Host Name", "Process Name")},
            "metrics": {},
            "issues": [],
        })
        if launch["kernel_name"] != row.get("Kernel Name", ""):
            launch["issues"].append("Conflicting kernel names for the same launch identity.")
        metric = row["Metric Name"]
        value, error = _canonical(metric, row["Metric Value"], row.get("Metric Unit", ""))
        sample = {"raw_value": row["Metric Value"], "raw_unit": row.get("Metric Unit", ""), "value": value}
        existing = launch["metrics"].get(metric)
        if existing is None:
            launch["metrics"][metric] = {"value": value, "observations": [sample]}
        else:
            existing["observations"].append(sample)
            if existing["value"] != value:
                existing["value"] = None
                launch["issues"].append(f"Conflicting duplicate metric '{metric}'.")
        if error:
            launch["issues"].append(f"{metric}: {error}")
    totals = {}
    for metric in COUNTERS:
        samples = [launch["metrics"].get(metric, {}).get("value") for launch in kernels.values()]
        totals[metric] = sum(samples) if samples and all(v is not None for v in samples) else None
    missing = [name for name, value in totals.items() if value is None]
    if missing:
        issues.append("Required metrics missing/invalid in at least one launch: " + ", ".join(missing))
    derived = _derived(totals)
    derived_warnings = []
    invalid_derived = {}
    for name, value in derived.items():
        if value is not None and (name.endswith("hit_fraction") and not 0 <= value <= 1 or name.endswith("miss_sectors") and value < 0):
            derived_warnings.append(
                f"Out-of-range derived metric '{name}': {value}; withheld. "
                "Raw counts retained; independent DRAM byte comparisons remain usable."
            )
            invalid_derived[name] = value
    for name in invalid_derived:
        derived[name] = None
    if any(launch["issues"] for launch in kernels.values()):
        issues.append("Some launch metric rows are invalid or ambiguous; inspect kernels[].issues.")
    if config is None:
        issues.append("Filename must contain global<levels>_none/all or regional<levels>_none/all.")
    program_metadata = _attach_program_metadata(path, config, list(kernels.values()))
    return {
        "path": str(path.resolve()),
        "program_metadata": program_metadata,
        "grid": config["grid"] if config else None,
        "levels": int(config["levels"]) if config else None,
        "cache_control": config["cache"] if config else None,
        "pair_key": str(path.resolve().parent / CONFIG_PATTERN.sub("GRID_LEVELS_CACHE", path.name)) if config else None,
        "launch_count": len(kernels),
        "totals": totals, "derived": derived,
        "kernels": list(kernels.values()), "issues": issues,
        "warnings": derived_warnings, "invalid_derived": invalid_derived,
        "valid": bool(kernels) and not issues,
    }


def compare_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for run in runs:
        if run["grid"]:
            groups.setdefault((run["grid"], run["levels"], run["pair_key"]), []).append(run)
    comparisons = []
    for (grid, levels, pair_key), members in sorted(groups.items()):
        none = [r for r in members if r["cache_control"] == "none"]
        all_ = [r for r in members if r["cache_control"] == "all"]
        result = {"grid": grid, "levels": levels, "pair_key": pair_key, "none_over_all_dram_bytes": None, "issues": []}
        if len(none) != 1 or len(all_) != 1:
            result["issues"].append(f"Need exactly one run per cache condition; found none={len(none)}, all={len(all_)}.")
        else:
            warm, cold = none[0], all_[0]
            result["none_path"], result["all_path"] = warm["path"], cold["path"]
            # Names alone cannot identify launches, but the full ordered launch signature
            # can expose missing/extra kernels and mismatched workloads between conditions.
            def signature(run):
                return [(k["kernel_name"], k["launch_metadata"]["Grid Size"], k["launch_metadata"]["Block Size"]) for k in run["kernels"]]
            if signature(warm) != signature(cold):
                result["issues"].append("Launch names/order/grid/block dimensions differ; workload matching is unverified.")
            if not warm["valid"] or not cold["valid"]:
                result["issues"].append("At least one input run is incomplete or invalid; ratio withheld.")
            if not result["issues"]:
                result["none_over_all_dram_bytes"] = _ratio(warm["derived"]["dram_total_bytes"], cold["derived"]["dram_total_bytes"])
                if result["none_over_all_dram_bytes"] is None:
                    result["issues"].append("Zero/missing flushed DRAM denominator; ratio undefined.")
        comparisons.append(result)
    return comparisons


def _fmt(value: int | float | None, percent: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.1%}" if percent else f"{value:,.4g}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# GH200 cache comparison", "", "Raw counters are summed across distinct launches; cache hit fractions are weighted by sectors.", "",
             "| Grid | Cache | Launches | DRAM bytes | L1 load hit | L2 read hit | Load sectors/request | Status |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for run in report["runs"]:
        d = run["derived"]
        lines.append(f"| {run['grid'] or '?'} / {run['levels'] or '?'} | {run['cache_control'] or '?'} | {run['launch_count']} | {_fmt(d['dram_total_bytes'])} | {_fmt(d['l1_load_hit_fraction'], True)} | {_fmt(d['l2_read_hit_fraction'], True)} | {_fmt(d['l1_load_sectors_per_request'])} | {('complete; rate warnings' if run.get('warnings') else 'complete') if run['valid'] else 'inspect issues'} |")
    lines += ["", "| Within-grid pair | DRAM bytes, none / all | Status |", "| --- | ---: | --- |"]
    for item in report["comparisons"]:
        lines.append(f"| {item['grid']} / {item['levels']} | {_fmt(item['none_over_all_dram_bytes'])} | {'; '.join(item['issues']) or 'matched launches'} |")
    lines += ["", "## Interpretation limits", ""]
    lines += [f"- {warning}" for warning in report["warnings"]]
    lines += ["", "## Inputs", ""]
    for run in report["runs"]:
        lines.append(f"- `{run['path']}`")
        for issue in run["issues"] + run.get("warnings", []):
            lines.append(f"  - {issue}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="JSON report path.")
    parser.add_argument("--markdown", type=pathlib.Path, help="Markdown report path.")
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        if item.is_dir():
            paths.extend(p for p in sorted(item.rglob("*.csv")) if CONFIG_PATTERN.search(p.name))
        elif item.is_file():
            paths.append(item)
        else:
            parser.error(f"Input does not exist: {item}")
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths:
        parser.error("No CSVs with global<levels>_none/all or regional<levels>_none/all filenames found.")
    runs = [load_run(path) for path in paths]
    report = {"schema_version": 1, "runs": runs, "comparisons": compare_runs(runs), "warnings": WARNINGS}
    output = args.output_dir or (args.inputs[0] if args.inputs[0].is_dir() else args.inputs[0].parent)
    output.mkdir(parents=True, exist_ok=True)
    json_path = args.output or output / "cache_comparison.json"
    markdown_path = args.markdown or output / "cache_comparison.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    markdown = render_markdown(report)
    markdown_path.write_text(markdown)
    print(markdown)
    print(f"Wrote {json_path} and {markdown_path}")
    return 0 if all(r["valid"] for r in runs) and all(not c["issues"] for c in report["comparisons"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
