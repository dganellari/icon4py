#!/usr/bin/env python3
"""Capture provenance and kernel ownership for a user-submitted dycore profile job.

Run from the icon4py checkout before and after profiling, using different output
names. This reads an existing GT4Py cache; it never builds or changes that cache.
Generated source is hashed, while tracked local code changes are saved as patches.
The kernel map deliberately keeps ambiguous names instead of guessing ownership.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict


CODE_SUFFIXES = {".py", ".sh", ".bash", ".toml", ".yaml", ".yml", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip", ".c", ".f90", ".cmake"}
SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}
ANNOTATIONS = {"__launch_bounds__", "__attribute__", "__declspec", "alignas", "decltype", "__maxnreg__"}
ENV_NAMES = (
    "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURMD_NODENAME", "SLURM_SUBMIT_DIR",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "GT4PY_BUILD_CACHE_DIR", "GT4PY_ADD_GPU_TRACE_MARKERS", "GT4PY_COLLECT_METRICS_LEVEL",
    "GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE", "ICON4PY_BACKEND_WORKSPACE_SIZE",
    "DACE_compiler_cuda_chiplet_number", "DACE_compiler_build_folder_mode",
    "CUDAFLAGS", "HIPFLAGS", "ROCM_HOME", "ROCM_PATH", "ROCM_VERSION",
    "GRID", "LEVELS", "VERSION_SUFFIX", "ROUNDS", "METRICS", "KERNEL_REGEX",
    "REPLAY", "CACHE_CONTROL", "MODE", "TAG",
    "ICON_GRID", "ICON4PY_PROFILE_WARMUP", "ICON4PY_PROFILE_ROUNDS",
    "ICON4PY_PROFILE_MODE", "ICON4PY_PROFILE_SEED", "ICON4PY_PROFILE_REPORT",
    "GT4PY_METRICS_OUTPUT_PATH", "PYTHONHASHSEED", "PYTHONOPTIMIZE",
    "ICON4PY_ROCTX_LIBRARY",
)


def command(args: list[str], *, cwd: Path | None = None, timeout: int = 20) -> dict:
    """Never turn a failed optional probe into missing provenance without a reason."""
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)
        return {"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "error": str(error)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_names(source: str) -> list[str]:
    """Find CUDA/HIP global functions, skipping balanced compiler annotations.

    Handles multiline signatures, template definitions, launch_bounds before or
    after the function, and either ordering of ``void`` / ``__global__``.
    """
    source = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', " ", source, flags=re.S)
    found = set()
    for global_token in re.finditer(r"\b__global__\b", source):
        header = re.split(r"[{};]", source[global_token.end():global_token.end() + 12000], maxsplit=1)[0]
        position = 0
        while match := re.search(r"\b([A-Za-z_]\w*)\s*\(", header[position:]):
            name = match.group(1)
            opening = position + match.end() - 1
            depth, closing = 1, opening + 1
            while closing < len(header) and depth:
                depth += (header[closing] == "(") - (header[closing] == ")")
                closing += 1
            if name not in ANNOTATIONS:
                found.add(name)
                break
            position = closing
    return sorted(found)


def program_name(entry: str, source_path: Path) -> str:
    if "_pyext_" in entry:
        return entry.split("_pyext_", 1)[0]
    # Older cache entries are <program>_<hash>; the generated source filename
    # preserves the program name without its configuration hash.
    return re.sub(r"_(?:cuda|hip|cpu)$", "", source_path.stem)


def cache_metadata(cache_dir: Path) -> dict:
    root = cache_dir if cache_dir.name == ".gt4py_cache" else cache_dir / ".gt4py_cache"
    result = {"requested_directory": str(cache_dir), "source_root": str(root), "exists": root.is_dir(), "source_files": [], "kernel_to_programs": {}, "kernel_to_cache_entries": {}, "errors": []}
    ownership: dict[str, set[str]] = defaultdict(set)
    entries: dict[str, set[str]] = defaultdict(set)
    if not root.is_dir():
        result["errors"].append("Existing '.gt4py_cache' directory was not found.")
        return result
    for entry in sorted(root.iterdir()):
        cuda_dir = entry / "src" / "cuda"
        if not cuda_dir.is_dir():
            continue
        for path in sorted(cuda_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                names = kernel_names(path.read_text(errors="replace"))
                owner = program_name(entry.name, path)
                result["source_files"].append({"path": str(path.relative_to(root)), "program": owner, "cache_entry": entry.name, "bytes": path.stat().st_size, "sha256": sha256(path), "kernels": names})
                for name in names:
                    ownership[name].add(owner)
                    entries[name].add(entry.name)
            except OSError as error:
                result["errors"].append({"path": str(path), "error": str(error)})
    result["kernel_to_programs"] = {k: sorted(v) for k, v in sorted(ownership.items())}
    result["kernel_to_cache_entries"] = {k: sorted(v) for k, v in sorted(entries.items())}
    result["ambiguous_kernels"] = [k for k, v in sorted(ownership.items()) if len(v) > 1]
    result["mapping_scope"] = "All generated CUDA/HIP sources in this cache; not proof that every cached variant executed. Join to observed dispatches and retain ambiguity."
    if not ownership:
        result["errors"].append("No '__global__' kernel definitions or declarations were found.")
    return result


def git_metadata(path: Path, output: Path, name: str) -> dict:
    result = {"requested_directory": str(path)}
    top = command(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if top.get("returncode") != 0:
        result["error"] = top
        return result
    root = Path(top["stdout"].strip())
    result["root"] = str(root)
    for key, args in {
        "commit": ["rev-parse", "HEAD"],
        "branch": ["symbolic-ref", "--quiet", "--short", "HEAD"],
        "status": ["status", "--porcelain=v1", "--untracked-files=normal"],
    }.items():
        probe = command(["git", "-C", str(root), *args])
        result[key] = probe["stdout"].strip() if probe.get("returncode") == 0 else probe
    result["dirty"] = bool(result["status"]) if isinstance(result["status"], str) else None
    diff = command(["git", "-C", str(root), "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--"])
    if diff.get("returncode") == 0:
        patch = output.with_name(f"{output.stem}.{name}.patch")
        patch.write_text(diff["stdout"])
        result["tracked_diff"] = {"path": str(patch), "bytes": patch.stat().st_size, "sha256": sha256(patch), "scope": "Tracked staged and unstaged changes relative to HEAD; untracked source files are hashed below."}
    else:
        result["tracked_diff_error"] = diff
    paths = set()
    for args in (["diff", "--name-only", "-z", "HEAD", "--"], ["ls-files", "--others", "--exclude-standard", "-z"]):
        probe = command(["git", "-C", str(root), *args])
        if probe.get("returncode") == 0:
            paths.update(probe["stdout"].split("\0"))
        else:
            result.setdefault("source_list_errors", []).append(probe)
    result["changed_source_files"] = []
    for relative in sorted(paths):
        path = root / relative
        if not relative or not path.is_file() or (path.suffix.lower() not in CODE_SUFFIXES and path.name not in {"CMakeLists.txt", "Makefile"}):
            continue
        try:
            result["changed_source_files"].append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
        except OSError as error:
            result.setdefault("source_hash_errors", []).append({"path": relative, "error": str(error)})
    return result


def versions() -> dict:
    packages = {}
    for name in ("gt4py", "dace", "cupy", "cupy-cuda12x", "cupy-cuda13x", "numpy", "pytest", "pytest-benchmark"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    probes = {}
    for name in ("ncu", "nsys", "nvcc", "hipcc", "rocprofv3"):
        executable = shutil.which(name)
        if executable:
            probes[name] = command([executable, "--version"], timeout=10)
    modules = {}
    for name in ("gt4py", "dace", "cupy"):
        try:
            spec = importlib.util.find_spec(name)
            modules[name] = spec.origin if spec else None
        except (ImportError, ValueError) as error:
            modules[name] = {"error": str(error)}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        probes["nvidia-smi"] = command([nvidia_smi, "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], timeout=10)
    return {"python": sys.version, "python_executable": sys.executable, "packages": packages, "module_origins": modules, "tools": probes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Icon4py checkout (default: current directory).")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root_probe = command(["git", "-C", str(args.repo_root), "rev-parse", "--show-toplevel"])
    repo = Path(root_probe["stdout"].strip()) if root_probe.get("returncode") == 0 else args.repo_root.resolve()
    cache = cache_metadata(args.cache_dir.resolve())
    report = {
        "schema_version": 1, "label": args.label,
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "node": platform.node(), "machine": platform.machine(), "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "collector": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "environment": {name: os.environ[name] for name in ENV_NAMES if name in os.environ},
        "versions": versions(),
        "repositories": {name: git_metadata(path, output, name) for name, path in (("icon4py", repo), ("gt4py", repo.parent / "gt4py"), ("dace", repo.parent / "dace"))},
        "cache": cache,
    }
    mapping_path = output.with_name(f"{output.stem}.kernel_program_map.csv")
    with mapping_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        # Preserve the existing kernel_program_map_{grid}120.csv layout: no
        # header, one (program, kernel) row per ownership, duplicates retained.
        for kernel, programs in cache["kernel_to_programs"].items():
            writer.writerows((program, kernel) for program in programs)
    report["legacy_kernel_program_map"] = {"path": str(mapping_path), "columns": ["program", "kernel"], "header": False, "sha256": sha256(mapping_path)}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}: {len(cache['source_files'])} source files, {len(cache['kernel_to_programs'])} kernel names, {len(cache.get('ambiguous_kernels', []))} ambiguous.")
    if cache["errors"]:
        print("Cache metadata warnings: " + json.dumps(cache["errors"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
