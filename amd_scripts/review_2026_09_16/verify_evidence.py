#!/usr/bin/env python3
"""Verify the review bundle and reconstruct paired measurements without GPU access."""

import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
THETA = "compute_rho_theta_pgrad_and_update_vn"
T_975_DF11 = 2.200985160082949


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)


def check_grid_levels():
    # Exercise the actual option/preset functions without loading grid data or Serialbox.
    path = ROOT.parent.parent / "model/testing/src/icon4py/model/testing/fixtures/stencil_tests.py"
    source = ast.parse(path.read_text())
    names = {"_evaluate_grid_option", "_get_grid_manager_from_preset"}
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    constants = {
        n.target.id: ast.literal_eval(n.value)
        for n in source.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id in {"DEFAULT_GRID", "DEFAULT_NUM_LEVELS", "BENCHMARK_DEFAULT_LEVELS"}
    }

    class GridNames:
        def __getattr__(self, name):
            return name

    context = dict(
        constants,
        test_defs=SimpleNamespace(Grids=GridNames()),
        grid_utils=SimpleNamespace(get_grid_manager_from_identifier=lambda *args, **kw: kw),
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + functions,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), context)
    for preset in ("icon_benchmark_regional", "icon_benchmark_global"):
        for levels in (None, 16, 80, 120):
            spec = preset if levels is None else f"{preset}:{levels}"
            request = SimpleNamespace(config=SimpleNamespace(getoption=lambda key: spec))
            name, parsed = context["_evaluate_grid_option"](request)
            result = context["_get_grid_manager_from_preset"](
                name, num_levels=parsed, allocator=None
            )
            expected = constants["BENCHMARK_DEFAULT_LEVELS"] if levels is None else levels
            require(result["num_levels"] == expected, f"Grid level override failed: {spec}")
    print("Grid-level prerequisite: 8 explicit/default preset checks passed")


def differences(blocks, metric):
    def value(block):
        if metric == "theta_device":
            return block["programs"][THETA]
        if metric == "solvers_device":
            return sum(
                v
                for n, v in block["programs"].items()
                if n.startswith("vertically_implicit_solver_at_")
            )
        return block["wall_ms" if metric == "granule_wall" else "device_ms"]

    require(len(blocks) == 48, "Expected 12 quartets.")
    result, arms = {}, {"A": [], "B": []}
    for i in range(0, len(blocks), 4):
        quartet = blocks[i : i + 4]
        require("".join(b["arm"] for b in quartet) in ("ABBA", "BAAB"), "Unbalanced quartet.")
        ids = {b["quartet"] for b in quartet}
        require(len(ids) == 1, "Quartet IDs differ.")
        key = ids.pop()
        require(key not in result, "Duplicate quartet.")
        pair = {a: statistics.mean(value(b) for b in quartet if b["arm"] == a) for a in arms}
        result[key] = pair["A"] - pair["B"]
        for arm in arms:
            arms[arm].append(pair[arm])
    return result, arms


def main():
    check_grid_levels()
    inventory = load(ROOT / "ARTIFACTS.json")
    for name, expected in inventory.items():
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
            f"Review artifact changed: {name}",
        )
    print(f"Artifact checksums: {len(inventory)} match")
    for job, bundle in (
        ("amd_639200", "theta_shared_timing"),
        ("amd_639284", "solver_scan_fusion"),
    ):
        folder = ROOT / "evidence" / job
        require((folder / "COMPLETE").exists(), f"Missing completion marker: {job}")
        require(load(folder / "STATUS.json")["status"] == "complete", "Run incomplete.")
        initial = load(folder / "source_hashes.json.gz")
        require(
            initial == load(folder / "source_hashes.final.json.gz"), "Run source hashes changed."
        )
        for name, expected in initial.items():
            if name.startswith("timing_bundle/"):
                path = ROOT.parent / bundle / name.split("/", 1)[1]
                require(
                    hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    f"Experiment differs from measured source: {path}",
                )
        summary = load(folder / "TIMING_SUMMARY.json")
        for case, item in summary.items():
            reports = [
                p
                for p in (folder / case).glob("timing.*.json.gz")
                if p.name.split(".")[1].isdigit()
            ]
            require(len(reports) == 1, "Expected one raw timing report.")
            raw = load(reports[0])
            require(
                raw["status"] == "complete" and raw["validation"]["status"] == "passed",
                "Numerical validation incomplete.",
            )
            require(
                raw["validation"]["fields"] == 148 and raw["validation"]["max_abs_error"] == 0,
                "Unexpected validation result.",
            )
            require(
                raw["grid_dimensions"]["levels"] == 120 and raw["grid_dimensions"]["limited_area"],
                "Wrong grid/levels.",
            )
            for block in raw["blocks"] + raw["placebo"]["blocks"]:
                require(len(block["samples"]) == 10, "Wrong sample count.")
                medians = {
                    name: statistics.median(s["programs"][name] for s in block["samples"])
                    for name in block["programs"]
                }
                require(
                    medians == block["programs"]
                    and close(sum(medians.values()), block["device_ms"]),
                    "Block summary differs from samples.",
                )
            for metric, reported in item["metrics"].items():
                x, arms = differences(raw["blocks"], metric)
                y, _ = differences(raw["placebo"]["blocks"], metric)
                require(x.keys() == y.keys(), "Missing matched controls.")
                saving = statistics.mean(x.values())
                require(close(saving, reported["saved_ms"]), "Reported saving differs.")
                require(close(statistics.mean(arms["A"]), reported["baseline_ms"]), "Wrong A mean.")
                require(
                    close(
                        statistics.mean(arms["B"]),
                        reported.get("variant_ms", reported.get("compiler_ms")),
                    ),
                    "Wrong B mean.",
                )
                for values, expected in (
                    (list(x.values()), reported["raw_interval"]["ci95_ms"]),
                    (
                        [x[k] - y[k] for k in x],
                        reported["order_review"]["control_adjusted"]["ci95_ms"],
                    ),
                ):
                    half = T_975_DF11 * statistics.stdev(values) / math.sqrt(12)
                    mean = statistics.mean(values)
                    require(
                        all(close(a, b) for a, b in zip((mean - half, mean + half), expected)),
                        "Confidence interval differs.",
                    )
                print(
                    f"{job}/{case}/{metric}: {100 * saving / statistics.mean(arms['A']):.3f}% less time"
                )
        print(f"{job}: validation, controls and {len(initial)} source hashes verified")
    original = ROOT.parent / "solver_scan_fusion" / "forward_sweep_fused.py"
    require(
        ast.dump(ast.parse(original.read_text()))
        == ast.dump(ast.parse((ROOT / "code/forward_sweep_fused.py").read_text())),
        "Readable solver copy differs semantically from measured source.",
    )
    print("Readable solver AST matches the measured prototype; all checks passed.")


if __name__ == "__main__":
    main()
