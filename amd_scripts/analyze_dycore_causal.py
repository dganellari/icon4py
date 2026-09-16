#!/usr/bin/env python3
"""Report measured interventions, counter coverage and cross-vendor gap closure."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import amd_cache_counters as amd
import analyze_cache_comparison as nv
from dycore_causal_core import CASES, MAIN_CASE_IDS, THETA, require, paired_summary


def mean(values):
    return statistics.mean(values)


def ratio(a, b):
    return a / b if b else None


def fmt(value, scale=1):
    return "unavailable" if value is None else f"{value * scale:.3f}"


def contrast_uncertainty(a, g):
    """Welch interval on independent vendor quartet means, not individual calls."""
    from scipy.stats import t

    if min(len(a), len(g)) < 2:
        return None
    va, vg = statistics.variance(a), statistics.variance(g)
    terms = (va / len(a), vg / len(g))
    se = math.sqrt(sum(terms))
    denominator = terms[0] ** 2 / (len(a) - 1) + terms[1] ** 2 / (len(g) - 1)
    df = sum(terms) ** 2 / denominator if denominator else None
    critical = float(t.ppf(0.975, df)) if df is not None else 1.96
    effect = mean(a) - mean(g)
    return dict(
        mean_difference_ms=effect,
        standard_error_ms=se,
        welch_df=df,
        ci95_ms=[effect - critical * se, effect + critical * se],
        quartet_difference_stdev_ms=math.sqrt(va + vg),
        approximate_mde80_ms=(critical + 0.8416) * se,
        n_amd=len(a),
        n_gh200=len(g),
    )


def assess_noise(a, g, pilot_a=None, pilot_g=None):
    samples = lambda item: item.get("timing", {}).get("quartet_saved_ms", [])
    current = contrast_uncertainty(samples(a), samples(g))
    result = dict(
        status="unassessed: missing quartet data or matched placebo",
        current=current,
        resolved=False,
        pilot_gate_passed=False,
    )
    if current is None or pilot_a is None or pilot_g is None:
        return result
    pilot = contrast_uncertainty(samples(pilot_a), samples(pilot_g))
    if pilot is None:
        return result
    gap = pilot_a["device_ms"]["A"] - pilot_g["device_ms"]["A"]
    target = 0.1 * gap if gap > 0 else 0
    bias = abs(pilot["mean_difference_ms"])
    gate = (
        min(pilot["n_amd"], pilot["n_gh200"]) >= 6
        and target > 0
        and bias + pilot["approximate_mde80_ms"] < target
    )
    # Conservatively use the individual quartet spread, not SE of its mean.
    threshold = bias + 2 * pilot["quartet_difference_stdev_ms"]
    lo, hi = current["ci95_ms"]
    resolved = gate and (lo > 0 or hi < 0) and abs(current["mean_difference_ms"]) > threshold
    result.update(
        pilot=pilot,
        target_ms=target,
        placebo_threshold_ms=threshold,
        pilot_gate_passed=gate,
        resolved=resolved,
        status=("resolved improvement" if lo > 0 else "resolved regression")
        if resolved
        else ("not resolved above noise" if gate else "pilot noise gate failed"),
    )
    return result


def order_review(item):
    timing = item.get("timing", {})
    samples = timing.get("quartet_saved_ms", [])
    orders = timing.get("quartet_orders", [])
    if len(samples) != len(orders) or not samples:
        return dict(status="unavailable: missing recorded quartet orders")
    groups = {
        order: [x for x, o in zip(samples, orders) if o == order] for order in ("ABBA", "BAAB")
    }
    difference = contrast_uncertainty(groups["ABBA"], groups["BAAB"])
    result = dict(
        saved_ms_by_order={o: mean(v) if v else None for o, v in groups.items()},
        order_difference=difference,
    )
    significant_order = difference and (
        difference["ci95_ms"][0] > 0 or difference["ci95_ms"][1] < 0
    )
    opposite = all(groups.values()) and mean(groups["ABBA"]) * mean(groups["BAAB"]) <= 0
    result["status"] = (
        "order-sensitive; review before claiming an effect"
        if significant_order or opposite
        else "no order dependence detected"
    )
    placebo = item.get("placebo", {}).get("timing", {})
    controls = placebo.get("quartet_saved_ms", [])
    # Sequential old phases are not independent of drift. Report their contrast
    # as a diagnostic, never subtract it silently from the primary estimate.
    if len(controls) == len(samples) and item.get("interleaved_controls"):
        adjusted = [x - y for x, y in zip(samples, controls)]
        result["control_adjusted"] = contrast_uncertainty(adjusted, [0.0] * len(adjusted))
        result["control_adjustment"] = "matched interleaved quartet differences"
    elif controls:
        result["control_adjusted"] = contrast_uncertainty(samples, controls)
        result["control_adjustment"] = "separate phases; drift may confound subtraction"
    return result


def interaction_uncertainty(items):
    """Delta-method interval for q4 + q5 - q8 - 1, keeping A/B covariance."""
    from scipy.stats import t

    terms, ratios, sizes = [], [], []
    for item in items:
        arms = item.get("timing", {}).get("quartet_arm_ms", {})
        a, b = arms.get("A", []), arms.get("B", [])
        if len(a) < 2 or len(a) != len(b):
            return dict(status="unavailable: missing quartet arm means")
        q = mean(b) / mean(a)
        residuals = [y - q * x for x, y in zip(a, b)]
        terms.append(statistics.variance(residuals) / (len(a) * mean(a) ** 2))
        ratios.append(q)
        sizes.append(len(a))
    variance = sum(terms)
    denominator = sum(v**2 / (n - 1) for v, n in zip(terms, sizes))
    df = variance**2 / denominator if denominator else None
    half = (float(t.ppf(0.975, df)) if df else 1.96) * math.sqrt(variance)
    effect = ratios[0] + ratios[1] - ratios[2] - 1
    nodes = [item.get("hostname") for item in items]
    return dict(
        estimate=effect,
        ci95=[effect - half, effect + half],
        same_node=all(nodes) and len(set(nodes)) == 1,
        nodes=nodes,
        status="exploratory; interval excludes between-run/node uncertainty",
        method="independent-case delta method with paired A/B covariance",
    )


def counter_table(programs, platform, arm="B"):
    lines = ["", f"### {arm}-arm counters per program", ""]
    if platform == "amd":
        lines += [
            "L2 counts all-client requests; L1 forwarding is a miss proxy. Traffic is L2-to-fabric, not physical HBM. Occupancy and VMEM latency retain the installed counter's units and use counter-duration weighting.",
            "",
            "| Program | L2 hit % | L1 forward % | Tags/read | Fabric read GB/call | Fabric write GB/call | Occupancy/CU counter | VMEM latency counter | L2 busy % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, p in programs.items():
            if "derived" not in p:
                continue
            d, rates = p["derived"], p["counter_duration_weighted_rates"]
            values = [
                fmt(d["l2_all_request_hit_fraction"], 100),
                fmt(d["l1_tcc_read_requests_per_access"], 100),
                fmt(d["l1_accesses_per_read_request"]),
                fmt(d["l2_fabric_read_bytes"], 1e-9),
                fmt(d["l2_fabric_write_bytes"], 1e-9),
                fmt(rates["occupancy"]["MeanOccupancyPerCU"]),
                fmt(rates["vmem_latency"]["VmemLatency"]),
                fmt(p["l2_busy_fraction"], 100),
            ]
            lines.append("| " + name + " | " + " | ".join(values) + " |")
    else:
        lines += [
            "L2 counts TEX sectors. Traffic is measured at HBM. Occupancy and scoreboard stalls use counter-duration weighting; the latter is a warp issue diagnostic, not a fraction of application wall time.",
            "",
            "| Program | L2 read hit % | L2 read+write hit % | L1 load hit % | Sectors/load request | HBM read GB/call | HBM write GB/call | Occupancy % | Long scoreboard % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, p in programs.items():
            if "derived" not in p:
                continue
            d, raw, rates = p["derived"], p["raw_per_call"], p["counter_duration_weighted_rates"]
            rw = ratio(
                sum(raw[f"{nv.L2}{op}_lookup_hit.sum"] for op in ("read", "write")),
                sum(raw[f"{nv.L2}{op}.sum"] for op in ("read", "write")),
            )
            values = [
                fmt(d["l2_read_hit_fraction"], 100),
                fmt(rw, 100),
                fmt(d["l1_load_hit_fraction"], 100),
                fmt(d["l1_load_sectors_per_request"]),
                fmt(raw["dram__bytes_read.sum"], 1e-9),
                fmt(raw["dram__bytes_write.sum"], 1e-9),
                fmt(rates["sm__warps_active.avg.pct_of_peak_sustained_active"]),
                fmt(rates["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]),
            ]
            lines.append("| " + name + " | " + " | ".join(values) + " |")
    return lines


def summarize_counters(counter, platform):
    by_program = defaultdict(list)
    for kernel in counter["kernels"]:
        by_program[kernel["program"]].append(kernel)
    programs = {}
    for name in by_program:
        program = dict(launches_per_call=len(by_program[name]) / counter["rounds"])
        kernels = by_program[name]
        if kernels and platform == "amd":
            raw = {
                g: {
                    m: sum(k["raw"][g][m] for k in kernels) / counter["rounds"]
                    for m in kernels[0]["raw"][g]
                }
                for g in kernels[0]["raw"]
            }
            derived = amd.derived(raw)
            # Derived occupancy/rate counters cannot be added. Preserve
            # every kernel value and label any duration-weighted summaries.
            rates = {}
            for group in ("occupancy", "occupancy_active", "vmem_latency", "issue"):
                weights = [k["group_duration_ns"][group] for k in kernels]
                rates[group] = {
                    metric: sum(k["raw"][group][metric] * w for k, w in zip(kernels, weights))
                    / sum(weights)
                    for metric in kernels[0]["raw"][group]
                }
            raw_additive = {g: values for g, values in raw.items() if g not in rates}
            program.update(
                raw_additive_per_call=raw_additive,
                derived=derived,
                counter_duration_weighted_rates=rates,
                raw_nonadditive_per_kernel=[
                    dict(kernel=k["kernel"], rates={g: k["raw"][g] for g in rates}) for k in kernels
                ],
            )
            program["l2_busy_fraction"] = ratio(
                raw["l2_busy"]["TCC_BUSY_sum"], raw["l2_busy"]["TCC_CYCLE_sum"]
            )
        elif kernels:
            additive = list(nv.COUNTERS) + ["smsp__inst_executed.sum"]
            raw = {m: sum(k["raw"][m] for k in kernels) / counter["rounds"] for m in additive}
            rates = {
                m: sum(k["raw"][m] * k["raw"]["gpu__time_duration.sum"] for k in kernels)
                / sum(k["raw"]["gpu__time_duration.sum"] for k in kernels)
                for m in kernels[0]["raw"]
                if m not in additive and not m.startswith("launch__")
            }
            program.update(
                raw_per_call=raw,
                derived=nv._derived(raw),
                counter_duration_weighted_rates=rates,
                launch_resources=[
                    {m: v for m, v in k["raw"].items() if m.startswith("launch__")} for k in kernels
                ],
            )
        programs[name] = program
    return programs


def summarize(root):
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["status"] == "collection_complete", "Collection is incomplete.")
    report = dict(case=manifest["case"], platform=manifest["platform"], path=str(root), grids={})
    lines = [
        f"# Dycore intervention: {manifest['case']['name']} ({manifest['platform']})",
        "",
        "Device timing is the sum of per-program median in-SDFG compute times in each ABBA block. Host wall time is reported separately.",
        "Counter durations are profiler diagnostics, never substituted for ordinary timings.",
        "",
    ]
    for grid, paths in manifest["grids"].items():
        timing = json.loads((root / paths["timing"]).read_text())
        counter = json.loads((root / paths["counters"]).read_text())
        require(
            timing["validation"]["status"] == "passed", "GPU equivalence validation did not pass."
        )
        blocks = timing["blocks"]
        arms = {arm: [b for b in blocks if b["arm"] == arm] for arm in ("A", "B")}
        names = set().union(*(b["programs"].keys() for b in blocks))
        times = {
            n: {arm: mean(b["programs"].get(n, 0) for b in arms[arm]) for arm in arms}
            for n in names
        }
        totals = {arm: mean(b["device_ms"] for b in arms[arm]) for arm in arms}
        item = dict(
            timing=paired_summary(blocks),
            interleaved_controls=timing.get("timing_design", {}).get("interleaved", False),
            device_ms=totals,
            wall_ms={a: mean(b["wall_ms"] for b in arms[a]) for a in arms},
            programs={},
            validation=timing["validation"],
            hostname=timing["hostname"],
            grid_dimensions=timing["grid_dimensions"],
            state_allocation_bytes=sum(a["bytes"] for a in timing["state_arrays"]),
            footprint_note="Sum of array-view sizes may overlap and includes untouched elements; not an active cache working-set measurement.",
        )
        if "placebo" in timing:
            placebo_blocks = timing["placebo"]["blocks"]
            item["placebo"] = dict(
                timing=paired_summary(placebo_blocks),
                device_ms={
                    arm: mean(b["device_ms"] for b in placebo_blocks if b["arm"] == arm)
                    for arm in ("A", "B")
                },
            )
        item["counter_programs"] = {"B": summarize_counters(counter, manifest["platform"])}
        if "A" in paths.get("counter_arms", {}):
            native = json.loads((root / paths["counter_arms"]["A"]).read_text())
            require(
                native.get("arm") == "A" and counter.get("arm") == "B",
                "Counter arm labels do not match the manifest.",
            )
            item["counter_programs"]["A"] = summarize_counters(native, manifest["platform"])
            require(
                set(item["counter_programs"]["A"]) == set(item["counter_programs"]["B"]),
                "Native and modified counter program coverage differs.",
            )
        for name in names:
            item["programs"][name] = dict(
                device_ms=times[name],
                saved_ms=times[name]["A"] - times[name]["B"],
                **item["counter_programs"]["B"].get(name, {"launches_per_call": 0}),
            )
        item["order_review"] = order_review(item)
        require(THETA in item["programs"], "Theta-rho timing attribution missing.")
        report["grids"][grid] = item
        lines += [
            f"## {grid}/{manifest['case']['levels']}",
            "",
            "A baseline: " + manifest["case"].get("baseline", {}).get("name", "native") + ".",
            "",
            f"A: {totals['A']:.6f} ms; B: {totals['B']:.6f} ms; B/A: {totals['B'] / totals['A']:.6f}.",
            f"Paired saved ms by quartet: {item['timing']['quartet_saved_ms']}",
            f"Quartet standard deviations (ms): {item['timing']['quartet_stdev_ms']}",
            "",
            "| Program | A ms | B ms | Saved ms | B kernels/call |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, p in sorted(item["programs"].items(), key=lambda kv: -kv[1]["device_ms"]["A"]):
            lines.append(
                f"| {name} | {p['device_ms']['A']:.6f} | {p['device_ms']['B']:.6f} | {p['saved_ms']:.6f} | {p['launches_per_call']:g} |"
            )
        lines += ["", "Order and control checks: " + json.dumps(item["order_review"]), ""]
        if "A" in item["counter_programs"]:
            lines += [
                "A and B counters come from this allocation, in separate profiler processes. Kernel mixtures and duration weights may change; these are diagnostic changes, not a unique causal decomposition."
            ]
            lines += counter_table(item["counter_programs"]["A"], manifest["platform"], "A")
        lines += counter_table(item["programs"], manifest["platform"])
        lines += [
            "",
            "All cache, traffic, occupancy, issue and latency counters are retained per program and per kernel in ANALYSIS.json and "
            + paths["counters"]
            + ".",
            "",
        ]
    lines += [
        "## Interpretation contract",
        "",
        "A faster validated B establishes the effect of that intervention on this benchmark. It does not assign a percentage to physical L2 capacity or L1 latency alone.",
        "Block traversal preserves mathematical work and launch counts, but may change registers, CU/XCD work distribution and memory scheduling. Inspect occupancy and resource changes.",
        "Fusion may change register pressure, intermediate traffic and kernel count together. VLB and block shape may change reuse and latency hiding together. Report the combined arm and interaction rather than adding gains.",
        "A level sweep changes size, parallelism and vertical branches. A hit-rate or timing knee alone does not isolate physical cache capacity.",
        "AMD L2 hits are all-client requests (including hit-on-miss); GH200 read/read+write metrics are TEX sectors. AMD fabric traffic is not physical HBM traffic. No AMD read-only L2 hit rate is inferred.",
        "",
    ]
    (root / "ANALYSIS.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (root / "ANALYSIS.md").write_text("\n".join(lines))
    return report


def compare(reports, output):
    by_case = defaultdict(dict)
    for report in reports:
        key, vendor = report["case"]["name"], report["platform"]
        require(
            vendor not in by_case[key],
            f"Duplicate {vendor}/{key} results; select one attempt explicitly.",
        )
        by_case[key][vendor] = report
    rows, effects = [], {}
    for name, vendors in by_case.items():
        if set(vendors) != {"amd", "nvidia"}:
            continue
        aroot, groot = (Path(vendors[v]["path"]) for v in ("amd", "nvidia"))
        require(vendors["amd"]["case"] == vendors["nvidia"]["case"], "Case configurations differ.")
        require(
            json.loads((aroot / "source_hashes.json").read_text())
            == json.loads((groot / "source_hashes.json").read_text()),
            "Cross-vendor source hashes differ; no matched-code conclusion permitted.",
        )
        require(
            set(vendors["amd"]["grids"]) == set(vendors["nvidia"]["grids"]),
            "Cross-vendor grid coverage differs.",
        )
        for grid in sorted(vendors["amd"]["grids"]):
            a, g = (vendors[v]["grids"][grid] for v in ("amd", "nvidia"))
            require(a["grid_dimensions"] == g["grid_dimensions"], "Actual mesh dimensions differ.")
            require(set(a["programs"]) == set(g["programs"]), "Timed program coverage differs.")
            ta, tg = a["device_ms"], g["device_ms"]
            d0, d1 = ta["A"] - tg["A"], ta["B"] - tg["B"]
            row = dict(
                case=name,
                levels=vendors["amd"]["case"]["levels"],
                grid=grid,
                baseline_ratio=ta["A"] / tg["A"],
                variant_ratio=ta["B"] / tg["B"],
                baseline_gap_ms=d0,
                variant_gap_ms=d1,
                differential_saved_ms=d0 - d1,
                fraction_gap_closed=(d0 - d1) / d0 if d0 > 0 else None,
                amd_saved_ms=ta["A"] - ta["B"],
                gh200_saved_ms=tg["A"] - tg["B"],
                program_differential_saved_ms={
                    n: a["programs"][n]["saved_ms"] - g["programs"][n]["saved_ms"]
                    for n in a["programs"].keys() & g["programs"].keys()
                },
            )
            pilot = by_case.get("native", {})
            pilot_a = pilot_g = None
            if (
                set(pilot) == {"amd", "nvidia"}
                and all(grid in pilot[v]["grids"] for v in ("amd", "nvidia"))
                and vendors["amd"]["case"]["levels"] == 120
            ):
                matched = all(
                    json.loads((Path(pilot[v]["path"]) / "source_hashes.json").read_text())
                    == json.loads((Path(vendors[v]["path"]) / "source_hashes.json").read_text())
                    and pilot[v]["grids"][grid]["grid_dimensions"]
                    == vendors[v]["grids"][grid]["grid_dimensions"]
                    for v in ("amd", "nvidia")
                )
                if matched:
                    pilot_a, pilot_g = (pilot[v]["grids"][grid] for v in ("amd", "nvidia"))
            if "placebo" in a and "placebo" in g:
                pilot_a, pilot_g = a["placebo"], g["placebo"]
                row["placebo_source"] = (
                    "same-process fused baseline A/A"
                    if vendors["amd"]["case"].get("baseline")
                    else "same-process native A/A"
                )
            else:
                row["placebo_source"] = "matched separate native case" if pilot_a else "unavailable"
            row["noise"] = assess_noise(a, g, pilot_a, pilot_g)
            row["order_review"] = {"amd": order_review(a), "nvidia": order_review(g)}
            if any(x["status"].startswith("order-sensitive") for x in row["order_review"].values()):
                row["noise"]["resolved"] = False
                row["noise"]["status"] = "order-sensitive; effect requires review"
            if name == "native":
                row["noise"]["status"] = "A/A placebo; no intervention"
                row["noise"]["resolved"] = False
            rows.append(row)
            effects[(name, grid)] = {"amd": ta["B"] / ta["A"], "nvidia": tg["B"] / tg["A"]}
    interactions = []
    for grid in ("global", "regional"):
        names = ("order4_all", "fused_theta", "order4_fused")
        if all((n, grid) in effects for n in names):
            interactions.append(
                dict(
                    grid=grid,
                    units="fraction of own same-node baseline; positive means extra combined saving",
                    uncertainty={
                        v: interaction_uncertainty([by_case[n][v]["grids"][grid] for n in names])
                        for v in ("amd", "nvidia")
                    },
                    same_sources={
                        v: all(
                            json.loads(
                                (Path(by_case[n][v]["path"]) / "source_hashes.json").read_text()
                            )
                            == json.loads(
                                (
                                    Path(by_case[names[0]][v]["path"]) / "source_hashes.json"
                                ).read_text()
                            )
                            for n in names
                        )
                        for v in ("amd", "nvidia")
                    },
                    **{
                        v: effects[(names[0], grid)][v]
                        + effects[(names[1], grid)][v]
                        - effects[(names[2], grid)][v]
                        - 1
                        for v in ("amd", "nvidia")
                    },
                )
            )
    missing = [
        f"{c['name']}/{v}"
        for c in (CASES[i] for i in MAIN_CASE_IDS)
        for v in ("amd", "nvidia")
        if v not in by_case[c["name"]]
    ]
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        rows=rows,
        normalized_interactions=interactions,
        missing_cases=missing,
        complete_matrix=not missing
        and all(set(report["grids"]) == {"global", "regional"} for report in reports),
        grid_coverage={
            name: {v: sorted(report["grids"]) for v, report in vendors.items()}
            for name, vendors in by_case.items()
        },
        physical_capacity_fraction=None,
        note="Quantified intervention effects, not a unique decomposition into L1/L2/occupancy causes. No clipping of negative savings or interactions.",
    )
    (output / "COMPARISON.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Cross-vendor intervention effects",
        "",
        "| Case | Grid/levels | Baseline AMD/GH | Variant AMD/GH | Differential saving ms | Raw gap closed | 95% CI ms | Noise assessment |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        closed = (
            f"{100 * r['fraction_gap_closed']:.1f}%"
            if r["fraction_gap_closed"] is not None
            else "undefined"
        )
        ci = r["noise"]["current"]
        interval = ", ".join(f"{v:.6f}" for v in ci["ci95_ms"]) if ci else "unavailable"
        lines.append(
            f"| {r['case']} | {r['grid']}/{r['levels']} | {r['baseline_ratio']:.4f} | {r['variant_ratio']:.4f} | {r['differential_saved_ms']:.6f} | {closed} | {interval} | {r['noise']['status']} |"
        )
    lines += [
        "",
        "Gap percentages are raw estimates, not evidence of a resolved effect. Welch intervals use vendor quartet contrasts; the noise gate also requires a matched native placebo (preferably the same-process A/A) and a differential larger than its absolute bias plus twice its quartet standard deviation. The pilot MDE screen targets 10% of the same-grid native gap (regional is the primary experiment; global receives a separate diagnostic screen). Node variability, timing drift and multiple comparisons require separate review. Per-program contributions remain descriptive. Interaction intervals retain paired A/B covariance within quartets but exclude between-run/node uncertainty; source, node and order checks remain necessary.",
        "",
        "Interaction estimates and uncertainty: " + json.dumps(interactions),
        "",
        "Missing cases: " + (", ".join(missing) or "none"),
        "",
    ]
    (output / "COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    roots = []
    for path in args.inputs:
        roots.extend(
            [path]
            if (path / "manifest.json").exists()
            else sorted(p.parent for p in path.rglob("manifest.json"))
        )
    require(roots, "No completed case manifests found.")
    reports = [summarize(root) for root in roots]
    if args.output:
        compare(reports, args.output)
