"""Restricted compiler selection and evidence for the theta timing experiment."""

from dycore_causal_core import THETA, alias, original_name, program_spec, require

SOLVERS = (
    "vertically_implicit_solver_at_predictor_step",
    "vertically_implicit_solver_at_corrector_step",
)


def comparisons():
    return [
        dict(
            name="solver_coefficients_in_scan",
            levels=120,
            compiler_fusion=True,
            solver_scan_fusion=True,
            scope="programs",
            targets=list(SOLVERS) + [THETA],
            baseline=dict(
                name="theta_compiler_baseline", levels=120, compiler_fusion=True, scope="theta"
            ),
        )
    ]


def configure_options(name, case, result):
    spec = program_spec(name, case)
    if original_name(name) != THETA or "__causal_" not in name or not spec.get("compiler_fusion"):
        return
    from compiler_hook import select_theta_split
    from gt4py.next.program_processors.runners.dace import transformations as tx
    from gt4py.next.program_processors.runners.dace.transformations.map_fusion_extended import (
        VerticalSplitMapRange,
    )

    require(
        "allow_shared_data" in VerticalSplitMapRange.__properties__,
        "Experimental GT4Py overlay is not active; shared-output splitting is unavailable.",
    )
    hooks = result.setdefault("optimization_args", {}).setdefault("optimization_hooks", {})
    key = tx.GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack
    require(key not in hooks, "An existing vertical-split callback would be overwritten.")
    hooks[key] = select_theta_split


def graph_evidence(sdfg):
    import dace

    return dict(
        gpu_global_transients={
            name: dict(shape=list(map(str, desc.shape)), dtype=str(desc.dtype))
            for name, desc in sdfg.arrays.items()
            if desc.transient
            and isinstance(desc, dace.data.Array)
            and desc.storage == dace.StorageType.GPU_Global
        },
    )


def check_graphs(spec, audits):
    evidence = {}
    theta = alias(THETA, spec["baseline"])
    ts = [a for a in audits if a["program"] == theta]
    require(
        ts and all(len(a["launches"]) == 5 for a in ts), "Theta compiler-fusion baseline missing."
    )
    evidence[theta] = ts
    for solver in SOLVERS:
        native = [a for a in audits if a["program"] == solver]
        variant = [a for a in audits if a["program"] == alias(solver, spec)]
        require(native and variant, "Missing original or modified solver code audit.")
        require(
            all(a['vertical_block_factors'] == [4] and
                all(k['block'] == [256, 1, 1] for k in a['launches'])
                for a in native + variant),
            'Solver block shape or vertical blocking changed.',
        )

        def counts(items):
            return sorted(
                (
                    len(a["launches"]),
                    sum(
                        d["shape"] == ["39788", "119"]
                        for d in a["theta_graph"]["gpu_global_transients"].values()
                    ),
                )
                for a in items
            )

        na, nb = counts(native), counts(variant)
        require(len(na) == len(nb), "Solver specialization counts changed.")
        for (ka, ba), (kb, bb) in zip(na, nb, strict=True):
            require(bb < ba, "Full solver did not reduce coefficient-sized intermediate storage.")
            require(kb <= ka, "Solver fusion unexpectedly increased kernel count.")
        evidence[solver] = native
        evidence[alias(solver, spec)] = variant
    return evidence
