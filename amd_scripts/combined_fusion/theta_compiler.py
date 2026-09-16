"""Restricted compiler selection and evidence for the theta timing experiment."""

from dycore_causal_core import THETA, alias, original_name, program_spec, require

SOLVERS = (
    "vertically_implicit_solver_at_predictor_step",
    "vertically_implicit_solver_at_corrector_step",
)


def comparisons():
    return [
        dict(
            name="combined_fusion",
            levels=120,
            compiler_fusion=True,
            solver_scan_fusion=True,
            scope="programs",
            targets=list(SOLVERS) + [THETA],
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


def check_graphs(spec, audits, platform):
    """Verify both interventions and preserve each vendor's native launch settings."""
    evidence = {}
    require("baseline" not in spec, "Combined comparison must start from native code.")
    require(platform in ("amd", "nvidia"), "Unknown platform.")
    allowed = {(256, 1, 1)} if platform == "amd" else {(128, 2, 1), (64, 1, 1)}
    factors = [4] if platform == "amd" else []
    for program in (THETA, *SOLVERS):
        native = [a for a in audits if a["program"] == program]
        variant = [a for a in audits if a["program"] == alias(program, spec)]
        require(native and variant, f"Missing A/B generated code: {program}.")
        require(len(native) == len(variant), "Specialization counts changed.")
        for audit in native + variant:
            require(
                audit["vertical_block_factors"] == factors,
                f"Unexpected vertical blocking: {program}.",
            )
            require(
                all(tuple(k["block"] or ()) in allowed for k in audit["launches"]),
                f"Unexpected launch shape: {program}.",
            )
        if program == THETA:
            require(
                all(len(a["launches"]) == 6 for a in native), "Native theta is not six kernels."
            )
            require(
                all(len(a["launches"]) == 5 for a in variant), "Fused theta is not five kernels."
            )
            require(
                all(len(a["fused_theta_pressure_kernels"]) == 2 for a in variant),
                "Expected two interior theta/pressure/wind kernels.",
            )
            for a in native + variant:
                require(
                    sum(
                        d["shape"] == ["42122", "120"]
                        for d in a["theta_graph"]["gpu_global_transients"].values()
                    )
                    == 4,
                    "Theta gradient buffers changed.",
                )
        else:

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

            for (ka, ba), (kb, bb) in zip(counts(native), counts(variant), strict=True):
                require(bb < ba, "Solver coefficient storage was not reduced.")
                require(kb < ka, "Solver kernel count was not reduced.")
        evidence[program] = native
        evidence[alias(program, spec)] = variant
    return evidence
