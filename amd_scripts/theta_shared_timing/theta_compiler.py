"""Restricted compiler selection and evidence for the theta timing experiment."""

from dycore_causal_core import THETA, alias, original_name, program_spec, require


def comparisons():
    common = dict(levels=120, compiler_fusion=True, scope="theta")
    return [
        dict(name="compiler_vs_native", **common),
        dict(
            name="compiler_vs_python",
            baseline=dict(name="python_fused_baseline", levels=120, fusion=True, scope="theta"),
            **common,
        ),
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
    """Fail on a no-op or unexpected code shape before reporting timings."""
    native = THETA
    variant = alias(THETA, spec)
    wanted = {native: 6, variant: 5}
    if spec.get("baseline"):
        wanted[alias(THETA, spec["baseline"])] = 7
    evidence = {}
    for program, count in wanted.items():
        matches = [a for a in audits if a["program"] == program]
        require(bool(matches), f"Missing generated-code evidence: {program}.")
        for audit in matches:
            require(len(audit["launches"]) == count, f"Unexpected kernel count: {program}.")
            require(
                all(k["block"] == [256, 1, 1] for k in audit["launches"])
                and audit["vertical_block_factors"] == [4],
                f"Unexpected block shape or vertical blocking: {program}.",
            )
            if program == variant:
                require(
                    len(audit["fused_theta_pressure_kernels"]) == 2,
                    "Expected two interior kernels combining theta, pressure and wind.",
                )
            if program in (native, variant):
                buffers = audit["theta_graph"]["gpu_global_transients"].values()
                require(
                    sum(b["shape"] == ["42122", "120"] for b in buffers) == 4,
                    f"Expected four unchanged gradient buffers: {program}.",
                )
        evidence[program] = matches
    return evidence
