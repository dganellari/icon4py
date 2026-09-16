"""Check treatment selection and fail-closed generated-code audits without a GPU."""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import dycore_causal_core as core
import pytest
from theta_compiler import SOLVERS, check_graphs, comparisons, configure_options


def test_native_vs_all_three_variants(monkeypatch):
    spec = comparisons()[0]
    assert core.arm_spec(spec, "A") is None
    assert spec["compiler_fusion"] and spec["solver_scan_fusion"]
    assert set(spec["targets"]) == {core.THETA, *SOLVERS}
    assert not core.selected("compute_exner_from_rhotheta", spec)
    tree = ast.parse((Path(__file__).parent / "dycore_causal_plugin.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProgramPair")
    state = dict(arm="A", programs={}, build=None)
    built = []

    def make_program(program, selected_spec, bounds):
        assert selected_spec == spec
        built.append(program.__name__)
        return SimpleNamespace(__name__=core.alias(program.__name__, selected_spec))

    ns = dict(
        STATE=state,
        THETA=core.THETA,
        arm_spec=core.arm_spec,
        case=lambda: spec,
        selected=core.selected,
        copy=copy,
        os=os,
        make_program=make_program,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<pair>", "exec"), ns)
    monkeypatch.setenv("ICON4PY_CAUSAL_MODE", "unit-test")
    for name in (core.THETA, *SOLVERS, "compute_exner_from_rhotheta"):
        pair = ns["ProgramPair"](
            lambda **kw: lambda: kw["program"].__name__,
            dict(program=SimpleNamespace(__name__=name), backend={}),
        )
        for arm in ("A", "B", "B", "A"):
            state["arm"] = arm
            expected = core.alias(name, spec) if arm == "B" and name in spec["targets"] else name
            assert pair() == expected
        assert pair.baseline is None
    assert built == [core.THETA, *SOLVERS]
    assert state["build"] is None


def test_no_compiler_hook_on_native_or_solvers():
    spec = comparisons()[0]
    for name in (core.THETA, *SOLVERS, *(core.alias(n, spec) for n in SOLVERS)):
        options = {"optimization_args": {"gpu_block_size_2d": (128, 2, 1)}}
        before = copy.deepcopy(options)
        configure_options(name, spec, options)
        assert options == before


def audits(platform):
    spec = comparisons()[0]
    result = []
    for program in (core.THETA, *SOLVERS):
        for arm in ("A", "B"):
            theta = program == core.THETA
            count = (6 if theta else 12) - (arm == "B")
            shape = ["42122", "120"] if theta else ["39788", "119"]
            buffers = 4 if theta or arm == "A" else 1
            result.append(
                dict(
                    program=program if arm == "A" else core.alias(program, spec),
                    launches=[
                        dict(block=[256, 1, 1] if platform == "amd" else [128, 2, 1])
                        for _ in range(count)
                    ],
                    vertical_block_factors=[4] if platform == "amd" else [],
                    fused_theta_pressure_kernels=["lower", "upper"] if theta and arm == "B" else [],
                    theta_graph=dict(
                        gpu_global_transients={str(i): dict(shape=shape) for i in range(buffers)}
                    ),
                )
            )
    return result


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_accepts_expected_vendor_graphs(platform):
    assert len(check_graphs(comparisons()[0], audits(platform), platform)) == 6


@pytest.mark.parametrize("damage", ["theta_noop", "solver_noop", "blocking", "missing"])
def test_rejects_invalid_intervention(damage):
    data = audits("amd")
    if damage == "theta_noop":
        data[1]["launches"].append(dict(block=[256, 1, 1]))
    elif damage == "solver_noop":
        data[3]["theta_graph"] = copy.deepcopy(data[2]["theta_graph"])
    elif damage == "blocking":
        data[0]["vertical_block_factors"] = [2]
    else:
        data.pop()
    with pytest.raises(ValueError):
        check_graphs(comparisons()[0], data, "amd")
