import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "model/common/src"), str(ROOT / "model/atmosphere/dycore/src")]

import dycore_causal_core as core
from solver_program import make_solver_program, replacement_source
from theta_compiler import SOLVERS, comparisons


def test_targets_and_shared_theta_implementation():
    spec = comparisons()[0]
    assert all(core.selected(name, spec) for name in (*SOLVERS, core.THETA))
    assert not core.selected("compute_exner_from_rhotheta", spec)
    tree = ast.parse((Path(__file__).parent / "dycore_causal_plugin.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProgramPair")
    state = dict(arm="B", programs={}, build=None)
    ns = dict(
        STATE=state,
        THETA=core.THETA,
        arm_spec=core.arm_spec,
        case=lambda: spec,
        selected=core.selected,
        copy=copy,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<pair>", "exec"), ns)
    pair = ns["ProgramPair"](
        lambda **kw: lambda: "native", dict(program=SimpleNamespace(__name__=core.THETA))
    )
    pair.baseline = lambda: "same compiler-fused theta"
    pair.baseline_checked = True
    assert pair() == "same compiler-fused theta"
    assert pair.variant is None


def test_solver_module_rebinds_only_forward_sweep(tmp_path):
    from forward_sweep_fused import (
        _solve_tridiagonal_matrix_for_w_forward_sweep as replacement,
    )
    from icon4py.model.atmosphere.dycore.stencils import (
        vertically_implicit_dycore_solver as source,
    )

    for name in SOLVERS:
        program = getattr(source, name)
        variant = make_solver_program(program, comparisons()[0], tmp_path)
        module = sys.modules[variant.definition.__module__]
        assert module._solve_tridiagonal_matrix_for_w_forward_sweep is replacement
        assert source._solve_tridiagonal_matrix_for_w_forward_sweep is not replacement
        assert variant.__name__ == core.alias(name, comparisons()[0])
        original_tree = ast.parse(Path(source.__file__).read_text())
        new_tree = ast.parse(
            replacement_source(Path(source.__file__).read_text(), name, variant.__name__)
        )
        for node in new_tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "forward_sweep_fused":
                node.module = "icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep"
            if isinstance(node, ast.FunctionDef) and node.name == variant.__name__:
                node.name = name
        assert ast.dump(original_tree) == ast.dump(new_tree)
