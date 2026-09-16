"""Build an isolated solver module with the coefficient arithmetic inside the scan."""

import ast
import importlib.util
import sys
from pathlib import Path

from dycore_causal_core import alias, require


def replacement_source(source, original, new_name):
    tree = ast.parse(source)
    imports = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        and n.module
        and n.module.endswith(".solve_tridiagonal_matrix_for_w_forward_sweep")
    ]
    require(len(imports) == 1, "Expected exactly one forward-sweep import in the solver module.")
    imports[0].module = "forward_sweep_fused"
    programs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == original]
    require(len(programs) == 1, "Solver program definition not found.")
    programs[0].name = new_name
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def make_solver_program(program, spec, directory):
    original_module = sys.modules[program.definition.__module__]
    source = Path(original_module.__file__).read_text()
    name = alias(program.__name__, spec)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".py")
    path.write_text(replacement_source(source, program.__name__, name))
    module_name = "_solver_scan_" + name
    loader = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[module_name] = module
    loader.loader.exec_module(module)
    return getattr(module, name)
