"""Pure helpers and preregistered cases for the dycore intervention experiment."""

from __future__ import annotations

import ast
import copy
import hashlib
import random
import re
import statistics


THETA = "compute_rho_theta_pgrad_and_update_vn"
CORRECTOR = "compute_advection_in_corrector_vertical_momentum"
CASES = [
    dict(name="native", levels=120),
    dict(name="footprint16", levels=16),
    dict(name="footprint64", levels=64),
    dict(name="footprint32", levels=32),
    dict(name="order4_all", levels=120, order_width=4, scope="all"),
    dict(name="fused_theta", levels=120, fusion=True, scope="theta"),
    dict(name="vlb1_theta", levels=120, vlb=1, scope="theta"),
    dict(name="block128_theta", levels=120, block=[128, 1, 1], scope="theta"),
    dict(name="order4_fused", levels=120, order_width=4, fusion=True, scope="all"),
    dict(name="order4_theta", levels=120, order_width=4, scope="theta"),
    dict(
        name="block64x4_corrector",
        levels=120,
        block_2d=[64, 4, 1],
        target=CORRECTOR,
        scope="program",
    ),
    dict(
        name="fused_theta_vlb2",
        levels=120,
        fusion=True,
        vlb=2,
        scope="theta",
        platform="amd",
        baseline=dict(name="fused_theta_baseline", levels=120, fusion=True, scope="theta"),
    ),
]
MAIN_CASE_IDS = (0, 4, 5, 8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected(name, case):
    return bool(case.keys() & {"order_width", "fusion", "vlb", "block", "block_2d"}) and (
        case.get("scope") == "all" or name == case.get("target", THETA)
    )


def alias(name, case):
    return name + "__causal_" + case["name"]


def original_name(name):
    return name.split("__causal_", 1)[0]


def arm_spec(spec, arm):
    require(arm in ("A", "B"), "Unknown intervention arm.")
    return spec.get("baseline") if arm == "A" else spec


def program_spec(name, spec):
    baseline = spec.get("baseline")
    if baseline and name == alias(original_name(name), baseline):
        return baseline
    return spec


def vertical_block_factors(code):
    """Recognise DaCe's generated coarse-to-fine vertical loop mapping."""
    return sorted(
        {int(n) for n in re.findall(r"\(\((\d+) \* __gtx_coarse_i_K_gtx_vertical\) \+", code)}
    )


def tile_coordinate(horizontal, vertical, nx, ny, width):
    """Bijective block traversal, including a short final horizontal tile."""
    require(nx > 0 and ny > 0 and width > 0, "Invalid tile dimensions.")
    require(0 <= horizontal < nx and 0 <= vertical < ny, "Invalid block coordinate.")
    width = min(width, nx)
    linear = horizontal + nx * vertical
    base = (linear // (width * ny)) * width
    actual = min(width, nx - base)
    remainder = linear - base * ny
    return base + remainder % actual, remainder // actual


def kernel_spans(code):
    """Find generated GPU function bodies, rejecting unbalanced source."""
    spans = []
    pattern = r"__global__\s+void\s+(?:__\w+__\([^\n]*?\)\s+)*([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
    for match in re.finditer(pattern, code):
        begin = match.end()
        depth, end = 1, begin
        while depth and end < len(code):
            depth += (code[end] == "{") - (code[end] == "}")
            end += 1
        require(depth == 0, "Unbalanced generated kernel source.")
        spans.append((match[1], begin, end - 1))
    return spans


def kernel_launches(code):
    """Read literal launch shapes and gather roles from generated CUDA/HIP."""
    shapes = {
        match[1]: dict(
            api=match[0], grid=[int(v) for v in match[2:5]], block=[int(v) for v in match[5:8]]
        )
        for match in re.findall(
            r"(hip|cuda)LaunchKernel\(\(void\*\)(\w+),\s*"
            r"dim3\(\s*(\d+),\s*(\d+),\s*(\d+)\),\s*"
            r"dim3\(\s*(\d+),\s*(\d+),\s*(\d+)\)",
            code,
        )
    }
    launches = []
    for name, begin, end in kernel_spans(code):
        if "fieldop" not in name:
            continue
        signature = code[code.rfind("__global__", 0, begin) : begin]
        body = code[begin:end]
        shape = shapes.get(name)
        vertical_axis = (
            2 if re.search(r"blockIdx\.x\s*\*\s*gridDim\.y\s*\+\s*blockIdx\.y", body) else 1
        )
        launches.append(
            dict(
                kernel=name,
                block=shape["block"] if shape else None,
                grid=shape["grid"] if shape else None,
                vertical_threads=bool(
                    shape and (shape["grid"][vertical_axis] > 1 or shape["block"][1] > 1)
                ),
                vertex_gather="gt_conn_V2C" in signature and "c_intp" in signature,
                edge_gather="gt_conn_E2V" in signature
                and "tangential_wind_on_half_levels" in signature,
            )
        )
    return launches


def reorder_kernels(code, width, platform):
    """Change only block visitation; preserve AMD's physical XCD x coordinate.

    AMD's current chiplet code maps H=x*gridDim.y+y and K=z. We permute
    y/z independently inside each x group. CUDA maps H=x and K=y.
    Unsupported layouts are recorded as untouched, never guessed.
    """
    audit = []
    for name, begin, end in reversed(kernel_spans(code)):
        body = code[begin:end]
        if "fieldop" not in name:
            continue
        if platform == "amd":
            supported = bool(re.search(r"blockIdx\.x\s*\*\s*gridDim\.y\s*\+\s*blockIdx\.y", body))
            supported &= "blockIdx.z" in body and "_K_gtx_vertical" in body
            h, k = "y", "z"
        else:
            supported = "blockIdx.x" in body and "blockIdx.y" in body
            supported &= "_K_gtx_vertical" in body and "blockIdx.z" not in body
            h, k = "x", "y"
        audit.append(dict(kernel=name, changed=bool(supported), platform=platform))
        if not supported:
            continue
        body = re.sub(r"\bblockIdx\b", "icon_causal_block", body)
        header = f"""
    // Causal experiment: bijective block traversal; arithmetic and launches unchanged.
    const unsigned icon_nh = gridDim.{h}, icon_nk = gridDim.{k};
    const unsigned icon_w = icon_nh < {width}u ? icon_nh : {width}u;
    const unsigned long long icon_l = blockIdx.{h} + (unsigned long long)icon_nh * blockIdx.{k};
    const unsigned icon_base = (icon_l / ((unsigned long long)icon_w * icon_nk)) * icon_w;
    const unsigned icon_actual = icon_nh - icon_base < icon_w ? icon_nh - icon_base : icon_w;
    const unsigned long long icon_r = icon_l - (unsigned long long)icon_base * icon_nk;
    dim3 icon_causal_block(blockIdx.x, blockIdx.y, blockIdx.z);
    icon_causal_block.{h} = icon_base + icon_r % icon_actual;
    icon_causal_block.{k} = icon_r / icon_actual;
"""
        code = code[:begin] + header + body + code[end:]
    return code, list(reversed(audit))


def output_bands(bounds):
    """Partition the original four output domains without changing boundary physics."""
    hs, he = int(bounds["horizontal_start"]), int(bounds["horizontal_end"])
    vs, ins, ie = (
        int(bounds[k])
        for k in ("start_edge_lateral_boundary", "start_edge_nudging_level_2", "end_edge_local")
    )
    require(hs <= vs <= ins < ie <= he, "Unsupported theta-rho output-domain ordering.")
    bands = [("interior", ins, ie, (0, 1, 2, 3))]
    if hs < ins:
        bands.append(("rho_left", hs, ins, (0, 1)))
    if ie < he:
        bands.append(("rho_right", ie, he, (0, 1)))
    if vs < ins:
        bands.append(("vn_boundary", vs, ins, (3,)))
    return bands


def program_source(module_source, name, new_name, *, fusion=False, bounds=None):
    """Build an experimental program; original field operators remain unchanged.

    Fusion aligns the four interior output domains. Separate exterior calls
    retain the original rho/theta and boundary-wind output domains exactly.
    Compilation and GPU equivalence validation decide whether it is usable.
    """
    module = ast.parse(module_source)
    program = copy.deepcopy(
        next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    program.name = new_name
    extra = []
    if fusion:
        require(name == THETA, "Fusion is defined only for theta-rho.")
        bands = output_bands(bounds)
        calls = [
            n.value
            for n in program.body
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        ]
        require(len(calls) == 1, "Theta-rho program structure changed.")
        call = calls[0]
        fo_name = call.func.id
        fo = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == fo_name)
        kwargs = [copy.deepcopy(k) for k in call.keywords if k.arg not in ("out", "domain")]
        outputs = next(k.value.elts for k in call.keywords if k.arg == "out")
        for wrapper_name, indices in (("_causal_rho_pair", [0, 1]), ("_causal_vn", [3])):
            wrapper = copy.deepcopy(fo)
            wrapper.name = new_name + wrapper_name
            names = [ast.Name(id=f"causal_output_{i}", ctx=ast.Store()) for i in range(4)]
            assignment = ast.Assign(
                targets=[ast.Tuple(elts=names, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=fo_name, ctx=ast.Load()),
                    args=[],
                    keywords=copy.deepcopy(kwargs),
                ),
            )
            ret = [ast.Name(id=f"causal_output_{i}", ctx=ast.Load()) for i in indices]
            wrapper.body = [
                assignment,
                ast.Return(value=ast.Tuple(elts=ret, ctx=ast.Load()) if len(ret) > 1 else ret[0]),
            ]
            wrapper.returns = (
                copy.deepcopy(fo.returns.slice.elts[0])
                if len(indices) == 1
                else ast.Subscript(
                    value=ast.Name(id="tuple", ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[copy.deepcopy(fo.returns.slice.elts[i]) for i in indices],
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                )
            )
            extra.append(wrapper)
        program.body = []
        for kind, start, end, indices in bands:
            f = (
                fo_name
                if kind == "interior"
                else new_name + ("_causal_vn" if kind == "vn_boundary" else "_causal_rho_pair")
            )
            domain = ast.parse(
                "{dims.EdgeDim: (%d, %d), dims.KDim: (vertical_start, vertical_end)}"
                % (start, end),
                mode="eval",
            ).body
            outs = [copy.deepcopy(outputs[i]) for i in indices]
            out = ast.Tuple(elts=outs, ctx=ast.Load()) if len(outs) > 1 else outs[0]
            program.body.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id=f, ctx=ast.Load()),
                        args=[],
                        keywords=copy.deepcopy(kwargs)
                        + [
                            ast.keyword(arg="out", value=out),
                            ast.keyword(arg="domain", value=domain),
                        ],
                    )
                )
            )
    result = ast.Module(body=extra + [program], type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(result)) + "\n"


def timing_schedule(quartets, seed, placebo=True):
    """Balance both arm orders locally and interleave matched A/A controls."""
    require(quartets >= 4 and quartets % 2 == 0, "Use an even quartet count of at least four.")
    rng = random.Random(seed)
    schedule = []
    for pair in range(quartets // 2):
        orders = ["ABBA", "BAAB"]
        rng.shuffle(orders)
        for offset, order in enumerate(orders):
            phases = ["placebo", "intervention"] if placebo else ["intervention"]
            rng.shuffle(phases)
            schedule.extend((phase, 2 * pair + offset, order) for phase in phases)
    return schedule


def paired_summary(blocks):
    """ABBA blocks are sampling units; preserve interaction and signed effects."""
    require(len(blocks) >= 4 and len(blocks) % 4 == 0, "Expected complete ABBA quartets.")
    contrasts, ratios, orders = [], [], []
    arm_quartets = {"A": [], "B": []}
    for i in range(0, len(blocks), 4):
        group = blocks[i : i + 4]
        require(
            [b["arm"] for b in group] in (["A", "B", "B", "A"], ["B", "A", "A", "B"]),
            "Unbalanced timing order.",
        )
        orders.append("".join(b["arm"] for b in group))
        values = {
            arm: statistics.mean(b["device_ms"] for b in group if b["arm"] == arm)
            for arm in ("A", "B")
        }
        require(values["A"] > 0 and values["B"] > 0, "Nonpositive device time.")
        contrasts.append(values["A"] - values["B"])
        ratios.append(values["B"] / values["A"])
        for arm in arm_quartets:
            arm_quartets[arm].append(values[arm])
    return dict(
        saved_ms=statistics.mean(contrasts),
        variant_over_baseline=statistics.mean(ratios),
        quartet_saved_ms=contrasts,
        quartet_orders=orders,
        quartet_ratios=ratios,
        quartet_arm_ms=arm_quartets,
        quartet_stdev_ms={
            name: statistics.stdev(values) if len(values) >= 2 else None
            for name, values in dict(arm_quartets, contrast=contrasts).items()
        },
        all_quartets_faster=all(d > 0 for d in contrasts),
    )


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()
