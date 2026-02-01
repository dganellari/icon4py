#!/usr/bin/env python3
"""
Export StableHLO IR from JAX scans for analysis and optimization.

Usage:
    python export_stablehlo.py --mode simple
    python export_stablehlo.py --mode simple-transpose
    python export_stablehlo.py --mode baseline
    python export_stablehlo.py --mode allinone
    python export_stablehlo.py --mode default --fused
"""

import argparse
import sys
import pathlib

import jax
import jax.numpy as jnp
from jax import lax
import netCDF4
import numpy as np

# Enable x64 precision for float64 support
jax.config.update("jax_enable_x64", True)


def export_simple_scan():
    """Export simple scan without transpose."""
    print("=" * 80)
    print("SIMPLE SCAN (no transpose)")
    print("=" * 80)

    def scan_fn(carry, x):
        return carry + x, carry

    # Simple version
    result_fn = jax.jit(lambda: lax.scan(scan_fn, 0, jnp.arange(10)))
    stablehlo_ir = result_fn.lower().as_text()

    output_file = "stablehlo_simple_scan.stablehlo"
    with open(output_file, 'w') as f:
        f.write(stablehlo_ir)

    print(f"✓ Exported to: {output_file}")
    print(f"  Size: {len(stablehlo_ir)} bytes")
    print(f"\nFirst 1000 chars:\n{stablehlo_ir[:1000]}\n")

    return output_file


def export_simple_transpose_scan():
    """Export simple scan with transpose (to check D2D)."""
    print("=" * 80)
    print("SIMPLE SCAN WITH TRANSPOSE")
    print("=" * 80)

    def scan_fn(carry, x):
        return carry + x, carry

    # With transpose
    result_fn = jax.jit(lambda x, y: lax.scan(scan_fn, y, x.T))
    lowered = result_fn.lower(jnp.zeros((5, 5)), jnp.zeros(5))

    # Get both lowered and compiled versions
    stablehlo_lowered = lowered.as_text()
    stablehlo_compiled = lowered.compile().as_text()

    # Save lowered
    output_lowered = "stablehlo_simple_transpose_lowered.stablehlo"
    with open(output_lowered, 'w') as f:
        f.write(stablehlo_lowered)
    print(f"✓ Lowered IR: {output_lowered} ({len(stablehlo_lowered)} bytes)")

    # Save compiled
    output_compiled = "stablehlo_simple_transpose_compiled.stablehlo"
    with open(output_compiled, 'w') as f:
        f.write(stablehlo_compiled)
    print(f"✓ Compiled IR: {output_compiled} ({len(stablehlo_compiled)} bytes)")

    # Count D2D operations
    d2d_lowered = stablehlo_lowered.count('dynamic_slice') + stablehlo_lowered.count('dynamic_update')
    d2d_compiled = stablehlo_compiled.count('dynamic_slice') + stablehlo_compiled.count('dynamic_update')

    print(f"\nD2D operations:")
    print(f"  Lowered:  {d2d_lowered}")
    print(f"  Compiled: {d2d_compiled}")

    return output_lowered, output_compiled


def load_real_inputs(input_file: str, timestep: int = 0):
    """Load real inputs from netCDF file."""
    print(f"Loading inputs from: {input_file}")

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.core.definitions import Q

    ds = netCDF4.Dataset(input_file, 'r')

    # Get dimensions (matching run_graupel_jax.py)
    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    print(f"  Grid: {ncells} cells × {nlev} levels")

    # Calculate dz from geometric height (matching run_graupel_jax.py)
    def _calc_dz(z: np.ndarray) -> np.ndarray:
        ksize = z.shape[0]
        dz = np.zeros(z.shape, np.float64)
        zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
        for k in range(ksize - 1, -1, -1):
            zh_new = 2.0 * z[k, :] - zh
            dz[k, :] = -zh + zh_new
            zh = zh_new
        return dz

    dz_calc = _calc_dz(ds.variables["zg"])
    dz = jnp.array(np.transpose(dz_calc), dtype=jnp.float64)  # (height, ncells) -> (ncells, height)

    # Load variables (transpose from (height, ncells) to (ncells, height))
    def load_var(varname: str) -> jnp.ndarray:
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[timestep, :, :]
        return jnp.array(np.transpose(var), dtype=jnp.float64)

    # Variable names from run_graupel_jax.py
    q = Q(
        v=load_var("hus"),   # specific humidity (vapor)
        c=load_var("clw"),   # cloud liquid water
        r=load_var("qr"),    # rain
        s=load_var("qs"),    # snow
        i=load_var("cli"),   # cloud ice
        g=load_var("qg"),    # graupel
    )

    t = load_var("ta")
    p = load_var("pfull")
    rho = load_var("rho")

    ds.close()

    return dz, t, p, rho, q, ncells, nlev


def export_input_data(input_file: str, output_dir: str = "inputs", timestep: int = 0):
    """Export input data as raw .bin files for run_hlo_module.

    The order matches how JAX flattens the graupel_run arguments:
    graupel_run(dz, te, p, rho, q_in, dt, qnc)
    where q_in = Q(v, c, r, s, i, g) is a namedtuple that gets flattened.

    JAX flattens pytrees in alphabetical order for namedtuples, so Q fields become:
    c, g, i, r, s, v (alphabetically sorted by field name)
    """
    print("=" * 80)
    print("EXPORTING INPUT DATA (raw .bin files for run_hlo_module)")
    print("=" * 80)

    dz, t, p, rho, q, ncells, nlev = load_real_inputs(input_file, timestep)

    # Create output directory
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(exist_ok=True)

    # Export each array as raw binary (little-endian float64)
    # Order MUST match JAX's pytree flattening of graupel_run arguments
    # The Q namedtuple fields are sorted alphabetically: c, g, i, r, s, v
    arrays = [
        ("dz", np.array(dz)),           # arg0
        ("te", np.array(t)),            # arg1 (te = temperature)
        ("p", np.array(p)),             # arg2
        ("rho", np.array(rho)),         # arg3
        ("q_c", np.array(q.c)),         # arg4 (Q.c - cloud water)
        ("q_g", np.array(q.g)),         # arg5 (Q.g - graupel)
        ("q_i", np.array(q.i)),         # arg6 (Q.i - ice)
        ("q_r", np.array(q.r)),         # arg7 (Q.r - rain)
        ("q_s", np.array(q.s)),         # arg8 (Q.s - snow)
        ("q_v", np.array(q.v)),         # arg9 (Q.v - vapor)
    ]
    # Note: dt and qnc are scalars, typically compiled as constants

    bin_files = []
    shapes = []
    total_size = 0

    for i, (name, arr) in enumerate(arrays):
        bin_file = out_path / f"input_{i}_{name}.bin"
        arr.astype('<f8').tofile(bin_file)  # little-endian float64
        bin_files.append(str(bin_file))
        shapes.append(f"f64[{arr.shape[0]},{arr.shape[1]}]")
        total_size += bin_file.stat().st_size
        print(f"  {i}: {bin_file.name} - shape {arr.shape} - {bin_file.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n✓ Exported {len(arrays)} input files to: {output_dir}/")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Grid: {ncells} cells × {nlev} levels")

    # Print run_hlo_module command
    print("\n" + "=" * 80)
    print("RUN_HLO_MODULE COMMAND:")
    print("=" * 80)

    bin_file_list = ",".join(bin_files)
    shape_list = ",".join(shapes)

    print(f"""
run_hlo_module \\
  --platform=cuda \\
  --input_format=stablehlo \\
  --input_shapes='{shape_list}' \\
  stablehlo_graupel_baseline_x64_lowered.stablehlo \\
  --input_files='{bin_file_list}'
""")

    print("NOTE: Verify input order matches the StableHLO function signature.")
    print("Check the first few lines of the .stablehlo file for the @main function args.")

    return bin_files, shapes


def export_graupel_scan(mode='baseline', input_file=None, skip_compile=False, with_args=False, **run_kwargs):
    """Export StableHLO from graupel implementations.

    Args:
        mode: Implementation mode (baseline, allinone, default)
        input_file: Input netCDF file for real data shapes
        skip_compile: Skip XLA compilation (only export lowered IR)
        with_args: If True, export with explicit function arguments instead of closure.
                   This allows run_hlo_module to provide inputs externally.
    """
    print("=" * 80)
    print(f"GRAUPEL SCAN - MODE: {mode.upper()}")
    if with_args:
        print("(with explicit arguments for run_hlo_module)")
    print("=" * 80)

    # Import here to avoid issues if module not available
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

    from muphys_jax.core.definitions import Q

    # Select implementation
    if mode == 'baseline':
        from muphys_jax.implementations.graupel_baseline import graupel_run
        output_prefix = "stablehlo_graupel_baseline"
    elif mode == 'allinone':
        from muphys_jax.implementations.graupel_allinone_fused import graupel_allinone_fused_run as graupel_run
        output_prefix = "stablehlo_graupel_allinone"
    else:  # default
        from muphys_jax.implementations.graupel import graupel_run
        output_prefix = f"stablehlo_graupel_default"
        if run_kwargs.get('use_fused_scans'):
            output_prefix += "_fused"
        if run_kwargs.get('use_triton'):
            output_prefix += "_triton"
        if run_kwargs.get('use_mlir'):
            output_prefix += "_mlir"

    if with_args:
        output_prefix += "_withargs"

    print(f"Implementation: {graupel_run.__module__}.{graupel_run.__name__}")
    print(f"Run kwargs: {run_kwargs}")

    # Load inputs - real file or dummy data
    if input_file:
        dz, t, p, rho, q, ncells, nlev = load_real_inputs(input_file)
        output_prefix += "_full"
    else:
        # Create minimal test inputs (small size for faster compilation)
        ncells = 100
        nlev = 10

        dz = jnp.ones((ncells, nlev), dtype=jnp.float64) * 100.0
        t = jnp.ones((ncells, nlev), dtype=jnp.float64) * 280.0
        p = jnp.ones((ncells, nlev), dtype=jnp.float64) * 80000.0
        rho = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1.0

        q = Q(
            v=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.01,
            c=jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-5,
            r=jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-6,
            s=jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-6,
            i=jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-7,
            g=jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-7,
        )

    dt = 30.0
    qnc = 100.0

    print(f"\nInputs: {ncells} cells × {nlev} levels")

    # Lower to StableHLO and HLO
    print("\nLowering...")

    if with_args:
        # Export with explicit arguments - allows run_hlo_module to provide inputs
        # Function signature: graupel_run(dz, te, p, rho, q_in, dt, qnc)
        # q_in is a Q namedtuple that JAX flattens alphabetically: c, g, i, r, s, v

        def run_fn_with_args(dz_arg, te_arg, p_arg, rho_arg, q_c_arg, q_g_arg, q_i_arg, q_r_arg, q_s_arg, q_v_arg):
            q_arg = Q(v=q_v_arg, c=q_c_arg, r=q_r_arg, s=q_s_arg, i=q_i_arg, g=q_g_arg)
            return graupel_run(dz_arg, te_arg, p_arg, rho_arg, q_arg, dt, qnc, **run_kwargs)

        jitted = jax.jit(run_fn_with_args)
        # Lower with abstract shapes matching the input data
        lowered = jitted.lower(dz, t, p, rho, q.c, q.g, q.i, q.r, q.s, q.v)

        print(f"  Function has 10 input arguments (scalars dt={dt}, qnc={qnc} are constants)")
    else:
        # Original closure-based export (inputs baked into HLO as constants)
        def run_fn():
            return graupel_run(dz, t, p, rho, q, dt, qnc, **run_kwargs)

        jitted = jax.jit(run_fn)
        lowered = jitted.lower()

        print("  ⚠ Closure-based export: inputs are baked into HLO as constants")
        print("    Use --with-args to export with explicit function arguments")

    # Get StableHLO (MLIR format)
    stablehlo_lowered = lowered.as_text()

    # Get HLO text format (what run_hlo_module expects)
    hlo_lowered = lowered.as_text(dialect='hlo')

    # Save both formats
    precision = "x64" if jax.config.jax_enable_x64 else "x32"

    # StableHLO format
    output_stablehlo = f"{output_prefix}_{precision}_lowered.stablehlo"
    with open(output_stablehlo, 'w') as f:
        f.write(stablehlo_lowered)
    print(f"✓ StableHLO IR: {output_stablehlo}")
    print(f"  Size: {len(stablehlo_lowered)} bytes ({len(stablehlo_lowered)/1024:.1f} KB)")

    # HLO text format (for run_hlo_module)
    output_hlo = f"{output_prefix}_{precision}_lowered.hlo"
    with open(output_hlo, 'w') as f:
        f.write(hlo_lowered)
    print(f"✓ HLO text:     {output_hlo}")
    print(f"  Size: {len(hlo_lowered)} bytes ({len(hlo_lowered)/1024:.1f} KB)")

    output_lowered = output_stablehlo  # Keep for backwards compat

    # Compile and get optimized IR (unless skipped)
    if skip_compile:
        print("\n⚠ Skipping compilation (--skip-compile flag)")
        print("\n" + "=" * 80)
        print("ANALYSIS (Lowered IR only)")
        print("=" * 80)

        while_count = stablehlo_lowered.count('stablehlo.while')
        dynamic_slice_lowered = stablehlo_lowered.count('dynamic_slice')
        dynamic_update_lowered = stablehlo_lowered.count('dynamic_update')

        print(f"\nLowered IR:")
        print(f"  While loops:      {while_count}")
        print(f"  dynamic_slice:    {dynamic_slice_lowered}")
        print(f"  dynamic_update:   {dynamic_update_lowered}")
        print(f"  D2D operations:   {dynamic_slice_lowered + dynamic_update_lowered}")

        return output_lowered, None

    print("\nCompiling (this may take a while)...")
    compiled = lowered.compile()
    stablehlo_compiled = compiled.as_text()

    # Save compiled
    output_compiled = f"{output_prefix}_{precision}_compiled.stablehlo"
    with open(output_compiled, 'w') as f:
        f.write(stablehlo_compiled)
    print(f"✓ Compiled IR: {output_compiled}")
    print(f"  Size: {len(stablehlo_compiled)} bytes ({len(stablehlo_compiled)/1024:.1f} KB)")

    # Analyze
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    while_count = stablehlo_lowered.count('stablehlo.while')
    dynamic_slice_lowered = stablehlo_lowered.count('dynamic_slice')
    dynamic_update_lowered = stablehlo_lowered.count('dynamic_update')

    dynamic_slice_compiled = stablehlo_compiled.count('dynamic_slice')
    dynamic_update_compiled = stablehlo_compiled.count('dynamic_update')

    print(f"\nLowered IR:")
    print(f"  While loops:      {while_count}")
    print(f"  dynamic_slice:    {dynamic_slice_lowered}")
    print(f"  dynamic_update:   {dynamic_update_lowered}")
    print(f"  D2D operations:   {dynamic_slice_lowered + dynamic_update_lowered}")

    print(f"\nCompiled IR (after XLA optimization):")
    print(f"  dynamic_slice:    {dynamic_slice_compiled}")
    print(f"  dynamic_update:   {dynamic_update_compiled}")
    print(f"  D2D operations:   {dynamic_slice_compiled + dynamic_update_compiled}")

    reduction = 0
    if dynamic_slice_lowered + dynamic_update_lowered > 0:
        total_lowered = dynamic_slice_lowered + dynamic_update_lowered
        total_compiled = dynamic_slice_compiled + dynamic_update_compiled
        reduction = 100 * (1 - total_compiled / total_lowered)

    print(f"\nD2D reduction: {reduction:.1f}%")

    return output_lowered, output_compiled


def main():
    parser = argparse.ArgumentParser(description="Export StableHLO IR from JAX scans")
    parser.add_argument(
        '--mode',
        choices=['simple', 'simple-transpose', 'baseline', 'allinone', 'default'],
        default='simple',
        help='Which scan to export'
    )
    parser.add_argument('--input', '-i', type=str, help='Input netCDF file (for full graupel export)')
    parser.add_argument('--skip-compile', action='store_true', help='Skip compilation (only export lowered IR)')
    parser.add_argument('--export-inputs', action='store_true', help='Export input data as .bin files for run_hlo_module')
    parser.add_argument('--output-inputs', type=str, default='inputs', help='Output directory for .bin files')
    parser.add_argument('--fused', action='store_true', help='Use fused scans (default mode only)')
    parser.add_argument('--triton', action='store_true', help='Use Triton (default mode only)')
    parser.add_argument('--mlir', action='store_true', help='Use MLIR (default mode only)')
    parser.add_argument('--with-args', action='store_true',
                       help='Export with explicit function arguments (for run_hlo_module with external inputs)')

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    if args.export_inputs:
        if not args.input:
            print("ERROR: --export-inputs requires --input <netcdf_file>")
            sys.exit(1)
        export_input_data(args.input, args.output_inputs)
    elif args.mode == 'simple':
        export_simple_scan()
    elif args.mode == 'simple-transpose':
        export_simple_transpose_scan()
    else:
        # Graupel modes
        run_kwargs = {}
        if args.mode == 'default':
            run_kwargs = {
                'use_fused_scans': args.fused,
                'use_triton': args.triton,
                'use_mlir': args.mlir,
            }
        export_graupel_scan(
            mode=args.mode,
            input_file=args.input,
            skip_compile=args.skip_compile,
            with_args=args.with_args,
            **run_kwargs
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
