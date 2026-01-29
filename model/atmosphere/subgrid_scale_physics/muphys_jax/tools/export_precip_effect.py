#!/usr/bin/env python3
"""
Export precipitation_effects function to StableHLO/HLO for optimization.

This isolates the core scan-heavy function for:
1. Export to HLO
2. Optimize with hlo-opt
3. Inject back via jax.extend.ffi or custom_call

Supports both:
- baseline: precipitation_effects (separate precip + temp scans)
- allinone: precipitation_effects_allinone_fused (single fused scan)

Usage:
    JAX_ENABLE_X64=1 python tools/export_precip_effect.py --input <netcdf_file>
    JAX_ENABLE_X64=1 python tools/export_precip_effect.py --mode allinone
    JAX_ENABLE_X64=1 python tools/export_precip_effect.py  # uses dummy data
"""

import argparse
import sys
import pathlib

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def load_precip_inputs(input_file: str = None, timestep: int = 0):
    """Load or create inputs for precipitation_effects."""

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.core.definitions import Q

    if input_file:
        import netCDF4
        print(f"Loading inputs from: {input_file}")

        ds = netCDF4.Dataset(input_file, 'r')

        try:
            ncells = len(ds.dimensions["cell"])
        except KeyError:
            ncells = len(ds.dimensions["ncells"])
        nlev = len(ds.dimensions["height"])

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
        dz = jnp.array(np.transpose(dz_calc), dtype=jnp.float64)

        def load_var(varname: str) -> jnp.ndarray:
            var = ds.variables[varname]
            if var.dimensions[0] == "time":
                var = var[timestep, :, :]
            return jnp.array(np.transpose(var), dtype=jnp.float64)

        q = Q(
            v=load_var("hus"),
            c=load_var("clw"),
            r=load_var("qr"),
            s=load_var("qs"),
            i=load_var("cli"),
            g=load_var("qg"),
        )

        t = load_var("ta")
        rho = load_var("rho")

        ds.close()
    else:
        # Create dummy data for shape analysis
        ncells = 327680
        nlev = 90
        print(f"Using dummy data: {ncells} cells × {nlev} levels")

        dz = jnp.ones((ncells, nlev), dtype=jnp.float64) * 100.0
        t = jnp.ones((ncells, nlev), dtype=jnp.float64) * 280.0
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
    last_lev = nlev - 1

    # Compute kmin masks (species present above threshold)
    from muphys_jax.core.common import constants as const
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    print(f"  Grid: {ncells} cells × {nlev} levels")

    return last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev


def export_precip_effect_hlo(input_file=None, skip_compile=False, output_dir=".", mode="baseline"):
    """Export precipitation_effects to HLO.

    Args:
        input_file: NetCDF file with input data (optional)
        skip_compile: Skip compilation step
        output_dir: Output directory for HLO files
        mode: "baseline" or "allinone"
    """
    print("=" * 80)
    print(f"EXPORTING: precipitation_effects ({mode})")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

    if mode == "allinone":
        from muphys_jax.implementations.graupel_allinone_fused import precipitation_effects_allinone_fused as precipitation_effects
        output_name = "precip_effect_allinone"
    else:
        from muphys_jax.implementations.graupel_baseline import precipitation_effects
        output_name = "precip_effect"

    # Load inputs
    last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev = \
        load_precip_inputs(input_file)

    print(f"\nFunction: precipitation_effects")
    print(f"  Inputs: last_lev (scalar), kmin_r/i/s/g (bool), q (Q namedtuple), t, rho, dz, dt")
    print(f"  Outputs: qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx")

    # Create function with explicit array arguments (no closure captures)
    # This ensures run_hlo_module can provide inputs
    def precip_effect_fn(
        kmin_r_arg, kmin_i_arg, kmin_s_arg, kmin_g_arg,
        q_v_arg, q_c_arg, q_r_arg, q_s_arg, q_i_arg, q_g_arg,
        t_arg, rho_arg, dz_arg
    ):
        from muphys_jax.core.definitions import Q
        q_arg = Q(v=q_v_arg, c=q_c_arg, r=q_r_arg, s=q_s_arg, i=q_i_arg, g=q_g_arg)
        # last_lev and dt are constants
        return precipitation_effects(
            last_lev, kmin_r_arg, kmin_i_arg, kmin_s_arg, kmin_g_arg,
            q_arg, t_arg, rho_arg, dz_arg, dt
        )

    print("\nLowering...")
    jitted = jax.jit(precip_effect_fn)

    # Lower with concrete shapes
    lowered = jitted.lower(
        kmin_r, kmin_i, kmin_s, kmin_g,
        q.v, q.c, q.r, q.s, q.i, q.g,
        t, rho, dz
    )

    # Get StableHLO and HLO
    stablehlo_text = lowered.as_text()
    hlo_text = lowered.as_text(dialect='hlo')

    precision = "x64" if jax.config.jax_enable_x64 else "x32"
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(exist_ok=True)

    # Save StableHLO
    stablehlo_file = out_path / f"{output_name}_{precision}_lowered.stablehlo"
    with open(stablehlo_file, 'w') as f:
        f.write(stablehlo_text)
    print(f"✓ StableHLO: {stablehlo_file}")
    print(f"  Size: {len(stablehlo_text) / 1024 / 1024:.2f} MB")

    # Save HLO
    hlo_file = out_path / f"{output_name}_{precision}_lowered.hlo"
    with open(hlo_file, 'w') as f:
        f.write(hlo_text)
    print(f"✓ HLO: {hlo_file}")
    print(f"  Size: {len(hlo_text) / 1024 / 1024:.2f} MB")

    # Export input data as .bin files
    print("\n" + "=" * 80)
    print("EXPORTING INPUT DATA (.bin files)")
    print("=" * 80)

    inputs_dir = out_path / f"{output_name}_inputs"
    inputs_dir.mkdir(exist_ok=True)

    # Order must match function signature
    arrays = [
        ("kmin_r", np.array(kmin_r)),
        ("kmin_i", np.array(kmin_i)),
        ("kmin_s", np.array(kmin_s)),
        ("kmin_g", np.array(kmin_g)),
        ("q_v", np.array(q.v)),
        ("q_c", np.array(q.c)),
        ("q_r", np.array(q.r)),
        ("q_s", np.array(q.s)),
        ("q_i", np.array(q.i)),
        ("q_g", np.array(q.g)),
        ("t", np.array(t)),
        ("rho", np.array(rho)),
        ("dz", np.array(dz)),
    ]

    bin_files = []
    for i, (name, arr) in enumerate(arrays):
        bin_file = inputs_dir / f"input_{i}_{name}.bin"
        if arr.dtype == bool:
            # HLO expects pred type - save as uint8
            arr.astype(np.uint8).tofile(bin_file)
            dtype_str = "pred"
        else:
            arr.astype('<f8').tofile(bin_file)
            dtype_str = "f64"
        bin_files.append(str(bin_file))
        print(f"  {i}: {bin_file.name} - {dtype_str}[{arr.shape[0]},{arr.shape[1]}]")

    # Analyze IR
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    while_count = stablehlo_text.count('stablehlo.while')
    scan_count = hlo_text.count('while (')
    dynamic_slice = stablehlo_text.count('dynamic_slice')
    dynamic_update = stablehlo_text.count('dynamic_update')

    print(f"  While loops (StableHLO): {while_count}")
    print(f"  While loops (HLO):       {scan_count}")
    print(f"  dynamic_slice:           {dynamic_slice}")
    print(f"  dynamic_update:          {dynamic_update}")
    print(f"  Total D2D ops:           {dynamic_slice + dynamic_update}")

    # Print run command
    print("\n" + "=" * 80)
    print("RUN COMMAND")
    print("=" * 80)
    print(f"""
run_hlo_module --platform=cuda --num_runs=100 {hlo_file}
""")

    if not skip_compile:
        print("\nCompiling (to verify correctness)...")
        try:
            compiled = lowered.compile()
            print("✓ Compilation successful")

            # Export serialized executable for injection
            serialized_file = out_path / f"{output_name}_{precision}.serialized"

            # Try different serialization methods based on JAX version
            try:
                # New API (JAX >= 0.4.20)
                serialized_hlo = compiled.runtime_executable().serialize()
            except AttributeError:
                try:
                    # Older API
                    serialized_hlo = compiled.as_serialized_hlo()
                except AttributeError:
                    # Alternative: get from executable
                    from jax._src import xla_bridge
                    backend = xla_bridge.get_backend()
                    # Compile and serialize directly
                    xla_comp = compiled.compiler_ir()
                    executable = backend.compile(xla_comp)
                    serialized_hlo = executable.serialize()

            with open(serialized_file, 'wb') as f:
                f.write(serialized_hlo)
            print(f"✓ Serialized executable: {serialized_file}")
            print(f"  Size: {len(serialized_hlo) / 1024 / 1024:.2f} MB")

            # Quick correctness test
            print("\nRunning correctness test...")
            result = jitted(
                kmin_r, kmin_i, kmin_s, kmin_g,
                q.v, q.c, q.r, q.s, q.i, q.g,
                t, rho, dz
            )
            print(f"✓ Execution successful, got {len(result)} outputs")

            # Print next steps
            print("\n" + "=" * 80)
            print("NEXT STEPS")
            print("=" * 80)
            print(f"""
1. Benchmark original HLO:
   run_hlo_module --platform=cuda --num_runs=100 {hlo_file.absolute()}

2. Transform with hlo-opt (unroll while loops to eliminate D2D):
   hlo-opt {stablehlo_file.absolute()} \\
       --stablehlo-legalize-to-hlo \\
       [your transforms] \\
       -o {out_path.absolute()}/{output_name}_optimized.hlo

3. Benchmark transformed HLO:
   run_hlo_module --platform=cuda --num_runs=100 {out_path.absolute()}/{output_name}_optimized.hlo

4. If faster, compile transformed HLO:
   python tools/compile_optimized_hlo.py \\
       -i {out_path.absolute()}/{output_name}_optimized.hlo \\
       -o {out_path.absolute()}/{output_name}_optimized.serialized
""")
        except Exception as e:
            print(f"⚠ Compilation/execution failed: {e}")

    return str(stablehlo_file), str(hlo_file)


def main():
    parser = argparse.ArgumentParser(
        description="Export precipitation_effects to StableHLO/HLO"
    )
    parser.add_argument('--input', '-i', type=str, help='Input netCDF file')
    parser.add_argument('--skip-compile', action='store_true', help='Skip compilation')
    parser.add_argument('--output-dir', '-o', type=str, default='shlo',
                       help='Output directory')
    parser.add_argument('--mode', '-m', choices=['baseline', 'allinone'], default='baseline',
                       help='Implementation mode: baseline (separate scans) or allinone (fused scan)')

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    export_precip_effect_hlo(
        input_file=args.input,
        skip_compile=args.skip_compile,
        output_dir=args.output_dir,
        mode=args.mode
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
