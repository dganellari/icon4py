#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Export precipitation_effects_native_transposed to StableHLO for optimization.

This exports the TRANSPOSED (nlev, ncells) version of precipitation_effects
for use with graupel_native_transposed.

The exported HLO:
- Expects inputs in (nlev, ncells) layout
- Returns outputs in (nlev, ncells) layout
- Can be optimized and injected via optimized_precip_transposed_p primitive

Usage:
    JAX_ENABLE_X64=1 python tools/export_precip_transposed.py --input <netcdf_file>
    JAX_ENABLE_X64=1 python tools/export_precip_transposed.py  # uses dummy data
"""

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp


# Enable x64 precision for float64 support
jax.config.update("jax_enable_x64", True)

# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

from muphys_jax.core.common import constants as const
from muphys_jax.core.definitions import Q


def load_precip_inputs_transposed(input_file: str = None, timestep: int = 0):
    """Load inputs in TRANSPOSED (nlev, ncells) layout."""

    if input_file:
        print(f"Loading inputs from: {input_file}")
        from muphys_jax.utils.data_loading import load_precip_inputs as _load_precip_inputs

        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev = (
            _load_precip_inputs(input_file, timestep)
        )

        # Transpose from (ncells, nlev) to (nlev, ncells)
        kmin_r = jnp.transpose(kmin_r)
        kmin_i = jnp.transpose(kmin_i)
        kmin_s = jnp.transpose(kmin_s)
        kmin_g = jnp.transpose(kmin_g)
        q = Q(
            v=jnp.transpose(q.v),
            c=jnp.transpose(q.c),
            r=jnp.transpose(q.r),
            s=jnp.transpose(q.s),
            i=jnp.transpose(q.i),
            g=jnp.transpose(q.g),
        )
        t = jnp.transpose(t)
        rho = jnp.transpose(rho)
        dz = jnp.transpose(dz)

        print(f"  Grid: {nlev} levels × {ncells} cells (transposed)")
        return last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev
    else:
        # Create dummy data in TRANSPOSED layout (nlev, ncells)
        ncells = 327680
        nlev = 90
        print(f"Using dummy data: {nlev} levels × {ncells} cells (transposed)")

        # Data in (nlev, ncells) layout
        dz = jnp.ones((nlev, ncells), dtype=jnp.float64) * 100.0
        t = jnp.ones((nlev, ncells), dtype=jnp.float64) * 280.0
        rho = jnp.ones((nlev, ncells), dtype=jnp.float64) * 1.0

        q = Q(
            v=jnp.ones((nlev, ncells), dtype=jnp.float64) * 0.01,
            c=jnp.ones((nlev, ncells), dtype=jnp.float64) * 1e-5,
            r=jnp.ones((nlev, ncells), dtype=jnp.float64) * 1e-6,
            s=jnp.ones((nlev, ncells), dtype=jnp.float64) * 1e-6,
            i=jnp.ones((nlev, ncells), dtype=jnp.float64) * 1e-7,
            g=jnp.ones((nlev, ncells), dtype=jnp.float64) * 1e-7,
        )

        dt = 30.0
        last_lev = nlev - 1

        # Compute kmin masks in transposed layout
        kmin_r = q.r > const.qmin
        kmin_i = q.i > const.qmin
        kmin_s = q.s > const.qmin
        kmin_g = q.g > const.qmin

        return last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev


def export_precip_transposed_hlo(input_file=None, skip_compile=False, output_dir="."):
    """Export transposed precipitation_effects to HLO.

    Args:
        input_file: NetCDF file with input data (optional)
        skip_compile: Skip compilation step
        output_dir: Output directory for HLO files
    """
    print("=" * 80)
    print("EXPORTING: precipitation_effects_native_transposed")
    print("Layout: (nlev, ncells) - TRANSPOSED")
    print("=" * 80)

    # Import the native transposed implementation
    from muphys_jax.implementations.graupel_native_transposed import (
        _precipitation_effects_native_transposed_jax,
    )

    # Load inputs in transposed layout
    last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev = (
        load_precip_inputs_transposed(input_file)
    )

    print("\nFunction: _precipitation_effects_native_transposed_jax")
    print(f"  Layout: (nlev={nlev}, ncells={ncells})")
    print("  Inputs: last_lev (scalar), kmin_r/i/s/g (bool), q (Q namedtuple), t, rho, dz, dt")
    print("  Outputs: qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx")

    # Create function with explicit array arguments (no closure captures)
    def precip_transposed_fn(
        kmin_r_arg,
        kmin_i_arg,
        kmin_s_arg,
        kmin_g_arg,
        q_v_arg,
        q_c_arg,
        q_r_arg,
        q_s_arg,
        q_i_arg,
        q_g_arg,
        t_arg,
        rho_arg,
        dz_arg,
    ):
        from muphys_jax.core.definitions import Q

        q_arg = Q(v=q_v_arg, c=q_c_arg, r=q_r_arg, s=q_s_arg, i=q_i_arg, g=q_g_arg)
        # last_lev and dt are constants baked into the HLO
        return _precipitation_effects_native_transposed_jax(
            last_lev,
            kmin_r_arg,
            kmin_i_arg,
            kmin_s_arg,
            kmin_g_arg,
            q_arg,
            t_arg,
            rho_arg,
            dz_arg,
            dt,
        )

    print("\nLowering...")
    jitted = jax.jit(precip_transposed_fn)

    # Lower with concrete shapes (transposed: nlev×ncells)
    lowered = jitted.lower(kmin_r, kmin_i, kmin_s, kmin_g, q.v, q.c, q.r, q.s, q.i, q.g, t, rho, dz)

    # Get StableHLO and HLO
    stablehlo_text = lowered.as_text()
    hlo_text = lowered.as_text(dialect="hlo")

    precision = "x64" if jax.config.jax_enable_x64 else "x32"
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(exist_ok=True)

    output_name = "precip_transposed"

    # Save StableHLO
    stablehlo_file = out_path / f"{output_name}_{precision}_lowered.stablehlo"
    with open(stablehlo_file, "w") as f:
        f.write(stablehlo_text)
    print(f"✓ StableHLO: {stablehlo_file}")
    print(f"  Size: {len(stablehlo_text) / 1024 / 1024:.2f} MB")

    # Save HLO
    hlo_file = out_path / f"{output_name}_{precision}_lowered.hlo"
    with open(hlo_file, "w") as f:
        f.write(hlo_text)
    print(f"✓ HLO: {hlo_file}")
    print(f"  Size: {len(hlo_text) / 1024 / 1024:.2f} MB")

    # Analyze IR
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    while_count = stablehlo_text.count("stablehlo.while")
    scan_count = hlo_text.count("while (")
    dynamic_slice = stablehlo_text.count("dynamic_slice")
    dynamic_update = stablehlo_text.count("dynamic_update")

    print(f"  While loops (StableHLO): {while_count}")
    print(f"  While loops (HLO):       {scan_count}")
    print(f"  dynamic_slice:           {dynamic_slice}")
    print(f"  dynamic_update:          {dynamic_update}")
    print(f"  Total D2D ops:           {dynamic_slice + dynamic_update}")

    if not skip_compile:
        print("\nCompiling (to verify correctness)...")
        try:
            compiled = lowered.compile()
            print("✓ Compilation successful")

            # Quick correctness test
            print("\nRunning correctness test...")
            result = jitted(
                kmin_r, kmin_i, kmin_s, kmin_g, q.v, q.c, q.r, q.s, q.i, q.g, t, rho, dz
            )
            print(f"✓ Execution successful, got {len(result)} outputs")

            # Verify output shapes are transposed
            print(f"\nOutput shapes (should be nlev×ncells = {nlev}×{ncells}):")
            output_names = [
                "qr",
                "qs",
                "qi",
                "qg",
                "t_new",
                "pflx_tot",
                "pr",
                "ps",
                "pi",
                "pg",
                "eflx",
            ]
            for name, arr in zip(output_names, result):
                print(f"  {name}: {arr.shape}")
                assert arr.shape == (nlev, ncells), f"Wrong shape for {name}!"

        except Exception as e:
            print(f"⚠ Compilation/execution failed: {e}")
            import traceback

            traceback.print_exc()

    # Print usage instructions
    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print(f"""
To use this HLO with graupel_native_transposed:

    from muphys_jax.core.optimized_precip import configure_optimized_precip

    # Configure HLO injection for transposed layout
    configure_optimized_precip(
        hlo_path="{stablehlo_file.absolute()}",
        use_optimized=True,
        transposed=True  # IMPORTANT: tells the primitive the HLO is transposed
    )

    # Then use graupel_native_transposed normally
    from muphys_jax.implementations.graupel_native_transposed import graupel_run_native_transposed
    result = graupel_run_native_transposed(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
""")

    return str(stablehlo_file), str(hlo_file)


def main():
    parser = argparse.ArgumentParser(
        description="Export precipitation_effects_native_transposed to StableHLO/HLO"
    )
    parser.add_argument("--input", "-i", type=str, help="Input netCDF file")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument(
        "--output-dir", "-o", type=str, default="stablehlo", help="Output directory"
    )

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    export_precip_transposed_hlo(
        input_file=args.input, skip_compile=args.skip_compile, output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
