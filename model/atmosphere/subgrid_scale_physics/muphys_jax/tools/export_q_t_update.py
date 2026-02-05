#!/usr/bin/env python3
"""
Export q_t_update (phase transitions) to StableHLO for optimization.

Similar to export_stablehlo.py but for the q_t_update function.

Usage:
    python export_q_t_update.py --input data.nc --output stablehlo/q_t_update_transposed.stablehlo --transposed
"""

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from muphys_jax.utils.data_loading import load_graupel_inputs
from muphys_jax.core.definitions import Q


def main():
    parser = argparse.ArgumentParser(description="Export q_t_update to StableHLO")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--output", "-o", default="stablehlo/q_t_update_transposed.stablehlo",
                       help="Output StableHLO file")
    parser.add_argument("--transposed", action="store_true",
                       help="Export in transposed (nlev, ncells) layout")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPORT q_t_update TO STABLEHLO")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Load data
    print("Loading data...")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(args.input)
    print(f"Grid: {ncells} cells x {nlev} levels")

    if args.transposed:
        print("Using transposed layout (nlev, ncells)")
        t = jnp.transpose(t)
        p = jnp.transpose(p)
        rho = jnp.transpose(rho)
        qnc = jnp.transpose(qnc)
        q = Q(
            v=jnp.transpose(q.v),
            c=jnp.transpose(q.c),
            r=jnp.transpose(q.r),
            s=jnp.transpose(q.s),
            i=jnp.transpose(q.i),
            g=jnp.transpose(q.g),
        )
        shape = (nlev, ncells)
    else:
        shape = (ncells, nlev)

    print(f"Shape: {shape}")

    # Import q_t_update
    from muphys_jax.implementations.graupel_native_transposed import q_t_update_native

    # Create wrapper that takes flat inputs and returns flat outputs
    # This makes the StableHLO signature cleaner
    def q_t_update_flat(t, p, rho, qv, qc, qr, qs, qi, qg, qnc):
        """Wrapper with flat inputs/outputs for cleaner StableHLO."""
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        q_out, t_out = q_t_update_native(t, p, rho, q_in, dt, qnc)
        return q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g, t_out

    # Lower to StableHLO
    print("\nLowering to StableHLO...")

    lowered = jax.jit(q_t_update_flat).lower(
        t, p, rho, q.v, q.c, q.r, q.s, q.i, q.g, qnc
    )

    # Get StableHLO text
    stablehlo_text = lowered.as_text()

    # Analyze
    num_while = stablehlo_text.count("stablehlo.while")
    num_select = stablehlo_text.count("stablehlo.select")
    num_multiply = stablehlo_text.count("stablehlo.multiply")
    num_add = stablehlo_text.count("stablehlo.add")
    num_divide = stablehlo_text.count("stablehlo.divide")
    num_compare = stablehlo_text.count("stablehlo.compare")
    num_power = stablehlo_text.count("stablehlo.power")
    num_exp = stablehlo_text.count("stablehlo.exponential")
    num_log = stablehlo_text.count("stablehlo.log")

    print(f"\nStableHLO analysis:")
    print(f"  Total size: {len(stablehlo_text):,} chars")
    print(f"  while loops: {num_while}")
    print(f"  select ops: {num_select}")
    print(f"  multiply ops: {num_multiply}")
    print(f"  add ops: {num_add}")
    print(f"  divide ops: {num_divide}")
    print(f"  compare ops: {num_compare}")
    print(f"  power ops: {num_power}")
    print(f"  exp ops: {num_exp}")
    print(f"  log ops: {num_log}")

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(stablehlo_text)

    print(f"\nSaved StableHLO to: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")

    # Print function signature
    print("\n" + "=" * 70)
    print("FUNCTION SIGNATURE")
    print("=" * 70)
    print(f"""
Inputs (10 arrays, all {shape}):
  %arg0: t       - temperature
  %arg1: p       - pressure
  %arg2: rho     - density
  %arg3: qv      - water vapor
  %arg4: qc      - cloud water
  %arg5: qr      - rain
  %arg6: qs      - snow
  %arg7: qi      - ice
  %arg8: qg      - graupel
  %arg9: qnc     - cloud number concentration

Outputs (7 arrays, all {shape}):
  qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, t_out

Note: dt is baked into the StableHLO as a constant (dt = {dt})
""")


if __name__ == "__main__":
    main()
