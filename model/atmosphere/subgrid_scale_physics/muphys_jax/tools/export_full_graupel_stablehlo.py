#!/usr/bin/env python3
"""
Export the FULL graupel function to StableHLO for analysis.

This exports the entire graupel computation (q_t_update + precipitation_effects)
as a single StableHLO module for potential end-to-end optimization.

Target: Full graupel from 33ms → 10-13ms

Usage:
    python export_full_graupel_stablehlo.py --input data.nc --output stablehlo/graupel_full_transposed.stablehlo
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
    parser = argparse.ArgumentParser(description="Export full graupel to StableHLO")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--output", "-o", default="stablehlo/graupel_full_transposed.stablehlo",
                       help="Output StableHLO file")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPORT FULL GRAUPEL TO STABLEHLO")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Load data
    print("Loading data...")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(args.input)
    print(f"Grid: {ncells} cells x {nlev} levels")

    # Transpose to (nlev, ncells)
    print("Using transposed layout (nlev, ncells)")
    t_t = jnp.transpose(t)
    p_t = jnp.transpose(p)
    rho_t = jnp.transpose(rho)
    dz_t = jnp.transpose(dz)
    qnc_t = jnp.transpose(qnc)
    q_t = Q(
        v=jnp.transpose(q.v),
        c=jnp.transpose(q.c),
        r=jnp.transpose(q.r),
        s=jnp.transpose(q.s),
        i=jnp.transpose(q.i),
        g=jnp.transpose(q.g),
    )

    # Import the native transposed graupel
    from muphys_jax.implementations.graupel_native_transposed import graupel_native_transposed

    # Create wrapper with flat inputs/outputs for cleaner StableHLO
    def graupel_flat(dz, t, p, rho, qv, qc, qr, qs, qi, qg, dt_scalar, qnc):
        """Wrapper with flat inputs."""
        last_level = t.shape[0] - 1  # nlev is first dim in transposed
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)

        t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_native_transposed(
            last_level, dz, t, p, rho, q_in, dt_scalar, qnc
        )

        return (t_out, q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g,
                pflx, pr, ps, pi, pg, pre)

    # Lower to StableHLO
    print("\nLowering full graupel to StableHLO...")

    lowered = jax.jit(graupel_flat).lower(
        dz_t, t_t, p_t, rho_t, q_t.v, q_t.c, q_t.r, q_t.s, q_t.i, q_t.g, dt, qnc_t
    )

    stablehlo_text = lowered.as_text()

    # Analyze
    num_while = stablehlo_text.count("stablehlo.while")
    num_select = stablehlo_text.count("stablehlo.select")
    num_multiply = stablehlo_text.count("stablehlo.multiply")
    num_add = stablehlo_text.count("stablehlo.add")
    num_divide = stablehlo_text.count("stablehlo.divide")
    num_compare = stablehlo_text.count("stablehlo.compare")
    num_slice = stablehlo_text.count("stablehlo.slice")
    num_concat = stablehlo_text.count("stablehlo.concatenate")
    num_power = stablehlo_text.count("stablehlo.power")

    print(f"\nStableHLO analysis:")
    print(f"  Total size: {len(stablehlo_text):,} chars")
    print(f"  while loops: {num_while}")
    print(f"  select ops: {num_select}")
    print(f"  multiply ops: {num_multiply}")
    print(f"  add ops: {num_add}")
    print(f"  divide ops: {num_divide}")
    print(f"  compare ops: {num_compare}")
    print(f"  slice ops: {num_slice}")
    print(f"  concatenate ops: {num_concat}")
    print(f"  power ops: {num_power}")

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(stablehlo_text)

    print(f"\nSaved StableHLO to: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")

    # Analysis summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 70)

    if num_while > 0:
        print(f"\n⚠️  Found {num_while} while loops")
        print("   These come from the lax.scan operations in:")
        print("   - 4 precipitation scans (one per species)")
        print("   - 1 temperature scan")
        print("\n   OPTIMIZATION: Unroll these 5 while loops for 2x+ speedup!")
    else:
        print("\n✓ No while loops - already fully unrolled")

    print(f"""
Current timing breakdown:
  - q_t_update: ~24ms (element-wise phase transitions)
  - precipitation scans: ~9-10ms (with HLO injection) or ~20ms (JAX)
  - Full graupel: ~33ms (with precip HLO) or ~45ms (pure JAX)

Target: 10-13ms

Options to reach target:
1. Unroll all 5 while loops in the StableHLO (precip scans + temp scan)
2. Fuse q_t_update operations into fewer kernels
3. Optimize the full graupel as one big unrolled StableHLO
4. Use XLA flags for better fusion: --xla_gpu_autofusion_size_threshold
""")

    # Also export just q_t_update if not present
    from muphys_jax.implementations.graupel_native_transposed import q_t_update_native

    qt_lowered = jax.jit(q_t_update_native).lower(t_t, p_t, rho_t, q_t, dt, qnc_t)
    qt_stablehlo = qt_lowered.as_text()

    qt_path = output_path.parent / "q_t_update_transposed.stablehlo"
    with open(qt_path, 'w') as f:
        f.write(qt_stablehlo)
    print(f"\nAlso saved q_t_update to: {qt_path}")


if __name__ == "__main__":
    main()
