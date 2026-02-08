#!/usr/bin/env python3
"""
Export q_t_update_fused as standalone StableHLO for HLO injection.

This script:
1. Traces q_t_update_fused through JAX
2. Lowers it to StableHLO MLIR
3. Saves it for later injection via merge_mlir_modules

The exported StableHLO can be:
- Injected standalone (for q_t_update only)
- Combined with precip StableHLO via generate_combined_graupel.py

Usage:
    python generate_qt_update_stablehlo.py --nlev 90 --ncells 327680
    python generate_qt_update_stablehlo.py -o stablehlo/qt_update.stablehlo
"""

import argparse
import sys
import pathlib

import jax
import jax.numpy as jnp
from jax import lax

# Enable x64
jax.config.update("jax_enable_x64", True)

# Add parent to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

from muphys_jax.core.definitions import Q
from muphys_jax.implementations.q_t_update_fused import q_t_update_fused


def export_qt_update(nlev: int = 90, ncells: int = 327680, output: str = None) -> str:
    """
    Export q_t_update_fused as StableHLO.

    The function signature will be:
        @main(t, p, rho, qv, qc, qr, qs, qi, qg, dt_scalar, qnc_scalar)
            -> (qv_new, qc_new, qr_new, qs_new, qi_new, qg_new, t_new)

    But since dt and qnc are scalars baked in as constants (30.0 and 100.0),
    the actual signature is:
        @main(t, p, rho, qv, qc, qr, qs, qi, qg)
            -> (qv_new, qc_new, qr_new, qs_new, qi_new, qg_new, t_new)
    """
    if output is None:
        output = "stablehlo/qt_update.stablehlo"

    print(f"Exporting q_t_update_fused as StableHLO...")
    print(f"  Layout: tensor<{nlev}x{ncells}xf64> (nlev x ncells, transposed)")
    print(f"  Constants baked in: dt=30.0, qnc=100.0")

    # Create abstract input shapes - transposed layout (nlev, ncells)
    shape = (nlev, ncells)
    t = jnp.zeros(shape, dtype=jnp.float64)
    p = jnp.zeros(shape, dtype=jnp.float64)
    rho = jnp.zeros(shape, dtype=jnp.float64)
    qv = jnp.zeros(shape, dtype=jnp.float64)
    qc = jnp.zeros(shape, dtype=jnp.float64)
    qr = jnp.zeros(shape, dtype=jnp.float64)
    qs = jnp.zeros(shape, dtype=jnp.float64)
    qi = jnp.zeros(shape, dtype=jnp.float64)
    qg = jnp.zeros(shape, dtype=jnp.float64)

    dt = 30.0
    qnc = 100.0

    # Wrapper that takes flat arrays and returns flat arrays
    # (no Q namedtuple in the signature - just plain tensors)
    def qt_update_flat(t_in, p_in, rho_in, qv_in, qc_in, qr_in, qs_in, qi_in, qg_in):
        q_in = Q(v=qv_in, c=qc_in, r=qr_in, s=qs_in, i=qi_in, g=qg_in)
        q_out, t_out = q_t_update_fused(t_in, p_in, rho_in, q_in, dt, qnc)
        # Return flat: qv, qc, qr, qs, qi, qg, t
        return q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g, t_out

    # JIT and lower
    jitted = jax.jit(qt_update_flat)
    lowered = jitted.lower(t, p, rho, qv, qc, qr, qs, qi, qg)

    # Get StableHLO text
    stablehlo_text = lowered.as_text()

    print(f"  StableHLO size: {len(stablehlo_text) / 1024:.1f} KB")
    print(f"  Lines: {stablehlo_text.count(chr(10))}")

    # Count ops
    print(f"  While loops: {stablehlo_text.count('stablehlo.while')}")
    print(f"  Dynamic slices: {stablehlo_text.count('dynamic_slice')}")
    print(f"  Power ops: {stablehlo_text.count('stablehlo.power')}")
    print(f"  Select ops: {stablehlo_text.count('stablehlo.select')}")
    print(f"  Multiply ops: {stablehlo_text.count('stablehlo.multiply')}")

    # Save
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(stablehlo_text)

    print(f"\n  Written to: {output}")
    return stablehlo_text


def main():
    parser = argparse.ArgumentParser(description="Export q_t_update_fused as StableHLO")
    parser.add_argument("-o", "--output", help="Output file (default: stablehlo/qt_update.stablehlo)")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    export_qt_update(args.nlev, args.ncells, args.output)


if __name__ == "__main__":
    main()
