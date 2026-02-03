#!/usr/bin/env python3
"""
Export full graupel function as StableHLO for optimization.

This exports the ENTIRE graupel computation (not just precipitation_effects)
as StableHLO, which can then be optimized and injected back.

Usage:
    # Export with transposed layout (recommended for GPU)
    python export_graupel_stablehlo.py --input data.nc -o shlo/graupel_full_transposed.stablehlo

    # Export with original layout
    python export_graupel_stablehlo.py --input data.nc --no-transpose -o shlo/graupel_full_original.stablehlo

    # Then benchmark:
    python benchmark_stablehlo.py shlo/graupel_full_transposed.stablehlo --input data.nc --transposed
"""

import argparse
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import netCDF4

jax.config.update("jax_enable_x64", True)

from muphys_jax.core.definitions import Q


def load_inputs(input_file: str, transposed: bool = False):
    """Load inputs from NetCDF file."""
    print(f"Loading inputs from: {input_file}")
    ds = netCDF4.Dataset(input_file, 'r')

    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    print(f"  Grid: {ncells} cells × {nlev} levels")

    def _calc_dz(z: np.ndarray) -> np.ndarray:
        ksize = z.shape[0]
        dz = np.zeros(z.shape, np.float64)
        zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
        for k in range(ksize - 1, -1, -1):
            zh_new = 2.0 * z[k, :] - zh
            dz[k, :] = -zh + zh_new
            zh = zh_new
        return dz

    dz = np.transpose(_calc_dz(ds.variables["zg"][:]))

    def load_var(varname: str) -> np.ndarray:
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[0, :, :]
        return np.transpose(np.array(var, dtype=np.float64))

    # Load variables - all in (ncells, nlev) layout
    qv = load_var("hus")
    qc = load_var("clw")
    qr = load_var("qr")
    qs = load_var("qs")
    qi = load_var("cli")
    qg = load_var("qg")
    t = load_var("ta")
    p = load_var("pfull")
    rho = load_var("rho")

    # QNC (cloud droplet number concentration)
    qnc = np.ones_like(qv) * 500.0e6

    dt = 30.0

    ds.close()

    # Transpose if requested
    if transposed:
        print("  Transposing to (nlev, ncells) layout...")
        dz = np.transpose(dz)
        t = np.transpose(t)
        p = np.transpose(p)
        rho = np.transpose(rho)
        qv = np.transpose(qv)
        qc = np.transpose(qc)
        qr = np.transpose(qr)
        qs = np.transpose(qs)
        qi = np.transpose(qi)
        qg = np.transpose(qg)
        qnc = np.transpose(qnc)

    q = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)

    return dz, t, p, rho, q, dt, qnc, ncells, nlev


def export_graupel_stablehlo(input_file: str, output_file: str, transposed: bool = True):
    """Export full graupel as StableHLO."""
    # Load inputs
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_inputs(input_file, transposed=transposed)

    if transposed:
        layout_str = f"nlev×ncells = {nlev}×{ncells}"
        from muphys_jax.implementations.graupel_transposed import graupel_transposed

        def graupel_fn(dz, te, p, rho, q_v, q_c, q_r, q_s, q_i, q_g, qnc):
            q = Q(v=q_v, c=q_c, r=q_r, s=q_s, i=q_i, g=q_g)
            last_level = nlev - 1
            return graupel_transposed(last_level, dz, te, p, rho, q, dt, qnc)
    else:
        layout_str = f"ncells×nlev = {ncells}×{nlev}"
        from muphys_jax.implementations.graupel_baseline import graupel

        def graupel_fn(dz, te, p, rho, q_v, q_c, q_r, q_s, q_i, q_g, qnc):
            q = Q(v=q_v, c=q_c, r=q_r, s=q_s, i=q_i, g=q_g)
            last_level = nlev - 1
            return graupel(last_level, dz, te, p, rho, q, dt, qnc)

    print(f"\nExporting full graupel with {layout_str} layout...")

    # Convert to JAX arrays
    dz_jax = jnp.array(dz)
    t_jax = jnp.array(t)
    p_jax = jnp.array(p)
    rho_jax = jnp.array(rho)
    qv_jax = jnp.array(q.v)
    qc_jax = jnp.array(q.c)
    qr_jax = jnp.array(q.r)
    qs_jax = jnp.array(q.s)
    qi_jax = jnp.array(q.i)
    qg_jax = jnp.array(q.g)
    qnc_jax = jnp.array(qnc)

    # Lower to StableHLO
    print("  Lowering to StableHLO...")
    lowered = jax.jit(graupel_fn).lower(
        dz_jax, t_jax, p_jax, rho_jax,
        qv_jax, qc_jax, qr_jax, qs_jax, qi_jax, qg_jax,
        qnc_jax
    )

    # Get StableHLO text
    stablehlo_text = lowered.as_text()

    # Write to file
    with open(output_file, 'w') as f:
        f.write(stablehlo_text)

    print(f"  Written to: {output_file}")
    print(f"  File size: {len(stablehlo_text) / 1024:.1f} KB")

    # Check for while loops
    if 'stablehlo.while' in stablehlo_text:
        print("  NOTE: Contains while loops (expected for precipitation scan)")
    else:
        print("  SUCCESS: No while loops")

    return stablehlo_text


def main():
    parser = argparse.ArgumentParser(description="Export full graupel as StableHLO")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("-o", "--output", help="Output StableHLO file")
    parser.add_argument("--no-transpose", action="store_true",
                       help="Use original (ncells, nlev) layout instead of transposed")
    args = parser.parse_args()

    transposed = not args.no_transpose

    # Default output filename
    if args.output:
        output_file = args.output
    else:
        layout = "transposed" if transposed else "original"
        output_file = f"shlo/graupel_full_{layout}.stablehlo"

    print("=" * 70)
    print("Export Full Graupel as StableHLO")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    export_graupel_stablehlo(args.input, output_file, transposed=transposed)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nTo benchmark:")
    if transposed:
        print(f"  python benchmark_stablehlo.py {output_file} --input {args.input} --transposed")
    else:
        print(f"  python benchmark_stablehlo.py {output_file} --input {args.input}")


if __name__ == "__main__":
    main()
