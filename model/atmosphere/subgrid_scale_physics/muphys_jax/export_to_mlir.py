#!/usr/bin/env python3
"""
Export JAX graupel to StableHLO MLIR for inspection and optimization.

Usage:
    PYTHONPATH=.:$PYTHONPATH python muphys_jax/export_to_mlir.py
"""

import os
import jax
import jax.numpy as jnp
from pathlib import Path

# Force float64
jax.config.update("jax_enable_x64", True)

from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.core.definitions import Q


def create_inputs(ncells=1024, nlev=90, dtype=jnp.float64):
    """Create inputs for graupel - smaller grid for readable MLIR."""
    dz = jnp.ones((ncells, nlev), dtype=dtype) * 100.0
    te = jnp.ones((ncells, nlev), dtype=dtype) * 273.15
    p = jnp.ones((ncells, nlev), dtype=dtype) * 101325.0
    rho = jnp.ones((ncells, nlev), dtype=dtype) * 1.2

    q_in = Q(
        v=jnp.ones((ncells, nlev), dtype=dtype) * 0.01,
        c=jnp.ones((ncells, nlev), dtype=dtype) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=dtype) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=dtype) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=dtype) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=dtype) * 0.0001,
    )

    return dz, te, p, rho, q_in, 30.0, 100.0


def export_graupel():
    """Export graupel to StableHLO MLIR (both unfused and fused versions)."""
    print("=" * 70)
    print("JAX Graupel → StableHLO MLIR Export")
    print("=" * 70)

    # Create output dir
    out_dir = Path("mlir_output")
    out_dir.mkdir(exist_ok=True)

    # Create inputs (float64)
    print("\n1. Creating float64 inputs (ncells=1024, nlev=90)...")
    args = create_inputs()
    print(f"   dtype: {args[0].dtype}")

    # Export both versions for comparison
    for use_fused in [False, True]:
        version = "fused" if use_fused else "unfused"
        print(f"\n2. Lowering to StableHLO ({version} scans)...")

        # Need to use partial to create a new function with use_fused_scans baked in
        from functools import partial
        graupel_fn = partial(graupel_run, use_fused_scans=use_fused)
        lowered = jax.jit(graupel_fn).lower(*args)

        # Get MLIR
        stablehlo = lowered.compiler_ir(dialect="stablehlo")
        hlo = lowered.compiler_ir(dialect="hlo")

        # Save
        stablehlo_file = out_dir / f"graupel_{version}_stablehlo.mlir"
        hlo_file = out_dir / f"graupel_{version}_hlo.mlir"

        print(f"   Saving to {out_dir}/...")
        with open(stablehlo_file, "w") as f:
            f.write(str(stablehlo))
        with open(hlo_file, "w") as f:
            f.write(str(hlo))

        print(f"   ✓ {stablehlo_file}")
        print(f"   ✓ {hlo_file}")

        # Analyze
        print(f"   Analyzing {version} MLIR...")
        analyze(str(stablehlo))

    # Also export simple scan
    export_simple_scan(out_dir)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Compare unfused vs fused:")
    print("     diff mlir_output/graupel_unfused_stablehlo.mlir mlir_output/graupel_fused_stablehlo.mlir")
    print("  2. grep 'stablehlo.while' mlir_output/graupel_*_stablehlo.mlir")
    print("  3. python mlir_passes/analyze_mlir.py mlir_output/graupel_fused_stablehlo.mlir")
    print("=" * 70)


def analyze(mlir_text):
    """Analyze MLIR operations."""
    lines = mlir_text.split('\n')

    # Count ops
    ops = {}
    for line in lines:
        for op in ['stablehlo.while', 'stablehlo.add', 'stablehlo.multiply',
                   'stablehlo.divide', 'stablehlo.power', 'stablehlo.select',
                   'stablehlo.compare', 'stablehlo.broadcast']:
            if op in line:
                ops[op] = ops.get(op, 0) + 1

    print(f"   Total lines: {len(lines)}")
    print(f"   Key operations:")
    for op, count in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"     {op}: {count}")

    # Count while loops (scans)
    whiles = mlir_text.count('stablehlo.while')
    print(f"\n   Vertical scans (stablehlo.while): {whiles}")
    print("   → These are the main optimization targets!")


def export_simple_scan(out_dir):
    """Export minimal scan for understanding."""
    print("\n5. Exporting simple scan example...")

    def simple_scan(xs):
        def step(carry, x):
            return carry + x, carry + x
        return jax.lax.scan(step, 0.0, xs)[1]

    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    lowered = jax.jit(simple_scan).lower(xs)
    mlir = lowered.compiler_ir(dialect="stablehlo")

    simple_file = out_dir / "simple_scan.mlir"
    with open(simple_file, "w") as f:
        f.write(str(mlir))
    print(f"   ✓ {simple_file}")


if __name__ == "__main__":
    export_graupel()
