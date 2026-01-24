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

    output_file = "stablehlo_simple_scan.mlir"
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
    output_lowered = "stablehlo_simple_transpose_lowered.mlir"
    with open(output_lowered, 'w') as f:
        f.write(stablehlo_lowered)
    print(f"✓ Lowered IR: {output_lowered} ({len(stablehlo_lowered)} bytes)")

    # Save compiled
    output_compiled = "stablehlo_simple_transpose_compiled.mlir"
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


def export_graupel_scan(mode='baseline', **run_kwargs):
    """Export StableHLO from graupel implementations."""
    print("=" * 80)
    print(f"GRAUPEL SCAN - MODE: {mode.upper()}")
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

    print(f"Implementation: {graupel_run.__module__}.{graupel_run.__name__}")
    print(f"Run kwargs: {run_kwargs}")

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

    print(f"\nTest inputs: {ncells} cells × {nlev} levels")

    # Create wrapper function for JIT
    def run_fn():
        return graupel_run(dz, t, p, rho, q, dt, qnc, **run_kwargs)

    # Lower to StableHLO
    print("\nLowering to StableHLO...")
    jitted = jax.jit(run_fn)
    lowered = jitted.lower()
    stablehlo_lowered = lowered.as_text()

    # Save lowered
    output_lowered = f"{output_prefix}_lowered.mlir"
    with open(output_lowered, 'w') as f:
        f.write(stablehlo_lowered)
    print(f"✓ Lowered IR: {output_lowered}")
    print(f"  Size: {len(stablehlo_lowered)} bytes ({len(stablehlo_lowered)/1024:.1f} KB)")

    # Compile and get optimized IR
    print("\nCompiling (this may take a while)...")
    compiled = lowered.compile()
    stablehlo_compiled = compiled.as_text()

    # Save compiled
    output_compiled = f"{output_prefix}_compiled.mlir"
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
    parser.add_argument('--fused', action='store_true', help='Use fused scans (default mode only)')
    parser.add_argument('--triton', action='store_true', help='Use Triton (default mode only)')
    parser.add_argument('--mlir', action='store_true', help='Use MLIR (default mode only)')

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    if args.mode == 'simple':
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
        export_graupel_scan(mode=args.mode, **run_kwargs)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
