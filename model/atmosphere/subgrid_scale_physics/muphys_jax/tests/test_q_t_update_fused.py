#!/usr/bin/env python3
"""
Test and benchmark the fused q_t_update implementation.

Compares:
1. Original q_t_update_native (uses separate transition/property functions)
2. Fused q_t_update_fused (all inlined with lax.select/lax.pow)

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python test_q_t_update_fused.py \\
        --input /path/to/graupel_input.nc
"""

import argparse
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from muphys_jax.utils.data_loading import load_graupel_inputs
from muphys_jax.core.definitions import Q


def benchmark(fn, args, name, num_warmup=10, num_runs=50):
    """Benchmark a function."""
    # Warmup
    for _ in range(num_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"  {name}: {np.median(times):.3f} ms (min: {np.min(times):.3f}, std: {np.std(times):.3f})")
    return np.median(times), result


def main():
    parser = argparse.ArgumentParser(description="Test fused q_t_update")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    args = parser.parse_args()

    print("=" * 70)
    print("Q_T_UPDATE FUSED vs ORIGINAL TEST")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Load data
    print("Loading data...")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(args.input)
    print(f"Grid: {ncells} cells x {nlev} levels")
    print(f"dt = {dt}, qnc = {qnc}")

    # Transpose to (nlev, ncells) layout
    t_t = jnp.transpose(t)
    p_t = jnp.transpose(p)
    rho_t = jnp.transpose(rho)
    q_t = Q(
        v=jnp.transpose(q.v),
        c=jnp.transpose(q.c),
        r=jnp.transpose(q.r),
        s=jnp.transpose(q.s),
        i=jnp.transpose(q.i),
        g=jnp.transpose(q.g),
    )
    print(f"Transposed shape: {t_t.shape}")
    print()

    # Import implementations
    from muphys_jax.implementations.graupel_native_transposed import q_t_update_native
    from muphys_jax.implementations.q_t_update_fused import q_t_update_fused

    # Test 1: Original
    print("=" * 70)
    print("TEST 1: Original q_t_update_native")
    print("=" * 70)

    original_fn = jax.jit(q_t_update_native)
    original_args = (t_t, p_t, rho_t, q_t, dt, qnc)
    time_original, result_original = benchmark(original_fn, original_args, "Original")

    # Test 2: Fused version
    print()
    print("=" * 70)
    print("TEST 2: Fused q_t_update_fused")
    print("=" * 70)

    fused_fn = jax.jit(q_t_update_fused)
    fused_args = (t_t, p_t, rho_t, q_t, dt, qnc)
    time_fused, result_fused = benchmark(fused_fn, fused_args, "Fused")

    # Verify results match
    q_orig, t_orig = result_original
    q_fused, t_fused = result_fused

    print()
    print("Verifying results...")
    max_diff_qv = float(jnp.max(jnp.abs(q_orig.v - q_fused.v)))
    max_diff_qc = float(jnp.max(jnp.abs(q_orig.c - q_fused.c)))
    max_diff_qr = float(jnp.max(jnp.abs(q_orig.r - q_fused.r)))
    max_diff_qs = float(jnp.max(jnp.abs(q_orig.s - q_fused.s)))
    max_diff_qi = float(jnp.max(jnp.abs(q_orig.i - q_fused.i)))
    max_diff_qg = float(jnp.max(jnp.abs(q_orig.g - q_fused.g)))
    max_diff_t = float(jnp.max(jnp.abs(t_orig - t_fused)))

    print(f"  Max diff qv: {max_diff_qv:.2e}")
    print(f"  Max diff qc: {max_diff_qc:.2e}")
    print(f"  Max diff qr: {max_diff_qr:.2e}")
    print(f"  Max diff qs: {max_diff_qs:.2e}")
    print(f"  Max diff qi: {max_diff_qi:.2e}")
    print(f"  Max diff qg: {max_diff_qg:.2e}")
    print(f"  Max diff t:  {max_diff_t:.2e}")

    max_diff = max(max_diff_qv, max_diff_qc, max_diff_qr, max_diff_qs, max_diff_qi, max_diff_qg, max_diff_t)
    if max_diff < 1e-5:
        print(f"  ✓ Results match within tolerance ({max_diff:.2e})")
    else:
        print(f"  ⚠ Results differ! Max diff: {max_diff:.2e}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original:  {time_original:.3f} ms")
    print(f"Fused:     {time_fused:.3f} ms")
    print(f"Speedup:   {time_original / time_fused:.2f}x")

    # Export both to StableHLO for analysis
    print()
    print("=" * 70)
    print("StableHLO ANALYSIS")
    print("=" * 70)

    def analyze_stablehlo(text):
        return {
            'size': len(text),
            'power': text.count('stablehlo.power'),
            'exp': text.count('stablehlo.exponential'),
            'mul': text.count('stablehlo.multiply'),
            'div': text.count('stablehlo.divide'),
            'select': text.count('stablehlo.select'),
            'compare': text.count('stablehlo.compare'),
            'func_calls': text.count('call @'),
        }

    def wrapper_orig(t, p, rho, qv, qc, qr, qs, qi, qg, qnc):
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        q_out, t_out = q_t_update_native(t, p, rho, q_in, dt, qnc)
        return q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g, t_out

    def wrapper_fused(t, p, rho, qv, qc, qr, qs, qi, qg, qnc):
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        q_out, t_out = q_t_update_fused(t, p, rho, q_in, dt, qnc)
        return q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g, t_out

    print("Lowering original to StableHLO...")
    lowered_orig = jax.jit(wrapper_orig).lower(
        t_t, p_t, rho_t, q_t.v, q_t.c, q_t.r, q_t.s, q_t.i, q_t.g, qnc
    )
    hlo_orig = lowered_orig.as_text()
    stats_orig = analyze_stablehlo(hlo_orig)

    print("Lowering fused to StableHLO...")
    lowered_fused = jax.jit(wrapper_fused).lower(
        t_t, p_t, rho_t, q_t.v, q_t.c, q_t.r, q_t.s, q_t.i, q_t.g, qnc
    )
    hlo_fused = lowered_fused.as_text()
    stats_fused = analyze_stablehlo(hlo_fused)

    print(f"\nOriginal StableHLO:")
    print(f"  Size: {stats_orig['size']:,} chars")
    print(f"  Power ops: {stats_orig['power']}")
    print(f"  Exp ops: {stats_orig['exp']}")
    print(f"  Multiply ops: {stats_orig['mul']}")
    print(f"  Divide ops: {stats_orig['div']}")
    print(f"  Select ops: {stats_orig['select']}")
    print(f"  Compare ops: {stats_orig['compare']}")
    print(f"  Function calls: {stats_orig['func_calls']}")

    print(f"\nFused StableHLO:")
    print(f"  Size: {stats_fused['size']:,} chars")
    print(f"  Power ops: {stats_fused['power']}")
    print(f"  Exp ops: {stats_fused['exp']}")
    print(f"  Multiply ops: {stats_fused['mul']}")
    print(f"  Divide ops: {stats_fused['div']}")
    print(f"  Select ops: {stats_fused['select']}")
    print(f"  Compare ops: {stats_fused['compare']}")
    print(f"  Function calls: {stats_fused['func_calls']}")

    # Save fused StableHLO
    output_path = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "stablehlo" / "q_t_update_fused.stablehlo"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(hlo_fused)
    print(f"\n✓ Saved fused StableHLO to: {output_path}")


if __name__ == "__main__":
    main()
