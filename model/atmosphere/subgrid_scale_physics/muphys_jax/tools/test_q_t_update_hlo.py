#!/usr/bin/env python3
"""
Test HLO injection for q_t_update optimization.

Usage:
    python test_q_t_update_hlo.py -i graupel_input_fortran_data.nc
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
    parser = argparse.ArgumentParser(description="Test q_t_update HLO injection")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--hlo", default=None, help="Path to optimized StableHLO file")
    args = parser.parse_args()

    # Default HLO path
    if args.hlo is None:
        args.hlo = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "stablehlo" / "q_t_update_transposed.stablehlo")

    print("=" * 70)
    print("Q_T_UPDATE HLO INJECTION TEST")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"HLO file: {args.hlo}")
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
    from muphys_jax.core.optimized_q_t_update import (
        configure_optimized_q_t_update,
        q_t_update_optimized,
        is_optimized_enabled
    )

    # Test 1: Baseline (pure JAX)
    print("=" * 70)
    print("TEST 1: Baseline (pure JAX)")
    print("=" * 70)

    baseline_fn = jax.jit(q_t_update_native)
    baseline_args = (t_t, p_t, rho_t, q_t, dt, qnc)
    time_baseline, result_baseline = benchmark(baseline_fn, baseline_args, "JAX baseline")

    # Test 2: With HLO injection (fallback - no HLO configured yet)
    print()
    print("=" * 70)
    print("TEST 2: Optimized primitive (fallback mode)")
    print("=" * 70)

    # Make sure HLO injection is disabled
    configure_optimized_q_t_update(use_optimized=False)
    print(f"HLO injection enabled: {is_optimized_enabled()}")

    optimized_fn = jax.jit(q_t_update_optimized)
    optimized_args = (t_t, p_t, rho_t, q_t, dt, qnc)
    time_fallback, result_fallback = benchmark(optimized_fn, optimized_args, "Fallback (no HLO)")

    # Verify results match
    q_base, t_base = result_baseline
    q_fall, t_fall = result_fallback

    max_diff_q = max(
        jnp.max(jnp.abs(q_base.v - q_fall.v)),
        jnp.max(jnp.abs(q_base.c - q_fall.c)),
        jnp.max(jnp.abs(q_base.r - q_fall.r)),
        jnp.max(jnp.abs(q_base.s - q_fall.s)),
        jnp.max(jnp.abs(q_base.i - q_fall.i)),
        jnp.max(jnp.abs(q_base.g - q_fall.g)),
    )
    max_diff_t = jnp.max(jnp.abs(t_base - t_fall))
    print(f"  Max diff (q): {max_diff_q:.2e}")
    print(f"  Max diff (t): {max_diff_t:.2e}")

    # Test 3: With HLO injection enabled
    print()
    print("=" * 70)
    print("TEST 3: With HLO injection")
    print("=" * 70)

    # Check if HLO file exists
    hlo_path = pathlib.Path(args.hlo)
    if not hlo_path.exists():
        print(f"WARNING: HLO file not found: {args.hlo}")
        print("Skipping HLO injection test")
        return

    # Configure HLO injection
    configure_optimized_q_t_update(
        hlo_path=str(hlo_path),
        use_optimized=True,
        dt=dt
    )
    print(f"HLO injection enabled: {is_optimized_enabled()}")

    # Need to re-JIT to pick up the new lowering
    optimized_fn_hlo = jax.jit(q_t_update_optimized)

    try:
        time_hlo, result_hlo = benchmark(optimized_fn_hlo, optimized_args, "With HLO injection")

        # Verify results match
        q_hlo, t_hlo = result_hlo

        max_diff_q = max(
            jnp.max(jnp.abs(q_base.v - q_hlo.v)),
            jnp.max(jnp.abs(q_base.c - q_hlo.c)),
            jnp.max(jnp.abs(q_base.r - q_hlo.r)),
            jnp.max(jnp.abs(q_base.s - q_hlo.s)),
            jnp.max(jnp.abs(q_base.i - q_hlo.i)),
            jnp.max(jnp.abs(q_base.g - q_hlo.g)),
        )
        max_diff_t = jnp.max(jnp.abs(t_base - t_hlo))
        print(f"  Max diff vs baseline (q): {max_diff_q:.2e}")
        print(f"  Max diff vs baseline (t): {max_diff_t:.2e}")

        # Summary
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"JAX baseline:     {time_baseline:.3f} ms")
        print(f"Fallback mode:    {time_fallback:.3f} ms")
        print(f"With HLO:         {time_hlo:.3f} ms")
        print(f"Speedup (HLO):    {time_baseline / time_hlo:.2f}x")

    except Exception as e:
        print(f"ERROR during HLO injection test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
