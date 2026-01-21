#!/usr/bin/env python3
"""
Benchmark broadcast hoisting optimization.

This script measures the performance improvement from hoisting broadcasts
out of the scan loop.

Usage:
    JAX_PLATFORMS=cuda PYTHONPATH=.:$PYTHONPATH python muphys_jax/benchmark_broadcast_opt.py <input.nc>
"""

import sys
import time
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def benchmark_with_real_data(input_file, n_iter=100):
    """Benchmark with real NetCDF data."""
    print("=" * 70)
    print("Broadcast Hoisting Optimization Benchmark")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Iterations: {n_iter}")

    # Load data using same function as optimize_xla_flags.py
    import netCDF4 as nc

    print(f"\nLoading: {input_file}")
    ds = nc.Dataset(input_file, 'r')

    # Get dimensions
    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    # Calculate dz from geometric height
    def calc_dz(z):
        ksize = z.shape[0]
        dz = np.zeros(z.shape, np.float64)
        zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
        for k in range(ksize - 1, -1, -1):
            zh_new = 2.0 * z[k, :] - zh
            dz[k, :] = -zh + zh_new
            zh = zh_new
        return dz

    dz_np = calc_dz(ds.variables["zg"][:])
    dz_np = np.transpose(dz_np)

    def load_var(varname):
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[0, :, :]
        else:
            var = var[:]
        return np.transpose(var).astype(np.float64)

    te_np = load_var("ta")
    p_np = load_var("pfull")
    rho_np = load_var("rho")

    from muphys_jax.core.definitions import Q
    q_in_np = Q(
        v=load_var("hus"),
        c=load_var("clw"),
        r=load_var("qr"),
        s=load_var("qs"),
        i=load_var("cli"),
        g=load_var("qg"),
    )

    ds.close()

    print(f"Grid: {ncells} cells × {nlev} levels")
    print(f"Temperature range: {te_np.min():.1f} - {te_np.max():.1f} K")

    # Convert to JAX arrays
    dz = jnp.array(dz_np)
    te = jnp.array(te_np)
    p = jnp.array(p_np)
    rho = jnp.array(rho_np)
    q_in = Q(
        v=jnp.array(q_in_np.v),
        c=jnp.array(q_in_np.c),
        r=jnp.array(q_in_np.r),
        s=jnp.array(q_in_np.s),
        i=jnp.array(q_in_np.i),
        g=jnp.array(q_in_np.g),
    )

    dt = 30.0
    qnc = 100.0

    # Benchmark with optimized (current) code
    print("\n" + "=" * 70)
    print("Benchmarking: WITH broadcast hoisting (current optimized version)")
    print("=" * 70)

    from muphys_jax.implementations.graupel import graupel_run

    # Warmup - use fused scans for best performance
    print("Compiling and warming up (with fused scans + broadcast hoisting)...")
    result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
    _ = jax.block_until_ready(result[0])

    # Benchmark
    print(f"Running {n_iter} iterations...")
    start = time.time()
    for _ in range(n_iter):
        result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
        _ = jax.block_until_ready(result[0])
    elapsed_optimized = time.time() - start

    time_per_iter_opt = elapsed_optimized / n_iter
    print(f"\nResults:")
    print(f"  Total time: {elapsed_optimized:.4f} s")
    print(f"  Time/iter: {time_per_iter_opt*1000:.2f} ms")

    # Verify correctness
    t_out = np.array(result[0])
    t_change = (t_out - te_np).mean()
    print(f"  T change: {t_change:.2e} K (should be ~-2.8e-04)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Optimized version (with broadcast hoisting):")
    print(f"  Time/iter: {time_per_iter_opt*1000:.2f} ms")
    print(f"\nCompare with your previous results:")
    print(f"  - Unfused baseline: ~72.8 ms")
    print(f"  - Fused (before broadcast opt): ~51.5 ms")
    print(f"  - Current (fused + broadcast): {time_per_iter_opt*1000:.2f} ms")

    # Calculate improvements
    baseline = 72.8 / 1000
    fused_before = 51.5 / 1000

    speedup_vs_baseline = baseline / time_per_iter_opt
    speedup_vs_fused = fused_before / time_per_iter_opt

    print(f"\nSpeedups:")
    print(f"  vs Unfused baseline: {speedup_vs_baseline:.2f}x")
    print(f"  vs Fused (before broadcast): {speedup_vs_fused:.2f}x")

    # Gap to DaCe
    dace_target = 0.0146  # 14.6ms
    gap = time_per_iter_opt / dace_target
    print(f"\nGap to DaCe GPU (14.6ms):")
    print(f"  Current: {gap:.2f}x slower")
    print(f"  Improvement needed: {time_per_iter_opt*1000:.1f}ms → 14.6ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_broadcast_opt.py <input.nc> [n_iter]")
        print("\nExample:")
        print("  JAX_PLATFORMS=cuda PYTHONPATH=.:$PYTHONPATH python muphys_jax/benchmark_broadcast_opt.py input.nc 100")
        sys.exit(1)

    input_file = sys.argv[1]
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    benchmark_with_real_data(input_file, n_iter)