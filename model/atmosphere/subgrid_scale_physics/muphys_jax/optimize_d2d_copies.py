#!/usr/bin/env python3
"""
Optimize D2D memory copies with XLA compiler flags.

Focus on reducing the 92.1% D2D memcpy overhead identified in profiling.

Usage:
    JAX_PLATFORMS=cuda PYTHONPATH=.:$PYTHONPATH python muphys_jax/optimize_d2d_copies.py <input.nc> [n_iter]
"""

import os
import sys
import time

# Force float64 BEFORE importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np


def get_d2d_optimization_flags():
    """Return XLA flag configurations specifically targeting D2D copy reduction."""
    return {
        "baseline": {},

        "reduce_copies_v1": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_deterministic_ops=false "  # Allow non-deterministic opts
                "--xla_gpu_enable_async_collectives=true"
            )
        },

        "reduce_copies_v2": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_enable_async_all_reduce=true "
                "--xla_gpu_all_reduce_combine_threshold_bytes=134217728"  # 128MB
            )
        },

        "memory_layout_opt": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_enable_latency_hiding_scheduler=true "
                "--xla_gpu_lhs_enable_gpu_async_tracker=true"
            )
        },

        "aggressive_inline": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_graph_level=0 "  # Disable graph capture that may add copies
                "--xla_gpu_enable_command_buffer="
            )
        },

        "scan_optimized": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_enable_while_loop_double_buffering=true "
                "--xla_gpu_enable_triton_softmax_fusion=true"
            )
        },

        "best_combination": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_deterministic_ops=false "
                "--xla_gpu_enable_async_collectives=true "
                "--xla_gpu_enable_latency_hiding_scheduler=true "
                "--xla_gpu_enable_while_loop_double_buffering=true"
            )
        },
    }


def load_input_data(filepath):
    """Load input data from NetCDF file."""
    import netCDF4 as nc

    print(f"Loading: {filepath}")
    ds = nc.Dataset(filepath, 'r')

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

    dz = calc_dz(ds.variables["zg"][:])
    dz = np.transpose(dz)

    def load_var(varname):
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[0, :, :]
        else:
            var = var[:]
        return np.transpose(var).astype(np.float64)

    te = jnp.array(load_var("ta"))
    p = jnp.array(load_var("pfull"))
    rho = jnp.array(load_var("rho"))
    dz = jnp.array(dz)

    from muphys_jax.core.definitions import Q
    q_in = Q(
        v=jnp.array(load_var("hus")),
        c=jnp.array(load_var("clw")),
        r=jnp.array(load_var("qr")),
        s=jnp.array(load_var("qs")),
        i=jnp.array(load_var("cli")),
        g=jnp.array(load_var("qg")),
    )

    ds.close()

    print(f"Grid: {ncells} cells × {nlev} levels")
    print(f"Temperature range: {float(te.min()):.1f} - {float(te.max()):.1f} K")
    return dz, te, p, rho, q_in


def run_benchmark(config_name, xla_flags, dz, te, p, rho, q_in, n_iter=20):
    """Run benchmark with specific XLA flags."""
    from muphys_jax.implementations.graupel import graupel_run

    # Set environment
    for key, val in xla_flags.items():
        os.environ[key] = val

    # Clear JIT cache to force recompilation with new flags
    jax.clear_caches()

    dt = 30.0
    qnc = 100.0

    # Warmup - use fused scans for best performance
    print(f"\n--- {config_name} ---")
    print(f"  Flags: {xla_flags.get('XLA_FLAGS', 'none')}")
    print(f"  Compiling...", end=" ", flush=True)

    start = time.time()
    result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
    _ = result[0].block_until_ready()
    compile_time = time.time() - start
    print(f"done ({compile_time:.2f}s)")

    # Benchmark
    print(f"  Running {n_iter} iterations...", end=" ", flush=True)
    jax.block_until_ready(result)

    start = time.time()
    for _ in range(n_iter):
        result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
        _ = result[0].block_until_ready()
    elapsed = time.time() - start

    time_per_iter = elapsed / n_iter
    print(f"done")
    print(f"  Time/iter: {time_per_iter*1000:.2f} ms")

    # Verify correctness
    t_out = np.array(result[0])
    t_change = (t_out - np.array(te)).mean()
    print(f"  T change: {t_change:.2e} K (expected ~-2.8e-04)")

    # Clean up env
    for key in xla_flags:
        if key in os.environ:
            del os.environ[key]

    return {
        'config': config_name,
        'time_per_iter': time_per_iter,
        'compile_time': compile_time,
        't_change': t_change,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python optimize_d2d_copies.py <input.nc> [n_iter]")
        print("\nExample:")
        print("  JAX_PLATFORMS=cuda PYTHONPATH=.:$PYTHONPATH python muphys_jax/optimize_d2d_copies.py input.nc 20")
        sys.exit(1)

    input_file = sys.argv[1]
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    print("=" * 70)
    print("D2D Memory Copy Optimization for JAX Graupel")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Float64 enabled: {jax.config.x64_enabled}")
    print(f"\nTarget: Reduce D2D memcpy from 92.1% of runtime")

    # Load data
    dz, te, p, rho, q_in = load_input_data(input_file)

    # Run benchmarks
    configs = get_d2d_optimization_flags()
    results = []

    for name, flags in configs.items():
        try:
            result = run_benchmark(name, flags, dz, te, p, rho, q_in, n_iter)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'config': name, 'time_per_iter': float('inf'), 'error': str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Time/iter (ms)':<15} {'Speedup':<10}")
    print("-" * 50)

    baseline_time = results[0]['time_per_iter']
    for r in sorted(results, key=lambda x: x['time_per_iter']):
        speedup = baseline_time / r['time_per_iter'] if r['time_per_iter'] > 0 else 0
        print(f"{r['config']:<25} {r['time_per_iter']*1000:<15.2f} {speedup:<10.2f}x")

    # Best config
    best = min(results, key=lambda x: x['time_per_iter'])
    print(f"\n✓ Best configuration: {best['config']}")
    print(f"  Time: {best['time_per_iter']*1000:.2f} ms/iter")
    print(f"  Improvement: {(1 - best['time_per_iter']/baseline_time)*100:.1f}%")

    # Compare to target
    current_baseline = 51.0  # ms (from previous benchmarks)
    dace_target = 14.6  # ms

    print(f"\nProgress:")
    print(f"  Previous best: {current_baseline:.1f} ms")
    print(f"  Current best:  {best['time_per_iter']*1000:.1f} ms")
    print(f"  DaCe target:   {dace_target:.1f} ms")
    print(f"  Gap remaining: {best['time_per_iter']*1000/dace_target:.2f}x")

    # Export best flags
    if best['time_per_iter'] < baseline_time * 0.95:  # >5% improvement
        print(f"\n✓ Recommended XLA_FLAGS:")
        print(f"  export XLA_FLAGS=\"{configs[best['config']].get('XLA_FLAGS', '')}\"")


if __name__ == "__main__":
    main()
