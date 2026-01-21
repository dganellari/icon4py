#!/usr/bin/env python3
"""
Experiment with XLA compiler flags to optimize graupel performance.

XLA has many tunable flags that can significantly impact performance.
This script tests different configurations.

Usage:
    JAX_PLATFORMS=cuda PYTHONPATH=.:$PYTHONPATH python muphys_jax/optimize_xla_flags.py <input_nc>
"""

import os
import sys
import time

# Force float64 BEFORE importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np


def get_xla_flags():
    """Return dict of XLA flag configurations to test."""
    return {
        "baseline": {},

        "aggressive_fusion": {
            "XLA_FLAGS": "--xla_gpu_enable_fast_min_max=true"
        },

        "more_fusion": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_all_reduce_combine_threshold_bytes=0"
            )
        },

        "layout_opt": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_force_compilation_parallelism=8"
            )
        },

        "triton_gemm": {
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=true"
        },

        "cudnn_fusion": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_cudnn_fmha=true "
                "--xla_gpu_fused_attention_use_cudnn_rng=true"
            )
        },

        "memory_opt": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_strict_conv_algorithm_picker=false"
            )
        },

        "all_optimizations": {
            "XLA_FLAGS": (
                "--xla_gpu_enable_fast_min_max=true "
                "--xla_gpu_all_reduce_combine_threshold_bytes=0 "
                "--xla_gpu_force_compilation_parallelism=8 "
                "--xla_gpu_enable_triton_gemm=true"
            )
        },
    }


def load_input_data(filepath):
    """Load input data from NetCDF file (same format as GT4Py)."""
    import netCDF4 as nc
    import numpy as np

    print(f"Loading: {filepath}")
    ds = nc.Dataset(filepath, 'r')

    # Get dimensions
    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    # Calculate dz from geometric height (same as GT4Py)
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
    dz = np.transpose(dz)  # (height, ncells) -> (ncells, height)

    # Load variables (transpose from (height, ncells) to (ncells, height))
    def load_var(varname):
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[0, :, :]
        else:
            var = var[:]
        return np.transpose(var).astype(np.float64)

    te = jnp.array(load_var("ta"))    # temperature
    p = jnp.array(load_var("pfull"))  # pressure
    rho = jnp.array(load_var("rho"))  # density
    dz = jnp.array(dz)

    from muphys_jax.core.definitions import Q
    q_in = Q(
        v=jnp.array(load_var("hus")),  # specific humidity (vapor)
        c=jnp.array(load_var("clw")),  # cloud liquid water
        r=jnp.array(load_var("qr")),   # rain
        s=jnp.array(load_var("qs")),   # snow
        i=jnp.array(load_var("cli")),  # cloud ice
        g=jnp.array(load_var("qg")),   # graupel
    )

    ds.close()

    print(f"Grid: {ncells} cells × {nlev} levels")
    print(f"Temperature range: {float(te.min()):.1f} - {float(te.max()):.1f} K")
    return dz, te, p, rho, q_in


def run_benchmark(config_name, xla_flags, dz, te, p, rho, q_in, n_iter=10):
    """Run benchmark with specific XLA flags."""
    from muphys_jax.implementations.graupel import graupel_run

    # Set environment
    for key, val in xla_flags.items():
        os.environ[key] = val

    # Clear JIT cache to force recompilation with new flags
    jax.clear_caches()

    dt = 30.0
    qnc = 100.0

    # Warmup (JIT compile) - use fused scans for best performance
    print(f"\n--- {config_name} ---")
    print(f"  Flags: {xla_flags.get('XLA_FLAGS', 'none')}")
    print(f"  Compiling...", end=" ", flush=True)

    start = time.time()
    result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
    # Force computation
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
    print(f"  T change: {t_change:.2e} K")

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
        print("Usage: python optimize_xla_flags.py <input.nc> [n_iter]")
        print("\nExample:")
        print("  JAX_PLATFORMS=cuda PYTHONPATH=.:$PYTHONPATH python muphys_jax/optimize_xla_flags.py input.nc 10")
        sys.exit(1)

    input_file = sys.argv[1]
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print("=" * 70)
    print("XLA Flag Optimization for JAX Graupel")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Float64 enabled: {jax.config.x64_enabled}")

    # Load data
    dz, te, p, rho, q_in = load_input_data(input_file)

    # Run benchmarks
    configs = get_xla_flags()
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
    print(f"\nBest configuration: {best['config']}")
    print(f"Best time: {best['time_per_iter']*1000:.2f} ms/iter")

    # Compare to target
    dace_target = 0.0146  # seconds
    gap = best['time_per_iter'] / dace_target
    print(f"\nDaCe GPU target: {dace_target*1000:.2f} ms/iter")
    print(f"Current gap: {gap:.2f}x slower")


if __name__ == "__main__":
    main()
