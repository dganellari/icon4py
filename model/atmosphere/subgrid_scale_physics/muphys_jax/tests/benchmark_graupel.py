# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive benchmark for graupel_jax: XLA vs IREE vs DaCe comparison.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import os


def create_test_data(ncells, nlev):
    """Create realistic atmospheric test data."""
    np.random.seed(42)

    # Temperature profile
    t_surface, t_top = 288.0, 220.0
    te = np.linspace(t_surface, t_top, nlev)[None, :] + np.random.randn(ncells, nlev) * 2.0

    # Pressure profile
    p_surface, p_top = 101325.0, 20000.0
    p = np.linspace(p_surface, p_top, nlev)[None, :] * (1.0 + np.random.randn(ncells, nlev) * 0.01)

    # Density
    rho = p / (287.04 * te)

    # Layer thickness
    dz = np.full((ncells, nlev), 500.0)

    # Water species
    from muphys_jax.core.definitions import Q
    q_in = Q(
        v=jnp.array(np.random.uniform(1e-4, 1e-3, (ncells, nlev))),
        c=jnp.array(np.random.uniform(0.0, 1e-5, (ncells, nlev))),
        r=jnp.array(np.random.uniform(0.0, 1e-6, (ncells, nlev))),
        s=jnp.array(np.random.uniform(0.0, 1e-6, (ncells, nlev))),
        i=jnp.array(np.random.uniform(0.0, 1e-7, (ncells, nlev))),
        g=jnp.array(np.random.uniform(0.0, 1e-7, (ncells, nlev))),
    )

    return jnp.array(dz), jnp.array(te), jnp.array(p), jnp.array(rho), q_in, 10.0, 1e8


def benchmark_xla(ncells=1000, nlev=65, num_runs=50):
    """Benchmark with XLA backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking XLA Backend")
    print(f"Grid: {ncells} cells × {nlev} levels")
    print(f"{'='*60}")

    # Import with XLA
    os.environ['JAX_BACKEND'] = 'xla'
    from muphys_jax.implementations.graupel import graupel_run

    # Create test data
    dz, te, p, rho, q_in, dt, qnc = create_test_data(ncells, nlev)

    # Warmup (compile)
    print("Warming up (JIT compilation)...")
    for _ in range(3):
        result = graupel_run(dz, te, p, rho, q_in, dt, qnc)
        result[0].block_until_ready()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = graupel_run(dz, te, p, rho, q_in, dt, qnc)
        result[0].block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    stats = {
        'backend': 'XLA',
        'ncells': ncells,
        'nlev': nlev,
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'median_ms': np.median(times) * 1000,
    }

    print(f"Mean:   {stats['mean_ms']:.2f} ms")
    print(f"Median: {stats['median_ms']:.2f} ms")
    print(f"Std:    {stats['std_ms']:.2f} ms")
    print(f"Min:    {stats['min_ms']:.2f} ms")
    print(f"Max:    {stats['max_ms']:.2f} ms")

    return stats


def benchmark_iree(ncells=1000, nlev=65, num_runs=50):
    """Benchmark with IREE backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking IREE Backend")
    print(f"Grid: {ncells} cells × {nlev} levels")
    print(f"{'='*60}")

    try:
        # Try to import IREE
        from jax.experimental.jax2iree import jax2iree_jit
        print("IREE is available!")
    except ImportError:
        print("IREE not available - install with: pip install iree-compiler iree-runtime")
        return None

    # Import with IREE
    os.environ['JAX_BACKEND'] = 'iree'
    import importlib
    from muphys_jax.implementations import graupel
    importlib.reload(graupel)
    from muphys_jax.implementations.graupel import graupel_run

    # Create test data
    dz, te, p, rho, q_in, dt, qnc = create_test_data(ncells, nlev)

    # Warmup (compile)
    print("Warming up (JIT compilation)...")
    try:
        for _ in range(3):
            result = graupel_run(dz, te, p, rho, q_in, dt, qnc)
            result[0].block_until_ready()
    except Exception as e:
        print(f"IREE compilation failed: {e}")
        return None

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = graupel_run(dz, te, p, rho, q_in, dt, qnc)
        result[0].block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    stats = {
        'backend': 'IREE',
        'ncells': ncells,
        'nlev': nlev,
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'median_ms': np.median(times) * 1000,
    }

    print(f"Mean:   {stats['mean_ms']:.2f} ms")
    print(f"Median: {stats['median_ms']:.2f} ms")
    print(f"Std:    {stats['std_ms']:.2f} ms")
    print(f"Min:    {stats['min_ms']:.2f} ms")
    print(f"Max:    {stats['max_ms']:.2f} ms")

    return stats


def print_comparison(stats_xla, stats_iree):
    """Print side-by-side comparison."""
    if stats_xla is None or stats_iree is None:
        return

    print(f"\n{'='*60}")
    print(f"XLA vs IREE Comparison")
    print(f"{'='*60}")

    speedup = stats_xla['mean_ms'] / stats_iree['mean_ms']
    faster = 'IREE' if speedup > 1 else 'XLA'

    print(f"XLA mean:   {stats_xla['mean_ms']:.2f} ms")
    print(f"IREE mean:  {stats_iree['mean_ms']:.2f} ms")
    print(f"Speedup:    {abs(speedup):.2f}x ({faster} is faster)")


if __name__ == '__main__':
    print("="*60)
    print("Graupel JAX Comprehensive Benchmark")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    # Test different grid sizes
    grid_sizes = [
        (100, 65),    # Small
        (1000, 65),   # Medium (realistic)
        (5000, 65),   # Large
    ]

    all_results = []

    for ncells, nlev in grid_sizes:
        print(f"\n{'#'*60}")
        print(f"# Grid Size: {ncells} cells × {nlev} levels")
        print(f"{'#'*60}")

        stats_xla = benchmark_xla(ncells, nlev, num_runs=50)
        all_results.append(stats_xla)

        stats_iree = benchmark_iree(ncells, nlev, num_runs=50)
        if stats_iree:
            all_results.append(stats_iree)
            print_comparison(stats_xla, stats_iree)

    # Summary table
    print(f"\n{'='*60}")
    print(f"Summary Table")
    print(f"{'='*60}")
    print(f"{'Backend':<10} {'Grid':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'Std (ms)':<10}")
    print(f"{'-'*60}")
    for stats in all_results:
        grid_str = f"{stats['ncells']}×{stats['nlev']}"
        print(f"{stats['backend']:<10} {grid_str:<15} {stats['mean_ms']:<12.2f} "
              f"{stats['median_ms']:<12.2f} {stats['std_ms']:<10.2f}")

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)
    print("\nNext step: Compare against GT4Py+DaCe backend!")
