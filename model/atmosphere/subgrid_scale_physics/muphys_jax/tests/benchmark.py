# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Benchmark JAX implementation with XLA and IREE backends.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import os

# Import JAX implementation
import transitions
import scans


def benchmark_cloud_to_rain(backend='xla', ncells=1000, nlev=65, num_runs=100):
    """
    Benchmark cloud_to_rain transition.

    Args:
        backend: 'xla' or 'iree'
        ncells: Number of horizontal grid cells
        nlev: Number of vertical levels
        num_runs: Number of benchmark iterations

    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking cloud_to_rain ({backend.upper()})")
    print(f"Grid size: {ncells} cells × {nlev} levels")
    print(f"{'='*60}")

    # Create test data
    np.random.seed(42)
    t = jnp.array(np.random.uniform(230.0, 300.0, (ncells, nlev)))
    qc = jnp.array(np.random.uniform(0.0, 1e-4, (ncells, nlev)))
    qr = jnp.array(np.random.uniform(0.0, 1e-4, (ncells, nlev)))
    nc = 1e8

    # JIT compile
    if backend == 'xla':
        cloud_to_rain_jit = jit(transitions.cloud_to_rain)
    elif backend == 'iree':
        try:
            from jax.experimental.jax2iree import jax2iree_jit
            cloud_to_rain_jit = jax2iree_jit(transitions.cloud_to_rain)
        except ImportError:
            print(f"IREE not available, skipping")
            return None
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Warmup (trigger compilation)
    print("Warming up (compiling)...")
    for _ in range(3):
        result = cloud_to_rain_jit(t, qc, qr, nc)
        result.block_until_ready()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = cloud_to_rain_jit(t, qc, qr, nc)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    stats = {
        'backend': backend,
        'function': 'cloud_to_rain',
        'ncells': ncells,
        'nlev': nlev,
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'median_ms': np.median(times) * 1000,
    }

    print(f"Mean:   {stats['mean_ms']:.3f} ms")
    print(f"Median: {stats['median_ms']:.3f} ms")
    print(f"Std:    {stats['std_ms']:.3f} ms")
    print(f"Min:    {stats['min_ms']:.3f} ms")
    print(f"Max:    {stats['max_ms']:.3f} ms")

    return stats


def benchmark_precip_scan(backend='xla', ncells=1000, nlev=65, num_runs=100):
    """
    Benchmark precip_scan operator.

    Args:
        backend: 'xla' or 'iree'
        ncells: Number of horizontal grid cells
        nlev: Number of vertical levels
        num_runs: Number of benchmark iterations

    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking precip_scan ({backend.upper()})")
    print(f"Grid size: {ncells} cells × {nlev} levels")
    print(f"{'='*60}")

    # Create test data
    np.random.seed(42)
    prefactor = jnp.ones((ncells, nlev)) * 25.0
    exponent = jnp.ones((ncells, nlev)) * 0.5
    offset = jnp.ones((ncells, nlev)) * 1e-10
    zeta = jnp.ones((ncells, nlev)) * 0.1
    vc = jnp.ones((ncells, nlev)) * 1.0
    q = jnp.array(np.random.uniform(1e-6, 1e-4, (ncells, nlev)))
    rho = jnp.ones((ncells, nlev)) * 1.2
    mask = jnp.ones((ncells, nlev), dtype=bool)

    # JIT compile
    if backend == 'xla':
        precip_scan_jit = jit(scans.precip_scan)
    elif backend == 'iree':
        try:
            from jax.experimental.jax2iree import jax2iree_jit
            precip_scan_jit = jax2iree_jit(scans.precip_scan)
        except ImportError:
            print(f"IREE not available, skipping")
            return None
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Warmup
    print("Warming up (compiling)...")
    for _ in range(3):
        result = precip_scan_jit(prefactor, exponent, offset, zeta, vc, q, rho, mask)
        result.q_update.block_until_ready()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = precip_scan_jit(prefactor, exponent, offset, zeta, vc, q, rho, mask)
        result.q_update.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    stats = {
        'backend': backend,
        'function': 'precip_scan',
        'ncells': ncells,
        'nlev': nlev,
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'median_ms': np.median(times) * 1000,
    }

    print(f"Mean:   {stats['mean_ms']:.3f} ms")
    print(f"Median: {stats['median_ms']:.3f} ms")
    print(f"Std:    {stats['std_ms']:.3f} ms")
    print(f"Min:    {stats['min_ms']:.3f} ms")
    print(f"Max:    {stats['max_ms']:.3f} ms")

    return stats


def benchmark_temperature_scan(backend='xla', ncells=1000, nlev=65, num_runs=100):
    """
    Benchmark temperature_scan operator.

    Args:
        backend: 'xla' or 'iree'
        ncells: Number of horizontal grid cells
        nlev: Number of vertical levels
        num_runs: Number of benchmark iterations

    Returns:
        dict with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking temperature_scan ({backend.upper()})")
    print(f"Grid size: {ncells} cells × {nlev} levels")
    print(f"{'='*60}")

    # Create test data
    np.random.seed(42)
    zeta = jnp.ones((ncells, nlev)) * 0.1
    lheat = jnp.ones((ncells, nlev)) * 2.5e6
    q_update = jnp.array(np.random.uniform(-1e-5, 1e-5, (ncells, nlev)))
    rho = jnp.ones((ncells, nlev)) * 1.2

    # JIT compile
    if backend == 'xla':
        temperature_scan_jit = jit(scans.temperature_scan)
    elif backend == 'iree':
        try:
            from jax.experimental.jax2iree import jax2iree_jit
            temperature_scan_jit = jax2iree_jit(scans.temperature_scan)
        except ImportError:
            print(f"IREE not available, skipping")
            return None
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Warmup
    print("Warming up (compiling)...")
    for _ in range(3):
        result = temperature_scan_jit(zeta, lheat, q_update, rho)
        result.te_update.block_until_ready()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = temperature_scan_jit(zeta, lheat, q_update, rho)
        result.te_update.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    stats = {
        'backend': backend,
        'function': 'temperature_scan',
        'ncells': ncells,
        'nlev': nlev,
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'median_ms': np.median(times) * 1000,
    }

    print(f"Mean:   {stats['mean_ms']:.3f} ms")
    print(f"Median: {stats['median_ms']:.3f} ms")
    print(f"Std:    {stats['std_ms']:.3f} ms")
    print(f"Min:    {stats['min_ms']:.3f} ms")
    print(f"Max:    {stats['max_ms']:.3f} ms")

    return stats


def print_comparison(stats_xla, stats_iree):
    """Print comparison between XLA and IREE."""
    if stats_xla is None or stats_iree is None:
        return

    print(f"\n{'='*60}")
    print(f"XLA vs IREE Comparison ({stats_xla['function']})")
    print(f"{'='*60}")

    speedup = stats_xla['mean_ms'] / stats_iree['mean_ms']
    faster = 'IREE' if speedup > 1 else 'XLA'

    print(f"XLA mean:  {stats_xla['mean_ms']:.3f} ms")
    print(f"IREE mean: {stats_iree['mean_ms']:.3f} ms")
    print(f"Speedup:   {abs(speedup):.2f}x ({faster} is faster)")


if __name__ == '__main__':
    print("="*60)
    print("JAX Muphys Backend Benchmark")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    # Grid sizes to test
    ncells = 1000
    nlev = 65
    num_runs = 100

    # Benchmark cloud_to_rain
    stats_c2r_xla = benchmark_cloud_to_rain('xla', ncells, nlev, num_runs)
    stats_c2r_iree = benchmark_cloud_to_rain('iree', ncells, nlev, num_runs)
    if stats_c2r_xla and stats_c2r_iree:
        print_comparison(stats_c2r_xla, stats_c2r_iree)

    # Benchmark precip_scan
    stats_ps_xla = benchmark_precip_scan('xla', ncells, nlev, num_runs)
    stats_ps_iree = benchmark_precip_scan('iree', ncells, nlev, num_runs)
    if stats_ps_xla and stats_ps_iree:
        print_comparison(stats_ps_xla, stats_ps_iree)

    # Benchmark temperature_scan
    stats_ts_xla = benchmark_temperature_scan('xla', ncells, nlev, num_runs)
    stats_ts_iree = benchmark_temperature_scan('iree', ncells, nlev, num_runs)
    if stats_ts_xla and stats_ts_iree:
        print_comparison(stats_ts_xla, stats_ts_iree)

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)
