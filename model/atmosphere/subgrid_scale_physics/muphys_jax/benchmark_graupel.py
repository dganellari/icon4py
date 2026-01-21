#!/usr/bin/env python3
"""
Benchmark graupel performance before/after optimizations.

Usage:
    python benchmark_graupel.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
import os

# Force float64
jax.config.update("jax_enable_x64", True)

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))
os.chdir(parent_dir)

from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.core.definitions import Q


def create_test_inputs(ncells=1024, nlev=90, dtype=jnp.float64):
    """Create realistic test inputs."""
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


def benchmark(n_warmup=10, n_iterations=100):
    """Run benchmark."""
    print("=" * 70)
    print("Graupel Performance Benchmark")
    print("=" * 70)
    
    # Create inputs
    print("\n1. Creating test inputs...")
    inputs = create_test_inputs()
    print(f"   Grid: {inputs[0].shape[0]} cells × {inputs[0].shape[1]} levels")
    print(f"   Dtype: {inputs[0].dtype}")
    
    # Compile
    print("\n2. Compiling (JIT)...")
    compile_start = time.perf_counter()
    graupel_jit = jax.jit(graupel_run)
    _ = graupel_jit(*inputs)
    compile_time = time.perf_counter() - compile_start
    print(f"   Compile time: {compile_time:.2f}s")
    
    # Warmup
    print(f"\n3. Warmup ({n_warmup} iterations)...")
    for i in range(n_warmup):
        result = graupel_jit(*inputs)
        result[0].block_until_ready()  # Wait for GPU
    print(f"   ✓ Warmup complete")
    
    # Benchmark
    print(f"\n4. Benchmarking ({n_iterations} iterations)...")
    times = []
    
    for i in range(n_iterations):
        start = time.perf_counter()
        result = graupel_jit(*inputs)
        result[0].block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{n_iterations}")
    
    times = np.array(times)
    
    # Statistics
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nTiming statistics (n={n_iterations}):")
    print(f"  Mean:   {np.mean(times)*1000:.2f} ms")
    print(f"  Median: {np.median(times)*1000:.2f} ms")
    print(f"  Std:    {np.std(times)*1000:.2f} ms")
    print(f"  Min:    {np.min(times)*1000:.2f} ms")
    print(f"  Max:    {np.max(times)*1000:.2f} ms")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [50, 90, 95, 99]:
        print(f"  p{p}:     {np.percentile(times, p)*1000:.2f} ms")
    
    # Throughput
    ncells = inputs[0].shape[0]
    nlev = inputs[0].shape[1]
    cells_per_sec = ncells / np.mean(times)
    columns_per_sec = ncells / np.mean(times)
    
    print(f"\nThroughput:")
    print(f"  {cells_per_sec:.1f} cells/sec")
    print(f"  {columns_per_sec:.1f} columns/sec")
    print(f"  {cells_per_sec * nlev:.1f} cell-levels/sec")
    
    return times, result


def compare_with_baseline(current_mean_ms, baseline_mean_ms=53.4):
    """Compare with baseline performance."""
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    
    speedup = baseline_mean_ms / current_mean_ms
    improvement_pct = (speedup - 1.0) * 100
    
    print(f"\nBaseline:  {baseline_mean_ms:.2f} ms")
    print(f"Current:   {current_mean_ms:.2f} ms")
    print(f"Speedup:   {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✓ {improvement_pct:.1f}% faster")
    elif speedup < 1.0:
        print(f"✗ {-improvement_pct:.1f}% slower")
    else:
        print(f"→ No change")
    
    # Expected vs actual
    print(f"\nOptimizations applied:")
    print(f"  ✓ Power-to-multiply (expected: 1.05-1.1x)")
    
    if 1.05 <= speedup <= 1.15:
        print(f"  ✓ Within expected range!")
    elif speedup > 1.15:
        print(f"  ✓ Better than expected!")
    else:
        print(f"  ⚠ Below expected range (may need investigation)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running graupel benchmark...")
    print("=" * 70)
    
    # Run benchmark
    times, result = benchmark(n_warmup=10, n_iterations=100)
    
    # Compare with baseline
    mean_ms = np.mean(times) * 1000
    compare_with_baseline(mean_ms, baseline_mean_ms=53.4)
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
