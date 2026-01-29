#!/usr/bin/env python3
"""
Benchmark precipitation_effects execution time (excluding compilation).

Usage:
    cd model/atmosphere/subgrid_scale_physics
    JAX_ENABLE_X64=1 python -m muphys_jax.tools.benchmark_execution_only --num-runs 100
"""

import argparse
import sys
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np


def create_test_inputs(ncells: int = 20480, nlev: int = 90):
    """Create test inputs for precipitation_effects."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.core.definitions import Q
    from muphys_jax.core.common import constants as const

    dz = jnp.ones((ncells, nlev), dtype=jnp.float64) * 100.0
    t = jnp.ones((ncells, nlev), dtype=jnp.float64) * 280.0
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
    last_lev = nlev - 1

    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    return last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev


def benchmark(num_warmup: int = 3, num_runs: int = 100):
    """Benchmark precipitation_effects."""
    print("=" * 80)
    print("BENCHMARKING precipitation_effects")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"x64: {jax.config.jax_enable_x64}")
    print()

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.graupel_baseline import precipitation_effects
    from muphys_jax.core.definitions import Q

    last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev = create_test_inputs()

    print(f"Grid: {ncells} cells x {nlev} levels")

    @jax.jit
    def run_precip(kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz):
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        return precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt)

    inputs = (kmin_r, kmin_i, kmin_s, kmin_g, q.v, q.c, q.r, q.s, q.i, q.g, t, rho, dz)

    # Compilation
    print("\n1. Compilation...")
    t0 = time.perf_counter()
    result = run_precip(*inputs)
    jax.block_until_ready(result)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"   Compilation: {compile_time:.2f} ms")

    # Warmup
    print(f"\n2. Warmup ({num_warmup} runs)...")
    for i in range(num_warmup):
        result = run_precip(*inputs)
        jax.block_until_ready(result)

    # Benchmark
    print(f"\n3. Benchmark ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        result = run_precip(*inputs)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Total:  {np.sum(times):.2f} ms for {num_runs} runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-warmup', type=int, default=3)
    parser.add_argument('--num-runs', type=int, default=100)
    args = parser.parse_args()

    benchmark(num_warmup=args.num_warmup, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
