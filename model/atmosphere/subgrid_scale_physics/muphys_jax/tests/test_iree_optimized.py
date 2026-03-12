#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test IREE-optimized implementation against baseline.

Run on CSCS Alps with:
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=iree_cuda python -m pytest test_iree_optimized.py -v

Or run directly:
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=iree_cuda python test_iree_optimized.py
"""

import os
import sys


# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import jax
import jax.numpy as jnp
import numpy as np

# Import implementations
from muphys_jax.core.definitions import Q
from muphys_jax.core.scans import (
    precip_scan_batched,
    precip_scan_fori,
    precip_scan_sequential,
    temperature_scan_iree as temperature_scan_fori,
    temperature_scan_step,
)
from muphys_jax.implementations.graupel_baseline import graupel_run_split
from muphys_jax.implementations.graupel_iree import graupel_run_iree


def make_test_data(ncells=1000, nlev=90, seed=42):
    """Create test data matching graupel typical ranges."""
    np.random.seed(seed)

    # Realistic atmospheric profile
    t = 280.0 + np.random.randn(ncells, nlev) * 10.0  # Temperature ~280K ± 10K
    p = 80000.0 + np.random.randn(ncells, nlev) * 5000.0  # Pressure ~800hPa
    rho = p / (287.0 * t)  # Ideal gas approximation
    dz = 200.0 + np.random.randn(ncells, nlev) * 20.0  # Layer thickness ~200m

    # Moisture fields (small positive values)
    q = Q(
        v=np.abs(np.random.randn(ncells, nlev) * 0.001 + 0.01),  # vapor
        c=np.abs(np.random.randn(ncells, nlev) * 0.0001),  # cloud water
        r=np.abs(np.random.randn(ncells, nlev) * 0.0001),  # rain
        s=np.abs(np.random.randn(ncells, nlev) * 0.0001),  # snow
        i=np.abs(np.random.randn(ncells, nlev) * 0.0001),  # ice
        g=np.abs(np.random.randn(ncells, nlev) * 0.0001),  # graupel
    )

    dt = 30.0
    qnc = 100.0

    # Convert to JAX arrays (float32 for IREE compatibility)
    return {
        "dz": jnp.array(dz, dtype=jnp.float32),
        "t": jnp.array(t, dtype=jnp.float32),
        "p": jnp.array(p, dtype=jnp.float32),
        "rho": jnp.array(rho, dtype=jnp.float32),
        "q": Q(
            v=jnp.array(q.v, dtype=jnp.float32),
            c=jnp.array(q.c, dtype=jnp.float32),
            r=jnp.array(q.r, dtype=jnp.float32),
            s=jnp.array(q.s, dtype=jnp.float32),
            i=jnp.array(q.i, dtype=jnp.float32),
            g=jnp.array(q.g, dtype=jnp.float32),
        ),
        "dt": dt,
        "qnc": qnc,
    }


def test_precip_scan_equivalence():
    """Test that fori_loop scan gives same results as vmap-based scan."""
    print("\n=== Testing Precipitation Scan Equivalence ===")

    ncells, nlev = 100, 30
    np.random.seed(42)

    # Test data
    zeta = jnp.array(np.random.rand(ncells, nlev) * 0.1 + 0.01, dtype=jnp.float32)
    rho = jnp.array(np.random.rand(ncells, nlev) * 0.5 + 0.5, dtype=jnp.float32)
    q = jnp.array(np.random.rand(ncells, nlev) * 0.001, dtype=jnp.float32)
    vc = jnp.array(np.random.rand(ncells, nlev) * 2 + 1, dtype=jnp.float32)
    mask = jnp.array(np.random.rand(ncells, nlev) > 0.5)

    params = jnp.array([14.58, 0.111, 1.0e-12], dtype=jnp.float32)

    # Baseline (scan-based)
    params_list = [(14.58, 0.111, 1.0e-12)]
    results_baseline = precip_scan_batched(params_list, zeta, rho, [q], [vc], [mask])
    q_base, flx_base = results_baseline[0]

    # IREE-optimized (fori_loop)
    q_iree, flx_iree = precip_scan_fori(params, zeta, rho, q, vc, mask)

    # Compare
    q_diff = jnp.max(jnp.abs(q_base - q_iree))
    flx_diff = jnp.max(jnp.abs(flx_base - flx_iree))

    print(f"  q max diff: {q_diff}")
    print(f"  flx max diff: {flx_diff}")

    assert float(q_diff) < 1e-5, f"q mismatch: {q_diff}"
    assert float(flx_diff) < 1e-5, f"flx mismatch: {flx_diff}"
    print("  ✓ Precipitation scan equivalence PASSED")


def test_graupel_equivalence():
    """Test that IREE-optimized graupel gives same results as split baseline."""
    print("\n=== Testing Graupel Equivalence ===")

    data = make_test_data(ncells=100, nlev=30)

    # Run baseline split version
    t_base, q_base, pflx_base, pr_base, ps_base, pi_base, pg_base, eflx_base = graupel_run_split(
        data["dz"], data["t"], data["p"], data["rho"], data["q"], data["dt"], data["qnc"]
    )

    # Run IREE-optimized version
    t_iree, q_iree, pflx_iree, pr_iree, ps_iree, pi_iree, pg_iree, eflx_iree = graupel_run_iree(
        data["dz"], data["t"], data["p"], data["rho"], data["q"], data["dt"], data["qnc"]
    )

    # Wait for computation
    t_base.block_until_ready()
    t_iree.block_until_ready()

    # Compare
    def max_diff(a, b, name):
        d = jnp.max(jnp.abs(a - b))
        print(f"  {name}: max diff = {d}")
        return float(d)

    t_diff = max_diff(t_base, t_iree, "temperature")
    pr_diff = max_diff(pr_base, pr_iree, "rain flux")
    ps_diff = max_diff(ps_base, ps_iree, "snow flux")
    eflx_diff = max_diff(eflx_base, eflx_iree, "energy flux")

    # Allow some tolerance due to float32 precision
    tolerance = 1e-4
    assert t_diff < tolerance, f"Temperature mismatch: {t_diff}"
    assert pr_diff < tolerance, f"Rain flux mismatch: {pr_diff}"
    print("  ✓ Graupel equivalence PASSED")


def benchmark_implementations():
    """Benchmark baseline split vs IREE-optimized."""
    print("\n=== Benchmarking Implementations ===")

    data = make_test_data(ncells=10000, nlev=90)
    n_warmup = 3
    n_iters = 10

    print(f"\nGrid size: {data['t'].shape[0]} cells × {data['t'].shape[1]} levels")
    print(f"Warmup: {n_warmup}, Iterations: {n_iters}")

    # Baseline split
    print("\n  Baseline (split JIT):")
    for _ in range(n_warmup):
        t, q, pflx, pr, ps, pi, pg, eflx = graupel_run_split(
            data["dz"], data["t"], data["p"], data["rho"], data["q"], data["dt"], data["qnc"]
        )
        t.block_until_ready()

    start = time.time()
    for _ in range(n_iters):
        t, q, pflx, pr, ps, pi, pg, eflx = graupel_run_split(
            data["dz"], data["t"], data["p"], data["rho"], data["q"], data["dt"], data["qnc"]
        )
        t.block_until_ready()
    baseline_time = time.time() - start
    print(f"    Total: {baseline_time:.3f}s, Per iter: {baseline_time/n_iters*1000:.2f}ms")

    # IREE-optimized
    print("\n  IREE-optimized (fori_loop scans):")
    for _ in range(n_warmup):
        t, q, pflx, pr, ps, pi, pg, eflx = graupel_run_iree(
            data["dz"], data["t"], data["p"], data["rho"], data["q"], data["dt"], data["qnc"]
        )
        t.block_until_ready()

    start = time.time()
    for _ in range(n_iters):
        t, q, pflx, pr, ps, pi, pg, eflx = graupel_run_iree(
            data["dz"], data["t"], data["p"], data["rho"], data["q"], data["dt"], data["qnc"]
        )
        t.block_until_ready()
    iree_time = time.time() - start
    print(f"    Total: {iree_time:.3f}s, Per iter: {iree_time/n_iters*1000:.2f}ms")

    # Summary
    speedup = baseline_time / iree_time
    print(f"\n  Speedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"  IREE-optimized is {(speedup-1)*100:.1f}% faster")
    else:
        print(f"  IREE-optimized is {(1/speedup-1)*100:.1f}% slower")


def main():
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    use_iree = "iree" in str(jax.devices()[0]).lower()
    print(f"Using IREE: {use_iree}")

    # Run tests
    test_precip_scan_equivalence()
    test_graupel_equivalence()
    benchmark_implementations()

    print("\n=== All Tests Passed! ===")


if __name__ == "__main__":
    main()
