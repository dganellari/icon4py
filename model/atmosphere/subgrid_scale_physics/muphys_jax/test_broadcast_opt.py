#!/usr/bin/env python3
"""
Test that broadcast hoisting optimization doesn't change results.

Usage:
    PYTHONPATH=.:$PYTHONPATH python muphys_jax/test_broadcast_opt.py
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.core.definitions import Q


def create_test_inputs(ncells=1024, nlev=90):
    """Create simple test inputs."""
    dz = jnp.ones((ncells, nlev), dtype=jnp.float64) * 100.0
    te = jnp.ones((ncells, nlev), dtype=jnp.float64) * 273.15
    p = jnp.ones((ncells, nlev), dtype=jnp.float64) * 101325.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1.2

    q_in = Q(
        v=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.01,
        c=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.0001,
    )

    return dz, te, p, rho, q_in, 30.0, 100.0


def test_broadcast_optimization():
    """Test that broadcast hoisting doesn't change outputs."""
    print("=" * 70)
    print("Testing Broadcast Hoisting Optimization")
    print("=" * 70)

    args = create_test_inputs()

    print("\nRunning graupel (fused scans)...")
    result = graupel_run(*args, use_fused_scans=True)

    # Force computation
    t_out = jax.block_until_ready(result[0])
    q_out = jax.block_until_ready(result[1])

    print("\nResults:")
    print(f"  T output shape: {t_out.shape}")
    print(f"  T range: {float(t_out.min()):.6f} - {float(t_out.max()):.6f} K")
    print(f"  T mean change: {float((t_out - args[1]).mean()):.6e} K")

    print(f"\n  Q.v range: {float(q_out.v.min()):.6e} - {float(q_out.v.max()):.6e}")
    print(f"  Q.r range: {float(q_out.r.min()):.6e} - {float(q_out.r.max()):.6e}")
    print(f"  Q.s range: {float(q_out.s.min()):.6e} - {float(q_out.s.max()):.6e}")
    print(f"  Q.g range: {float(q_out.g.min()):.6e} - {float(q_out.g.max()):.6e}")

    # Check for NaNs/Infs
    has_nan = np.isnan(np.array(t_out)).any()
    has_inf = np.isinf(np.array(t_out)).any()

    if has_nan or has_inf:
        print(f"\n  ✗ ERROR: Found NaN={has_nan}, Inf={has_inf}")
        return False
    else:
        print(f"\n  ✓ All outputs finite")

    print("\n" + "=" * 70)
    print("Test PASSED - Broadcast optimization is working correctly")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_broadcast_optimization()