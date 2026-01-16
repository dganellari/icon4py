# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test complete graupel_jax implementation.
"""

import jax.numpy as jnp
import numpy as np
from muphys_jax.core.definitions import Q
from muphys_jax.implementations.graupel import graupel_run


def test_graupel_basic():
    """Test that graupel_run executes without errors."""
    print("\n=== Testing Complete Graupel Implementation ===")

    # Create realistic atmospheric profile
    ncells, nlev = 100, 65
    np.random.seed(42)

    # Temperature profile (decreasing with height)
    t_surface = 288.0  # K
    t_top = 220.0  # K
    te = np.linspace(t_surface, t_top, nlev)[None, :] + np.random.randn(ncells, nlev) * 2.0

    # Pressure profile (decreasing with height)
    p_surface = 101325.0  # Pa
    p_top = 20000.0  # Pa
    p = np.linspace(p_surface, p_top, nlev)[None, :] * (1.0 + np.random.randn(ncells, nlev) * 0.01)

    # Density (from ideal gas law approximation)
    rho = p / (287.04 * te)

    # Layer thickness (uniform for simplicity)
    dz = np.full((ncells, nlev), 500.0)  # 500m layers

    # Initial water species (small random values)
    q_in = Q(
        v=np.random.uniform(1e-4, 1e-3, (ncells, nlev)),  # vapor
        c=np.random.uniform(0.0, 1e-5, (ncells, nlev)),  # cloud
        r=np.random.uniform(0.0, 1e-6, (ncells, nlev)),  # rain
        s=np.random.uniform(0.0, 1e-6, (ncells, nlev)),  # snow
        i=np.random.uniform(0.0, 1e-7, (ncells, nlev)),  # ice
        g=np.random.uniform(0.0, 1e-7, (ncells, nlev)),  # graupel
    )

    # Convert to JAX arrays
    q_in_jax = Q(
        v=jnp.array(q_in.v),
        c=jnp.array(q_in.c),
        r=jnp.array(q_in.r),
        s=jnp.array(q_in.s),
        i=jnp.array(q_in.i),
        g=jnp.array(q_in.g),
    )

    # Time step and cloud number concentration
    dt = 10.0  # 10 second timestep
    qnc = 1e8  # 1e8 droplets per m^3

    print(f"Grid: {ncells} cells × {nlev} levels")
    print(f"Temperature range: {te.min():.1f} - {te.max():.1f} K")
    print(f"Pressure range: {p.min():.0f} - {p.max():.0f} Pa")
    print(f"Initial vapor: {q_in.v.mean():.2e} kg/kg")

    # Run graupel
    print("\nRunning graupel_run (JIT compilation + execution)...")
    import time

    start = time.time()

    t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
        jnp.array(dz), jnp.array(te), jnp.array(p), jnp.array(rho), q_in_jax, dt, qnc
    )

    # Block until computation completes
    t_out_np = np.array(t_out)
    end = time.time()

    print(f"Completed in {(end - start)*1000:.1f} ms")

    # Verify outputs
    print("\n=== Output Verification ===")
    print(f"Temperature range: {t_out_np.min():.1f} - {t_out_np.max():.1f} K")
    print(f"Temperature change: {(t_out_np - te).mean():.2e} K")

    q_out_np = Q(
        v=np.array(q_out.v),
        c=np.array(q_out.c),
        r=np.array(q_out.r),
        s=np.array(q_out.s),
        i=np.array(q_out.i),
        g=np.array(q_out.g),
    )

    print("\nWater species (mean):")
    print(f"  Vapor:   {q_out_np.v.mean():.2e} kg/kg (change: {(q_out_np.v - q_in.v).mean():.2e})")
    print(f"  Cloud:   {q_out_np.c.mean():.2e} kg/kg (change: {(q_out_np.c - q_in.c).mean():.2e})")
    print(f"  Rain:    {q_out_np.r.mean():.2e} kg/kg (change: {(q_out_np.r - q_in.r).mean():.2e})")
    print(f"  Snow:    {q_out_np.s.mean():.2e} kg/kg (change: {(q_out_np.s - q_in.s).mean():.2e})")
    print(f"  Ice:     {q_out_np.i.mean():.2e} kg/kg (change: {(q_out_np.i - q_in.i).mean():.2e})")
    print(f"  Graupel: {q_out_np.g.mean():.2e} kg/kg (change: {(q_out_np.g - q_in.g).mean():.2e})")

    pflx_np = np.array(pflx)
    print(
        f"\nPrecipitation flux: {pflx_np.mean():.2e} (range: {pflx_np.min():.2e} - {pflx_np.max():.2e})"
    )

    # Numerical stability checks
    assert np.all(np.isfinite(t_out_np)), "NaN/Inf in temperature output"
    assert np.all(np.isfinite(q_out_np.v)), "NaN/Inf in vapor"
    assert np.all(np.isfinite(q_out_np.c)), "NaN/Inf in cloud"
    assert np.all(np.isfinite(q_out_np.r)), "NaN/Inf in rain"
    assert np.all(np.isfinite(q_out_np.s)), "NaN/Inf in snow"
    assert np.all(np.isfinite(q_out_np.i)), "NaN/Inf in ice"
    assert np.all(np.isfinite(q_out_np.g)), "NaN/Inf in graupel"
    assert np.all(np.isfinite(pflx_np)), "NaN/Inf in precipitation flux"

    # Physical constraints
    assert np.all(q_out_np.v >= 0), "Negative vapor"
    assert np.all(q_out_np.c >= 0), "Negative cloud"
    assert np.all(q_out_np.r >= 0), "Negative rain"
    assert np.all(q_out_np.s >= 0), "Negative snow"
    assert np.all(q_out_np.i >= 0), "Negative ice"
    assert np.all(q_out_np.g >= 0), "Negative graupel"

    print("\n✓ All numerical stability and physical constraint checks passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Complete Graupel JAX Implementation Test")
    print("=" * 60)

    test_graupel_basic()

    print("\n" + "=" * 60)
    print("Test completed successfully! ✓")
    print("=" * 60)
