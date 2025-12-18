# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive numerical comparison: GT4Py vs JAX graupel implementations.

This script verifies that the JAX implementation produces the same numerical
results as the GT4Py reference implementation.

Usage:
    python compare_gt4py_jax.py [--grid NCELLS NLEV] [--tolerance RTOL]

Example:
    python compare_gt4py_jax.py --grid 100 65 --tolerance 1e-6
"""

import argparse
import numpy as np
import sys
import os

print("="*80)
print("GT4Py vs JAX Numerical Validation")
print("="*80)

# ============================================================================
# Import GT4Py Implementation
# ============================================================================

try:
    import gt4py.next as gtx
    from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import (
        graupel_run as gt4py_graupel_run
    )
    from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q as GT4PyQ
    from icon4py.model.common.model_backends import get_backend
    from icon4py.model.common import dimension as dims
    GT4PY_AVAILABLE = True
    print("✓ GT4Py implementation loaded")
except ImportError as e:
    GT4PY_AVAILABLE = False
    print(f"✗ GT4Py not available: {e}")
    print("\nThis script requires GT4Py to be installed.")
    print("Install with: uv sync --extra all")
    sys.exit(1)

# ============================================================================
# Import JAX Implementation
# ============================================================================

try:
    import jax
    import jax.numpy as jnp
    from graupel_jax import graupel_run as jax_graupel_run, Q as JaxQ
    JAX_AVAILABLE = True
    print(f"✓ JAX implementation loaded (version {jax.__version__})")
except ImportError as e:
    JAX_AVAILABLE = False
    print(f"✗ JAX not available: {e}")
    sys.exit(1)


# ============================================================================
# Test Data Generation
# ============================================================================

def create_test_data(ncells, nlev, seed=42):
    """
    Create realistic atmospheric test data.

    Args:
        ncells: Number of horizontal grid cells
        nlev: Number of vertical levels
        seed: Random seed for reproducibility

    Returns:
        Dictionary with all input fields
    """
    np.random.seed(seed)

    print(f"\nGenerating test data:")
    print(f"  Grid: {ncells} cells × {nlev} levels")
    print(f"  Random seed: {seed}")

    # Temperature profile (realistic troposphere)
    t_surface = 288.0  # K
    lapse_rate = 6.5 / 1000  # K/m
    heights = np.linspace(0, 16000, nlev)  # 0-16km
    te_profile = t_surface - lapse_rate * heights
    te = te_profile[None, :] + np.random.randn(ncells, nlev) * 1.0  # Small perturbations

    # Pressure profile (hydrostatic)
    p_surface = 101325.0  # Pa
    scale_height = 8500.0  # m
    p_profile = p_surface * np.exp(-heights / scale_height)
    p = p_profile[None, :] * (1.0 + np.random.randn(ncells, nlev) * 0.01)

    # Density from ideal gas law
    R_d = 287.04  # J/(kg·K)
    rho = p / (R_d * te)

    # Layer thickness (uniform for simplicity)
    dz = np.full((ncells, nlev), 500.0)

    # Water species (physically realistic ranges)
    q_v = np.random.uniform(1e-4, 5e-3, (ncells, nlev))  # Vapor: 0.1-5 g/kg
    q_c = np.random.uniform(0.0, 1e-4, (ncells, nlev))   # Cloud: 0-0.1 g/kg
    q_r = np.random.uniform(0.0, 5e-5, (ncells, nlev))   # Rain: 0-0.05 g/kg
    q_s = np.random.uniform(0.0, 1e-5, (ncells, nlev))   # Snow: 0-0.01 g/kg
    q_i = np.random.uniform(0.0, 5e-6, (ncells, nlev))   # Ice: 0-0.005 g/kg
    q_g = np.random.uniform(0.0, 1e-6, (ncells, nlev))   # Graupel: 0-0.001 g/kg

    # Parameters
    dt = 10.0   # 10 second timestep
    qnc = 1e8   # Cloud droplet number concentration [1/m³]

    print(f"  Temperature: {te.min():.1f} - {te.max():.1f} K")
    print(f"  Pressure: {p.min():.0f} - {p.max():.0f} Pa")
    print(f"  Vapor: {q_v.mean():.2e} ± {q_v.std():.2e} kg/kg")

    return {
        'dz': dz,
        'te': te,
        'p': p,
        'rho': rho,
        'q_v': q_v,
        'q_c': q_c,
        'q_r': q_r,
        'q_s': q_s,
        'q_i': q_i,
        'q_g': q_g,
        'dt': dt,
        'qnc': qnc,
    }


# ============================================================================
# Run GT4Py Implementation
# ============================================================================

def run_gt4py(data, backend='gtfn_cpu'):
    """
    Run GT4Py implementation.

    Args:
        data: Dictionary with input fields
        backend: GT4Py backend to use

    Returns:
        Dictionary with output fields
    """
    print(f"\n--- Running GT4Py (backend={backend}) ---")

    ncells, nlev = data['te'].shape

    # Get backend
    try:
        gtx_backend = get_backend(backend)
    except Exception as e:
        print(f"Error getting backend '{backend}': {e}")
        raise

    # Create GT4Py fields
    # Note: This requires setting up proper GT4Py field infrastructure
    # The exact implementation depends on how GT4Py fields are created in your setup

    from gt4py.next import as_field
    from icon4py.model.common import dimension as dims

    # Convert to GT4Py fields (simplified - actual implementation may differ)
    dz_field = as_field((dims.CellDim, dims.KDim), data['dz'])
    te_field = as_field((dims.CellDim, dims.KDim), data['te'])
    p_field = as_field((dims.CellDim, dims.KDim), data['p'])
    rho_field = as_field((dims.CellDim, dims.KDim), data['rho'])

    # Create Q input
    q_in = GT4PyQ(
        v=as_field((dims.CellDim, dims.KDim), data['q_v']),
        c=as_field((dims.CellDim, dims.KDim), data['q_c']),
        r=as_field((dims.CellDim, dims.KDim), data['q_r']),
        s=as_field((dims.CellDim, dims.KDim), data['q_s']),
        i=as_field((dims.CellDim, dims.KDim), data['q_i']),
        g=as_field((dims.CellDim, dims.KDim), data['q_g']),
    )

    # Allocate output fields
    q_out = GT4PyQ(
        v=gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend),
        c=gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend),
        r=gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend),
        s=gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend),
        i=gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend),
        g=gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend),
    )
    t_out = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)
    pflx = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)
    pr = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)
    ps = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)
    pi = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)
    pg = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)
    pre = gtx.zeros((dims.CellDim, dims.KDim), gtx.float64, allocator=gtx_backend)

    # Run GT4Py graupel
    import time
    start = time.time()

    gt4py_graupel_run(
        dz=dz_field,
        te=te_field,
        p=p_field,
        rho=rho_field,
        q_in=q_in,
        dt=data['dt'],
        qnc=data['qnc'],
        q_out=q_out,
        t_out=t_out,
        pflx=pflx,
        pr=pr,
        ps=ps,
        pi=pi,
        pg=pg,
        pre=pre,
        horizontal_start=0,
        horizontal_end=ncells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={},  # May need proper offset provider
    )

    end = time.time()
    print(f"Execution time: {(end-start)*1000:.1f} ms")

    # Convert outputs to numpy
    results = {
        't_out': np.asarray(t_out),
        'q_v': np.asarray(q_out.v),
        'q_c': np.asarray(q_out.c),
        'q_r': np.asarray(q_out.r),
        'q_s': np.asarray(q_out.s),
        'q_i': np.asarray(q_out.i),
        'q_g': np.asarray(q_out.g),
        'pflx': np.asarray(pflx),
    }

    print(f"Temperature: mean={results['t_out'].mean():.3f} K")
    print(f"Vapor: mean={results['q_v'].mean():.2e} kg/kg")
    print(f"Precip flux: mean={results['pflx'].mean():.2e}")

    return results


# ============================================================================
# Run JAX Implementation
# ============================================================================

def run_jax(data):
    """
    Run JAX implementation.

    Args:
        data: Dictionary with input fields

    Returns:
        Dictionary with output fields
    """
    print(f"\n--- Running JAX (backend={os.getenv('JAX_BACKEND', 'xla')}) ---")

    # Create JAX Q input
    q_in = JaxQ(
        v=jnp.array(data['q_v']),
        c=jnp.array(data['q_c']),
        r=jnp.array(data['q_r']),
        s=jnp.array(data['q_s']),
        i=jnp.array(data['q_i']),
        g=jnp.array(data['q_g']),
    )

    # Run JAX graupel
    import time
    start = time.time()

    t_out, q_out, pflx, pr, ps, pi, pg, pre = jax_graupel_run(
        jnp.array(data['dz']),
        jnp.array(data['te']),
        jnp.array(data['p']),
        jnp.array(data['rho']),
        q_in,
        data['dt'],
        data['qnc']
    )

    # Block until ready
    t_out_np = np.array(t_out)
    end = time.time()
    print(f"Execution time: {(end-start)*1000:.1f} ms")

    # Convert outputs to numpy
    results = {
        't_out': t_out_np,
        'q_v': np.array(q_out.v),
        'q_c': np.array(q_out.c),
        'q_r': np.array(q_out.r),
        'q_s': np.array(q_out.s),
        'q_i': np.array(q_out.i),
        'q_g': np.array(q_out.g),
        'pflx': np.array(pflx),
    }

    print(f"Temperature: mean={results['t_out'].mean():.3f} K")
    print(f"Vapor: mean={results['q_v'].mean():.2e} kg/kg")
    print(f"Precip flux: mean={results['pflx'].mean():.2e}")

    return results


# ============================================================================
# Compare Results
# ============================================================================

def compare_results(gt4py_results, jax_results, rtol=1e-6, atol=1e-10):
    """
    Compare GT4Py and JAX results.

    Args:
        gt4py_results: Dictionary with GT4Py outputs
        jax_results: Dictionary with JAX outputs
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if all fields match within tolerance
    """
    print("\n" + "="*80)
    print("NUMERICAL COMPARISON")
    print("="*80)

    all_pass = True

    fields = [
        ('t_out', 'Temperature', 'K'),
        ('q_v', 'Vapor', 'kg/kg'),
        ('q_c', 'Cloud', 'kg/kg'),
        ('q_r', 'Rain', 'kg/kg'),
        ('q_s', 'Snow', 'kg/kg'),
        ('q_i', 'Ice', 'kg/kg'),
        ('q_g', 'Graupel', 'kg/kg'),
        ('pflx', 'Precip flux', 'kg/m²/s'),
    ]

    for field_key, field_name, units in fields:
        gt4py_field = gt4py_results[field_key]
        jax_field = jax_results[field_key]

        # Compute differences
        abs_diff = np.abs(jax_field - gt4py_field)
        rel_diff = abs_diff / (np.abs(gt4py_field) + atol)

        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()

        # Check tolerance
        passed = (max_abs_diff < atol) or (max_rel_diff < rtol)
        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\n{field_name} [{units}]:")
        print(f"  Max abs diff:  {max_abs_diff:.2e} {status}")
        print(f"  Mean abs diff: {mean_abs_diff:.2e}")
        print(f"  Max rel diff:  {max_rel_diff:.2e}")
        print(f"  Mean rel diff: {mean_rel_diff:.2e}")

        if not passed:
            all_pass = False
            # Show where largest differences occur
            worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
            print(f"  Worst at index {worst_idx}:")
            print(f"    GT4Py: {gt4py_field[worst_idx]:.6e}")
            print(f"    JAX:   {jax_field[worst_idx]:.6e}")

    print("\n" + "="*80)
    if all_pass:
        print("✓ ALL FIELDS MATCH within tolerance!")
        print(f"  Relative tolerance: {rtol}")
        print(f"  Absolute tolerance: {atol}")
    else:
        print("✗ SOME FIELDS DO NOT MATCH")
        print("  This may indicate:")
        print("  - Implementation differences")
        print("  - Numerical precision issues")
        print("  - Different optimization orders")
    print("="*80)

    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare GT4Py and JAX graupel implementations')
    parser.add_argument('--grid', nargs=2, type=int, default=[100, 65],
                        help='Grid size: ncells nlev (default: 100 65)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Relative tolerance for comparison (default: 1e-6)')
    parser.add_argument('--backend', type=str, default='gtfn_cpu',
                        help='GT4Py backend (default: gtfn_cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    ncells, nlev = args.grid

    # Generate test data
    data = create_test_data(ncells, nlev, seed=args.seed)

    # Run both implementations
    try:
        gt4py_results = run_gt4py(data, backend=args.backend)
    except Exception as e:
        print(f"\n✗ GT4Py execution failed: {e}")
        print("\nNote: GT4Py requires proper field setup and backend configuration.")
        print("This comparison script is a template that may need adjustments")
        print("based on your specific GT4Py setup.")
        sys.exit(1)

    jax_results = run_jax(data)

    # Compare results
    match = compare_results(gt4py_results, jax_results, rtol=args.tolerance)

    sys.exit(0 if match else 1)


if __name__ == '__main__':
    main()
