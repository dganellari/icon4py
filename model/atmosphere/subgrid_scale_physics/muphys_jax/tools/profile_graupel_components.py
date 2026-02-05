#!/usr/bin/env python3
"""
Profile individual components of graupel to identify bottlenecks.

This script measures each major component separately to understand
where time is spent and identify optimization opportunities.

Target: Reduce full graupel from 33ms to 10-13ms (DaCe-GPU level)

Usage:
    python profile_graupel_components.py --input /path/to/data.nc
"""

import argparse
import time
import sys
import pathlib

# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

import jax
# Enable x64 before any JAX operations
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from muphys_jax.utils.data_loading import load_graupel_inputs


def load_inputs(input_file: str, timestep: int = 0):
    """Load graupel inputs from netCDF (wrapper for compatibility)."""
    print(f"Loading inputs from: {input_file}")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(input_file, timestep)
    print(f"  Grid: {ncells} cells × {nlev} levels")
    return dz, t, p, rho, q, dt, qnc, ncells, nlev


def benchmark_function(fn, args, name, num_warmup=10, num_runs=20):
    """Benchmark a single function."""
    # Warmup
    for _ in range(num_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    print(f"  {name}: {np.median(times):.2f} ms (min: {np.min(times):.2f}, max: {np.max(times):.2f})")
    return np.median(times), result


def main():
    parser = argparse.ArgumentParser(description="Profile graupel components")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of benchmark runs")
    args = parser.parse_args()

    print("=" * 80)
    print("GRAUPEL COMPONENT PROFILING")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    # Load data
    print("Loading data...")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_inputs(args.input)
    print(f"Grid: {ncells} cells x {nlev} levels")

    # Pre-transpose to (nlev, ncells)
    print("\nPre-transposing data to (nlev, ncells)...")
    from muphys_jax.core.definitions import Q

    dz_t = jnp.transpose(dz)
    t_t = jnp.transpose(t)
    p_t = jnp.transpose(p)
    rho_t = jnp.transpose(rho)
    qnc_t = jnp.transpose(qnc)
    q_t = Q(
        v=jnp.transpose(q.v),
        c=jnp.transpose(q.c),
        r=jnp.transpose(q.r),
        s=jnp.transpose(q.s),
        i=jnp.transpose(q.i),
        g=jnp.transpose(q.g),
    )
    jax.block_until_ready((dz_t, t_t, p_t, rho_t, qnc_t, q_t))
    print(f"Transposed shape: {t_t.shape}")

    # Import components
    from muphys_jax.core.common import constants as const
    from muphys_jax.implementations.graupel_native_transposed import (
        q_t_update_native,
        precipitation_effects_native_transposed,
        _precipitation_effects_native_transposed_jax,
    )
    from muphys_jax.core.scans_transposed import (
        precip_scan_batched_transposed,
        temperature_update_scan_transposed,
    )
    from muphys_jax.core import thermo

    print("\n" + "=" * 80)
    print("COMPONENT TIMING (all in transposed layout)")
    print("=" * 80)

    # 1. Full graupel (baseline)
    print("\n1. FULL GRAUPEL (end-to-end)")
    from muphys_jax.implementations.graupel_native_transposed import graupel_run_native_transposed
    graupel_jit = jax.jit(graupel_run_native_transposed)

    full_time, _ = benchmark_function(
        graupel_jit,
        (dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t),
        "Full graupel",
        num_runs=args.num_runs
    )

    # 2. q_t_update (phase transitions)
    print("\n2. q_t_update (phase transitions)")
    q_t_update_jit = jax.jit(q_t_update_native)

    qt_update_time, (q_updated, t_updated) = benchmark_function(
        q_t_update_jit,
        (t_t, p_t, rho_t, q_t, dt, qnc_t),
        "q_t_update",
        num_runs=args.num_runs
    )

    # 3. precipitation_effects (with JAX scans, no HLO)
    print("\n3. precipitation_effects (JAX scans, no HLO injection)")
    kmin_r = q_t.r > const.qmin
    kmin_i = q_t.i > const.qmin
    kmin_s = q_t.s > const.qmin
    kmin_g = q_t.g > const.qmin
    last_level = nlev - 1

    precip_jax_jit = jax.jit(_precipitation_effects_native_transposed_jax, static_argnums=(0,))

    precip_jax_time, precip_result = benchmark_function(
        precip_jax_jit,
        (last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho_t, dz_t, dt),
        "precipitation_effects (JAX)",
        num_runs=args.num_runs
    )

    # 4. Isolated precipitation scans (just the 4 species scans)
    print("\n4. ISOLATED PRECIPITATION SCANS (4 species via batched vmap)")

    zeta = dt / (2.0 * dz_t)
    xrho = jnp.sqrt(const.rho_00 / rho_t)

    # Compute velocity coefficients
    vc_r = const.v0r * xrho
    vc_s = const.v0s * xrho
    vc_i = const.v0i * xrho
    vc_g = const.v0g * xrho

    params_list = [
        (const.alf, const.bet, const.qmin),  # rain
        (const.alfs, const.bets, const.qmin),  # snow
        (const.alfi, const.beti, const.qmin),  # ice
        (const.alfg, const.betg, 1e-8),  # graupel
    ]
    q_list = [q_updated.r, q_updated.s, q_updated.i, q_updated.g]
    vc_list = [vc_r, vc_s, vc_i, vc_g]
    mask_list = [kmin_r, kmin_s, kmin_i, kmin_g]

    def run_precip_scans():
        return precip_scan_batched_transposed(params_list, zeta, rho_t, q_list, vc_list, mask_list)

    precip_scans_jit = jax.jit(run_precip_scans)

    precip_scans_time, scan_results = benchmark_function(
        precip_scans_jit,
        (),
        "4 precipitation scans (batched)",
        num_runs=args.num_runs
    )

    # 5. Temperature scan
    print("\n5. ISOLATED TEMPERATURE SCAN")

    # Prepare temperature scan inputs
    qr_out, qs_out, qi_out, qg_out = [r[0] for r in scan_results]
    pr = scan_results[0][1]
    ps = scan_results[1][1]
    pi = scan_results[2][1]
    pg = scan_results[3][1]
    pflx_tot = pr + ps + pi + pg

    qliq = q_updated.c + q_updated.r
    qice = q_updated.s + q_updated.i + q_updated.g
    ei_old = thermo.internal_energy(t_updated, q_updated.v, qliq, qice, rho_t, dz_t)

    # t_kp1: shifted temperature
    t_kp1 = jnp.concatenate([t_updated[1:], t_updated[-1:]], axis=0)

    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    def run_temp_scan():
        return temperature_update_scan_transposed(
            t_updated, t_kp1, ei_old, pr, pflx_tot,
            q_updated.v, qliq, qice, rho_t, dz_t, dt, kmin_rsig
        )

    temp_scan_jit = jax.jit(run_temp_scan)

    temp_scan_time, temp_result = benchmark_function(
        temp_scan_jit,
        (),
        "temperature scan",
        num_runs=args.num_runs
    )

    # 6. Internal energy computation
    print("\n6. INTERNAL ENERGY COMPUTATION")

    def compute_internal_energy():
        return thermo.internal_energy(t_updated, q_updated.v, qliq, qice, rho_t, dz_t)

    ei_jit = jax.jit(compute_internal_energy)

    ei_time, _ = benchmark_function(
        ei_jit,
        (),
        "internal_energy",
        num_runs=args.num_runs
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFull graupel:              {full_time:.2f} ms (100%)")
    print(f"\nBreakdown:")
    print(f"  q_t_update:              {qt_update_time:.2f} ms ({qt_update_time/full_time*100:.1f}%)")
    print(f"  precipitation_effects:   {precip_jax_time:.2f} ms ({precip_jax_time/full_time*100:.1f}%)")
    print(f"    - 4 precip scans:      {precip_scans_time:.2f} ms ({precip_scans_time/full_time*100:.1f}%)")
    print(f"    - temperature scan:    {temp_scan_time:.2f} ms ({temp_scan_time/full_time*100:.1f}%)")
    print(f"    - internal_energy:     {ei_time:.2f} ms ({ei_time/full_time*100:.1f}%)")

    accounted = qt_update_time + precip_jax_time
    overhead = full_time - accounted
    print(f"\nAccounted time:            {accounted:.2f} ms")
    print(f"Overhead/other:            {overhead:.2f} ms ({overhead/full_time*100:.1f}%)")

    print("\n" + "=" * 80)
    print("OPTIMIZATION TARGETS")
    print("=" * 80)
    target = 12.0  # Target: 12ms
    current = full_time
    reduction_needed = current - target

    print(f"\nCurrent:  {current:.2f} ms")
    print(f"Target:   {target:.2f} ms")
    print(f"Reduce:   {reduction_needed:.2f} ms ({reduction_needed/current*100:.1f}%)")

    print(f"\nLargest opportunities:")
    components = [
        ("q_t_update (phase transitions)", qt_update_time),
        ("precipitation scans", precip_scans_time),
        ("temperature scan", temp_scan_time),
    ]
    components.sort(key=lambda x: x[1], reverse=True)
    for name, time_ms in components:
        print(f"  {name}: {time_ms:.2f} ms")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
