#!/usr/bin/env python3
"""
Benchmark the generated/unrolled precipitation_effects vs original.

Supports both dummy data and real input data from NetCDF files.

Usage:
    cd model/atmosphere/subgrid_scale_physics

    # With dummy data (default 20480x90 grid)
    JAX_ENABLE_X64=1 python -m muphys_jax.tools.benchmark_generated --num-runs 50

    # With real input data from NetCDF file
    JAX_ENABLE_X64=1 python -m muphys_jax.tools.benchmark_generated --input /path/to/input.nc

    # Benchmark full graupel (not just precipitation_effects)
    JAX_ENABLE_X64=1 python -m muphys_jax.tools.benchmark_generated --input /path/to/input.nc --full-graupel

    # Compare against reference output
    JAX_ENABLE_X64=1 python -m muphys_jax.tools.benchmark_generated --input /path/to/input.nc --reference /path/to/reference.nc
"""

import argparse
import sys
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# Input data loading
# ============================================================================

def _calc_dz(z: np.ndarray) -> np.ndarray:
    """Calculate layer thickness from geometric height."""
    ksize = z.shape[0]
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


def load_input_from_netcdf(filename: str):
    """Load input data from NetCDF file (same format as test data)."""
    import netCDF4

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.core.definitions import Q
    from muphys_jax.core.common import constants as const

    with netCDF4.Dataset(filename, mode="r") as ncfile:
        # Get dimensions
        try:
            ncells = len(ncfile.dimensions["cell"])
        except KeyError:
            ncells = len(ncfile.dimensions["ncells"])
        nlev = len(ncfile.dimensions["height"])

        # Calculate layer thickness
        dz = _calc_dz(ncfile.variables["zg"][:])
        dz = np.transpose(dz)  # (height, ncells) -> (ncells, height)

        # Load variables (transpose from (height, ncells) to (ncells, height))
        def load_var(varname: str) -> np.ndarray:
            var = ncfile.variables[varname]
            if var.dimensions[0] == "time":
                var = var[0, :, :]
            return np.transpose(var).astype(np.float64)

        # Create Q structure
        q = Q(
            v=jnp.array(load_var("hus")),
            c=jnp.array(load_var("clw")),
            r=jnp.array(load_var("qr")),
            s=jnp.array(load_var("qs")),
            i=jnp.array(load_var("cli")),
            g=jnp.array(load_var("qg")),
        )

        t = jnp.array(load_var("ta"))
        p = jnp.array(load_var("pfull"))
        rho = jnp.array(load_var("rho"))
        dz = jnp.array(dz)

    dt = 30.0
    qnc = 100.0
    last_lev = nlev - 1

    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    print(f"  Loaded input: {ncells} cells x {nlev} levels")

    return {
        'last_lev': last_lev,
        'kmin_r': kmin_r,
        'kmin_i': kmin_i,
        'kmin_s': kmin_s,
        'kmin_g': kmin_g,
        'q': q,
        't': t,
        'p': p,
        'rho': rho,
        'dz': dz,
        'dt': dt,
        'qnc': qnc,
        'ncells': ncells,
        'nlev': nlev,
    }


def load_reference_from_netcdf(filename: str):
    """Load reference output from NetCDF file."""
    import netCDF4

    with netCDF4.Dataset(filename, mode="r") as nc:
        # Transpose from (height, ncells) to (ncells, height)
        return {
            "t": np.array(nc.variables["ta"][:]).T,
            "qv": np.array(nc.variables["hus"][:]).T,
            "qc": np.array(nc.variables["clw"][:]).T,
            "qi": np.array(nc.variables["cli"][:]).T,
            "qr": np.array(nc.variables["qr"][:]).T,
            "qs": np.array(nc.variables["qs"][:]).T,
            "qg": np.array(nc.variables["qg"][:]).T,
        }


def create_dummy_inputs(ncells: int = 20480, nlev: int = 90):
    """Create dummy test inputs for precipitation_effects."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.core.definitions import Q
    from muphys_jax.core.common import constants as const

    dz = jnp.ones((ncells, nlev), dtype=jnp.float64) * 100.0
    t = jnp.ones((ncells, nlev), dtype=jnp.float64) * 280.0
    p = jnp.ones((ncells, nlev), dtype=jnp.float64) * 101325.0
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
    qnc = 100.0
    last_lev = nlev - 1

    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    print(f"  Created dummy input: {ncells} cells x {nlev} levels")

    return {
        'last_lev': last_lev,
        'kmin_r': kmin_r,
        'kmin_i': kmin_i,
        'kmin_s': kmin_s,
        'kmin_g': kmin_g,
        'q': q,
        't': t,
        'p': p,
        'rho': rho,
        'dz': dz,
        'dt': dt,
        'qnc': qnc,
        'ncells': ncells,
        'nlev': nlev,
    }


# ============================================================================
# Precipitation-only benchmarks
# ============================================================================

def benchmark_precip_original(inputs: dict, num_warmup: int = 3, num_runs: int = 50):
    """Benchmark original precipitation_effects with lax.scan."""
    print("\n" + "=" * 80)
    print("ORIGINAL precipitation_effects (lax.scan)")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.graupel_baseline import precipitation_effects
    from muphys_jax.core.definitions import Q

    last_lev = inputs['last_lev']
    dt = inputs['dt']

    @jax.jit
    def run_precip(kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz):
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        return precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt)

    q = inputs['q']
    args = (inputs['kmin_r'], inputs['kmin_i'], inputs['kmin_s'], inputs['kmin_g'],
            q.v, q.c, q.r, q.s, q.i, q.g, inputs['t'], inputs['rho'], inputs['dz'])

    # Compilation
    print("Compiling...")
    t0 = time.perf_counter()
    result = run_precip(*args)
    jax.block_until_ready(result)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"  Compilation: {compile_time:.2f} ms")

    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = run_precip(*args)
        jax.block_until_ready(result)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = run_precip(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")

    return np.mean(times), compile_time


def benchmark_precip_unrolled(inputs: dict, num_warmup: int = 3, num_runs: int = 50):
    """Benchmark unrolled precipitation_effects with Python for-loops."""
    print("\n" + "=" * 80)
    print("UNROLLED precipitation_effects (Python for-loop)")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.generated_precip import precipitation_effects_unrolled
    from muphys_jax.core.definitions import Q

    last_lev = inputs['last_lev']
    dt = inputs['dt']

    @jax.jit
    def run_precip(kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz):
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        return precipitation_effects_unrolled(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt)

    q = inputs['q']
    args = (inputs['kmin_r'], inputs['kmin_i'], inputs['kmin_s'], inputs['kmin_g'],
            q.v, q.c, q.r, q.s, q.i, q.g, inputs['t'], inputs['rho'], inputs['dz'])

    # Compilation
    print("Compiling...")
    t0 = time.perf_counter()
    result = run_precip(*args)
    jax.block_until_ready(result)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"  Compilation: {compile_time:.2f} ms")

    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = run_precip(*args)
        jax.block_until_ready(result)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = run_precip(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")

    return np.mean(times), compile_time


# ============================================================================
# Full graupel benchmarks
# ============================================================================

def benchmark_graupel_original(inputs: dict, num_warmup: int = 3, num_runs: int = 50):
    """Benchmark full graupel with original precipitation_effects."""
    print("\n" + "=" * 80)
    print("FULL GRAUPEL - Original (lax.scan)")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.graupel_baseline import graupel_run

    @jax.jit
    def run_graupel(dz, t, p, rho, qv, qc, qr, qs, qi, qg, dt, qnc):
        from muphys_jax.core.definitions import Q
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        return graupel_run(dz, t, p, rho, q_in, dt, qnc)

    q = inputs['q']
    args = (inputs['dz'], inputs['t'], inputs['p'], inputs['rho'],
            q.v, q.c, q.r, q.s, q.i, q.g, inputs['dt'], inputs['qnc'])

    # Compilation
    print("Compiling...")
    t0 = time.perf_counter()
    result = run_graupel(*args)
    jax.block_until_ready(result)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"  Compilation: {compile_time:.2f} ms")

    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = run_graupel(*args)
        jax.block_until_ready(result)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = run_graupel(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")

    return np.mean(times), compile_time, result


def benchmark_graupel_unrolled(inputs: dict, num_warmup: int = 3, num_runs: int = 50):
    """Benchmark full graupel with unrolled precipitation_effects."""
    print("\n" + "=" * 80)
    print("FULL GRAUPEL - Unrolled (Python for-loop)")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.generated_precip import graupel_unrolled_run

    @jax.jit
    def run_graupel(dz, t, p, rho, qv, qc, qr, qs, qi, qg, dt, qnc):
        from muphys_jax.core.definitions import Q
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        return graupel_unrolled_run(dz, t, p, rho, q_in, dt, qnc)

    q = inputs['q']
    args = (inputs['dz'], inputs['t'], inputs['p'], inputs['rho'],
            q.v, q.c, q.r, q.s, q.i, q.g, inputs['dt'], inputs['qnc'])

    # Compilation
    print("Compiling...")
    t0 = time.perf_counter()
    result = run_graupel(*args)
    jax.block_until_ready(result)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"  Compilation: {compile_time:.2f} ms")

    # Warmup
    print(f"Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = run_graupel(*args)
        jax.block_until_ready(result)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        result = run_graupel(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  Std:    {np.std(times):.3f} ms")
    print(f"  Min:    {np.min(times):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")

    return np.mean(times), compile_time, result


# ============================================================================
# Correctness verification
# ============================================================================

def verify_precip_correctness(inputs: dict):
    """Verify that unrolled precipitation produces same results as original."""
    print("\n" + "=" * 80)
    print("CORRECTNESS CHECK - precipitation_effects")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.graupel_baseline import precipitation_effects
    from muphys_jax.implementations.generated_precip import precipitation_effects_unrolled

    q = inputs['q']
    last_lev = inputs['last_lev']
    dt = inputs['dt']

    # Run original
    result_orig = precipitation_effects(
        last_lev, inputs['kmin_r'], inputs['kmin_i'], inputs['kmin_s'], inputs['kmin_g'],
        q, inputs['t'], inputs['rho'], inputs['dz'], dt
    )

    # Run unrolled
    result_unrolled = precipitation_effects_unrolled(
        last_lev, inputs['kmin_r'], inputs['kmin_i'], inputs['kmin_s'], inputs['kmin_g'],
        q, inputs['t'], inputs['rho'], inputs['dz'], dt
    )

    # Compare outputs
    all_close = True
    output_names = ['qr', 'qs', 'qi', 'qg', 't_new', 'pflx_tot', 'pr', 'ps', 'pi', 'pg', 'eflx']

    for i, name in enumerate(output_names):
        orig = result_orig[i]
        unrolled = result_unrolled[i]
        max_diff = float(jnp.max(jnp.abs(orig - unrolled)))
        rel_diff = float(jnp.max(jnp.abs(orig - unrolled) / (jnp.abs(orig) + 1e-10)))

        if max_diff > 1e-10:
            print(f"  {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
            if max_diff > 1e-6:
                all_close = False
        else:
            print(f"  {name}: OK (max_diff={max_diff:.2e})")

    if all_close:
        print("\n  ✓ All outputs match within tolerance!")
    else:
        print("\n  ✗ Some outputs differ significantly!")

    return all_close


def verify_graupel_correctness(inputs: dict):
    """Verify that graupel with unrolled precip produces same results as original."""
    print("\n" + "=" * 80)
    print("CORRECTNESS CHECK - Full graupel")
    print("=" * 80)

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.implementations.graupel_baseline import graupel_run
    from muphys_jax.implementations.generated_precip import graupel_unrolled_run

    q = inputs['q']

    # Run original
    result_orig = graupel_run(
        inputs['dz'], inputs['t'], inputs['p'], inputs['rho'],
        q, inputs['dt'], inputs['qnc']
    )

    # Run unrolled
    result_unrolled = graupel_unrolled_run(
        inputs['dz'], inputs['t'], inputs['p'], inputs['rho'],
        q, inputs['dt'], inputs['qnc']
    )

    # Compare outputs: (t, q_out, pflx, pr, ps, pi, pg, pre)
    t_orig, q_orig, pflx_orig, pr_orig, ps_orig, pi_orig, pg_orig, pre_orig = result_orig
    t_unrolled, q_unrolled, pflx_unrolled, pr_unrolled, ps_unrolled, pi_unrolled, pg_unrolled, pre_unrolled = result_unrolled

    all_close = True

    def check_field(name, orig, unrolled):
        nonlocal all_close
        max_diff = float(jnp.max(jnp.abs(orig - unrolled)))
        rel_diff = float(jnp.max(jnp.abs(orig - unrolled) / (jnp.abs(orig) + 1e-10)))

        if max_diff > 1e-10:
            print(f"  {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
            if max_diff > 1e-6:
                all_close = False
        else:
            print(f"  {name}: OK (max_diff={max_diff:.2e})")

    check_field("t", t_orig, t_unrolled)
    check_field("qv", q_orig.v, q_unrolled.v)
    check_field("qc", q_orig.c, q_unrolled.c)
    check_field("qr", q_orig.r, q_unrolled.r)
    check_field("qs", q_orig.s, q_unrolled.s)
    check_field("qi", q_orig.i, q_unrolled.i)
    check_field("qg", q_orig.g, q_unrolled.g)
    check_field("pflx", pflx_orig, pflx_unrolled)
    check_field("pr", pr_orig, pr_unrolled)
    check_field("ps", ps_orig, ps_unrolled)
    check_field("pi", pi_orig, pi_unrolled)
    check_field("pg", pg_orig, pg_unrolled)
    check_field("pre", pre_orig, pre_unrolled)

    if all_close:
        print("\n  ✓ All outputs match within tolerance!")
    else:
        print("\n  ✗ Some outputs differ significantly!")

    return all_close


def verify_against_reference(result, reference: dict, rtol: float = 1e-7, atol: float = 1e-8):
    """Verify graupel results against reference NetCDF data."""
    print("\n" + "=" * 80)
    print("VERIFICATION AGAINST REFERENCE")
    print("=" * 80)

    t_out, q_out, pflx, pr, ps, pi, pg, pre = result

    all_close = True

    def check_field(name, computed, ref_name):
        nonlocal all_close
        ref = reference[ref_name]
        max_diff = float(np.max(np.abs(np.array(computed) - ref)))
        try:
            np.testing.assert_allclose(ref, np.array(computed), atol=atol, rtol=rtol)
            print(f"  {name}: OK (max_diff={max_diff:.2e})")
        except AssertionError as e:
            print(f"  {name}: FAILED (max_diff={max_diff:.2e})")
            all_close = False

    check_field("t", t_out, "t")
    check_field("qv", q_out.v, "qv")
    check_field("qc", q_out.c, "qc")
    check_field("qr", q_out.r, "qr")
    check_field("qs", q_out.s, "qs")
    check_field("qi", q_out.i, "qi")
    check_field("qg", q_out.g, "qg")

    if all_close:
        print("\n  ✓ All outputs match reference!")
    else:
        print("\n  ✗ Some outputs differ from reference!")

    return all_close


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark generated vs original precipitation_effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Benchmark with dummy data
    python -m muphys_jax.tools.benchmark_generated --num-runs 50

    # Benchmark with real input data
    python -m muphys_jax.tools.benchmark_generated --input input.nc

    # Benchmark full graupel with reference verification
    python -m muphys_jax.tools.benchmark_generated --input input.nc --reference reference.nc --full-graupel
        """
    )
    parser.add_argument('--input', '-i', type=str, help='Input NetCDF file path')
    parser.add_argument('--reference', '-r', type=str, help='Reference NetCDF file path for verification')
    parser.add_argument('--num-warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=50, help='Number of benchmark runs')
    parser.add_argument('--skip-verify', action='store_true', help='Skip correctness verification')
    parser.add_argument('--only-unrolled', action='store_true', help='Only benchmark unrolled version')
    parser.add_argument('--only-original', action='store_true', help='Only benchmark original version')
    parser.add_argument('--full-graupel', action='store_true', help='Benchmark full graupel (not just precipitation)')
    parser.add_argument('--ncells', type=int, default=20480, help='Number of cells for dummy data')
    parser.add_argument('--nlev', type=int, default=90, help='Number of levels for dummy data')
    args = parser.parse_args()

    print("=" * 80)
    print("BENCHMARK: Generated vs Original precipitation_effects")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"Devices: {jax.devices()}")

    # Load or create inputs
    print("\n" + "-" * 40)
    print("Loading input data...")
    if args.input:
        inputs = load_input_from_netcdf(args.input)
    else:
        inputs = create_dummy_inputs(args.ncells, args.nlev)

    # Load reference if provided
    reference = None
    if args.reference:
        print(f"  Loading reference from: {args.reference}")
        reference = load_reference_from_netcdf(args.reference)

    # Verify correctness first
    if not args.skip_verify and not args.only_original:
        if args.full_graupel:
            correct = verify_graupel_correctness(inputs)
        else:
            correct = verify_precip_correctness(inputs)
        if not correct:
            print("\nWARNING: Results differ! Benchmark results may not be comparable.")

    # Run benchmarks
    if args.full_graupel:
        # Full graupel benchmarks
        if not args.only_unrolled:
            orig_mean, orig_compile, result_orig = benchmark_graupel_original(
                inputs, args.num_warmup, args.num_runs
            )
            if reference:
                verify_against_reference(result_orig, reference)
        else:
            orig_mean, orig_compile = None, None

        if not args.only_original:
            unroll_mean, unroll_compile, result_unrolled = benchmark_graupel_unrolled(
                inputs, args.num_warmup, args.num_runs
            )
            if reference:
                verify_against_reference(result_unrolled, reference)
        else:
            unroll_mean, unroll_compile = None, None
    else:
        # Precipitation-only benchmarks
        if not args.only_unrolled:
            orig_mean, orig_compile = benchmark_precip_original(
                inputs, args.num_warmup, args.num_runs
            )
        else:
            orig_mean, orig_compile = None, None

        if not args.only_original:
            unroll_mean, unroll_compile = benchmark_precip_unrolled(
                inputs, args.num_warmup, args.num_runs
            )
        else:
            unroll_mean, unroll_compile = None, None

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    mode = "Full graupel" if args.full_graupel else "precipitation_effects"
    print(f"  Mode: {mode}")
    print(f"  Grid: {inputs['ncells']} cells x {inputs['nlev']} levels")
    print()

    if orig_mean is not None:
        print(f"  Original (lax.scan):      {orig_mean:.3f} ms (compile: {orig_compile:.0f} ms)")
    if unroll_mean is not None:
        print(f"  Unrolled (Python for):    {unroll_mean:.3f} ms (compile: {unroll_compile:.0f} ms)")

    if orig_mean is not None and unroll_mean is not None:
        speedup = orig_mean / unroll_mean
        if speedup > 1:
            print(f"\n  Unrolled is {speedup:.2f}x FASTER")
        else:
            print(f"\n  Unrolled is {1/speedup:.2f}x SLOWER")


if __name__ == "__main__":
    main()
