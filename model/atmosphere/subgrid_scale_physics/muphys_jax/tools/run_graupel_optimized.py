#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver script to benchmark graupel implementations with HLO injection support.

Supports both .hlo text files and .serialized binary files for HLO injection.

Usage:
    # Run baseline (no optimization)
    JAX_ENABLE_X64=1 python tools/run_graupel_optimized.py \
        --input data.nc \
        --mode baseline

    # Run with HLO text file injection
    JAX_ENABLE_X64=1 python tools/run_graupel_optimized.py \
        --input data.nc \
        --optimized-hlo shlo/precip_effect_x64_batched_fused.hlo \
        --mode baseline

    # Compare baseline vs optimized
    JAX_ENABLE_X64=1 python tools/run_graupel_optimized.py \
        --input data.nc \
        --optimized-hlo shlo/precip_effect_x64_batched_fused.hlo \
        --compare

    # Benchmark precipitation_effects only (not full graupel)
    JAX_ENABLE_X64=1 python tools/run_graupel_optimized.py \
        --input data.nc \
        --precip-only \
        --optimized-hlo shlo/precip_effect_x64_batched_fused.hlo
"""

import argparse
import pathlib
import sys
import time

import jax


# Enable x64 before any JAX operations
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


# Add parent to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

from muphys_jax.utils.data_loading import load_graupel_inputs


def load_inputs(input_file: str, timestep: int = 0):
    """Load graupel inputs from netCDF (wrapper for compatibility)."""
    print(f"Loading inputs from: {input_file}")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(input_file, timestep)
    print(f"  Grid: {ncells} cells × {nlev} levels")
    return dz, t, p, rho, q, dt, qnc, ncells, nlev


def load_hlo_module(hlo_path: str):
    """
    Load HLO from file (supports .hlo text and .serialized binary).

    Returns the HLO content (text or bytes).
    """
    path = pathlib.Path(hlo_path)

    if path.suffix == ".serialized":
        with open(path, "rb") as f:
            return f.read()
    elif path.suffix in (".hlo", ".stablehlo", ".txt"):
        with open(path) as f:
            return f.read()
    else:
        # Try as text first
        try:
            with open(path) as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                return f.read()


def compile_hlo_text(hlo_text: str, platform: str = "cuda"):
    """
    Compile HLO/StableHLO text to an executable.

    Supports both HLO and StableHLO (MLIR) formats.
    Returns (executable, client).
    """
    import jaxlib._jax as jax_cpp

    # Get GPU client using newer API
    backend_name = "gpu" if platform.lower() == "cuda" else "cpu"
    try:
        client = jax.extend.backend.get_backend(backend_name)
    except (AttributeError, ModuleNotFoundError):
        # Fallback for older JAX versions
        from jax.lib import xla_bridge

        client = xla_bridge.get_backend(backend_name)

    devices = client.local_devices()[:1]
    device_list = jax_cpp.DeviceList(tuple(devices))
    compile_options = jax_cpp.CompileOptions()

    # Compile (works for both HLO and StableHLO)
    executable = client.compile(hlo_text, device_list, compile_options)

    # Serialize and deserialize to get LoadedExecutable with execute method
    serialized = executable.serialize()
    loaded = client.deserialize_executable(serialized, device_list, compile_options)

    return loaded, client


def benchmark_hlo_direct(
    hlo_path: str,
    input_file: str,
    num_warmup: int = 3,
    num_runs: int = 10,
    transposed: bool = False,
):
    """
    Benchmark an HLO/StableHLO file directly using XLA client (most accurate).

    This bypasses JAX's JIT and custom_call mechanism, giving pure
    execution time without any JAX overhead.

    Supports both HLO (.hlo) and StableHLO (.stablehlo) formats.

    Args:
        hlo_path: Path to HLO/StableHLO file
        input_file: Path to NetCDF input file
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        transposed: If True, transpose inputs to nlev×ncells layout (for optimized HLO)
    """
    from muphys_jax.core.common import constants as const

    print("=" * 80)
    print(f"DIRECT HLO BENCHMARK: {hlo_path}")
    if transposed:
        print("LAYOUT: TRANSPOSED (nlev×ncells) - coalesced GPU memory access")
    else:
        print("LAYOUT: ORIGINAL (ncells×nlev)")
    print("=" * 80)

    # Load and compile HLO
    print("\nLoading HLO...")
    hlo_content = load_hlo_module(hlo_path)

    print("Compiling HLO...")
    start = time.perf_counter()
    executable, client = compile_hlo_text(hlo_content, "cuda")
    compile_time = time.perf_counter() - start
    print(f"Compilation time: {compile_time:.2f} s")

    # Load real inputs
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_inputs(input_file)

    # Prepare inputs matching the HLO signature
    # kmin masks
    kmin_r = np.array(q.r > const.qmin)
    kmin_i = np.array(q.i > const.qmin)
    kmin_s = np.array(q.s > const.qmin)
    kmin_g = np.array(q.g > const.qmin)

    # Convert to numpy arrays with correct layout
    inputs = [
        kmin_r.astype(np.bool_),
        kmin_i.astype(np.bool_),
        kmin_s.astype(np.bool_),
        kmin_g.astype(np.bool_),
        np.array(q.v, dtype=np.float64),
        np.array(q.c, dtype=np.float64),
        np.array(q.r, dtype=np.float64),
        np.array(q.s, dtype=np.float64),
        np.array(q.i, dtype=np.float64),
        np.array(q.g, dtype=np.float64),
        np.array(t, dtype=np.float64),
        np.array(rho, dtype=np.float64),
        np.array(dz, dtype=np.float64),
    ]

    # Transpose inputs if needed for transposed HLO
    if transposed:
        print(f"  Transposing inputs from {inputs[0].shape} to ", end="")
        inputs = [np.ascontiguousarray(np.transpose(inp)) for inp in inputs]
        print(f"{inputs[0].shape}")

    # Transfer to device
    device = client.local_devices()[0]
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]
    device_inputs = [x.addressable_data(0) for x in jax_inputs]

    # Warmup - do more warmup runs to stabilize GPU clocks
    actual_warmup = max(num_warmup, 10)  # At least 10 warmup runs
    print(f"\nWarming up ({actual_warmup} runs to stabilize GPU clocks)...")
    for i in range(actual_warmup):
        results = executable.execute(device_inputs)
        jax.block_until_ready(results)
        if i < 3 or i >= actual_warmup - 3:
            print(f"  Warmup {i+1}/{actual_warmup} complete")

    # Benchmark
    print(f"\nBenchmarking ({num_runs} runs)...")
    times = []
    total_start = time.perf_counter()
    for i in range(num_runs):
        start = time.perf_counter()
        results = executable.execute(device_inputs)
        jax.block_until_ready(results)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        # Print first 10 runs as they happen
        if i < 10:
            print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")
    total_elapsed = time.perf_counter() - total_start

    times = np.array(times)

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)

    # Detect outliers (values > 1.5x median)
    outliers = times > (median_time * 1.5)
    num_outliers = np.sum(outliers)

    # Print summary
    print(
        f"\n  Min: {min_time:.2f} ms, Max: {max_time:.2f} ms, Median: {median_time:.2f} ms, Mean: {mean_time:.2f} ms"
    )
    print(f"  Total time for {num_runs} runs: {total_elapsed:.3f} s")
    if num_outliers > 0:
        print(f"  ⚠ WARNING: {num_outliers}/{num_runs} runs were outliers (>1.5x median)")

    print("\n" + "=" * 80)
    print("RESULTS (direct HLO execution, no JAX overhead)")
    print("=" * 80)
    print(f"HLO file:   {hlo_path}")
    print(f"Grid size:  {ncells} cells × {nlev} levels")
    print(f"Layout:     {'transposed (nlev×ncells)' if transposed else 'original (ncells×nlev)'}")
    print("\nExecution time (ms):")
    print(f"  Mean:      {mean_time:.2f} ± {std_time:.2f}")
    print(f"  Median:    {median_time:.2f}  ← use this for stable performance")
    print(f"  Min:       {min_time:.2f}")
    print(f"  Max:       {max_time:.2f}")
    print(f"  95th %ile: {p95_time:.2f}")
    print(f"  99th %ile: {p99_time:.2f}")
    if num_outliers > 0:
        print("\nOutlier Analysis:")
        print(f"  {num_outliers}/{num_runs} runs exceeded 1.5x median ({median_time*1.5:.2f} ms)")
        print("  This indicates GPU throttling or context switching")
        print(f"  → Use MEDIAN ({median_time:.2f} ms) as reliable performance metric")

    # Transpose outputs back if needed
    if transposed:
        results = [np.transpose(np.asarray(r)) for r in results]

    return np.mean(times), np.std(times), results


def run_graupel_native_transposed(
    input_file: str, num_warmup: int = 3, num_runs: int = 10, optimized_hlo: str = None
):
    """
    Run graupel with NATIVE TRANSPOSED layout - ZERO transposes during computation.

    Option 3: Data is loaded and transposed ONCE (not timed), then all computation
    happens in (nlev, ncells) layout without any transposes.

    When optimized_hlo is provided, injects the transposed HLO for 2x speedup.
    """
    from muphys_jax.core.definitions import Q
    from muphys_jax.core.optimized_precip import configure_optimized_precip
    from muphys_jax.implementations.graupel_native_transposed import graupel_run_native_transposed

    print("=" * 80)
    print("RUNNING GRAUPEL - MODE: NATIVE-TRANSPOSED")
    print("Option 3: ZERO transposes during computation")
    print("Data is pre-transposed once (not measured)")
    if optimized_hlo:
        print(f"WITH OPTIMIZED HLO: {optimized_hlo}")
        configure_optimized_precip(hlo_path=optimized_hlo, use_optimized=True, transposed=True)
        print("✓ Configured optimized HLO for transposed layout (nlev×ncells)")
    else:
        print("NO OPTIMIZED HLO (using pure JAX scans)")
    print("=" * 80)

    # Load inputs in original layout
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_inputs(input_file)

    # Pre-transpose ALL data BEFORE timing starts
    print(f"\nPre-transposing data from ({ncells}, {nlev}) to ({nlev}, {ncells})...")
    print("(This transpose is NOT included in timing)")

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

    # Block to ensure transpose is complete
    jax.block_until_ready((dz_t, t_t, p_t, rho_t, qnc_t, q_t))
    print(f"  Data transposed: ({nlev}, {ncells}) = {dz_t.shape}")

    print("\nImplementation: graupel_native_transposed")
    print(f"Grid: {ncells} cells × {nlev} levels")
    print("Layout: TRANSPOSED (nlev×ncells) - ZERO transposes during computation")

    # Warmup
    print(f"\nWarming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        result = graupel_run_native_transposed(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
        jax.block_until_ready(result)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark full graupel
    print(f"\nBenchmarking full graupel ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = graupel_run_native_transposed(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        if i < 10:
            print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)

    # =========================================================================
    # Isolated precipitation_effects_native_transposed timing
    # =========================================================================
    print("\n" + "-" * 60)
    print("ISOLATED precipitation_effects TIMING (NATIVE TRANSPOSED)")
    print("-" * 60)

    from muphys_jax.core.common import constants as const
    from muphys_jax.implementations.graupel_native_transposed import (
        precipitation_effects_native_transposed,
        q_t_update_native,
    )

    # Prepare inputs for precipitation_effects (same as in graupel function)
    kmin_r = q_t.r > const.qmin
    kmin_i = q_t.i > const.qmin
    kmin_s = q_t.s > const.qmin
    kmin_g = q_t.g > const.qmin
    last_level = nlev - 1

    # Run phase transitions first to get intermediate state
    q_updated, t_updated = q_t_update_native(t_t, p_t, rho_t, q_t, dt, qnc_t)
    jax.block_until_ready((q_updated, t_updated))

    # JIT compile precipitation_effects_native_transposed
    precip_jit = jax.jit(precipitation_effects_native_transposed, static_argnums=(0,))

    # Warmup precipitation_effects
    print("  Warming up isolated precipitation_effects...")
    for _ in range(max(num_warmup, 10)):
        precip_result = precip_jit(
            last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho_t, dz_t, dt
        )
        jax.block_until_ready(precip_result)

    # Benchmark isolated precipitation_effects
    print(f"  Benchmarking ({num_runs} runs)...")
    precip_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        precip_result = precip_jit(
            last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho_t, dz_t, dt
        )
        jax.block_until_ready(precip_result)
        elapsed = time.perf_counter() - start
        precip_times.append(elapsed * 1000)
        if i < 10:
            print(f"    Run {i+1}: {elapsed*1000:.2f} ms")

    precip_times = np.array(precip_times)
    precip_mean = np.mean(precip_times)
    precip_median = np.median(precip_times)
    precip_min = np.min(precip_times)

    print("\n  Isolated precipitation_effects (NATIVE TRANSPOSED):")
    print(f"    Median: {precip_median:.2f} ms")
    print(f"    Mean:   {precip_mean:.2f} ms")
    print(f"    Min:    {precip_min:.2f} ms")
    print(f"    % of full graupel: {precip_median/median_time*100:.1f}%")

    # =========================================================================
    # Full results summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("Implementation: graupel_native_transposed")
    print(f"Optimized HLO:  {optimized_hlo if optimized_hlo else 'None (pure JAX)'}")
    print("Layout:         TRANSPOSED (nlev×ncells) - ZERO transposes during computation")
    print(f"Grid size:      {ncells} cells × {nlev} levels")
    print("\nFull Graupel Timing (ms):")
    print(f"  Mean:   {mean_time:.2f} ± {std_time:.2f}")
    print(f"  Min:    {min_time:.2f}")
    print(f"  Max:    {max_time:.2f}")
    print(f"  Median: {median_time:.2f}")
    print("\nIsolated precipitation_effects (ms):")
    print(f"  Median: {precip_median:.2f} ({precip_median/median_time*100:.1f}% of full graupel)")

    # Check output shapes (should be nlev, ncells)
    t_out, q_out, pflx, pr, ps, pi, pg, pre = result
    print(f"\nOutput shapes (should be {nlev}×{ncells}):")
    print(f"  t_out: {t_out.shape}")
    print(f"  q.v:   {q_out.v.shape}")
    print(f"  pflx:  {pflx.shape}")

    return mean_time, std_time, result


def run_graupel(
    mode: str,
    optimized_hlo: str = None,
    input_file: str = None,
    num_warmup: int = 3,
    num_runs: int = 10,
    transposed: bool = False,
):
    """Run graupel with or without optimization."""
    from muphys_jax.core.common import constants as const

    print("=" * 80)
    print(f"RUNNING GRAUPEL - MODE: {mode.upper()}")
    if optimized_hlo:
        print(f"WITH OPTIMIZATION: {optimized_hlo}")
        if transposed:
            print("LAYOUT: TRANSPOSED (nlev×ncells)")
    else:
        print("WITHOUT OPTIMIZATION (baseline)")
    print("=" * 80)

    # Configure optimization if requested
    if optimized_hlo:
        from muphys_jax.core.optimized_precip import configure_optimized_precip

        configure_optimized_precip(
            hlo_path=optimized_hlo, use_optimized=True, transposed=transposed
        )
        layout = "transposed" if transposed else "original"
        print(f"✓ Configured optimized HLO: {optimized_hlo} [{layout}]")

    # Load inputs
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_inputs(input_file)

    # Select implementation
    if mode == "baseline":
        from muphys_jax.implementations.graupel_baseline import graupel_run

        impl_name = "graupel_baseline"
    elif mode == "native-transposed":
        impl_name = "graupel_native_transposed"
        # This mode requires special handling - return early with custom benchmark
        return run_graupel_native_transposed(input_file, num_warmup, num_runs, optimized_hlo)
    else:
        raise ValueError(f"Unknown mode: {mode}. Valid: baseline, native-transposed")

    print(f"\nImplementation: {impl_name}")
    print(f"Grid: {ncells} cells × {nlev} levels")

    # Warmup
    print(f"\nWarming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        result = graupel_run(dz, t, p, rho, q, dt, qnc)
        jax.block_until_ready(result)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark full graupel
    print(f"\nBenchmarking full graupel ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = graupel_run(dz, t, p, rho, q, dt, qnc)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        if i < 10:
            print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")

    # Statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)

    # =========================================================================
    # Isolated precipitation_effects timing (for apples-to-apples comparison)
    # =========================================================================
    print("\n" + "-" * 60)
    print("ISOLATED precipitation_effects TIMING (JAX baseline)")
    print("-" * 60)

    from muphys_jax.implementations.graupel_baseline import precipitation_effects, q_t_update

    # Prepare inputs for precipitation_effects (same as in graupel function)
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin
    last_level = nlev - 1

    # Run phase transitions first to get intermediate state
    q_updated, t_updated = q_t_update(t, p, rho, q, dt, qnc)
    jax.block_until_ready((q_updated, t_updated))

    # JIT compile precipitation_effects
    precip_jit = jax.jit(precipitation_effects, static_argnums=(0,))

    # Warmup precipitation_effects
    print("  Warming up isolated precipitation_effects...")
    for _ in range(max(num_warmup, 10)):
        precip_result = precip_jit(
            last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt
        )
        jax.block_until_ready(precip_result)

    # Benchmark isolated precipitation_effects
    print(f"  Benchmarking ({num_runs} runs)...")
    precip_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        precip_result = precip_jit(
            last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt
        )
        jax.block_until_ready(precip_result)
        elapsed = time.perf_counter() - start
        precip_times.append(elapsed * 1000)
        if i < 10:
            print(f"    Run {i+1}: {elapsed*1000:.2f} ms")

    precip_times = np.array(precip_times)
    precip_mean = np.mean(precip_times)
    precip_median = np.median(precip_times)
    precip_min = np.min(precip_times)

    print("\n  Isolated precipitation_effects (JAX baseline):")
    print(f"    Median: {precip_median:.2f} ms")
    print(f"    Mean:   {precip_mean:.2f} ms")
    print(f"    Min:    {precip_min:.2f} ms")
    print(f"    % of full graupel: {precip_median/median_time*100:.1f}%")

    # Also benchmark the OPTIMIZED precipitation_effects if enabled
    if optimized_hlo:
        print("\n" + "-" * 60)
        print("ISOLATED precipitation_effects TIMING (OPTIMIZED)")
        print("-" * 60)

        from muphys_jax.core.optimized_precip import precipitation_effects_optimized

        # JIT compile optimized precipitation_effects
        precip_opt_jit = jax.jit(precipitation_effects_optimized, static_argnums=(0,))

        # Warmup
        print("  Warming up isolated OPTIMIZED precipitation_effects...")
        for _ in range(max(num_warmup, 10)):
            precip_opt_result = precip_opt_jit(
                last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt
            )
            jax.block_until_ready(precip_opt_result)

        # Benchmark
        print(f"  Benchmarking ({num_runs} runs)...")
        precip_opt_times = []
        for i in range(num_runs):
            start = time.perf_counter()
            precip_opt_result = precip_opt_jit(
                last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt
            )
            jax.block_until_ready(precip_opt_result)
            elapsed = time.perf_counter() - start
            precip_opt_times.append(elapsed * 1000)
            if i < 10:
                print(f"    Run {i+1}: {elapsed*1000:.2f} ms")

        precip_opt_times = np.array(precip_opt_times)
        precip_opt_median = np.median(precip_opt_times)

        speedup = precip_median / precip_opt_median
        print("\n  Isolated precipitation_effects (OPTIMIZED):")
        print(f"    Median: {precip_opt_median:.2f} ms")
        print(f"    Speedup vs baseline: {speedup:.2f}x")

    # =========================================================================
    # Full results summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Implementation: {impl_name}")
    print(f"Optimized HLO:  {optimized_hlo if optimized_hlo else 'None'}")
    print(f"Grid size:      {ncells} cells × {nlev} levels")
    print("\nFull Graupel Timing (ms):")
    print(f"  Mean:   {mean_time:.2f} ± {std_time:.2f}")
    print(f"  Min:    {min_time:.2f}")
    print(f"  Max:    {max_time:.2f}")
    print(f"  Median: {median_time:.2f}")
    print("\nIsolated precipitation_effects (ms):")
    print(f"  Median: {precip_median:.2f} ({precip_median/median_time*100:.1f}% of full graupel)")

    # Check output shapes
    t_out, q_out, pflx, pr, ps, pi, pg, pre = result
    print("\nOutput shapes:")
    print(f"  t_out: {t_out.shape}")
    print(f"  q.v:   {q_out.v.shape}")
    print(f"  pflx:  {pflx.shape}")

    return mean_time, std_time, result


def compare_hlo_files(
    input_file: str,
    hlo_files: list,
    num_warmup: int = 3,
    num_runs: int = 10,
    transposed: bool = False,
):
    """Compare multiple HLO files directly."""
    print("=" * 80)
    print("COMPARING HLO FILES (direct execution)")
    if transposed:
        print("LAYOUT: TRANSPOSED (nlev×ncells)")
    print("=" * 80)

    results = {}
    for hlo_file in hlo_files:
        print("\n" + "-" * 60)
        try:
            mean, std, _ = benchmark_hlo_direct(
                hlo_file, input_file, num_warmup, num_runs, transposed=transposed
            )
            results[hlo_file] = {"mean": mean, "std": std}
        except Exception as e:
            print(f"ERROR benchmarking {hlo_file}: {e}")
            import traceback

            traceback.print_exc()
            results[hlo_file] = None

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    baseline_time = None
    for hlo_file, result in results.items():
        name = pathlib.Path(hlo_file).name
        if result is None:
            print(f"{name:45s}: FAILED")
        else:
            mean = result["mean"]
            std = result["std"]

            if baseline_time is None:
                baseline_time = mean
                speedup_str = "(baseline)"
            else:
                speedup = baseline_time / mean
                if speedup > 1:
                    speedup_str = f"({speedup:.2f}x faster)"
                else:
                    speedup_str = f"({1/speedup:.2f}x slower)"

            print(f"{name:45s}: {mean:6.2f} ± {std:5.2f} ms  {speedup_str}")

    return results


def compare_modes(
    input_file: str,
    optimized_hlo: str = None,
    mode: str = "baseline",
    num_warmup: int = 3,
    num_runs: int = 10,
    transposed: bool = False,
):
    """Compare optimized vs unoptimized."""
    print("=" * 80)
    print("COMPARISON: OPTIMIZED vs BASELINE")
    if transposed:
        print("LAYOUT: TRANSPOSED (nlev×ncells)")
    print("=" * 80)

    # Run without optimization
    print("\n[1/2] Running baseline (no optimization)...")
    time_baseline, std_baseline, result_baseline = run_graupel(
        mode=mode,
        optimized_hlo=None,
        input_file=input_file,
        num_warmup=num_warmup,
        num_runs=num_runs,
        transposed=False,
    )

    # Run with optimization
    print("\n[2/2] Running with optimization...")
    time_optimized, std_optimized, result_optimized = run_graupel(
        mode=mode,
        optimized_hlo=optimized_hlo,
        input_file=input_file,
        num_warmup=num_warmup,
        num_runs=num_runs,
        transposed=transposed,
    )

    # Compare
    speedup = time_baseline / time_optimized
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Baseline:   {time_baseline:.2f} ± {std_baseline:.2f} ms")
    print(f"Optimized:  {time_optimized:.2f} ± {std_optimized:.2f} ms")
    print(f"Speedup:    {speedup:.2f}x")
    if speedup > 1:
        print(f"            ({(speedup-1)*100:.1f}% faster)")
    else:
        print(f"            ({(1-speedup)*100:.1f}% slower)")

    # Check correctness (outputs should be close)
    print("\nCorrectness check:")
    t_base, q_base, _, _, _, _, _, _ = result_baseline
    t_opt, q_opt, _, _, _, _, _, _ = result_optimized

    t_diff = jnp.max(jnp.abs(t_base - t_opt))
    qv_diff = jnp.max(jnp.abs(q_base.v - q_opt.v))

    print(f"  Max |t_base - t_opt|:   {t_diff}")
    print(f"  Max |qv_base - qv_opt|: {qv_diff}")

    if t_diff < 1e-10 and qv_diff < 1e-10:
        print("  ✓ Results match (numerical precision)")
    else:
        print("  ⚠ Results differ!")


def main():
    parser = argparse.ArgumentParser(
        description="Run graupel with optimized HLO injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Native-transposed (default, best performance)
    python run_graupel_optimized.py --input data.nc

    # With combined graupel HLO injection
    python run_graupel_optimized.py --input data.nc --mode native-transposed \\
        --optimized-hlo stablehlo/graupel_combined.stablehlo

    # Baseline (original ncells x nlev layout)
    python run_graupel_optimized.py --input data.nc --mode baseline

    # Direct HLO benchmark (most accurate, no JAX overhead)
    python run_graupel_optimized.py --input data.nc --hlo-direct shlo/precip_transposed.stablehlo --transposed

    # Compare multiple HLO files
    python run_graupel_optimized.py --input data.nc --compare-hlo \\
        shlo/baseline.stablehlo shlo/optimized.stablehlo --transposed

Implementation modes:
    baseline          - Original graupel with (ncells, nlev) layout
    native-transposed - Data pre-transposed ONCE, then ZERO transposes during computation
                        Best performance - all computation in (nlev, ncells) layout (~32ms)
""",
    )
    parser.add_argument("--input", "-i", required=True, help="Input netCDF file")
    parser.add_argument(
        "--optimized-hlo", type=str, help="Path to optimized HLO file (.hlo or .serialized)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["baseline", "native-transposed"],
        default="native-transposed",
        help="Implementation mode (baseline, native-transposed)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare optimized vs baseline (full graupel)"
    )
    parser.add_argument(
        "--hlo-direct", type=str, help="Benchmark HLO file directly (no JAX overhead)"
    )
    parser.add_argument("--compare-hlo", nargs="+", help="Compare multiple HLO files directly")
    parser.add_argument(
        "--transposed",
        action="store_true",
        help="Use transposed layout (nlev×ncells) for HLO - 2x faster on GPU. "
        "Use with both --hlo-direct and --optimized-hlo when the HLO "
        "was exported with transposed layout.",
    )
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    if args.compare_hlo:
        # Compare multiple HLO files directly
        compare_hlo_files(
            args.input, args.compare_hlo, args.num_warmup, args.num_runs, transposed=args.transposed
        )
    elif args.hlo_direct:
        # Benchmark single HLO file directly
        benchmark_hlo_direct(
            args.hlo_direct, args.input, args.num_warmup, args.num_runs, transposed=args.transposed
        )
    elif args.compare:
        if not args.optimized_hlo:
            print("ERROR: --compare requires --optimized-hlo")
            sys.exit(1)
        compare_modes(
            args.input,
            args.optimized_hlo,
            args.mode,
            args.num_warmup,
            args.num_runs,
            transposed=args.transposed,
        )
    else:
        run_graupel(
            args.mode,
            args.optimized_hlo,
            args.input,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            transposed=args.transposed,
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
