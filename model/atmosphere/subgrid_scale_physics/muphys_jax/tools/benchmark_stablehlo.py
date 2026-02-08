#!/usr/bin/env python3
"""
Benchmark StableHLO execution time (compilation vs execution separated).

Uses JAX's deserialize_and_execute to run compiled StableHLO.

Usage:
    # With NetCDF input (original ncells×nlev layout)
    python benchmark_stablehlo.py shlo/precip_effect.stablehlo --input data.nc --num-runs 10

    # With synthetic data (specify dimensions directly)
    python benchmark_stablehlo.py shlo/precip_effect.stablehlo --ncells 327680 --nlev 90 --num-runs 10

    # For transposed layout (nlev×ncells)
    python benchmark_stablehlo.py shlo/transposed.stablehlo --ncells 327680 --nlev 90 --transposed

    # Compare multiple files
    python benchmark_stablehlo.py shlo/baseline.stablehlo --compare shlo/optimized.stablehlo --ncells 327680 --nlev 90
"""

import argparse
import pathlib
import time
import sys

import numpy as np
import netCDF4
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import jaxlib._jax as jax_cpp

jax.config.update("jax_enable_x64", True)


def load_stablehlo(path: str) -> str:
    """Load StableHLO text from file."""
    with open(path, 'r') as f:
        return f.read()


def compile_stablehlo(stablehlo_text: str, client):
    """Compile StableHLO and return a LoadedExecutable with execute method."""
    # Use only the first device to avoid device mismatch
    devices = client.local_devices()[:1]
    device_list = jax_cpp.DeviceList(tuple(devices))
    compile_options = jax_cpp.CompileOptions()

    # Compile to get Executable (no execute method)
    executable = client.compile(stablehlo_text, device_list, compile_options)

    # Serialize and deserialize to get LoadedExecutable (has execute method)
    serialized = executable.serialize()
    loaded = client.deserialize_executable(serialized, device_list, compile_options)

    return loaded


def detect_input_count(stablehlo_text: str) -> int:
    """Detect number of inputs from StableHLO @main signature."""
    import re
    match = re.search(r'func\.func\s+public\s+@main\s*\(([^)]*)\)', stablehlo_text, re.DOTALL)
    if not match:
        return 13  # default to precip-only
    args_str = match.group(1)
    return len(re.findall(r'%arg\d+', args_str))


def load_inputs_from_netcdf(input_file: str, num_inputs: int = 13):
    """Load real inputs from NetCDF file.

    Args:
        input_file: Path to NetCDF file
        num_inputs: Number of inputs expected by the StableHLO module.
            13 = precip-only: [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz]
            14 = combined graupel: [kmin_r, kmin_i, kmin_s, kmin_g, t, p, rho, dz, qv, qc, qr, qs, qi, qg]
    """
    print(f"Loading inputs from: {input_file}")
    ds = netCDF4.Dataset(input_file, 'r')

    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    print(f"  Grid: {ncells} cells × {nlev} levels")

    # Calculate dz from z
    def _calc_dz(z: np.ndarray) -> np.ndarray:
        ksize = z.shape[0]
        dz = np.zeros(z.shape, np.float64)
        zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
        for k in range(ksize - 1, -1, -1):
            zh_new = 2.0 * z[k, :] - zh
            dz[k, :] = -zh + zh_new
            zh = zh_new
        return dz

    dz = np.transpose(_calc_dz(ds.variables["zg"][:]))

    def load_var(varname: str) -> np.ndarray:
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[0, :, :]
        return np.transpose(np.array(var, dtype=np.float64))

    # Load variables
    qv = load_var("hus")
    qc = load_var("clw")
    qr = load_var("qr")
    qs = load_var("qs")
    qi = load_var("cli")
    qg = load_var("qg")
    t = load_var("ta")
    p = load_var("pfull")
    rho = load_var("rho")

    # Compute kmin masks (same logic as graupel)
    qmin = 1e-8
    kmin_r = qr > qmin
    kmin_i = qi > qmin
    kmin_s = qs > qmin
    kmin_g = qg > qmin

    ds.close()

    if num_inputs == 14:
        # Combined graupel: kmin_r, kmin_i, kmin_s, kmin_g, t, p, rho, dz, qv, qc, qr, qs, qi, qg
        print(f"  Input layout: combined graupel (14 inputs)")
        return [kmin_r, kmin_i, kmin_s, kmin_g, t, p, rho, dz, qv, qc, qr, qs, qi, qg], ncells, nlev
    else:
        # Precip-only: kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz
        print(f"  Input layout: precip-only (13 inputs)")
        return [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz], ncells, nlev


def generate_synthetic_inputs(ncells: int, nlev: int, transposed: bool = False, num_inputs: int = 13):
    """Generate synthetic inputs for benchmarking without NetCDF file.

    Args:
        ncells: Number of horizontal cells
        nlev: Number of vertical levels
        transposed: If True, generate tensors as nlev×ncells (transposed layout)
                   If False, generate tensors as ncells×nlev (original layout)
        num_inputs: 13 for precip-only, 14 for combined graupel
    """
    print(f"Generating synthetic inputs: {ncells} cells × {nlev} levels")
    if transposed:
        print(f"  Layout: tensor<{nlev}×{ncells}> (transposed - nlev×ncells)")
        shape = (nlev, ncells)
    else:
        print(f"  Layout: tensor<{ncells}×{nlev}> (original - ncells×nlev)")
        shape = (ncells, nlev)

    # Boolean masks - activation at ~level 10
    kmin_r = np.zeros(shape, dtype=bool)
    kmin_i = np.zeros(shape, dtype=bool)
    kmin_s = np.zeros(shape, dtype=bool)
    kmin_g = np.zeros(shape, dtype=bool)
    if transposed:
        kmin_r[10, :] = True
        kmin_i[12, :] = True
        kmin_s[8, :] = True
        kmin_g[15, :] = True
    else:
        kmin_r[:, 10] = True
        kmin_i[:, 12] = True
        kmin_s[:, 8] = True
        kmin_g[:, 15] = True

    # Physical quantities
    qv = np.ones(shape, dtype=np.float64) * 1e-3
    qc = np.ones(shape, dtype=np.float64) * 1e-5
    qr = np.ones(shape, dtype=np.float64) * 1e-6
    qs = np.ones(shape, dtype=np.float64) * 1e-6
    qi = np.ones(shape, dtype=np.float64) * 1e-6
    qg = np.ones(shape, dtype=np.float64) * 1e-6
    t = np.ones(shape, dtype=np.float64) * 273.0
    p = np.ones(shape, dtype=np.float64) * 80000.0
    rho = np.ones(shape, dtype=np.float64) * 1.0
    dz = np.ones(shape, dtype=np.float64) * 100.0

    if num_inputs == 14:
        print(f"  Input layout: combined graupel (14 inputs)")
        return [kmin_r, kmin_i, kmin_s, kmin_g, t, p, rho, dz, qv, qc, qr, qs, qi, qg], ncells, nlev
    else:
        print(f"  Input layout: precip-only (13 inputs)")
        return [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz], ncells, nlev


def benchmark_execution(executable, client, inputs, num_warmup=3, num_runs=10):
    """Benchmark execution time (excluding compilation)."""
    device = client.local_devices()[0]
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]

    try:
        device_inputs = [x.addressable_data(0) for x in jax_inputs]
    except:
        device_inputs = jax_inputs

    # Warmup - do more warmup runs to stabilize GPU clocks
    actual_warmup = max(num_warmup, 10)  # At least 10 warmup runs
    print(f"  Warmup ({actual_warmup} runs to stabilize GPU clocks)...")
    for _ in range(actual_warmup):
        results = executable.execute(device_inputs)
        jax.block_until_ready(results)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    total_start = time.perf_counter()
    for i in range(num_runs):
        start = time.perf_counter()
        results = executable.execute(device_inputs)
        jax.block_until_ready(results)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        # Print first 10 runs as they happen
        if i < 10:
            print(f"    Run {i+1}: {elapsed:.2f} ms")
    total_elapsed = time.perf_counter() - total_start

    times_arr = np.array(times)
    median_time = np.median(times_arr)
    outliers = times_arr > (median_time * 1.5)
    num_outliers = np.sum(outliers)

    print(f"  Min: {np.min(times_arr):.2f} ms, Max: {np.max(times_arr):.2f} ms, Median: {median_time:.2f} ms, Mean: {np.mean(times_arr):.2f} ms")
    print(f"  Total time for {num_runs} runs: {total_elapsed:.3f} s")
    if num_outliers > 0:
        print(f"  ⚠ {num_outliers}/{num_runs} runs were outliers (>1.5x median) - GPU throttling detected")

    return times_arr


def main():
    parser = argparse.ArgumentParser(description="Benchmark StableHLO execution")
    parser.add_argument("stablehlo_file", help="Input StableHLO file")
    parser.add_argument("--input", "-i", help="NetCDF input file (optional if --ncells/--nlev provided)")
    parser.add_argument("--ncells", type=int, help="Number of cells (for synthetic data)")
    parser.add_argument("--nlev", type=int, help="Number of levels (for synthetic data)")
    parser.add_argument("--transposed", action="store_true",
                       help="Use transposed layout (nlev×ncells) for synthetic data")
    parser.add_argument("--compare", nargs="+", help="Additional files to compare")
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--num-runs", type=int, default=10, help="Benchmark runs")
    args = parser.parse_args()

    # Validate arguments
    if args.input is None and (args.ncells is None or args.nlev is None):
        parser.error("Either --input or both --ncells and --nlev are required")

    print("=" * 70)
    print("StableHLO Execution Benchmark")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    client = xla_bridge.get_backend("gpu")

    all_files = [args.stablehlo_file]
    if args.compare:
        all_files.extend(args.compare)

    results = {}

    for filepath in all_files:
        print("=" * 70)
        print(f"File: {filepath}")
        print("=" * 70)

        try:
            shlo_text = load_stablehlo(filepath)
            print(f"  Size: {len(shlo_text) / 1024:.1f} KB")

            # Detect input count from the StableHLO signature
            num_inputs = detect_input_count(shlo_text)
            print(f"  Detected {num_inputs} inputs")

            # Load inputs matching this file's signature
            if args.input:
                inputs, ncells, nlev = load_inputs_from_netcdf(args.input, num_inputs)
                if args.transposed:
                    print("  Transposing inputs to nlev×ncells layout...")
                    inputs = [np.transpose(inp) for inp in inputs]
            else:
                inputs, ncells, nlev = generate_synthetic_inputs(
                    args.ncells, args.nlev, args.transposed, num_inputs)

            print("  Compiling...")
            compile_start = time.perf_counter()
            executable = compile_stablehlo(shlo_text, client)
            compile_time = time.perf_counter() - compile_start
            print(f"  Compilation time: {compile_time:.2f}s")

            times = benchmark_execution(executable, client, inputs,
                                        args.num_warmup, args.num_runs)

            results[filepath] = {
                'compile_time': compile_time,
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
            }
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[filepath] = None
            print()

    # Summary
    if len(all_files) > 1:
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        baseline_exec = None
        for filepath in all_files:
            result = results.get(filepath)
            name = pathlib.Path(filepath).stem[:40]
            if result is None:
                print(f"{name:<45} FAILED")
            else:
                exec_ms = result['mean']
                if baseline_exec is None:
                    baseline_exec = exec_ms
                    speedup = "(baseline)"
                else:
                    ratio = baseline_exec / exec_ms
                    speedup = f"{ratio:.2f}x faster" if ratio > 1 else f"{1/ratio:.2f}x slower"
                print(f"{name:<45} {exec_ms:7.2f} ms  {speedup}")


if __name__ == "__main__":
    main()
