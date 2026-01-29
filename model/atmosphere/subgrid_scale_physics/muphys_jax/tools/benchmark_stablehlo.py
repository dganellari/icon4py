#!/usr/bin/env python3
"""
Benchmark StableHLO execution time (compilation vs execution separated).

Uses JAX's deserialize_and_execute to run compiled StableHLO.

Usage:
    python benchmark_stablehlo.py shlo/precip_effect_x64_optimered.stablehlo --num-runs 10
"""

import argparse
import pathlib
import time
import sys

import numpy as np
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


def create_test_inputs(ncells: int = 20480, nlev: int = 90):
    """Create test inputs matching the precipitation_effects signature."""
    np.random.seed(42)

    kmin_r = np.random.rand(ncells, nlev) > 0.5
    kmin_i = np.random.rand(ncells, nlev) > 0.5
    kmin_s = np.random.rand(ncells, nlev) > 0.5
    kmin_g = np.random.rand(ncells, nlev) > 0.5

    qv = np.random.rand(ncells, nlev).astype(np.float64) * 0.01
    qc = np.random.rand(ncells, nlev).astype(np.float64) * 1e-5
    qr = np.random.rand(ncells, nlev).astype(np.float64) * 1e-6
    qs = np.random.rand(ncells, nlev).astype(np.float64) * 1e-6
    qi = np.random.rand(ncells, nlev).astype(np.float64) * 1e-7
    qg = np.random.rand(ncells, nlev).astype(np.float64) * 1e-7

    t = np.random.rand(ncells, nlev).astype(np.float64) * 50 + 230
    rho = np.random.rand(ncells, nlev).astype(np.float64) * 0.5 + 0.5
    dz = np.random.rand(ncells, nlev).astype(np.float64) * 100 + 50

    return [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz]


def benchmark_execution(executable, client, inputs, num_warmup=3, num_runs=10):
    """Benchmark execution time (excluding compilation)."""
    device = client.local_devices()[0]
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]

    try:
        device_inputs = [x.addressable_data(0) for x in jax_inputs]
    except:
        device_inputs = jax_inputs

    # Warmup
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        results = executable.execute(device_inputs)
        jax.block_until_ready(results)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        results = executable.execute(device_inputs)
        jax.block_until_ready(results)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.2f} ms")

    return np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark StableHLO execution")
    parser.add_argument("stablehlo_file", help="Input StableHLO file")
    parser.add_argument("--compare", nargs="+", help="Additional files to compare")
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--num-runs", type=int, default=10, help="Benchmark runs")
    args = parser.parse_args()

    print("=" * 70)
    print("StableHLO Execution Benchmark")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    client = xla_bridge.get_backend("gpu")
    inputs = create_test_inputs()

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

            print(f"\n  Execution: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
            print(f"  Min: {np.min(times):.2f} ms")
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
