#!/usr/bin/env python3
"""
Benchmark HLO execution time separating compile from execution.

Usage:
    python benchmark_hlo_execution.py shlo/precip_effect_x64_batched_fused.hlo
    python benchmark_hlo_execution.py shlo/precip_effect_x64_unrolled.hlo --compare shlo/precip_effect_x64_batched_fused.hlo
"""

import argparse
import pathlib
import time
import sys

import numpy as np


def load_hlo_text(hlo_path: str) -> str:
    """Load HLO text from file."""
    with open(hlo_path, 'r') as f:
        return f.read()


def get_client():
    """Get the GPU client."""
    import jax
    try:
        return jax.extend.backend.get_backend("gpu")
    except AttributeError:
        from jax.lib import xla_bridge
        return xla_bridge.get_backend("gpu")


def compile_hlo(hlo_text: str, client):
    """Compile HLO text to executable using jaxlib._jax.hlo_module_from_text."""
    import jaxlib._jax as jax_cpp

    # Parse HLO text to HloModule
    hlo_module = jax_cpp.hlo_module_from_text(hlo_text)

    # Get devices and create DeviceList
    devices = client.local_devices()
    device_list = jax_cpp.DeviceList(tuple(devices))
    compile_options = jax_cpp.CompileOptions()

    # Compile HloModule - need to serialize to bytes first
    hlo_bytes = hlo_module.as_serialized_hlo_module_proto()

    return client.compile(hlo_bytes, device_list, compile_options)


def create_test_inputs(ncells: int = 20480, nlev: int = 90):
    """Create test inputs matching the precipitation_effects signature."""
    np.random.seed(42)  # Reproducible

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


def benchmark_execution(executable, client, inputs, num_warmup=5, num_runs=20):
    """Benchmark execution time (excluding compilation)."""
    import jax

    # Transfer inputs to device
    device = client.local_devices()[0]

    # Use JAX arrays for proper device placement
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]

    # Get the underlying buffers
    try:
        device_inputs = [x.addressable_data(0) for x in jax_inputs]
    except:
        device_inputs = [client.buffer_from_pyval(inp, device=device) for inp in inputs]

    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        results = executable.execute(device_inputs)
        # Block until complete
        for r in results:
            _ = np.asarray(r)

    # Synchronize
    jax.block_until_ready(results)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []

    for i in range(num_runs):
        # Synchronize before timing
        if hasattr(device, 'synchronize'):
            device.synchronize()

        start = time.perf_counter()
        results = executable.execute(device_inputs)

        # Block until complete
        for r in results:
            _ = np.asarray(r)

        # Alternative sync
        jax.block_until_ready(results)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    return np.array(times)


def benchmark_with_cuda_events(executable, client, inputs, num_warmup=5, num_runs=20):
    """Benchmark using CUDA events for more accurate GPU timing."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    import jax

    device = client.local_devices()[0]
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]

    try:
        device_inputs = [x.addressable_data(0) for x in jax_inputs]
    except:
        device_inputs = [client.buffer_from_pyval(inp, device=device) for inp in inputs]

    # Warmup
    for _ in range(num_warmup):
        results = executable.execute(device_inputs)
        for r in results:
            _ = np.asarray(r)

    # Benchmark with CUDA events
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        results = executable.execute(device_inputs)
        for r in results:
            _ = np.asarray(r)
        end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)  # Already in ms
        times.append(elapsed)

    return np.array(times)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HLO execution time (separating compilation)"
    )
    parser.add_argument("hlo_file", help="Input HLO text file")
    parser.add_argument("--compare", nargs="+", help="Additional HLO files to compare")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--num-runs", type=int, default=20, help="Benchmark runs")
    parser.add_argument("--use-cuda-events", action="store_true",
                       help="Use CUDA events for timing (requires torch)")

    args = parser.parse_args()

    # Enable x64
    import jax
    jax.config.update("jax_enable_x64", True)

    print("=" * 70)
    print("HLO Execution Benchmark")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    # Get client
    client = get_client()
    print(f"Using client: {client.platform}")
    print()

    # Create inputs once
    inputs = create_test_inputs()
    print(f"Input shapes: {[x.shape for x in inputs[:4]]} (masks), {[x.shape for x in inputs[4:]]}")
    print()

    # Collect all files
    all_files = [args.hlo_file]
    if args.compare:
        all_files.extend(args.compare)

    results = {}

    for hlo_file in all_files:
        print("=" * 70)
        print(f"File: {hlo_file}")
        print("=" * 70)

        try:
            # Load and compile
            hlo_text = load_hlo_text(hlo_file)
            print(f"  HLO size: {len(hlo_text) / 1024:.1f} KB")

            print("  Compiling...")
            compile_start = time.perf_counter()
            executable = compile_hlo(hlo_text, client)
            compile_time = time.perf_counter() - compile_start
            print(f"  Compilation time: {compile_time:.2f}s")

            # Benchmark execution
            if args.use_cuda_events:
                times = benchmark_with_cuda_events(
                    executable, client, inputs,
                    args.num_warmup, args.num_runs
                )
                if times is None:
                    print("  CUDA events not available, falling back to host timing")
                    times = benchmark_execution(
                        executable, client, inputs,
                        args.num_warmup, args.num_runs
                    )
                else:
                    print("  (Using CUDA events for timing)")
            else:
                times = benchmark_execution(
                    executable, client, inputs,
                    args.num_warmup, args.num_runs
                )

            # Statistics
            results[hlo_file] = {
                'compile_time': compile_time,
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'median': np.median(times),
            }

            print()
            print(f"  Execution time (ms):")
            print(f"    Mean:   {np.mean(times):7.2f} ± {np.std(times):.2f}")
            print(f"    Median: {np.median(times):7.2f}")
            print(f"    Min:    {np.min(times):7.2f}")
            print(f"    Max:    {np.max(times):7.2f}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[hlo_file] = None
            print()

    # Summary comparison
    if len(all_files) > 1:
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print(f"{'File':<45} {'Compile(s)':<12} {'Exec(ms)':<12} {'Speedup':<10}")
        print("-" * 70)

        baseline_exec = None
        for hlo_file in all_files:
            result = results.get(hlo_file)
            name = pathlib.Path(hlo_file).stem[:40]

            if result is None:
                print(f"{name:<45} {'FAILED':<12}")
            else:
                compile_s = result['compile_time']
                exec_ms = result['mean']

                if baseline_exec is None:
                    baseline_exec = exec_ms
                    speedup_str = "(baseline)"
                else:
                    speedup = baseline_exec / exec_ms
                    if speedup > 1:
                        speedup_str = f"{speedup:.2f}x faster"
                    else:
                        speedup_str = f"{1/speedup:.2f}x slower"

                print(f"{name:<45} {compile_s:<12.2f} {exec_ms:<12.2f} {speedup_str:<10}")

        print()


if __name__ == "__main__":
    main()
