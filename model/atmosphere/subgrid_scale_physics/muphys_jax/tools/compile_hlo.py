#!/usr/bin/env python3
"""
Compile HLO text files to serialized executables for JAX injection.

This script takes an HLO text file and compiles it to a serialized format
that can be loaded by JAX's custom_call mechanism.

Usage:
    python compile_hlo.py input.hlo -o output.serialized
    python compile_hlo.py input.hlo --benchmark  # Compile and benchmark execution time
    python compile_hlo.py input.hlo --profile    # Run with profiling
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


def compile_hlo_to_executable(hlo_text: str, platform: str = "cuda"):
    """
    Compile HLO text to an XLA executable.

    Returns the compiled executable and the client.
    """
    import jax
    from jax._src.lib import xla_client

    # Get the appropriate client - API changed in newer JAX versions
    if platform.lower() == "cuda":
        # Try different methods based on JAX version
        try:
            # Newer JAX (0.4.20+): use jax.extend.backend
            client = jax.extend.backend.get_backend("gpu")
        except AttributeError:
            try:
                # Try xla_client methods
                client = xla_client.make_gpu_client()
            except AttributeError:
                # Even newer JAX: use get_local_backend
                try:
                    client = xla_client.get_local_backend("gpu")
                except:
                    # Fallback: get default backend
                    client = jax.lib.xla_bridge.get_backend("gpu")
    elif platform.lower() == "cpu":
        try:
            client = jax.extend.backend.get_backend("cpu")
        except AttributeError:
            try:
                client = xla_client.make_cpu_client()
            except AttributeError:
                try:
                    client = xla_client.get_local_backend("cpu")
                except:
                    client = jax.lib.xla_bridge.get_backend("cpu")
    else:
        raise ValueError(f"Unknown platform: {platform}")

    # Get devices for this client
    devices = client.local_devices()
    compile_options = xla_client.CompileOptions()

    # Try different compilation approaches based on JAX version
    executable = None

    # Try Method 1: New API with string HLO and DeviceList (JAX 0.6+)
    try:
        device_list = jax.sharding.DeviceList(devices)
        executable = client.compile(hlo_text, device_list, compile_options)
    except (TypeError, AttributeError):
        pass

    # Try Method 2: New API with bytes
    if executable is None:
        try:
            device_list = jax.sharding.DeviceList(devices)
            executable = client.compile(hlo_text.encode('utf-8'), device_list, compile_options)
        except (TypeError, AttributeError):
            pass

    # Try Method 3: Old API with XlaComputation
    if executable is None:
        try:
            computation = xla_client.XlaComputation(hlo_text)
            executable = client.compile(computation, compile_options)
        except (TypeError, AttributeError):
            pass

    # Try Method 4: XlaComputation with bytes
    if executable is None:
        try:
            computation = xla_client.XlaComputation(hlo_text.encode('utf-8'))
            executable = client.compile(computation, compile_options)
        except (TypeError, AttributeError):
            pass

    # Try Method 5: hlo_module_from_text
    if executable is None:
        try:
            hlo_module = xla_client.hlo_module_from_text(hlo_text)
            executable = client.compile(hlo_module, compile_options)
        except (TypeError, AttributeError) as e:
            raise RuntimeError(f"Could not compile HLO with any known API: {e}")

    return executable, client


def serialize_executable(executable, output_path: str):
    """Serialize a compiled executable to disk."""
    serialized = executable.serialize()
    with open(output_path, 'wb') as f:
        f.write(serialized)
    print(f"Serialized executable to: {output_path}")
    print(f"Size: {len(serialized) / 1024:.1f} KB")
    return serialized


def create_dummy_inputs(ncells: int = 20480, nlev: int = 90):
    """Create dummy inputs matching the precipitation_effects signature."""
    import jax.numpy as jnp

    # Input signature:
    # kmin_r, kmin_i, kmin_s, kmin_g: pred[20480,90]
    # qv, qc, qr, qs, qi, qg: f64[20480,90]
    # t, rho, dz: f64[20480,90]

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

    t = np.random.rand(ncells, nlev).astype(np.float64) * 50 + 230  # 230-280 K
    rho = np.random.rand(ncells, nlev).astype(np.float64) * 0.5 + 0.5  # 0.5-1.0
    dz = np.random.rand(ncells, nlev).astype(np.float64) * 100 + 50  # 50-150 m

    return [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz]


def benchmark_executable(executable, client, num_warmup: int = 3, num_runs: int = 10):
    """Benchmark an executable's execution time (excluding compilation)."""
    from jax._src.lib import xla_client

    # Create dummy inputs
    inputs = create_dummy_inputs()

    # Convert to device buffers
    device = client.local_devices()[0]
    device_inputs = [
        client.buffer_from_pyval(inp, device=device)
        for inp in inputs
    ]

    # Warmup runs
    print(f"\nWarming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        results = executable.execute(device_inputs)
        # Block until complete
        for r in results:
            _ = np.asarray(r)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark runs
    print(f"\nBenchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        results = executable.execute(device_inputs)
        # Block until complete
        for r in results:
            _ = np.asarray(r)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
        print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")

    # Statistics
    times = np.array(times)
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (execution only, no compilation)")
    print("=" * 60)
    print(f"Mean:   {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    print(f"Min:    {np.min(times):.2f} ms")
    print(f"Max:    {np.max(times):.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")

    return times


def benchmark_hlo_file(hlo_path: str, platform: str = "cuda",
                       num_warmup: int = 3, num_runs: int = 10):
    """Compile and benchmark an HLO file."""
    print(f"Loading HLO from: {hlo_path}")
    hlo_text = load_hlo_text(hlo_path)

    print(f"\nCompiling for platform: {platform}")
    start = time.perf_counter()
    executable, client = compile_hlo_to_executable(hlo_text, platform)
    compile_time = time.perf_counter() - start
    print(f"Compilation time: {compile_time:.2f} s")

    times = benchmark_executable(executable, client, num_warmup, num_runs)

    return executable, client, times


def main():
    parser = argparse.ArgumentParser(
        description="Compile HLO text to serialized executable"
    )
    parser.add_argument("hlo_file", help="Input HLO text file")
    parser.add_argument("-o", "--output", help="Output serialized file")
    parser.add_argument("--platform", default="cuda",
                       choices=["cuda", "cpu"],
                       help="Target platform")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark execution time after compilation")
    parser.add_argument("--num-warmup", type=int, default=3,
                       help="Number of warmup runs")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of benchmark runs")
    parser.add_argument("--compare", nargs="+",
                       help="Compare multiple HLO files")

    args = parser.parse_args()

    # Enable x64
    import jax
    jax.config.update("jax_enable_x64", True)

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    if args.compare:
        # Compare multiple HLO files
        all_files = [args.hlo_file] + args.compare
        results = {}

        for hlo_file in all_files:
            print("\n" + "=" * 70)
            print(f"BENCHMARKING: {hlo_file}")
            print("=" * 70)

            try:
                _, _, times = benchmark_hlo_file(
                    hlo_file, args.platform,
                    args.num_warmup, args.num_runs
                )
                results[hlo_file] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                }
            except Exception as e:
                print(f"ERROR: {e}")
                results[hlo_file] = None

        # Summary comparison
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        baseline_time = None
        for hlo_file, result in results.items():
            if result is None:
                print(f"{pathlib.Path(hlo_file).name}: FAILED")
            else:
                name = pathlib.Path(hlo_file).name
                mean = result['mean']
                std = result['std']

                if baseline_time is None:
                    baseline_time = mean
                    speedup_str = "(baseline)"
                else:
                    speedup = baseline_time / mean
                    if speedup > 1:
                        speedup_str = f"({speedup:.2f}x faster)"
                    else:
                        speedup_str = f"({1/speedup:.2f}x slower)"

                print(f"{name:40s}: {mean:6.2f} ± {std:5.2f} ms  {speedup_str}")

    elif args.benchmark:
        executable, client, _ = benchmark_hlo_file(
            args.hlo_file, args.platform,
            args.num_warmup, args.num_runs
        )

        # Optionally save
        if args.output:
            serialize_executable(executable, args.output)

    else:
        # Just compile and optionally serialize
        print(f"Loading HLO from: {args.hlo_file}")
        hlo_text = load_hlo_text(args.hlo_file)

        print(f"\nCompiling for platform: {args.platform}")
        start = time.perf_counter()
        executable, client = compile_hlo_to_executable(hlo_text, args.platform)
        compile_time = time.perf_counter() - start
        print(f"Compilation time: {compile_time:.2f} s")

        if args.output:
            serialize_executable(executable, args.output)
        else:
            # Default output name
            output_path = pathlib.Path(args.hlo_file).with_suffix('.serialized')
            serialize_executable(executable, str(output_path))


if __name__ == "__main__":
    main()
