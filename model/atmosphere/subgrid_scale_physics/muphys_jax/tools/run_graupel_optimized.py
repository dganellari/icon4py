#!/usr/bin/env python3
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
import sys
import pathlib
import time

import jax
# Enable x64 before any JAX operations
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import netCDF4
import numpy as np


def load_inputs(input_file: str, timestep: int = 0):
    """Load graupel inputs from netCDF."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
    from muphys_jax.core.definitions import Q

    print(f"Loading inputs from: {input_file}")
    ds = netCDF4.Dataset(input_file, 'r')

    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    print(f"  Grid: {ncells} cells × {nlev} levels")

    # Calculate dz
    def _calc_dz(z: np.ndarray) -> np.ndarray:
        ksize = z.shape[0]
        dz = np.zeros(z.shape, np.float64)
        zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
        for k in range(ksize - 1, -1, -1):
            zh_new = 2.0 * z[k, :] - zh
            dz[k, :] = -zh + zh_new
            zh = zh_new
        return dz

    dz_calc = _calc_dz(ds.variables["zg"])
    dz = jnp.array(np.transpose(dz_calc), dtype=jnp.float64)

    def load_var(varname: str) -> jnp.ndarray:
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[timestep, :, :]
        return jnp.array(np.transpose(var), dtype=jnp.float64)

    q = Q(
        v=load_var("hus"),
        c=load_var("clw"),
        r=load_var("qr"),
        s=load_var("qs"),
        i=load_var("cli"),
        g=load_var("qg"),
    )

    t = load_var("ta")
    p = load_var("pfull")
    rho = load_var("rho")

    ds.close()

    dt = 30.0
    qnc = 100.0

    return dz, t, p, rho, q, dt, qnc, ncells, nlev


def load_hlo_module(hlo_path: str):
    """
    Load HLO from file (supports .hlo text and .serialized binary).

    Returns the HLO content (text or bytes).
    """
    path = pathlib.Path(hlo_path)

    if path.suffix == '.serialized':
        with open(path, 'rb') as f:
            return f.read()
    elif path.suffix in ('.hlo', '.stablehlo', '.txt'):
        with open(path, 'r') as f:
            return f.read()
    else:
        # Try as text first
        try:
            with open(path, 'r') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, 'rb') as f:
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


def benchmark_hlo_direct(hlo_path: str, input_file: str,
                         num_warmup: int = 3, num_runs: int = 10):
    """
    Benchmark an HLO/StableHLO file directly using XLA client (most accurate).

    This bypasses JAX's JIT and custom_call mechanism, giving pure
    execution time without any JAX overhead.

    Supports both HLO (.hlo) and StableHLO (.stablehlo) formats.
    """
    from muphys_jax.core.definitions import Q
    from muphys_jax.core.common import constants as const

    print("=" * 80)
    print(f"DIRECT HLO BENCHMARK: {hlo_path}")
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

    # Transfer to device
    device = client.local_devices()[0]
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]
    device_inputs = [x.addressable_data(0) for x in jax_inputs]

    # Warmup
    print(f"\nWarming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        results = executable.execute(device_inputs)
        for r in results:
            _ = np.asarray(r)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark
    print(f"\nBenchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        results = executable.execute(device_inputs)
        for r in results:
            _ = np.asarray(r)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")

    times = np.array(times)
    print("\n" + "=" * 80)
    print("RESULTS (direct HLO execution, no JAX overhead)")
    print("=" * 80)
    print(f"HLO file:   {hlo_path}")
    print(f"Grid size:  {ncells} cells × {nlev} levels")
    print(f"\nExecution time (ms):")
    print(f"  Mean:   {np.mean(times):.2f} ± {np.std(times):.2f}")
    print(f"  Min:    {np.min(times):.2f}")
    print(f"  Max:    {np.max(times):.2f}")
    print(f"  Median: {np.median(times):.2f}")

    return np.mean(times), np.std(times), results


def run_graupel(mode: str, optimized_hlo: str = None, input_file: str = None,
                num_warmup: int = 3, num_runs: int = 10):
    """Run graupel with or without optimization."""

    print("=" * 80)
    print(f"RUNNING GRAUPEL - MODE: {mode.upper()}")
    if optimized_hlo:
        print(f"WITH OPTIMIZATION: {optimized_hlo}")
    else:
        print("WITHOUT OPTIMIZATION (baseline)")
    print("=" * 80)

    # Configure optimization if requested
    if optimized_hlo:
        from muphys_jax.core.optimized_precip import configure_optimized_precip
        configure_optimized_precip(hlo_path=optimized_hlo, use_optimized=True)
        print(f"✓ Configured optimized HLO: {optimized_hlo}")

    # Load inputs
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_inputs(input_file)

    # Select implementation
    if mode == "baseline":
        if optimized_hlo:
            from muphys_jax.implementations.graupel_optimized import graupel_run
            impl_name = "graupel_optimized"
        else:
            from muphys_jax.implementations.graupel_baseline import graupel_run
            impl_name = "graupel_baseline"
    elif mode == "allinone":
        if optimized_hlo:
            from muphys_jax.implementations.graupel_allinone_optimized import graupel_allinone_fused_run as graupel_run
            impl_name = "graupel_allinone_optimized"
        else:
            from muphys_jax.implementations.graupel_allinone_fused import graupel_allinone_fused_run as graupel_run
            impl_name = "graupel_allinone_fused"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\nImplementation: {impl_name}")
    print(f"Grid: {ncells} cells × {nlev} levels")

    # Warmup
    print(f"\nWarming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        result = graupel_run(dz, t, p, rho, q, dt, qnc)
        jax.block_until_ready(result)
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark
    print(f"\nBenchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = graupel_run(dz, t, p, rho, q, dt, qnc)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        print(f"  Run {i+1}/{num_runs}: {elapsed*1000:.2f} ms")

    # Statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Implementation: {impl_name}")
    print(f"Optimized HLO:  {optimized_hlo if optimized_hlo else 'None'}")
    print(f"Grid size:      {ncells} cells × {nlev} levels")
    print(f"\nTiming (ms):")
    print(f"  Mean:   {mean_time:.2f} ± {std_time:.2f}")
    print(f"  Min:    {min_time:.2f}")
    print(f"  Max:    {max_time:.2f}")
    print(f"  Median: {np.median(times):.2f}")

    # Check output shapes
    t_out, q_out, pflx, pr, ps, pi, pg, pre = result
    print(f"\nOutput shapes:")
    print(f"  t_out: {t_out.shape}")
    print(f"  q.v:   {q_out.v.shape}")
    print(f"  pflx:  {pflx.shape}")

    return mean_time, std_time, result


def compare_hlo_files(input_file: str, hlo_files: list, num_warmup: int = 3, num_runs: int = 10):
    """Compare multiple HLO files directly."""
    print("=" * 80)
    print("COMPARING HLO FILES (direct execution)")
    print("=" * 80)

    results = {}
    for hlo_file in hlo_files:
        print("\n" + "-" * 60)
        try:
            mean, std, _ = benchmark_hlo_direct(
                hlo_file, input_file, num_warmup, num_runs
            )
            results[hlo_file] = {'mean': mean, 'std': std}
        except Exception as e:
            print(f"ERROR benchmarking {hlo_file}: {e}")
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

            print(f"{name:45s}: {mean:6.2f} ± {std:5.2f} ms  {speedup_str}")

    return results


def compare_modes(input_file: str, optimized_hlo: str = None, mode: str = "baseline",
                  num_warmup: int = 3, num_runs: int = 10):
    """Compare optimized vs unoptimized."""
    print("=" * 80)
    print("COMPARISON: OPTIMIZED vs BASELINE")
    print("=" * 80)

    # Run without optimization
    print("\n[1/2] Running baseline (no optimization)...")
    time_baseline, std_baseline, result_baseline = run_graupel(
        mode=mode, optimized_hlo=None, input_file=input_file,
        num_warmup=num_warmup, num_runs=num_runs
    )

    # Run with optimization
    print("\n[2/2] Running with optimization...")
    time_optimized, std_optimized, result_optimized = run_graupel(
        mode=mode, optimized_hlo=optimized_hlo, input_file=input_file,
        num_warmup=num_warmup, num_runs=num_runs
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
    # Benchmark baseline
    python run_graupel_optimized.py --input data.nc --mode baseline

    # Direct HLO benchmark (most accurate, no JAX overhead)
    python run_graupel_optimized.py --input data.nc --hlo-direct shlo/precip.hlo

    # Compare multiple HLO files
    python run_graupel_optimized.py --input data.nc --compare-hlo \\
        shlo/lowered.hlo shlo/fused.hlo shlo/batched_fused.hlo

    # Full graupel with HLO injection
    python run_graupel_optimized.py --input data.nc --optimized-hlo shlo/precip.hlo
"""
    )
    parser.add_argument('--input', '-i', required=True, help='Input netCDF file')
    parser.add_argument('--optimized-hlo', type=str,
                       help='Path to optimized HLO file (.hlo or .serialized)')
    parser.add_argument('--mode', '-m', choices=['baseline', 'allinone'],
                       default='baseline', help='Implementation mode')
    parser.add_argument('--compare', action='store_true',
                       help='Compare optimized vs baseline (full graupel)')
    parser.add_argument('--hlo-direct', type=str,
                       help='Benchmark HLO file directly (no JAX overhead)')
    parser.add_argument('--compare-hlo', nargs='+',
                       help='Compare multiple HLO files directly')
    parser.add_argument('--num-warmup', type=int, default=3,
                       help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of benchmark runs')

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    if args.compare_hlo:
        # Compare multiple HLO files directly
        compare_hlo_files(args.input, args.compare_hlo,
                         args.num_warmup, args.num_runs)
    elif args.hlo_direct:
        # Benchmark single HLO file directly
        benchmark_hlo_direct(args.hlo_direct, args.input,
                            args.num_warmup, args.num_runs)
    elif args.compare:
        if not args.optimized_hlo:
            print("ERROR: --compare requires --optimized-hlo")
            sys.exit(1)
        compare_modes(args.input, args.optimized_hlo, args.mode,
                     args.num_warmup, args.num_runs)
    else:
        run_graupel(args.mode, args.optimized_hlo, args.input,
                   num_warmup=args.num_warmup, num_runs=args.num_runs)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
