#!/usr/bin/env python3
"""
Benchmark full graupel with StableHLO-injected precipitation scan.

Compares:
1. Baseline (lax.scan)
2. Unrolled (Python loop)
3. StableHLO (pre-compiled unrolled IR)

Usage:
    python benchmark_graupel_stablehlo.py --num-runs 20
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.core.definitions import Q


def create_test_inputs(ncells=20480, nlev=90, dtype=jnp.float64):
    """Create realistic test inputs."""
    np.random.seed(42)

    dz = jnp.array(np.random.rand(ncells, nlev) * 100 + 50, dtype=dtype)
    te = jnp.array(np.random.rand(ncells, nlev) * 50 + 230, dtype=dtype)
    p = jnp.ones((ncells, nlev), dtype=dtype) * 101325.0
    rho = jnp.array(np.random.rand(ncells, nlev) * 0.5 + 0.5, dtype=dtype)

    q_in = Q(
        v=jnp.array(np.random.rand(ncells, nlev) * 0.01, dtype=dtype),
        c=jnp.array(np.random.rand(ncells, nlev) * 1e-5, dtype=dtype),
        r=jnp.array(np.random.rand(ncells, nlev) * 1e-6, dtype=dtype),
        s=jnp.array(np.random.rand(ncells, nlev) * 1e-6, dtype=dtype),
        i=jnp.array(np.random.rand(ncells, nlev) * 1e-7, dtype=dtype),
        g=jnp.array(np.random.rand(ncells, nlev) * 1e-7, dtype=dtype),
    )

    return dz, te, p, rho, q_in, 30.0, 100.0


def benchmark_mode(inputs, mode_name, num_warmup, num_runs, **kwargs):
    """Benchmark a specific mode."""
    print(f"\n{'='*70}")
    print(f"Mode: {mode_name}")
    print(f"{'='*70}")

    # JIT compile with specific flags
    graupel_jit = jax.jit(
        lambda *args: graupel_run(*args, **kwargs),
        static_argnames=['use_fused_scans', 'use_tiled_scans', 'tile_size',
                        'optimize_layout', 'use_unrolled', 'use_pallas',
                        'use_triton', 'use_mlir']
    )

    # Compile
    print("  Compiling...")
    compile_start = time.perf_counter()
    try:
        result = graupel_jit(*inputs)
        result[0].block_until_ready()
        compile_time = time.perf_counter() - compile_start
        print(f"  Compilation time: {compile_time:.2f}s")
    except Exception as e:
        print(f"  ERROR during compilation: {e}")
        return None

    # Warmup
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = graupel_jit(*inputs)
        result[0].block_until_ready()

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = graupel_jit(*inputs)
        result[0].block_until_ready()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"    Run {i+1}: {elapsed:.2f} ms")

    times = np.array(times)
    print(f"\n  Execution: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")

    return {
        'compile_time': compile_time,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'times': times,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark full graupel with StableHLO")
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--num-runs", type=int, default=20, help="Benchmark runs")
    parser.add_argument("--ncells", type=int, default=20480, help="Number of cells")
    parser.add_argument("--nlev", type=int, default=90, help="Number of levels")
    parser.add_argument("--modes", nargs="+", default=["baseline", "unrolled"],
                       choices=["baseline", "unrolled", "fused", "tiled", "stablehlo"],
                       help="Modes to benchmark")
    args = parser.parse_args()

    print("=" * 70)
    print("Full Graupel Benchmark")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Grid: {args.ncells} cells × {args.nlev} levels")
    print()

    # Create inputs
    inputs = create_test_inputs(args.ncells, args.nlev)

    results = {}

    # Baseline (lax.scan)
    if "baseline" in args.modes:
        results["baseline"] = benchmark_mode(
            inputs, "Baseline (lax.scan)",
            args.num_warmup, args.num_runs,
            use_fused_scans=False, use_unrolled=False
        )

    # Unrolled (Python loop)
    if "unrolled" in args.modes:
        results["unrolled"] = benchmark_mode(
            inputs, "Unrolled (Python loop)",
            args.num_warmup, args.num_runs,
            use_fused_scans=False, use_unrolled=True
        )

    # Fused scan
    if "fused" in args.modes:
        results["fused"] = benchmark_mode(
            inputs, "Fused (single scan)",
            args.num_warmup, args.num_runs,
            use_fused_scans=True, use_unrolled=False
        )

    # Tiled scan
    if "tiled" in args.modes:
        results["tiled"] = benchmark_mode(
            inputs, "Tiled (tile_size=4)",
            args.num_warmup, args.num_runs,
            use_fused_scans=False, use_tiled_scans=True, tile_size=4
        )

    # StableHLO injection (if available)
    if "stablehlo" in args.modes:
        try:
            from muphys_jax.core.scans_stablehlo import STABLEHLO_AVAILABLE, STABLEHLO_PATH
            if STABLEHLO_AVAILABLE:
                print(f"\nStableHLO path: {STABLEHLO_PATH}")
                # Note: Need to modify graupel.py to support stablehlo flag
                # For now, benchmark the StableHLO directly
                from muphys_jax.core.scans_stablehlo import precip_scan_stablehlo_direct
                print("  StableHLO module available - benchmark precip_scan directly")
            else:
                print(f"\nStableHLO not available: {STABLEHLO_PATH}")
        except ImportError as e:
            print(f"\nCould not import StableHLO module: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Mode':<25} {'Compile(s)':<12} {'Exec(ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline_exec = None
    for mode, result in results.items():
        if result is None:
            print(f"{mode:<25} {'FAILED':<12}")
            continue

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

        print(f"{mode:<25} {compile_s:<12.2f} {exec_ms:<12.2f} {speedup_str:<10}")

    print()


if __name__ == "__main__":
    main()
