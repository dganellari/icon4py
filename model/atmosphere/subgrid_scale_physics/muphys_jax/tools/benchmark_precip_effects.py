#!/usr/bin/env python3
"""
Benchmark precipitation_effects execution time (compilation vs execution separated).

Compares baseline JAX implementation vs unrolled version.

Usage:
    python benchmark_precip_effects.py --num-runs 10
    python benchmark_precip_effects.py --mode unrolled --num-runs 10
"""

import argparse
import time
import sys
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# Add parent path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))


def create_test_inputs(ncells: int = 20480, nlev: int = 90):
    """Create test inputs matching the precipitation_effects signature."""
    np.random.seed(42)

    kmin_r = jnp.array(np.random.rand(ncells, nlev) > 0.5)
    kmin_i = jnp.array(np.random.rand(ncells, nlev) > 0.5)
    kmin_s = jnp.array(np.random.rand(ncells, nlev) > 0.5)
    kmin_g = jnp.array(np.random.rand(ncells, nlev) > 0.5)

    qv = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 0.01)
    qc = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 1e-5)
    qr = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 1e-6)
    qs = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 1e-6)
    qi = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 1e-7)
    qg = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 1e-7)

    t = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 50 + 230)
    rho = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 0.5 + 0.5)
    dz = jnp.array(np.random.rand(ncells, nlev).astype(np.float64) * 100 + 50)

    return kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz


def benchmark_fn(fn, inputs, num_warmup=3, num_runs=10):
    """Benchmark function execution time."""
    # Warmup
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        result = fn(*inputs)
        jax.block_until_ready(result)

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        jax.block_until_ready(inputs)  # Ensure inputs ready
        start = time.perf_counter()
        result = fn(*inputs)
        jax.block_until_ready(result)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.2f} ms")

    return np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark precipitation_effects")
    parser.add_argument("--mode", choices=["baseline", "unrolled", "both"],
                       default="both", help="Which version to benchmark")
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--num-runs", type=int, default=10, help="Benchmark runs")
    parser.add_argument("--nlev", type=int, default=90, help="Number of levels")
    parser.add_argument("--ncells", type=int, default=20480, help="Number of cells")
    args = parser.parse_args()

    print("=" * 70)
    print("Precipitation Effects Benchmark")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Grid: {args.ncells} cells × {args.nlev} levels")
    print()

    # Create inputs
    inputs = create_test_inputs(args.ncells, args.nlev)

    results = {}

    # Import the real function
    try:
        from muphys_jax.core.graupel.precipitation_effects import precip_effect_fn
        has_baseline = True
    except ImportError as e:
        print(f"Could not import baseline: {e}")
        has_baseline = False

    # Baseline version
    if has_baseline and args.mode in ["baseline", "both"]:
        print("=" * 70)
        print("BASELINE (lax.scan)")
        print("=" * 70)

        jitted_fn = jax.jit(precip_effect_fn)

        # Compile
        print("  Compiling...")
        compile_start = time.perf_counter()
        _ = jitted_fn(*inputs)  # Trigger compilation
        jax.block_until_ready(_)
        compile_time = time.perf_counter() - compile_start
        print(f"  Compilation time: {compile_time:.2f}s")

        times = benchmark_fn(jitted_fn, inputs, args.num_warmup, args.num_runs)
        results["baseline"] = {
            "compile_time": compile_time,
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
        }
        print(f"\n  Execution: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
        print(f"  Min: {np.min(times):.2f} ms")
        print()

    # Unrolled version
    if args.mode in ["unrolled", "both"]:
        print("=" * 70)
        print("UNROLLED (static indexing)")
        print("=" * 70)

        # Import or define unrolled version
        try:
            from muphys_jax.core.graupel.precipitation_effects_unrolled import precip_effect_unrolled
            jitted_unrolled = jax.jit(precip_effect_unrolled)
        except ImportError:
            print("  Unrolled version not found, creating inline...")
            # Create unrolled version inline (simplified for benchmark)
            jitted_unrolled = create_unrolled_version(args.nlev)

        if jitted_unrolled is not None:
            # Compile
            print("  Compiling...")
            compile_start = time.perf_counter()
            _ = jitted_unrolled(*inputs)
            jax.block_until_ready(_)
            compile_time = time.perf_counter() - compile_start
            print(f"  Compilation time: {compile_time:.2f}s")

            times = benchmark_fn(jitted_unrolled, inputs, args.num_warmup, args.num_runs)
            results["unrolled"] = {
                "compile_time": compile_time,
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
            }
            print(f"\n  Execution: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
            print(f"  Min: {np.min(times):.2f} ms")
            print()

    # Summary
    if len(results) > 1:
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        baseline_exec = results.get("baseline", {}).get("mean")
        for name, result in results.items():
            exec_ms = result["mean"]
            compile_s = result["compile_time"]
            if name == "baseline":
                speedup = "(baseline)"
            elif baseline_exec:
                ratio = baseline_exec / exec_ms
                speedup = f"{ratio:.2f}x faster" if ratio > 1 else f"{1/ratio:.2f}x slower"
            else:
                speedup = ""
            print(f"{name:<15} compile: {compile_s:5.2f}s  exec: {exec_ms:7.2f} ms  {speedup}")


def create_unrolled_version(nlev):
    """Create an unrolled version of precipitation_effects."""
    try:
        from muphys_jax.core.graupel.precipitation_effects import precip_effect_body_batched
    except ImportError:
        print("  Could not import precip_effect_body_batched")
        return None

    def unrolled_fn(kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz):
        """Fully unrolled precipitation effects."""
        ncells = qv.shape[0]

        # Initialize outputs
        qv_out = jnp.zeros_like(qv)
        qc_out = jnp.zeros_like(qc)
        qr_out = jnp.zeros_like(qr)
        qs_out = jnp.zeros_like(qs)
        qi_out = jnp.zeros_like(qi)
        qg_out = jnp.zeros_like(qg)
        t_out = jnp.zeros_like(t)
        prr = jnp.zeros_like(qv)
        prs = jnp.zeros_like(qv)
        prg = jnp.zeros_like(qv)
        pri = jnp.zeros_like(qv)

        # Initialize fluxes
        pflx_r = jnp.zeros(ncells)
        pflx_s = jnp.zeros(ncells)
        pflx_i = jnp.zeros(ncells)
        pflx_g = jnp.zeros(ncells)

        # Precompute dz * 2
        dz2 = dz * 2.0

        # Unroll all levels
        for k in range(nlev):
            # Static slice inputs for level k
            km_r = kmin_r[:, k]
            km_i = kmin_i[:, k]
            km_s = kmin_s[:, k]
            km_g = kmin_g[:, k]
            qv_k = qv[:, k]
            qc_k = qc[:, k]
            qr_k = qr[:, k]
            qs_k = qs[:, k]
            qi_k = qi[:, k]
            qg_k = qg[:, k]
            t_k = t[:, k]
            rho_k = rho[:, k]
            dz_k = dz[:, k]
            dz2_k = dz2[:, k]

            # Process level
            result = precip_effect_body_batched(
                (km_r, km_i, km_s, km_g, pflx_r, pflx_s, pflx_i, pflx_g),
                (qv_k, qc_k, qr_k, qs_k, qi_k, qg_k, t_k, rho_k, dz_k, dz2_k)
            )
            carry, outputs = result

            # Update fluxes
            _, _, _, _, pflx_r, pflx_s, pflx_i, pflx_g = carry

            # Store outputs
            qv_new, qc_new, qr_new, qs_new, qi_new, qg_new, t_new, prr_k, prs_k, prg_k, pri_k = outputs
            qv_out = qv_out.at[:, k].set(qv_new)
            qc_out = qc_out.at[:, k].set(qc_new)
            qr_out = qr_out.at[:, k].set(qr_new)
            qs_out = qs_out.at[:, k].set(qs_new)
            qi_out = qi_out.at[:, k].set(qi_new)
            qg_out = qg_out.at[:, k].set(qg_new)
            t_out = t_out.at[:, k].set(t_new)
            prr = prr.at[:, k].set(prr_k)
            prs = prs.at[:, k].set(prs_k)
            prg = prg.at[:, k].set(prg_k)
            pri = pri.at[:, k].set(pri_k)

        return qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, t_out, prr, prs, prg, pri

    return unrolled_fn


if __name__ == "__main__":
    main()
