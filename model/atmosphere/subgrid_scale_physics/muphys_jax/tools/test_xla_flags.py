#!/usr/bin/env python3
"""
Test various XLA flags to improve kernel fusion and performance.

This script tests different XLA optimization flags to find the best
configuration for graupel performance.

Usage:
    python test_xla_flags.py --input data.nc
"""

import argparse
import subprocess
import sys
import os

# Different XLA flag configurations to test
XLA_CONFIGS = {
    "baseline": "",

    "autotune_high": "--xla_gpu_autotune_level=4",

    "graph_level_3": "--xla_gpu_graph_level=3",

    "fusion_aggressive": (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_enable_priority_fusion=true"
    ),

    "memory_opt": (
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true"
    ),

    "all_optimizations": (
        "--xla_gpu_autotune_level=4 "
        "--xla_gpu_graph_level=3 "
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_enable_priority_fusion=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true"
    ),

    "cudnn_fusion": (
        "--xla_gpu_enable_cudnn_fmha=true "
        "--xla_gpu_fused_attention_use_cudnn_rng=true"
    ),
}


def run_benchmark(input_file: str, xla_flags: str, config_name: str):
    """Run the graupel benchmark with specific XLA flags."""
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    if xla_flags:
        env["XLA_FLAGS"] = xla_flags

    # Run a quick benchmark script
    script = f'''
import sys
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))}")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import time
import numpy as np

from muphys_jax.utils.data_loading import load_graupel_inputs
from muphys_jax.core.definitions import Q
from muphys_jax.implementations.graupel_native_transposed import (
    graupel_run_native_transposed,
    q_t_update_native,
)

# Load data
dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs("{input_file}")

# Transpose
dz_t = jnp.transpose(dz)
t_t = jnp.transpose(t)
p_t = jnp.transpose(p)
rho_t = jnp.transpose(rho)
qnc_t = jnp.transpose(qnc)
q_t = Q(
    v=jnp.transpose(q.v), c=jnp.transpose(q.c), r=jnp.transpose(q.r),
    s=jnp.transpose(q.s), i=jnp.transpose(q.i), g=jnp.transpose(q.g),
)

# Benchmark q_t_update
q_t_update_jit = jax.jit(q_t_update_native)
for _ in range(5):
    result = q_t_update_jit(t_t, p_t, rho_t, q_t, dt, qnc_t)
    jax.block_until_ready(result)

times = []
for _ in range(10):
    start = time.perf_counter()
    result = q_t_update_jit(t_t, p_t, rho_t, q_t, dt, qnc_t)
    jax.block_until_ready(result)
    times.append((time.perf_counter() - start) * 1000)

qt_median = np.median(times)

# Benchmark full graupel
graupel_jit = jax.jit(graupel_run_native_transposed)
for _ in range(5):
    result = graupel_jit(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
    jax.block_until_ready(result)

times = []
for _ in range(10):
    start = time.perf_counter()
    result = graupel_jit(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
    jax.block_until_ready(result)
    times.append((time.perf_counter() - start) * 1000)

full_median = np.median(times)

print(f"RESULT: q_t_update={{qt_median:.2f}}ms full_graupel={{full_median:.2f}}ms")
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse output
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                return line

        if result.returncode != 0:
            return f"ERROR: {result.stderr[:200]}"

        return "ERROR: No result found"

    except subprocess.TimeoutExpired:
        return "ERROR: Timeout"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Test XLA flags")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    parser.add_argument("--config", "-c", choices=list(XLA_CONFIGS.keys()),
                       help="Specific config to test (default: test all)")
    args = parser.parse_args()

    print("=" * 70)
    print("XLA FLAGS OPTIMIZATION TEST")
    print("=" * 70)
    print()

    configs_to_test = [args.config] if args.config else list(XLA_CONFIGS.keys())

    results = {}
    for config_name in configs_to_test:
        xla_flags = XLA_CONFIGS[config_name]
        print(f"\nTesting: {config_name}")
        if xla_flags:
            print(f"  Flags: {xla_flags[:60]}...")
        else:
            print("  Flags: (none)")

        result = run_benchmark(args.input, xla_flags, config_name)
        print(f"  {result}")
        results[config_name] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'q_t_update':<15} {'Full graupel':<15}")
    print("-" * 55)

    for config_name, result in results.items():
        if result.startswith("RESULT:"):
            # Parse "RESULT: q_t_update=X.XXms full_graupel=Y.YYms"
            parts = result.replace("RESULT:", "").strip().split()
            qt = parts[0].split("=")[1]
            full = parts[1].split("=")[1]
            print(f"{config_name:<25} {qt:<15} {full:<15}")
        else:
            print(f"{config_name:<25} {result[:40]}")


if __name__ == "__main__":
    main()
