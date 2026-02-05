#!/usr/bin/env python3
"""
Analyze XLA kernel count and fusion for graupel functions.

This helps understand if operations are being properly fused.
"""

import argparse
import sys
import pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

# Enable XLA dump for analysis
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"

import jax.numpy as jnp
import time

from muphys_jax.utils.data_loading import load_graupel_inputs
from muphys_jax.core.definitions import Q


def count_hlo_instructions(hlo_text: str):
    """Count various HLO instruction types."""
    lines = hlo_text.split('\n')

    # Count fusion blocks
    fusion_count = hlo_text.count('%fusion')
    custom_call_count = hlo_text.count('custom-call')

    # Count basic ops
    counts = {
        'multiply': hlo_text.count('multiply('),
        'add': hlo_text.count('add('),
        'divide': hlo_text.count('divide('),
        'select': hlo_text.count('select('),
        'compare': hlo_text.count('compare('),
        'exp': hlo_text.count('exponential('),
        'log': hlo_text.count('log('),
        'sqrt': hlo_text.count('sqrt('),
        'power': hlo_text.count('power('),
        'fusion': fusion_count,
        'custom-call': custom_call_count,
    }

    return counts


def main():
    parser = argparse.ArgumentParser(description="Analyze XLA kernels")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    args = parser.parse_args()

    print("=" * 70)
    print("XLA KERNEL ANALYSIS")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Load data
    print("Loading data...")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(args.input)
    print(f"Grid: {ncells} cells x {nlev} levels")

    # Transpose to (nlev, ncells)
    print("\nUsing transposed layout (nlev, ncells)")
    t_t = jnp.transpose(t)
    p_t = jnp.transpose(p)
    rho_t = jnp.transpose(rho)
    dz_t = jnp.transpose(dz)
    qnc_t = jnp.transpose(qnc)
    q_t = Q(
        v=jnp.transpose(q.v),
        c=jnp.transpose(q.c),
        r=jnp.transpose(q.r),
        s=jnp.transpose(q.s),
        i=jnp.transpose(q.i),
        g=jnp.transpose(q.g),
    )

    # Import functions
    from muphys_jax.implementations.graupel_native_transposed import (
        q_t_update_native,
        graupel_run_native_transposed,
    )

    # Analyze q_t_update
    print("\n" + "=" * 70)
    print("1. q_t_update (phase transitions)")
    print("=" * 70)

    q_t_update_jit = jax.jit(q_t_update_native)

    # Compile and get HLO
    lowered = q_t_update_jit.lower(t_t, p_t, rho_t, q_t, dt, qnc_t)
    compiled = lowered.compile()

    # Get the HLO text
    hlo_text = compiled.as_text()

    counts = count_hlo_instructions(hlo_text)
    print(f"\nHLO instruction counts:")
    for op, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {op}: {count}")

    # Check number of kernels
    print(f"\nTotal fusions (likely separate kernels): {counts['fusion']}")
    print(f"Custom calls: {counts['custom-call']}")

    # Benchmark
    print("\nBenchmarking...")
    for _ in range(10):  # warmup
        result = q_t_update_jit(t_t, p_t, rho_t, q_t, dt, qnc_t)
        jax.block_until_ready(result)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        result = q_t_update_jit(t_t, p_t, rho_t, q_t, dt, qnc_t)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - start) * 1000)

    import numpy as np
    times = np.array(times)
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")

    # Analyze full graupel
    print("\n" + "=" * 70)
    print("2. Full graupel (end-to-end)")
    print("=" * 70)

    graupel_jit = jax.jit(graupel_run_native_transposed)

    lowered_full = graupel_jit.lower(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
    compiled_full = lowered_full.compile()

    hlo_full = compiled_full.as_text()
    counts_full = count_hlo_instructions(hlo_full)

    print(f"\nHLO instruction counts (full graupel):")
    for op, count in sorted(counts_full.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {op}: {count}")

    print(f"\nTotal fusions: {counts_full['fusion']}")

    # Estimate memory bandwidth
    print("\n" + "=" * 70)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("=" * 70)

    # Each array is nlev x ncells x 8 bytes (float64)
    array_size_bytes = nlev * ncells * 8
    array_size_mb = array_size_bytes / (1024 * 1024)

    print(f"\nSingle array size: {array_size_mb:.1f} MB")
    print(f"q_t_update inputs: t, p, rho, qv, qc, qr, qs, qi, qg, qnc = 10 arrays")
    print(f"q_t_update outputs: qv, qc, qr, qs, qi, qg, t = 7 arrays")
    print(f"Total I/O: 17 arrays = {17 * array_size_mb:.1f} MB")

    # A100 bandwidth: ~2 TB/s
    bandwidth_tbs = 2.0
    theoretical_time_ms = (17 * array_size_mb / 1024) / bandwidth_tbs * 1000

    print(f"\nTheoretical minimum (A100 @ 2 TB/s): {theoretical_time_ms:.2f} ms")
    print(f"Actual time: {np.median(times):.2f} ms")
    print(f"Efficiency: {theoretical_time_ms / np.median(times) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print(f"""
Based on the analysis:

1. q_t_update has {counts['fusion']} fusion blocks
   - Each fusion is likely a separate kernel launch
   - More fusions = more kernel launch overhead + memory round-trips

2. Memory bandwidth theoretical minimum: {theoretical_time_ms:.2f} ms
   - Current time: {np.median(times):.2f} ms
   - This suggests we're NOT memory-bound but compute or launch-bound

3. Optimization strategies:
   a. Reduce kernel count by improving fusion
   b. Use XLA_FLAGS for aggressive fusion
   c. Export to StableHLO and manually optimize
   d. Consider Triton or custom CUDA kernels
""")


if __name__ == "__main__":
    main()
