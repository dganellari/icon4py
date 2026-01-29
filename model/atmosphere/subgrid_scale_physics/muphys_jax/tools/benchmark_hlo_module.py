#!/usr/bin/env python3
"""
Benchmark HLO module execution time (excluding compilation).

This loads a serialized HLO executable and measures pure execution time,
matching the JAX benchmarking methodology.

Usage:
    python tools/benchmark_hlo_module.py shlo/precip_effect_x64.serialized --num-runs 100
"""

import argparse
import time
import sys

import jax
from jax._src import xla_bridge


def benchmark_hlo_executable(serialized_path: str, num_warmup: int = 3, num_runs: int = 100):
    """Benchmark a serialized HLO executable."""

    print(f"Loading serialized executable: {serialized_path}")

    # Load the serialized executable
    backend = xla_bridge.get_backend()
    print(f"Backend: {backend.platform}")

    with open(serialized_path, 'rb') as f:
        serialized = f.read()

    print(f"Deserializing executable ({len(serialized) / 1024 / 1024:.2f} MB)...")
    executable = backend.deserialize_executable(serialized)

    print(f"Executable: {executable.name()}")

    # Get input shapes from the executable
    # We need to provide dummy inputs matching the expected shapes
    # For precipitation_effects: 13 inputs
    # 4 bool masks (ncells, nlev) + 9 float64 arrays (ncells, nlev)

    # Extract shapes from executable metadata if available
    # For now, we'll need to know the shapes ahead of time
    # or provide them as arguments

    print("\nWARNING: This script requires input data to execute.")
    print("Use the .bin files exported by export_precip_effect.py")
    print("Or load them from a netCDF file.")

    return None


def benchmark_with_inputs(serialized_path: str, input_files: list, num_warmup: int = 3, num_runs: int = 100):
    """Benchmark with actual input data."""
    import numpy as np

    print(f"Loading serialized executable: {serialized_path}")
    backend = xla_bridge.get_backend()

    with open(serialized_path, 'rb') as f:
        serialized = f.read()

    executable = backend.deserialize_executable(serialized)
    print(f"Backend: {backend.platform}")
    print(f"Executable: {executable.name()}")

    # Load input data
    print(f"\nLoading {len(input_files)} input arrays...")
    inputs = []
    for input_file in input_files:
        # Read binary file
        # Need to know dtype and shape
        print(f"  {input_file}")
        # This is complex without metadata - skip for now

    print("\nNOTE: Direct executable benchmarking with run_hlo_module measures compilation + execution.")
    print("For execution-only timing, the best approach is to use JAX's compiled function.")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HLO executable execution time"
    )
    parser.add_argument('executable', help='Path to .serialized executable')
    parser.add_argument('--num-warmup', type=int, default=3, help='Warmup runs')
    parser.add_argument('--num-runs', type=int, default=100, help='Benchmark runs')
    parser.add_argument('--input-dir', help='Directory with .bin input files')

    args = parser.parse_args()

    print("=" * 80)
    print("IMPORTANT NOTE")
    print("=" * 80)
    print("""
run_hlo_module measures compilation + execution time, not pure execution.

For accurate execution-only benchmarking:

1. Use JAX directly:
   python tools/run_graupel_optimized.py --input data.nc --mode baseline

2. The .serialized file is for injection, not standalone benchmarking.

3. To compare:
   - JAX baseline: Use run_graupel_optimized.py
   - Transformed HLO: Transform, inject, then use run_graupel_optimized.py

The transformation workflow is:
1. Export to HLO (done)
2. Transform with hlo-opt
3. Inject back into JAX
4. Benchmark the full graupel pipeline
""")

    sys.exit(0)


if __name__ == "__main__":
    main()
