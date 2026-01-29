#!/usr/bin/env python3
"""
Compile StableHLO files and benchmark execution time.

This script:
1. Loads a StableHLO file
2. Compiles it to an XLA executable
3. Benchmarks execution time (excluding compilation)
4. Compares original vs optimized versions

Usage:
    # Compile and verify a StableHLO file
    python tools/compile_and_benchmark_stablehlo.py shlo/precip_effect_x64_lowered.stablehlo

    # Benchmark original vs optimized
    python tools/compile_and_benchmark_stablehlo.py --compare \
        shlo/precip_effect_x64_lowered.stablehlo \
        shlo/precip_effect_x64_optimized.stablehlo

    # Just verify compilation
    python tools/compile_and_benchmark_stablehlo.py --verify-only shlo/precip_effect_x64_optimized.stablehlo
"""

import argparse
import subprocess
import sys
import time
import pathlib

import numpy as np


def run_hlo_opt(input_file: str, output_file: str = None, passes: list = None) -> str:
    """Run hlo-opt on a StableHLO file."""
    if output_file is None:
        output_file = input_file.replace('.stablehlo', '_compiled.hlo')

    cmd = ['hlo-opt', input_file]

    if passes:
        cmd.extend(passes)
    else:
        # Default: just legalize to HLO
        cmd.extend(['--stablehlo-legalize-to-hlo'])

    cmd.extend(['-o', output_file])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"hlo-opt failed:")
        print(result.stderr)
        return None

    print(f"Output: {output_file}")
    return output_file


def run_hlo_module_benchmark(hlo_file: str, platform: str = 'cuda', num_runs: int = 100) -> dict:
    """Run run_hlo_module to benchmark an HLO file."""
    cmd = [
        'run_hlo_module',
        f'--platform={platform}',
        f'--num_runs={num_runs}',
        hlo_file
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"run_hlo_module failed:")
        print(result.stderr)
        return None

    # Parse output for timing information
    output = result.stdout + result.stderr
    print(output)

    # Try to extract timing
    timing = {}
    for line in output.split('\n'):
        if 'time' in line.lower() or 'ms' in line.lower():
            timing['raw'] = line

    return timing


def verify_stablehlo_syntax(stablehlo_file: str) -> bool:
    """Verify StableHLO file syntax using mlir-opt."""
    cmd = ['mlir-opt', '--verify-diagnostics', stablehlo_file]

    print(f"Verifying syntax: {stablehlo_file}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Syntax verification failed:")
        print(result.stderr[:2000])  # First 2000 chars of error
        return False

    print("Syntax OK")
    return True


def compile_stablehlo_with_jax(stablehlo_file: str):
    """Attempt to compile StableHLO using JAX/XLA."""
    try:
        import jax
        from jax._src import xla_bridge
        from jax._src.lib import xla_client

        print(f"Loading StableHLO: {stablehlo_file}")
        with open(stablehlo_file, 'r') as f:
            stablehlo_text = f.read()

        print(f"  Size: {len(stablehlo_text)} bytes")

        # Get backend
        backend = xla_bridge.get_backend()
        print(f"  Backend: {backend.platform}")

        # Try to compile
        print("Compiling...")

        # Method 1: Use xla_client to compile HLO text
        # This requires converting StableHLO to HLO first
        try:
            computation = xla_client.XlaComputation(stablehlo_text.encode())
            executable = backend.compile(computation)
            print("Compilation successful (method 1)")
            return executable
        except Exception as e1:
            print(f"Method 1 failed: {e1}")

        # Method 2: Use mlir_api if available
        try:
            from jax._src.interpreters import mlir
            # This is more complex and requires proper MLIR handling
            print("Method 2: MLIR API not directly usable for raw text")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")

        return None

    except Exception as e:
        print(f"JAX compilation failed: {e}")
        return None


def analyze_stablehlo(stablehlo_file: str):
    """Analyze StableHLO file for key metrics."""
    with open(stablehlo_file, 'r') as f:
        content = f.read()

    metrics = {
        'file': stablehlo_file,
        'size_bytes': len(content),
        'lines': content.count('\n'),
        'while_loops': content.count('stablehlo.while'),
        'dynamic_slice': content.count('stablehlo.dynamic_slice'),
        'dynamic_update_slice': content.count('stablehlo.dynamic_update_slice'),
        'static_slice': content.count('stablehlo.slice') - content.count('dynamic_slice'),
        'add': content.count('stablehlo.add'),
        'multiply': content.count('stablehlo.multiply'),
        'func_call': content.count('func.call'),
    }

    metrics['total_d2d'] = metrics['dynamic_slice'] + metrics['dynamic_update_slice']

    return metrics


def print_comparison(original_metrics: dict, optimized_metrics: dict):
    """Print comparison between original and optimized versions."""
    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Optimized")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Original':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 75)

    for key in ['size_bytes', 'lines', 'while_loops', 'dynamic_slice',
                'dynamic_update_slice', 'total_d2d', 'static_slice',
                'add', 'multiply', 'func_call']:
        orig = original_metrics.get(key, 0)
        opt = optimized_metrics.get(key, 0)

        if orig > 0:
            change = f"{(opt - orig) / orig * 100:+.1f}%"
        else:
            change = "N/A"

        print(f"{key:<30} {orig:>15} {opt:>15} {change:>15}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compile and benchmark StableHLO files"
    )
    parser.add_argument('input', nargs='+', help='Input StableHLO file(s)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare two files (first=original, second=optimized)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify syntax, do not benchmark')
    parser.add_argument('--platform', default='cuda',
                       help='Platform for run_hlo_module (default: cuda)')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--use-hlo-opt', action='store_true',
                       help='Use hlo-opt to convert to HLO before benchmarking')

    args = parser.parse_args()

    print("=" * 80)
    print("StableHLO Compilation and Benchmarking")
    print("=" * 80)

    if args.compare and len(args.input) >= 2:
        # Compare two files
        original_file = args.input[0]
        optimized_file = args.input[1]

        print(f"\nOriginal:  {original_file}")
        print(f"Optimized: {optimized_file}")

        # Analyze both
        orig_metrics = analyze_stablehlo(original_file)
        opt_metrics = analyze_stablehlo(optimized_file)

        print_comparison(orig_metrics, opt_metrics)

        if not args.verify_only:
            print("\nBenchmarking...")

            if args.use_hlo_opt:
                # Convert to HLO first
                orig_hlo = run_hlo_opt(original_file)
                opt_hlo = run_hlo_opt(optimized_file)

                if orig_hlo and opt_hlo:
                    print("\n--- Original ---")
                    run_hlo_module_benchmark(orig_hlo, args.platform, args.num_runs)

                    print("\n--- Optimized ---")
                    run_hlo_module_benchmark(opt_hlo, args.platform, args.num_runs)
            else:
                print("\nNote: Direct benchmarking requires hlo-opt conversion.")
                print("Use --use-hlo-opt flag to convert and benchmark.")

    else:
        # Single file analysis
        for input_file in args.input:
            print(f"\n--- Analyzing: {input_file} ---")

            metrics = analyze_stablehlo(input_file)

            print(f"\nMetrics:")
            for key, value in metrics.items():
                if key != 'file':
                    print(f"  {key}: {value}")

            if args.verify_only:
                verify_stablehlo_syntax(input_file)
            elif args.use_hlo_opt:
                hlo_file = run_hlo_opt(input_file)
                if hlo_file:
                    run_hlo_module_benchmark(hlo_file, args.platform, args.num_runs)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
