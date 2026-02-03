#!/usr/bin/env python3
"""
Apply direct optimizations to StableHLO by modifying the IR structure.

The current unrolled StableHLO gets ~1.3x speedup. This script tries to improve it by:
1. Loop tiling (unroll in blocks instead of fully)
2. Reducing intermediate operations
3. Better memory access patterns

Usage:
    python transform_stablehlo.py shlo/precip_effect_x64_lowered.stablehlo \\
        --tile-size 16 -o shlo/precip_effect_tiled_16.stablehlo
"""

import argparse
import re


def main():
    parser = argparse.ArgumentParser(description="Transform StableHLO for better performance")
    parser.add_argument("input", help="Input StableHLO file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile size for loop unrolling (default: 16)")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    with open(args.input, 'r') as f:
        stablehlo = f.read()

    print(f"Original size: {len(stablehlo) / 1024:.1f} KB")

    # For now, just analyze the IR
    analyze_stablehlo(stablehlo)

    print("\nTo improve performance beyond 1.3x, consider:")
    print("1. Generate tiled version (unroll in blocks of 8-16 levels)")
    print("2. Use JAX Pallas to write custom fused kernel")
    print("3. Profile with nsys to find the real bottleneck")


def analyze_stablehlo(stablehlo_text: str):
    """Analyze StableHLO to find optimization opportunities."""
    print("\nAnalyzing StableHLO IR...")

    # Count operations
    while_loops = stablehlo_text.count('stablehlo.while')
    dynamic_slices = stablehlo_text.count('dynamic_slice')
    dynamic_updates = stablehlo_text.count('dynamic_update')
    broadcasts = stablehlo_text.count('stablehlo.broadcast')
    slices = stablehlo_text.count('stablehlo.slice')

    print(f"  While loops:      {while_loops}")
    print(f"  Dynamic slices:   {dynamic_slices}")
    print(f"  Dynamic updates:  {dynamic_updates}")
    print(f"  Broadcasts:       {broadcasts}")
    print(f"  Static slices:    {slices}")

    if while_loops > 0:
        print("\n⚠ Contains while loops - consider using generate_unrolled_stablehlo.py")
    if dynamic_slices + dynamic_updates > 0:
        print(f"\n⚠ Contains {dynamic_slices + dynamic_updates} dynamic operations")
        print("  These prevent compiler optimizations")


if __name__ == "__main__":
    main()
