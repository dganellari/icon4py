#!/usr/bin/env python3
"""
Analyze power operations in MLIR and suggest code optimizations.

Power operations (stablehlo.power) are expensive in float64.
This script identifies them and suggests faster alternatives.

Usage:
    python mlir_passes/optimize_power_ops.py mlir_output/graupel_fused_stablehlo.mlir
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def analyze_power_ops(filepath):
    """Extract and analyze power operations from MLIR."""
    print("=" * 70)
    print(f"Power Operation Analysis: {filepath}")
    print("=" * 70)

    with open(filepath) as f:
        content = f.read()

    # Find all power operations
    # Pattern: %result = stablehlo.power %base, %exponent : tensor<...>
    power_pattern = r'(%\w+)\s*=\s*stablehlo\.power\s+(%\w+),\s*(%\w+)\s*:\s*tensor<[^>]+>'
    powers = re.findall(power_pattern, content)

    print(f"\nFound {len(powers)} power operations")
    print("\nLet's trace back to find exponent values...\n")

    # For each power, try to find the exponent value
    exponent_values = defaultdict(list)

    for result, base, exponent in powers[:10]:  # Show first 10
        # Try to find the definition of the exponent
        # Look for: %exponent = stablehlo.constant dense<VALUE>
        const_pattern = rf'{re.escape(exponent)}\s*=\s*stablehlo\.constant\s+dense<([\d.e+-]+)>'
        match = re.search(const_pattern, content)

        if match:
            exp_val = float(match.group(1))
            exponent_values[exp_val].append((result, base, exponent))
            print(f"  {result} = power({base}, {exponent}) = {base}^{exp_val}")

            # Suggest optimization
            if abs(exp_val - 2.0) < 1e-6:
                print(f"    → OPTIMIZE: Replace with multiply: {base} * {base}")
            elif abs(exp_val - 0.5) < 1e-6:
                print(f"    → OPTIMIZE: Replace with sqrt: sqrt({base})")
            elif abs(exp_val - 3.0) < 1e-6:
                print(f"    → OPTIMIZE: Replace with: {base} * {base} * {base}")
            elif abs(exp_val - (-1.0)) < 1e-6:
                print(f"    → OPTIMIZE: Replace with divide: 1.0 / {base}")
        else:
            print(f"  {result} = power({base}, {exponent}) [dynamic exponent]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Exponent Distribution")
    print("=" * 70)

    for exp_val, ops in sorted(exponent_values.items()):
        count = len(ops)
        print(f"  Exponent {exp_val}: {count} occurrences")

        if abs(exp_val - 2.0) < 1e-6:
            print(f"    → Use lax.square() or x * x")
        elif abs(exp_val - 0.5) < 1e-6:
            print(f"    → Use lax.sqrt()")
        elif abs(exp_val - 3.0) < 1e-6:
            print(f"    → Use x * x * x")
        elif abs(exp_val - (-1.0)) < 1e-6:
            print(f"    → Use 1.0 / x")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Search graupel.py for jnp.power() or ** operators
2. Replace with equivalent operations:
   - x**2 → x * x  (or lax.square(x))
   - x**0.5 → jnp.sqrt(x)  (or lax.sqrt(x))
   - x**3 → x * x * x
   - x**(-1) → 1.0 / x

3. For fractional exponents like 0.16666 (1/6) or 0.217:
   - These are used in velocity calculations
   - Keep as power() but document they're necessary
   - Consider precomputing if used repeatedly

4. After making changes:
   - Re-export MLIR
   - Count power ops again (should decrease)
   - Benchmark performance improvement
""")


def find_source_code_powers(source_dir="../"):
    """Scan Python source for power operations."""
    print("\n" + "=" * 70)
    print("Scanning Python source for power operations...")
    print("=" * 70)

    patterns = [
        r'\*\*\s*([\d.]+)',  # x ** 2.0
        r'jnp\.power\(',  # jnp.power(x, y)
        r'lax\.pow\(',  # lax.pow(x, y)
    ]

    import glob
    py_files = glob.glob(f"{source_dir}/**/*.py", recursive=True)

    findings = []
    for filepath in py_files:
        if 'mlir_passes' in filepath:
            continue

        with open(filepath) as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    findings.append((filepath, i, line.strip()))

    print(f"\nFound {len(findings)} potential power operations in source:\n")
    for filepath, lineno, line in findings[:20]:  # Show first 20
        print(f"  {filepath}:{lineno}")
        print(f"    {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_power_ops.py <mlir_file>")
        print("\nExample:")
        print("  python mlir_passes/optimize_power_ops.py mlir_output/graupel_fused_stablehlo.mlir")
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    analyze_power_ops(filepath)
    find_source_code_powers("../")