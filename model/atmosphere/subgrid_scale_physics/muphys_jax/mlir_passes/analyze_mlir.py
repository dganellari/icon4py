#!/usr/bin/env python3
"""
Analyze exported MLIR to understand the structure and identify optimization targets.

Usage:
    python mlir_passes/analyze_mlir.py mlir_output/graupel_stablehlo.mlir
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def analyze_mlir_file(filepath):
    """Analyze MLIR file for optimization opportunities."""
    print("=" * 70)
    print(f"MLIR Analysis: {filepath}")
    print("=" * 70)

    with open(filepath) as f:
        content = f.read()

    lines = content.split('\n')
    print(f"\nBasic Stats:")
    print(f"  Total lines: {len(lines)}")
    print(f"  File size: {len(content) / 1024:.1f} KB")

    # 1. Count operations by type
    print(f"\n1. Operation Counts:")
    ops = defaultdict(int)
    for line in lines:
        # Match stablehlo.xxx or arith.xxx etc
        matches = re.findall(r'(stablehlo\.\w+|arith\.\w+|tensor\.\w+|func\.\w+)', line)
        for m in matches:
            ops[m] += 1

    for op, count in sorted(ops.items(), key=lambda x: -x[1])[:20]:
        print(f"    {op}: {count}")

    # 2. Find while loops (scans)
    print(f"\n2. Vertical Scans (stablehlo.while):")
    while_count = content.count('stablehlo.while')
    print(f"    Found {while_count} while loops")

    # Extract while loop regions
    while_matches = list(re.finditer(r'stablehlo\.while\(', content))
    for i, match in enumerate(while_matches[:5]):  # Show first 5
        start = match.start()
        # Get context around the while
        context_start = max(0, start - 100)
        context_end = min(len(content), start + 500)
        context = content[context_start:context_end]
        print(f"\n    While #{i+1} at position {start}:")
        print(f"    ...{context[:200]}...")

    # 3. Find tensor types (to check precision)
    print(f"\n3. Tensor Types (check precision):")
    f64_count = content.count('f64')
    f32_count = content.count('f32')
    print(f"    f64 (double): {f64_count} occurrences")
    print(f"    f32 (float): {f32_count} occurrences")

    # 4. Find function definitions
    print(f"\n4. Functions:")
    func_matches = re.findall(r'func\.func @(\w+)', content)
    print(f"    Found {len(func_matches)} functions:")
    for fn in func_matches[:10]:
        print(f"      - {fn}")
    if len(func_matches) > 10:
        print(f"      ... and {len(func_matches) - 10} more")

    # 5. Identify scan patterns
    print(f"\n5. Scan Pattern Analysis:")
    analyze_scan_patterns(content)

    # 6. Memory access patterns
    print(f"\n6. Memory Operations:")
    broadcast_count = content.count('stablehlo.broadcast')
    reshape_count = content.count('stablehlo.reshape')
    slice_count = content.count('stablehlo.slice')
    print(f"    broadcast: {broadcast_count}")
    print(f"    reshape: {reshape_count}")
    print(f"    slice: {slice_count}")

    # 7. Compute operations (potential fusion targets)
    print(f"\n7. Compute Operations (fusion targets):")
    add_count = content.count('stablehlo.add')
    mul_count = content.count('stablehlo.multiply')
    div_count = content.count('stablehlo.divide')
    pow_count = content.count('stablehlo.power')
    print(f"    add: {add_count}")
    print(f"    multiply: {mul_count}")
    print(f"    divide: {div_count}")
    print(f"    power: {pow_count}")

    print(f"\n8. Optimization Opportunities:")
    print_optimization_opportunities(ops, while_count)


def analyze_scan_patterns(content):
    """Analyze the structure of scan (while) operations."""
    # Look for tuple types in while loops (carry state)
    tuple_pattern = r'tuple<([^>]+)>'
    tuples = re.findall(tuple_pattern, content)

    # Count unique tuple signatures
    tuple_counts = defaultdict(int)
    for t in tuples:
        # Simplify: just count element count
        elem_count = t.count(',') + 1
        tuple_counts[elem_count] += 1

    print(f"    Tuple sizes (carry state complexity):")
    for size, count in sorted(tuple_counts.items()):
        print(f"      {size}-element tuples: {count} occurrences")

    # Look for iteration bounds
    bounds = re.findall(r'arith\.constant\s+(\d+)', content)
    if bounds:
        print(f"    Iteration bounds found: {set(bounds)}")
        if '90' in bounds or '89' in bounds:
            print(f"    ✓ Found 90-level vertical scan pattern!")


def print_optimization_opportunities(ops, while_count):
    """Print potential optimization opportunities."""
    opportunities = []

    # 1. Scan fusion
    if while_count > 1:
        opportunities.append(
            f"SCAN FUSION: {while_count} while loops could potentially be fused into fewer kernels"
        )

    # 2. CSE
    if ops.get('stablehlo.multiply', 0) > 50:
        opportunities.append(
            "CSE: Many multiply ops - common subexpression elimination may help"
        )

    # 3. Broadcast optimization
    if ops.get('stablehlo.broadcast', 0) > 20:
        opportunities.append(
            "BROADCAST: Many broadcasts - could be hoisted out of loops"
        )

    # 4. Power optimization
    if ops.get('stablehlo.power', 0) > 5:
        opportunities.append(
            "POWER: power operations are expensive - check if exponents are constants"
        )

    print(f"    Found {len(opportunities)} potential optimizations:")
    for i, opp in enumerate(opportunities, 1):
        print(f"    {i}. {opp}")

    if not opportunities:
        print("    No obvious opportunities found - need deeper analysis")


def compare_with_xla_hlo(stablehlo_file, hlo_file=None):
    """Compare StableHLO with XLA HLO to understand transformations."""
    if hlo_file is None:
        hlo_file = stablehlo_file.replace('stablehlo', 'hlo')

    if not Path(hlo_file).exists():
        print(f"\nNo HLO file found at {hlo_file}")
        return

    print(f"\n" + "=" * 70)
    print(f"Comparing StableHLO vs HLO")
    print("=" * 70)

    with open(stablehlo_file) as f:
        stablehlo_size = len(f.read())
    with open(hlo_file) as f:
        hlo_size = len(f.read())

    print(f"  StableHLO: {stablehlo_size / 1024:.1f} KB")
    print(f"  HLO: {hlo_size / 1024:.1f} KB")
    print(f"  Ratio: {hlo_size / stablehlo_size:.2f}x")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_mlir.py <mlir_file>")
        print("\nExample:")
        print("  python mlir_passes/analyze_mlir.py mlir_output/graupel_stablehlo.mlir")
        sys.exit(1)

    filepath = sys.argv[0] if len(sys.argv) == 1 else sys.argv[1]

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    analyze_mlir_file(filepath)

    # Also compare with HLO if available
    compare_with_xla_hlo(filepath)
