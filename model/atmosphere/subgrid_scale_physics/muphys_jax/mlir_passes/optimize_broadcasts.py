#!/usr/bin/env python3
"""
Analyze broadcast operations in MLIR to find hoisting opportunities.

Broadcasts inside loops are expensive. This script identifies broadcasts
that could be moved outside the scan loops.

Usage:
    python mlir_passes/optimize_broadcasts.py mlir_output/graupel_fused_stablehlo.mlir
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def analyze_broadcasts(filepath):
    """Identify broadcast patterns and hoisting opportunities."""
    print("=" * 70)
    print(f"Broadcast Optimization Analysis: {filepath}")
    print("=" * 70)

    with open(filepath) as f:
        content = f.read()

    lines = content.split('\n')

    # Find while loop regions
    while_starts = []
    for i, line in enumerate(lines):
        if 'stablehlo.while' in line:
            while_starts.append(i)

    print(f"\nFound {len(while_starts)} while loops")

    # Count broadcasts
    total_broadcasts = content.count('stablehlo.broadcast_in_dim')
    print(f"Total broadcasts: {total_broadcasts}")

    # Analyze broadcasts in while loops
    broadcasts_in_loops = 0
    for start_idx in while_starts:
        # Find the loop body (rough heuristic: next 500 lines)
        loop_region = '\n'.join(lines[start_idx:start_idx+500])
        loop_broadcasts = loop_region.count('stablehlo.broadcast_in_dim')
        broadcasts_in_loops += loop_broadcasts

        print(f"\n  Loop at line {start_idx}:")
        print(f"    Broadcasts in loop: {loop_broadcasts}")

    broadcasts_outside = total_broadcasts - broadcasts_in_loops
    print(f"\nBroadcasts outside loops: {broadcasts_outside}")
    print(f"Broadcasts inside loops: {broadcasts_in_loops}")
    print(f"Hoist ratio: {broadcasts_in_loops}/{total_broadcasts} = {broadcasts_in_loops/total_broadcasts*100:.1f}%")

    # Look for constant broadcasts
    const_broadcast_pattern = r'(%\w+)\s*=\s*stablehlo\.constant\s+dense<([\d.e+-]+)>.*\n.*stablehlo\.broadcast_in_dim\s+\1'
    const_broadcasts = re.findall(const_broadcast_pattern, content, re.MULTILINE)

    print(f"\nConstant broadcasts (easy to hoist): {len(const_broadcasts)}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
Broadcasts inside scan loops hurt performance in 3 ways:
1. Repeated memory allocation
2. Repeated data movement
3. Prevents kernel fusion

OPTIMIZATION STRATEGY:

1. **Hoist constant broadcasts**:
   - Move constant arrays out of scan functions
   - Precompute them in graupel_run() before the scan
   - Pass as additional inputs to the scan

2. **Shape-invariant broadcasts**:
   - If broadcasting same shape repeatedly, do it once
   - Example: If broadcasting (ncells,) → (ncells, nlev) in every iteration

3. **JAX code patterns to check**:

   # BEFORE (broadcast in loop)
   def scan_step(carry, inputs):
       const_array = jnp.full(shape, CONST_VALUE)  # broadcast
       result = inputs * const_array
       return carry, result

   # AFTER (hoist broadcast)
   const_array = jnp.full(shape, CONST_VALUE)  # before scan
   def scan_step(carry, inputs_with_const):
       inputs, const_array = inputs_with_const
       result = inputs * const_array
       return carry, result

4. **Specific constants to check in graupel.py**:
   - Physical constants (const.cvd, const.lvc, etc.)
   - dt, qnc (if broadcasted repeatedly)
   - Velocity prefactors (14.58, 57.80, etc.)

5. **After optimizing**:
   - Re-export MLIR
   - Check broadcast count (should decrease)
   - Measure performance improvement (expect 5-10% gain)
""")


def find_repeated_constants(filepath):
    """Find constants that are used multiple times."""
    print("\n" + "=" * 70)
    print("Finding repeated constant broadcasts...")
    print("=" * 70)

    with open(filepath) as f:
        content = f.read()

    # Extract all constant values
    const_pattern = r'stablehlo\.constant\s+dense<([\d.e+-]+)>'
    constants = re.findall(const_pattern, content)

    # Count occurrences
    from collections import Counter
    const_counts = Counter(constants)

    print(f"\nMost frequently broadcast constants:\n")
    for value, count in const_counts.most_common(20):
        if count > 5:  # Only show constants used more than 5 times
            print(f"  {value}: used {count} times")
            # Try to identify what this constant might be
            try:
                val = float(value)
                if abs(val - 0.0) < 1e-10:
                    print(f"    → Zero (consider using jnp.zeros_like)")
                elif abs(val - 1.0) < 1e-10:
                    print(f"    → One (consider using jnp.ones_like)")
                elif abs(val - 2.0) < 1e-10:
                    print(f"    → Two (common in formulas)")
            except:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_broadcasts.py <mlir_file>")
        print("\nExample:")
        print("  python mlir_passes/optimize_broadcasts.py mlir_output/graupel_fused_stablehlo.mlir")
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    analyze_broadcasts(filepath)
    find_repeated_constants(filepath)