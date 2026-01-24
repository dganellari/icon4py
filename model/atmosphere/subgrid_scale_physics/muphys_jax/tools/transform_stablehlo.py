#!/usr/bin/env python3
"""
Transform StableHLO IR from JAX scan to eliminate D2D copies.

Strategy:
1. Parse stablehlo.while loop with scan pattern
2. Unroll into static sequence of operations
3. Keep carry state in SSA values (registers, not memory)
4. Replace dynamic_slice/dynamic_update_slice with static indexing
5. Emit only final write operations

This transforms:
  stablehlo.while (i=0; i<90; i++) {
    x = dynamic_slice(input, i)      // D2D READ
    carry_new = compute(carry, x)
    output = dynamic_update(out, i, carry_new)  // D2D WRITE
  }

Into:
  x0 = input[0]; carry1 = compute(carry0, x0)
  x1 = input[1]; carry2 = compute(carry1, x1)
  ...
  x89 = input[89]; carry90 = compute(carry89, x89)
  return carry90  // Single write at end

Usage:
  python transform_stablehlo.py input.mlir output.mlir
  python transform_stablehlo.py input.mlir output.mlir --loop-bound 10
  python transform_stablehlo.py input.mlir --analyze-only
"""

import argparse
import re
import sys
import os
from typing import List, Tuple, Dict, Optional


def find_loop_bound(mlir_text: str) -> Optional[int]:
    """
    Try multiple patterns to extract loop bound from StableHLO IR.
    Returns None if no bound can be determined.
    """
    # Pattern 1: stablehlo.constant dense<N> : tensor<i32> followed by compare LT
    pattern1 = re.search(
        r'stablehlo\.constant\s+dense<(\d+)>\s*:\s*tensor<i32>.*?stablehlo\.compare\s+LT',
        mlir_text, re.DOTALL
    )
    if pattern1:
        return int(pattern1.group(1))

    # Pattern 2: stablehlo.constant dense<N> : tensor<i64> followed by compare LT
    pattern2 = re.search(
        r'stablehlo\.constant\s+dense<(\d+)>\s*:\s*tensor<i64>.*?stablehlo\.compare\s+LT',
        mlir_text, re.DOTALL
    )
    if pattern2:
        return int(pattern2.group(1))

    # Pattern 3: Look for any constant that looks like a loop bound (small integer)
    # in the condition block
    cond_match = re.search(r'cond\s*\{(.*?)\}\s*do', mlir_text, re.DOTALL)
    if cond_match:
        cond_block = cond_match.group(1)
        # Look for constants in condition
        constants = re.findall(r'stablehlo\.constant\s+dense<(\d+)>', cond_block)
        for c in constants:
            val = int(c)
            # Loop bounds are typically small-ish integers
            if 1 < val <= 10000:
                return val

    # Pattern 4: Look for loop counter comparison anywhere
    pattern4 = re.search(
        r'%\w+\s*=\s*stablehlo\.compare\s+LT.*?%\w+,\s*%(\w+)',
        mlir_text, re.DOTALL
    )
    if pattern4:
        # Try to find the constant that the counter is compared to
        const_name = pattern4.group(1)
        const_match = re.search(
            rf'%{const_name}\s*=\s*stablehlo\.constant\s+dense<(\d+)>',
            mlir_text
        )
        if const_match:
            return int(const_match.group(1))

    # Pattern 5: Look for tensor shape that indicates loop bound
    # e.g., tensor<10x100xf64> - first dimension is often the loop bound
    shape_match = re.search(r'tensor<(\d+)x\d+xf\d+>', mlir_text)
    if shape_match:
        potential_bound = int(shape_match.group(1))
        if 1 < potential_bound <= 1000:
            return potential_bound

    return None


def parse_while_loop(mlir_text: str, manual_bound: Optional[int] = None) -> Dict:
    """Extract while loop structure from StableHLO IR"""
    info = {
        'found': False,
        'loop_bound': None,
        'num_iterargs': 0,
        'body_start': None,
        'body_end': None,
        'cond_start': None,
        'cond_end': None,
        'num_while_loops': 0,
    }

    # Count while loops
    info['num_while_loops'] = mlir_text.count('stablehlo.while')

    # Find while loop
    while_match = re.search(r'stablehlo\.while\((.*?)\) : (.*?)\n\s+cond \{', mlir_text, re.DOTALL)
    if not while_match:
        return info

    info['found'] = True

    # Use manual bound if provided, otherwise try to detect
    if manual_bound is not None:
        info['loop_bound'] = manual_bound
    else:
        info['loop_bound'] = find_loop_bound(mlir_text)

    # Find body boundaries
    do_match = re.search(r'} do \{', mlir_text)
    if do_match:
        info['body_start'] = do_match.end()
        # Find matching closing brace
        brace_count = 1
        pos = info['body_start']
        while brace_count > 0 and pos < len(mlir_text):
            if mlir_text[pos] == '{':
                brace_count += 1
            elif mlir_text[pos] == '}':
                brace_count -= 1
            pos += 1
        info['body_end'] = pos - 1

    return info


def extract_loop_body_pattern(mlir_text: str, info: Dict) -> List[str]:
    """Extract operations from loop body"""
    if not info['found'] or info['body_start'] is None:
        return []

    body_text = mlir_text[info['body_start']:info['body_end']]
    lines = [line.strip() for line in body_text.split('\n') if line.strip() and not line.strip().startswith('stablehlo.return')]

    return lines


def generate_unrolled_stablehlo(loop_bound: int, body_lines: List[str], input_shape: Tuple[int, int]) -> str:
    """
    Generate unrolled StableHLO IR by parsing and replicating actual loop body.

    Strategy:
    1. Parse each operation in body_lines
    2. For each iteration k, replace:
       - %iterArg_N (loop counter) → constant k
       - dynamic_slice(..., %iterArg_N, ...) → static slice at index k
       - %varN → %varN_k (SSA renaming per iteration)
    3. Track carry variables across iterations
    """
    nlev, ncells = input_shape

    output_lines = []
    output_lines.append(f"// Unrolled scan: {loop_bound} iterations")
    output_lines.append("// D2D operations replaced with static slicing")
    output_lines.append("")

    # Parse body to identify variables
    # Track SSA variable mappings across iterations
    var_map = {}  # Maps original var → current iteration var

    for k in range(loop_bound):
        output_lines.append(f"// === Iteration {k} ===")

        # Reset variable mapping for this iteration
        iter_var_map = {}

        # Process each operation in the loop body
        for line in body_lines:
            if not line.strip() or line.strip().startswith('//'):
                continue

            # Parse operation: %result = op %inputs...
            transformed_line = line

            # Replace loop counter references with constant k
            # Pattern: %iterArg_141 (common loop counter name)
            transformed_line = re.sub(r'%iterArg_\d+', f'%c_iter_{k}', transformed_line)

            # Replace dynamic_slice with static slice
            if 'dynamic_slice' in transformed_line:
                # Convert dynamic_slice to static slice
                # Extract: %out = stablehlo.dynamic_slice %input, %idx, sizes = [...]
                match = re.search(r'(%\w+)\s*=\s*stablehlo\.dynamic_slice\s+(%\w+),\s*([^,]+),\s*(.+?),\s*sizes\s*=\s*\[([^\]]+)\]', transformed_line)
                if match:
                    result_var = match.group(1)
                    input_var = match.group(2)
                    sizes = match.group(5)

                    # Generate static slice for iteration k
                    # Assuming first dimension is the iteration dimension
                    transformed_line = f"{result_var}_{k} = stablehlo.slice {input_var} [{k}:{k+1}, ...] sizes = [{sizes}]"

            # Replace dynamic_update_slice (eliminate - will write at end)
            if 'dynamic_update_slice' in transformed_line:
                # Skip dynamic updates - we'll write outputs at the end
                output_lines.append(f"  // Skipped: {transformed_line[:80]}...")
                continue

            # Rename SSA variables with iteration suffix
            # %1234 → %1234_k
            def rename_var(match):
                var = match.group(0)
                if var.startswith('%iterArg'):
                    return var  # Already handled
                elif var.startswith('%arg'):
                    return var  # Function arguments don't change
                elif var.startswith('%c_'):
                    return var  # Constants can be shared (or rename if needed)
                else:
                    return f"{var}_{k}"

            transformed_line = re.sub(r'%\w+', rename_var, transformed_line)

            # Add to output
            output_lines.append(f"  {transformed_line}")

        output_lines.append("")

    # Note about outputs
    output_lines.append("// NOTE: Output tensor construction not implemented")
    output_lines.append("// Need to collect all iteration results and build output tensors")
    output_lines.append("// For now, this eliminates D2D ops within the loop")

    return "\n".join(output_lines)


def analyze_stablehlo_ir(input_file: str) -> Dict:
    """Analyze StableHLO IR without transforming."""
    print(f"Reading StableHLO IR from {input_file}")

    with open(input_file, 'r') as f:
        mlir_text = f.read()

    print(f"  Size: {len(mlir_text)} bytes ({len(mlir_text) / 1024:.1f} KB)")
    print(f"  Lines: {mlir_text.count(chr(10))}")

    # Count operations
    ops = {
        'while': mlir_text.count('stablehlo.while'),
        'dynamic_slice': mlir_text.count('stablehlo.dynamic_slice'),
        'dynamic_update_slice': mlir_text.count('stablehlo.dynamic_update_slice'),
        'add': mlir_text.count('stablehlo.add'),
        'multiply': mlir_text.count('stablehlo.multiply'),
        'constant': mlir_text.count('stablehlo.constant'),
        'broadcast': mlir_text.count('stablehlo.broadcast'),
    }

    print("\n=== OPERATION COUNTS ===")
    for op, count in ops.items():
        print(f"  {op}: {count}")

    print(f"\n  Total D2D operations: {ops['dynamic_slice'] + ops['dynamic_update_slice']}")

    # Try to detect loop bound
    bound = find_loop_bound(mlir_text)
    print(f"\n=== LOOP ANALYSIS ===")
    print(f"  While loops: {ops['while']}")
    print(f"  Detected loop bound: {bound}")

    if bound and ops['dynamic_slice'] > 0:
        print(f"  Estimated D2D per iteration: {ops['dynamic_slice'] // bound} reads, {ops['dynamic_update_slice'] // bound} writes")

    return {'ops': ops, 'bound': bound, 'mlir_text': mlir_text}


def transform_stablehlo_ir(input_file: str, output_file: str,
                           manual_bound: Optional[int] = None,
                           input_shape: Tuple[int, int] = (90, 1000)) -> bool:
    """Main transformation pipeline. Returns True on success."""
    print(f"Reading StableHLO IR from {input_file}")

    with open(input_file, 'r') as f:
        mlir_text = f.read()

    print("\n=== ANALYSIS ===")
    info = parse_while_loop(mlir_text, manual_bound)

    if not info['found']:
        print("ERROR: No stablehlo.while loop found!")
        return False

    print(f"  Found {info['num_while_loops']} while loop(s)")
    print(f"  Loop bound: {info['loop_bound']}")

    if info['loop_bound'] is None:
        print("\nERROR: Could not determine loop bound!")
        print("Please specify the loop bound manually with --loop-bound N")
        print("\nHint: Look for tensor shapes like 'tensor<NxMxf64>' in the IR")
        print("      The first dimension (N) is often the loop bound.")
        return False

    body_lines = extract_loop_body_pattern(mlir_text, info)
    print(f"  Body operations: {len(body_lines)}")
    print(f"\nFirst 10 body operations:")
    for i, line in enumerate(body_lines[:10]):
        print(f"    {i}: {line[:80]}")

    print("\n=== TRANSFORMATION ===")
    print("Strategy:")
    print("  1. Unroll while loop into static sequence")
    print("  2. Replace dynamic_slice with stablehlo.slice")
    print("  3. Keep carry state as SSA values (no memref)")
    print("  4. Eliminate dynamic_update_slice (write outputs at end)")

    # Generate unrolled version
    nlev, ncells = input_shape
    unrolled = generate_unrolled_stablehlo(
        loop_bound=info['loop_bound'],
        body_lines=body_lines,
        input_shape=input_shape
    )

    print(f"\n=== GENERATED UNROLLED IR ===")
    print(f"Lines: {len(unrolled.split(chr(10)))}")
    print("\nFirst 500 chars:")
    print(unrolled[:500])
    print("...")

    # Save transformed IR
    with open(output_file, 'w') as f:
        f.write("module @unrolled_scan {\n")
        f.write(f"  func.func public @main(%arg0: tensor<{ncells}xf32>, %arg1: tensor<{ncells}xf32>, %arg2: tensor<{nlev}x{ncells}xf32>) -> (tensor<{ncells}xf32>, tensor<{ncells}xf32>) {{\n")
        for line in unrolled.split('\n'):
            f.write(f"    {line}\n")
        f.write("  }\n")
        f.write("}\n")

    print(f"\n  Saved transformed IR to {output_file}")

    print("\n=== NEXT STEPS ===")
    print("1. Fix output tensor construction (currently only returns final carry)")
    print("2. Apply MLIR optimization passes:")
    print("   mlir-opt --canonicalize --cse stablehlo_unrolled.mlir")
    print("3. Lower to GPU dialect:")
    print("   mlir-opt --convert-stablehlo-to-linalg --linalg-fuse-elementwise-ops")
    print("4. Compare performance against JAX baseline")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Transform StableHLO IR to eliminate D2D copies by unrolling while loops"
    )
    parser.add_argument('input_file', help='Input StableHLO MLIR file')
    parser.add_argument('output_file', nargs='?', help='Output transformed MLIR file')
    parser.add_argument('--loop-bound', type=int, dest='loop_bound',
                       help='Manually specify loop bound (required if auto-detection fails)')
    parser.add_argument('--analyze-only', action='store_true', dest='analyze_only',
                       help='Only analyze the IR, do not transform')
    parser.add_argument('--input-shape', type=str, dest='input_shape', default='90,1000',
                       help='Input tensor shape as "nlev,ncells" (default: 90,1000)')

    args = parser.parse_args()

    print("=" * 80)
    print("StableHLO Transformation: Eliminate D2D Copies")
    print("=" * 80)
    print()

    if args.analyze_only:
        analyze_stablehlo_ir(args.input_file)
        return

    if args.output_file is None:
        base = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base}_unrolled.mlir"

    # Parse input shape
    try:
        nlev, ncells = map(int, args.input_shape.split(','))
    except ValueError:
        print(f"ERROR: Invalid input shape '{args.input_shape}'. Use format 'nlev,ncells'")
        sys.exit(1)

    success = transform_stablehlo_ir(
        args.input_file,
        args.output_file,
        manual_bound=args.loop_bound,
        input_shape=(nlev, ncells)
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
