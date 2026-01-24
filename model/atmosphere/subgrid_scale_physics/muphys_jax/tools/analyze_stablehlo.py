#!/usr/bin/env python3
"""
Analyze StableHLO IR to understand D2D copy patterns.

This script provides detailed analysis of:
1. While loop structure and bounds
2. dynamic_slice/dynamic_update_slice patterns
3. Tensor shapes and data flow
4. Comparison between different implementations

Usage:
    python analyze_stablehlo.py stablehlo_graupel_baseline_lowered.mlir
    python analyze_stablehlo.py --compare baseline allinone
"""

import argparse
import re
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def count_operations(mlir_text: str) -> Dict[str, int]:
    """Count StableHLO operations by type."""
    # Match stablehlo.operation_name
    ops = re.findall(r'stablehlo\.([a-z_]+)', mlir_text)
    counts = defaultdict(int)
    for op in ops:
        counts[op] += 1
    return dict(counts)


def find_while_loops(mlir_text: str) -> List[Dict]:
    """Find all while loops and extract their structure."""
    loops = []

    # Pattern for while loop with iter_args
    pattern = r'(%\w+(?:,\s*%\w+)*)\s*=\s*stablehlo\.while\(([^)]*)\)'

    for match in re.finditer(pattern, mlir_text):
        loop_info = {
            'outputs': match.group(1),
            'inputs': match.group(2),
            'start': match.start(),
            'end': None,
            'bound': None,
        }

        # Look for loop bound in the condition block
        # Pattern: stablehlo.compare LT with a constant
        cond_text = mlir_text[match.end():match.end()+2000]

        # Try different patterns for loop bound
        bound_patterns = [
            r'stablehlo\.constant\s+dense<(\d+)>\s*:\s*tensor<i32>.*?stablehlo\.compare\s+LT',
            r'stablehlo\.constant\s+dense<(\d+)>\s*:\s*tensor<i64>.*?stablehlo\.compare\s+LT',
            r'%c_?\d*\s*=\s*stablehlo\.constant\s+dense<(\d+)>',
        ]

        for pattern in bound_patterns:
            bound_match = re.search(pattern, cond_text, re.DOTALL)
            if bound_match:
                loop_info['bound'] = int(bound_match.group(1))
                break

        loops.append(loop_info)

    return loops


def find_dynamic_operations(mlir_text: str) -> Dict[str, List[Dict]]:
    """Find all dynamic_slice and dynamic_update_slice operations."""
    result = {
        'dynamic_slice': [],
        'dynamic_update_slice': [],
    }

    # dynamic_slice pattern
    ds_pattern = r'(%\w+)\s*=\s*stablehlo\.dynamic_slice\s+(%\w+),\s*([^:]+),\s*sizes\s*=\s*\[([^\]]+)\]'
    for match in re.finditer(ds_pattern, mlir_text):
        result['dynamic_slice'].append({
            'output': match.group(1),
            'input': match.group(2),
            'indices': match.group(3).strip(),
            'sizes': match.group(4),
        })

    # dynamic_update_slice pattern
    dus_pattern = r'(%\w+)\s*=\s*stablehlo\.dynamic_update_slice\s+(%\w+),\s*(%\w+),\s*([^:]+)'
    for match in re.finditer(dus_pattern, mlir_text):
        result['dynamic_update_slice'].append({
            'output': match.group(1),
            'input': match.group(2),
            'update': match.group(3),
            'indices': match.group(4).strip(),
        })

    return result


def find_tensor_shapes(mlir_text: str) -> Dict[str, str]:
    """Extract tensor shapes from type annotations."""
    shapes = {}

    # Pattern: tensor<NxMxf64> or tensor<Nxf64>
    pattern = r'(%\w+)\s*:\s*(tensor<[^>]+>)'
    for match in re.finditer(pattern, mlir_text):
        shapes[match.group(1)] = match.group(2)

    return shapes


def analyze_file(filepath: str) -> Dict:
    """Analyze a single StableHLO file."""
    with open(filepath, 'r') as f:
        content = f.read()

    return {
        'filepath': filepath,
        'size_bytes': len(content),
        'size_kb': len(content) / 1024,
        'line_count': content.count('\n'),
        'op_counts': count_operations(content),
        'while_loops': find_while_loops(content),
        'dynamic_ops': find_dynamic_operations(content),
        'tensor_shapes': find_tensor_shapes(content),
    }


def print_analysis(analysis: Dict, verbose: bool = False):
    """Print analysis results."""
    print("=" * 80)
    print(f"ANALYSIS: {os.path.basename(analysis['filepath'])}")
    print("=" * 80)

    print(f"\nFile size: {analysis['size_kb']:.1f} KB ({analysis['line_count']} lines)")

    # Operation counts
    print(f"\n--- Operation Counts ---")
    ops = analysis['op_counts']
    total_ops = sum(ops.values())
    print(f"Total operations: {total_ops}")

    # Sort by count
    sorted_ops = sorted(ops.items(), key=lambda x: -x[1])
    for op, count in sorted_ops[:15]:
        print(f"  {op}: {count}")

    # While loops
    print(f"\n--- While Loops ---")
    loops = analysis['while_loops']
    print(f"Found {len(loops)} while loop(s)")
    for i, loop in enumerate(loops):
        print(f"  Loop {i+1}:")
        print(f"    Bound: {loop['bound']}")
        print(f"    Outputs: {loop['outputs'][:50]}...")

    # D2D operations
    print(f"\n--- D2D Operations ---")
    dyn = analysis['dynamic_ops']
    ds_count = len(dyn['dynamic_slice'])
    dus_count = len(dyn['dynamic_update_slice'])
    print(f"dynamic_slice: {ds_count}")
    print(f"dynamic_update_slice: {dus_count}")
    print(f"Total D2D: {ds_count + dus_count}")

    if verbose and ds_count > 0:
        print(f"\n  First 5 dynamic_slice operations:")
        for i, op in enumerate(dyn['dynamic_slice'][:5]):
            print(f"    {i+1}. {op['output']} = dynamic_slice {op['input']}")
            print(f"       indices: {op['indices'][:40]}")
            print(f"       sizes: [{op['sizes']}]")

    # Key metrics summary
    print(f"\n--- Summary ---")
    print(f"While loops: {len(loops)}")
    print(f"D2D reads (dynamic_slice): {ds_count}")
    print(f"D2D writes (dynamic_update_slice): {dus_count}")

    if loops and loops[0]['bound']:
        bound = loops[0]['bound']
        print(f"Expected D2D per iteration: {ds_count // bound} reads, {dus_count // bound} writes")

    return analysis


def compare_files(file1: str, file2: str):
    """Compare two StableHLO files."""
    a1 = analyze_file(file1)
    a2 = analyze_file(file2)

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'File 1':>15} {'File 2':>15} {'Diff':>15}")
    print("-" * 75)

    # Size
    print(f"{'Size (KB)':<30} {a1['size_kb']:>15.1f} {a2['size_kb']:>15.1f} {a2['size_kb']-a1['size_kb']:>+15.1f}")

    # Lines
    print(f"{'Lines':<30} {a1['line_count']:>15} {a2['line_count']:>15} {a2['line_count']-a1['line_count']:>+15}")

    # While loops
    l1 = len(a1['while_loops'])
    l2 = len(a2['while_loops'])
    print(f"{'While loops':<30} {l1:>15} {l2:>15} {l2-l1:>+15}")

    # D2D operations
    ds1 = len(a1['dynamic_ops']['dynamic_slice'])
    ds2 = len(a2['dynamic_ops']['dynamic_slice'])
    print(f"{'dynamic_slice':<30} {ds1:>15} {ds2:>15} {ds2-ds1:>+15}")

    dus1 = len(a1['dynamic_ops']['dynamic_update_slice'])
    dus2 = len(a2['dynamic_ops']['dynamic_update_slice'])
    print(f"{'dynamic_update_slice':<30} {dus1:>15} {dus2:>15} {dus2-dus1:>+15}")

    total1 = ds1 + dus1
    total2 = ds2 + dus2
    print(f"{'Total D2D':<30} {total1:>15} {total2:>15} {total2-total1:>+15}")

    # Key operations
    print(f"\n--- Operation Comparison ---")
    all_ops = set(a1['op_counts'].keys()) | set(a2['op_counts'].keys())
    key_ops = ['add', 'multiply', 'select', 'broadcast', 'reshape', 'transpose',
               'dynamic_slice', 'dynamic_update_slice', 'slice', 'while', 'constant']

    for op in key_ops:
        if op in all_ops:
            c1 = a1['op_counts'].get(op, 0)
            c2 = a2['op_counts'].get(op, 0)
            diff = c2 - c1
            if diff != 0:
                print(f"  {op:<25} {c1:>10} {c2:>10} {diff:>+10}")


def main():
    parser = argparse.ArgumentParser(description="Analyze StableHLO IR for D2D patterns")
    parser.add_argument('files', nargs='*', help='StableHLO files to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare two files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if not args.files:
        # Default: analyze all stablehlo files in current directory
        import glob
        args.files = glob.glob('stablehlo_*.mlir')
        if not args.files:
            print("No StableHLO files found. Specify files or run export first.")
            sys.exit(1)

    if args.compare and len(args.files) >= 2:
        compare_files(args.files[0], args.files[1])
    else:
        for filepath in args.files:
            if os.path.exists(filepath):
                analysis = analyze_file(filepath)
                print_analysis(analysis, args.verbose)
                print()
            else:
                print(f"File not found: {filepath}")


if __name__ == "__main__":
    main()
