#!/usr/bin/env python3
"""
Transform stablehlo.power operations with constant exponents to multiplications.

This pass identifies power operations like x^2, x^3, x^0.5 and transforms them:
- x^2 → x * x
- x^3 → x * x * x  
- x^0.5 → sqrt(x)
- x^-1 → 1/x

Usage:
    python mlir_passes/power_to_multiply.py mlir_output/graupel_stablehlo.mlir
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def analyze_power_operations(mlir_content):
    """Analyze power operations to find optimization opportunities."""
    print("=" * 70)
    print("Power Operation Analysis")
    print("=" * 70)
    
    # Find all power operations
    power_pattern = r'(%\w+) = stablehlo\.power (%\w+), (%\w+) : (tensor<[^>]+>)'
    powers = re.findall(power_pattern, mlir_content)
    
    print(f"\nFound {len(powers)} power operations")
    
    # Track exponents
    exponent_values = defaultdict(list)
    exponent_sources = defaultdict(list)
    
    # For each power, try to trace back the exponent to a constant
    for result, base, exp, dtype in powers:
        # Look for the exponent definition (immediate constant)
        exp_pattern = rf'{re.escape(exp)} = stablehlo\.constant dense<([^>]+)> :'
        match = re.search(exp_pattern, mlir_content)
        
        if match:
            value = match.group(1)
            exponent_values[value].append((result, base, exp, dtype))
        else:
            # Exponent is computed - try to find its definition
            exp_def_pattern = rf'{re.escape(exp)} = ([^\n]+)'
            def_match = re.search(exp_def_pattern, mlir_content)
            if def_match:
                definition = def_match.group(1)
                exponent_sources['computed'].append((result, base, exp, dtype, definition))
            else:
                exponent_sources['unknown'].append((result, base, exp, dtype))
    
    if exponent_values:
        print(f"\nConstant exponent values:")
        for value, ops in sorted(exponent_values.items(), key=lambda x: -len(x[1])):
            print(f"  {value}: {len(ops)} occurrences")
            
            # Suggest optimization
            try:
                val = float(value)
                if val == 2.0:
                    print(f"    → Optimize to multiply: x * x")
                elif val == 3.0:
                    print(f"    → Optimize to multiply: x * x * x")
                elif val == 0.5:
                    print(f"    → Optimize to sqrt: stablehlo.sqrt(x)")
                elif val == -1.0:
                    print(f"    → Optimize to divide: 1.0 / x")
                elif val == 1.0:
                    print(f"    → Optimize to identity: x")
                elif val > 0 and val == int(val):
                    n = int(val)
                    print(f"    → Optimize to {n} multiplies")
            except ValueError:
                pass
    
    # Show examples of computed exponents
    if exponent_sources.get('computed'):
        print(f"\nComputed exponents (first 10 examples):")
        for i, (result, base, exp, dtype, definition) in enumerate(exponent_sources['computed'][:10]):
            print(f"  {i+1}. {exp} = {definition[:80]}")
            if len(definition) > 80:
                print(f"     ...")
    
    return exponent_values, exponent_sources


def generate_transformation_plan(exponent_values):
    """Generate a transformation plan for power operations."""
    print("\n" + "=" * 70)
    print("Transformation Plan")
    print("=" * 70)
    
    transformations = []
    
    for value, ops in exponent_values.items():
        try:
            val = float(value)
            
            if val == 2.0:
                for result, base, exp, dtype in ops:
                    transformations.append({
                        'type': 'square',
                        'original': f'{result} = stablehlo.power {base}, {exp} : {dtype}',
                        'replacement': f'{result} = stablehlo.multiply {base}, {base} : {dtype}',
                        'speedup': '~2-3x (multiply vs power)'
                    })
            elif val == 0.5:
                for result, base, exp, dtype in ops:
                    transformations.append({
                        'type': 'sqrt',
                        'original': f'{result} = stablehlo.power {base}, {exp} : {dtype}',
                        'replacement': f'{result} = stablehlo.sqrt {base} : {dtype}',
                        'speedup': '~2x (sqrt vs power)'
                    })
            elif val == -1.0:
                for result, base, exp, dtype in ops:
                    transformations.append({
                        'type': 'reciprocal',
                        'original': f'{result} = stablehlo.power {base}, {exp} : {dtype}',
                        'replacement': f'%one = stablehlo.constant dense<1.0> : {dtype}\n' +
                                     f'{result} = stablehlo.divide %one, {base} : {dtype}',
                        'speedup': '~2x (divide vs power)'
                    })
            elif val == 3.0:
                for result, base, exp, dtype in ops:
                    transformations.append({
                        'type': 'cube',
                        'original': f'{result} = stablehlo.power {base}, {exp} : {dtype}',
                        'replacement': f'%tmp = stablehlo.multiply {base}, {base} : {dtype}\n' +
                                     f'{result} = stablehlo.multiply %tmp, {base} : {dtype}',
                        'speedup': '~2-3x (multiply vs power)'
                    })
        except ValueError:
            continue
    
    print(f"\nTotal transformations: {len(transformations)}")
    print("\nBreakdown by type:")
    type_counts = defaultdict(int)
    for t in transformations:
        type_counts[t['type']] += 1
    
    for ttype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ttype}: {count}")
    
    # Show examples
    if transformations:
        print(f"\nExample transformations:")
        shown_types = set()
        for t in transformations:
            if t['type'] not in shown_types:
                print(f"\n  {t['type'].upper()}:")
                print(f"    Before: {t['original']}")
                print(f"    After:  {t['replacement']}")
                print(f"    Speedup: {t['speedup']}")
                shown_types.add(t['type'])
                if len(shown_types) >= 3:
                    break
    
    return transformations


def apply_transformations(mlir_content, transformations, output_file):
    """Apply power-to-multiply transformations to MLIR."""
    print("\n" + "=" * 70)
    print("Applying Transformations")
    print("=" * 70)
    
    modified = mlir_content
    applied = 0
    
    # For now, just generate a report - actual MLIR rewriting would need proper IR handling
    print(f"\nWARNING: Automatic MLIR rewriting requires proper IR manipulation.")
    print(f"This script generates a transformation plan.")
    print(f"\nTo apply these optimizations:")
    print(f"  1. Use MLIR dialect conversion framework")
    print(f"  2. Implement a custom StableHLO pass in C++")
    print(f"  3. Or use JAX XLA custom calls with optimized kernels")
    
    # For demonstration, show what would be replaced
    print(f"\nTransformation opportunities identified: {len(transformations)}")
    
    return modified, applied


def estimate_speedup(transformations):
    """Estimate overall speedup from transformations."""
    print("\n" + "=" * 70)
    print("Performance Impact Estimate")
    print("=" * 70)
    
    # Rough estimates based on operation counts
    speedups = {
        'square': 2.5,
        'cube': 2.0,
        'sqrt': 2.0,
        'reciprocal': 2.0,
    }
    
    type_counts = defaultdict(int)
    for t in transformations:
        type_counts[t['type']] += 1
    
    print("\nPer-operation speedup estimates:")
    total_ops = sum(type_counts.values())
    weighted_speedup = 0
    
    for ttype, count in type_counts.items():
        speedup = speedups.get(ttype, 1.0)
        weight = count / total_ops
        weighted_speedup += speedup * weight
        print(f"  {ttype}: {speedup:.1f}x speedup × {count} ops = {speedup * count:.1f}x total")
    
    print(f"\nEstimated overall impact:")
    print(f"  Operations affected: {total_ops}")
    print(f"  Average speedup: {weighted_speedup:.2f}x")
    print(f"  Note: Actual speedup depends on overall kernel performance")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mlir_passes/power_to_multiply.py <mlir_file>")
        print("\nExample:")
        print("  python mlir_passes/power_to_multiply.py mlir_output/graupel_stablehlo.mlir")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    # Read MLIR
    with open(filepath) as f:
        mlir_content = f.read()
    
    # Analyze
    exponent_values, exponent_sources = analyze_power_operations(mlir_content)
    
    # Generate plan
    transformations = generate_transformation_plan(exponent_values)
    
    # Estimate impact
    if transformations:
        estimate_speedup(transformations)
    else:
        print("\n" + "=" * 70)
        print("No Simple Constant Exponents Found")
        print("=" * 70)
        print("\nThe power operations use computed exponents, not simple constants.")
        print("This means the exponents are calculated at runtime.")
        print("\nOptimization strategy:")
        print("  1. Identify patterns in the JAX source code")
        print("  2. Look for x**2, x**3, x**0.5 in Python code")
        print("  3. Replace with x*x, x*x*x, jnp.sqrt(x) at source level")
        print("  4. XLA/StableHLO will then generate optimized code")
        print("\nRun: python mlir_passes/optimize_source_powers.py")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Review the transformation plan above")
    print("  2. Run: python mlir_passes/optimize_source_powers.py")
    print("  3. Apply optimizations to JAX source code")
    print("  4. Re-export and verify optimizations applied")
    print("=" * 70)
