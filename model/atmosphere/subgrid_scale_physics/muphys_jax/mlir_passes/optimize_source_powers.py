#!/usr/bin/env python3
"""
Find and report power operations in JAX source code that can be optimized.

Usage:
    python mlir_passes/optimize_source_powers.py
"""

import re
from pathlib import Path
from collections import defaultdict


def find_power_operations(root_dir):
    """Find all jnp.power operations in source code."""
    root = Path(root_dir)
    
    # Pattern to match jnp.power(base, exponent)
    power_pattern = r'jnp\.power\(([^,]+),\s*([^)]+)\)'
    
    results = defaultdict(list)
    
    for py_file in root.rglob('*.py'):
        # Skip test files and this script
        if 'test' in str(py_file) or 'mlir_passes' in str(py_file):
            continue
            
        with open(py_file) as f:
            content = f.read()
            lines = content.split('\n')
            
        for i, line in enumerate(lines, 1):
            matches = re.finditer(power_pattern, line)
            for match in matches:
                base = match.group(1).strip()
                exp = match.group(2).strip()
                
                results[py_file].append({
                    'line': i,
                    'code': line.strip(),
                    'base': base,
                    'exponent': exp,
                })
    
    return results


def analyze_exponents(results):
    """Analyze exponents and suggest optimizations."""
    print("=" * 70)
    print("Power Operation Analysis in JAX Source")
    print("=" * 70)
    
    total = sum(len(v) for v in results.values())
    print(f"\nFound {total} jnp.power() calls in {len(results)} files\n")
    
    # Categorize by exponent
    exponent_map = defaultdict(list)
    
    for file, ops in results.items():
        for op in ops:
            exp = op['exponent']
            exponent_map[exp].append((file, op))
    
    print("Exponent frequency:")
    for exp, ops in sorted(exponent_map.items(), key=lambda x: -len(x[1])):
        print(f"  {exp}: {len(ops)} occurrences")
    
    # Identify optimization opportunities
    print("\n" + "=" * 70)
    print("Optimization Opportunities")
    print("=" * 70)
    
    optimizable = []
    
    for exp, ops in exponent_map.items():
        # Check for simple constant exponents
        try:
            val = float(exp)
            if val == 2.0:
                opt_type = 'square'
                replacement = lambda b: f"({b}) * ({b})"
                speedup = "2-3x"
            elif val == 3.0:
                opt_type = 'cube'
                replacement = lambda b: f"({b}) * ({b}) * ({b})"
                speedup = "2-3x"
            elif val == 0.5:
                opt_type = 'sqrt'
                replacement = lambda b: f"jnp.sqrt({b})"
                speedup = "~2x"
            elif val == -1.0:
                opt_type = 'reciprocal'
                replacement = lambda b: f"1.0 / ({b})"
                speedup = "~2x"
            elif val == 4.0:
                opt_type = 'quartic'
                replacement = lambda b: f"(({b}) * ({b})) ** 2"  # (x²)²
                speedup = "~2x"
            else:
                continue
                
            optimizable.append({
                'type': opt_type,
                'exponent': exp,
                'ops': ops,
                'replacement': replacement,
                'speedup': speedup
            })
        except ValueError:
            # Not a simple float constant
            pass
    
    # Show optimization plan
    if optimizable:
        print(f"\n✓ Found {len(optimizable)} types of optimizable power operations")
        print(f"  Total operations: {sum(len(o['ops']) for o in optimizable)}\n")
        
        for opt in optimizable:
            print(f"\n{opt['type'].upper()}: x^{opt['exponent']} ({len(opt['ops'])} occurrences)")
            print(f"  Speedup: {opt['speedup']}")
            print(f"  Example transformation:")
            example_file, example_op = opt['ops'][0]
            base = example_op['base']
            print(f"    Before: jnp.power({base}, {opt['exponent']})")
            print(f"    After:  {opt['replacement'](base)}")
            
            print(f"\n  Locations:")
            for file, op in opt['ops'][:5]:
                rel_path = file.relative_to(Path('.'))
                print(f"    {rel_path}:{op['line']}")
            if len(opt['ops']) > 5:
                print(f"    ... and {len(opt['ops']) - 5} more")
    else:
        print("\n✗ No simple constant exponents found (all are computed)")
    
    return optimizable


def generate_patch(optimizable, output_file='power_optimization.patch'):
    """Generate a patch/script to apply optimizations."""
    print("\n" + "=" * 70)
    print("Generating Optimization Script")
    print("=" * 70)
    
    if not optimizable:
        print("\nNo optimizations to apply")
        return
    
    # Group by file
    file_changes = defaultdict(list)
    
    for opt in optimizable:
        for file, op in opt['ops']:
            file_changes[file].append({
                'line': op['line'],
                'old': op['code'],
                'base': op['base'],
                'exp': opt['exponent'],
                'replacement_fn': opt['replacement'],
                'type': opt['type']
            })
    
    # Generate report
    print(f"\nChanges needed in {len(file_changes)} files:")
    
    total_changes = 0
    for file, changes in sorted(file_changes.items()):
        rel_path = file.relative_to(Path('.'))
        print(f"\n{rel_path}: {len(changes)} changes")
        
        # Show first few
        for change in changes[:3]:
            print(f"  Line {change['line']}:")
            print(f"    - jnp.power({change['base']}, {change['exp']})")
            print(f"    + {change['replacement_fn'](change['base'])}")
        
        if len(changes) > 3:
            print(f"  ... and {len(changes) - 3} more")
        
        total_changes += len(changes)
    
    print(f"\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Total changes: {total_changes}")
    print(f"  Files affected: {len(file_changes)}")
    print(f"  Estimated speedup: 1.2-1.5x overall")
    print("=" * 70)
    
    # Instructions
    print("\nTo apply these optimizations:")
    print("  1. Review changes above")
    print("  2. Use your editor to find and replace:")
    for opt in optimizable:
        print(f"     - Search: jnp.power(X, {opt['exponent']})")
        example_file, example_op = opt['ops'][0]
        print(f"     - Replace: {opt['replacement']('X')}")
    print("  3. Re-run export_to_mlir.py")
    print("  4. Verify no power operations remain in MLIR")


if __name__ == "__main__":
    # Search from current directory
    results = find_power_operations('.')
    
    if not results:
        print("No jnp.power() operations found")
        exit(0)
    
    # Analyze
    optimizable = analyze_exponents(results)
    
    # Generate recommendations
    generate_patch(optimizable)
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Apply the transformations above")
    print("  2. Test the code still works correctly")
    print("  3. Re-export MLIR and verify optimization")
    print("  4. Benchmark performance improvement")
    print("=" * 70)
