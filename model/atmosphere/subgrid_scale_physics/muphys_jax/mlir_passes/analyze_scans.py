#!/usr/bin/env python3
"""
Analyze vertical scan patterns in StableHLO MLIR.

Identifies scan structure, carry state, and fusion opportunities.

Usage:
    python mlir_passes/analyze_scans.py mlir_output/graupel_stablehlo.mlir
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def extract_while_loops(mlir_content):
    """Extract all while loop (scan) operations."""
    print("=" * 70)
    print("Scan Structure Analysis")
    print("=" * 70)
    
    # Find while loops
    while_pattern = r'(%\w+(?::\d+)?) = stablehlo\.while\(([^)]+)\) : ([^{]+)\s*cond \{([^}]+)\} do \{([^}]+)\}'
    
    # More robust: find while statements and extract their blocks
    while_starts = []
    for match in re.finditer(r'(%\w+(?::\d+)?) = stablehlo\.while\(', mlir_content):
        while_starts.append(match.start())
    
    print(f"\nFound {len(while_starts)} while loops")
    
    scans = []
    for i, start in enumerate(while_starts):
        print(f"\n{'='*70}")
        print(f"Scan #{i+1} (position {start})")
        print(f"{'='*70}")
        
        # Extract context
        context_start = max(0, start - 500)
        context_end = min(len(mlir_content), start + 3000)
        context = mlir_content[context_start:context_end]
        
        # Parse while signature
        sig_match = re.search(
            r'(%\w+(?::\d+)?) = stablehlo\.while\(([^)]+)\) : ([^\n]+)',
            context
        )
        
        if sig_match:
            result, args, types = sig_match.groups()
            
            # Count arguments
            arg_list = [a.strip() for a in args.split(',')]
            type_list = types.strip().split(',')
            
            print(f"\nCarry State:")
            print(f"  Result: {result}")
            print(f"  Arguments: {len(arg_list)}")
            print(f"  Types: {len(type_list)}")
            
            # Identify the iterator
            iterator_pattern = r'%iterArg_\d+ = %c_\d+'
            if re.search(iterator_pattern, args):
                print(f"  ✓ Found iterator variable")
            
            # Extract types
            tensor_types = defaultdict(int)
            for t in type_list:
                t = t.strip()
                if 'tensor' in t:
                    # Extract shape
                    shape_match = re.search(r'tensor<([^>]+)>', t)
                    if shape_match:
                        shape = shape_match.group(1)
                        tensor_types[shape] += 1
            
            print(f"\n  Tensor shapes in carry:")
            for shape, count in sorted(tensor_types.items(), key=lambda x: -x[1]):
                print(f"    {shape}: {count} tensors")
                if '90x' in shape or 'x90' in shape:
                    print(f"      → Likely 90-level vertical dimension")
        
        # Find condition block
        cond_match = re.search(r'cond \{([^}]+)\}', context)
        if cond_match:
            cond_body = cond_match.group(1)
            print(f"\n  Condition:")
            # Extract iteration limit
            limit_match = re.search(r'constant dense<(\d+)>', cond_body)
            if limit_match:
                limit = limit_match.group(1)
                print(f"    Iteration limit: {limit}")
                if limit == '90':
                    print(f"    ✓ Confirmed: 90-level vertical scan")
        
        # Find do block and analyze operations
        do_start = context.find('do {')
        if do_start >= 0:
            # Find matching brace (simplified)
            do_end = context.find('stablehlo.return', do_start)
            if do_end >= 0:
                do_body = context[do_start:do_end+100]
                
                print(f"\n  Loop Body Operations:")
                ops = defaultdict(int)
                for op in ['dynamic_slice', 'dynamic_update_slice', 'reshape', 
                          'broadcast_in_dim', 'func.call', 'add', 'multiply']:
                    count = do_body.count(f'stablehlo.{op}')
                    if count > 0:
                        ops[op] = count
                
                for op, count in sorted(ops.items(), key=lambda x: -x[1]):
                    print(f"    {op}: {count}")
                
                # Check for function calls
                call_match = re.search(r'func\.call @(\w+)', do_body)
                if call_match:
                    func_name = call_match.group(1)
                    print(f"\n  ✓ Calls function: @{func_name}")
                    
                    # Count function args
                    args_match = re.search(r'func\.call @\w+\(([^)]+)\)', do_body)
                    if args_match:
                        args_str = args_match.group(1)
                        arg_count = args_str.count(',') + 1
                        print(f"    Arguments: {arg_count}")
        
        scans.append({
            'index': i + 1,
            'position': start,
            'result': result if sig_match else None,
            'arg_count': len(arg_list) if sig_match else 0,
        })
    
    return scans


def analyze_scan_dependencies(mlir_content, scans):
    """Check if scans can be fused based on dependencies."""
    print("\n" + "=" * 70)
    print("Scan Fusion Analysis")
    print("=" * 70)
    
    if len(scans) < 2:
        print("\nOnly one scan found - no fusion opportunities")
        return
    
    print(f"\nAnalyzing {len(scans)} scans for fusion opportunities...")
    
    # For each pair of scans, check if second scan uses outputs of first
    for i in range(len(scans) - 1):
        scan1 = scans[i]
        scan2 = scans[i + 1]
        
        print(f"\nScan #{scan1['index']} → Scan #{scan2['index']}:")
        
        # Extract the region between scans
        gap_start = scan1['position']
        gap_end = scan2['position']
        gap = mlir_content[gap_start:gap_end]
        
        # Check if scan1 outputs are used in scan2 inputs
        if scan1['result']:
            result_base = scan1['result'].split(':')[0]
            # Check for uses like %1092#14
            uses = re.findall(rf'{re.escape(result_base)}#\d+', gap)
            
            if uses:
                print(f"  Dependencies found: {len(uses)} uses of scan1 outputs")
                print(f"  Examples: {uses[:3]}")
                print(f"  ✗ CANNOT FUSE: Scan2 depends on Scan1")
            else:
                print(f"  No dependencies found")
                print(f"  ✓ COULD POTENTIALLY FUSE")
                print(f"  → Would reduce kernel launches and memory traffic")


def analyze_memory_patterns(mlir_content, scans):
    """Analyze memory access patterns in scans."""
    print("\n" + "=" * 70)
    print("Memory Access Pattern Analysis")
    print("=" * 70)
    
    for scan in scans:
        print(f"\nScan #{scan['index']}:")
        
        # Extract scan region
        start = scan['position']
        end = min(len(mlir_content), start + 3000)
        region = mlir_content[start:end]
        
        # Count memory operations
        dynamic_slices = region.count('dynamic_slice')
        dynamic_updates = region.count('dynamic_update_slice')
        broadcasts = region.count('broadcast_in_dim')
        
        print(f"  dynamic_slice: {dynamic_slices} (reads from 3D tensors)")
        print(f"  dynamic_update_slice: {dynamic_updates} (writes to 3D tensors)")
        print(f"  broadcast_in_dim: {broadcasts} (expand dimensions)")
        
        print(f"\n  Pattern:")
        print(f"    1. Slice level k from input tensors (90×...)")
        print(f"    2. Process 2D slice")
        print(f"    3. Broadcast results back to 3D")
        print(f"    4. Update output tensors at level k")
        
        print(f"\n  Memory Traffic per Iteration:")
        print(f"    Read: {dynamic_slices} slices")
        print(f"    Write: {dynamic_updates} slices")
        print(f"    Total: ~{(dynamic_slices + dynamic_updates) * 1024 * 8} bytes")
        print(f"           (assuming 1024 cells × f64)")


def suggest_optimizations(scans):
    """Suggest optimizations based on scan analysis."""
    print("\n" + "=" * 70)
    print("Optimization Recommendations")
    print("=" * 70)
    
    print("\n1. VERTICAL FUSION:")
    print("   Instead of 90 iterations × 2 scans = 180 kernel launches")
    print("   → Fuse into 1 scan = 90 kernel launches")
    print("   Savings: ~50% kernel launch overhead")
    
    print("\n2. MEMORY ACCESS:")
    print("   Current: Read all input slices, write all output slices")
    print("   → Pre-transpose tensors to (cells, species, levels)")
    print("   → Process columns directly without slicing")
    print("   Savings: ~30-40% memory bandwidth")
    
    print("\n3. CUSTOM KERNEL:")
    print("   JAX scan is general-purpose")
    print("   → Write specialized GPU kernel for vertical column")
    print("   → Process all levels in single thread block")
    print("   Savings: ~2-3x speedup")
    
    print("\n4. PRECISION:")
    print("   All operations use f64")
    print("   → Check if f32 is sufficient for physics")
    print("   → Mixed precision: f32 compute, f64 accumulate")
    print("   Savings: ~2x memory bandwidth, ~1.5x compute")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mlir_passes/analyze_scans.py <mlir_file>")
        print("\nExample:")
        print("  python mlir_passes/analyze_scans.py mlir_output/graupel_stablehlo.mlir")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    # Read MLIR
    with open(filepath) as f:
        mlir_content = f.read()
    
    # Extract and analyze scans
    scans = extract_while_loops(mlir_content)
    
    # Analyze dependencies
    analyze_scan_dependencies(mlir_content, scans)
    
    # Analyze memory patterns
    analyze_memory_patterns(mlir_content, scans)
    
    # Suggest optimizations
    suggest_optimizations(scans)
    
    print("\n" + "=" * 70)
