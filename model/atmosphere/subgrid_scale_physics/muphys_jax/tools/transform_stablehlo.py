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
from dataclasses import dataclass


@dataclass
class WhileLoopInfo:
    """Information about a single while loop."""
    loop_id: int
    start_line: int
    end_line: int
    cond_start: int
    cond_end: int
    body_start: int
    body_end: int
    num_iterations: int
    result_name: str
    iter_args: List[str]
    iter_types: List[str]
    dynamic_slices: List[Tuple[int, str]]
    dynamic_updates: List[Tuple[int, str]]
    func_calls: List[str]


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


def find_all_while_loops(lines: List[str]) -> List[WhileLoopInfo]:
    """Find all while loops in the StableHLO IR."""
    loops = []
    i = 0
    loop_id = 0

    while i < len(lines):
        line = lines[i]

        if 'stablehlo.while' in line:
            loop_info = parse_single_while_loop(lines, i, loop_id)
            if loop_info:
                loops.append(loop_info)
                i = loop_info.end_line + 1
                loop_id += 1
            else:
                i += 1
        else:
            i += 1

    return loops


def parse_single_while_loop(lines: List[str], start_idx: int, loop_id: int) -> Optional[WhileLoopInfo]:
    """Parse a single while loop starting at the given line."""
    line = lines[start_idx]
    if 'stablehlo.while' not in line:
        return None

    # Parse result name (e.g., %138:16)
    result_match = re.match(r'\s*(%\d+:\d+|\%\w+)\s*=', line)
    result_name = result_match.group(1) if result_match else "unknown"

    # Find the loop structure
    idx = start_idx
    cond_start = cond_end = body_start = body_end = -1
    brace_depth = 0

    # Find cond block
    while idx < len(lines):
        if 'cond {' in lines[idx]:
            cond_start = idx
            break
        idx += 1

    if cond_start == -1:
        return None

    # Find end of cond block (} do {)
    idx = cond_start + 1
    while idx < len(lines):
        if '} do {' in lines[idx]:
            cond_end = idx
            body_start = idx
            break
        idx += 1

    if body_start == -1:
        return None

    # Find end of body block
    idx = body_start + 1
    brace_depth = 1
    while idx < len(lines) and brace_depth > 0:
        line = lines[idx]
        brace_depth += line.count('{') - line.count('}')
        if brace_depth == 0:
            body_end = idx
            break
        idx += 1

    if body_end == -1:
        return None

    # Extract iteration count from cond block
    num_iterations = 90  # Default
    for i in range(cond_start, cond_end + 1):
        match = re.search(r'constant\s+dense<(\d+)>', lines[i])
        if match:
            val = int(match.group(1))
            if val > 1:
                num_iterations = val
                break

    # Find dynamic_slice and dynamic_update_slice in body
    dynamic_slices = []
    dynamic_updates = []
    func_calls = []

    for i in range(body_start, body_end + 1):
        body_line = lines[i]
        if 'stablehlo.dynamic_slice' in body_line and 'dynamic_update' not in body_line:
            dynamic_slices.append((i, body_line.strip()))
        elif 'stablehlo.dynamic_update_slice' in body_line:
            dynamic_updates.append((i, body_line.strip()))
        elif 'func.call' in body_line:
            func_calls.append(body_line.strip())

    # Parse iteration arguments (simplified)
    iter_args = []
    iter_types = []

    return WhileLoopInfo(
        loop_id=loop_id,
        start_line=start_idx,
        end_line=body_end,
        cond_start=cond_start,
        cond_end=cond_end,
        body_start=body_start,
        body_end=body_end,
        num_iterations=num_iterations,
        result_name=result_name,
        iter_args=iter_args,
        iter_types=iter_types,
        dynamic_slices=dynamic_slices,
        dynamic_updates=dynamic_updates,
        func_calls=func_calls,
    )


def analyze_stablehlo_ir(input_file: str) -> Dict:
    """Analyze StableHLO IR and print detailed report."""
    print(f"Reading StableHLO IR from {input_file}")

    with open(input_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    print(f"  File size: {len(content)} bytes ({len(content) / 1024:.1f} KB)")
    print(f"  Total lines: {len(lines)}")

    # Count operations
    ops = {
        'while': content.count('stablehlo.while'),
        'dynamic_slice': content.count('stablehlo.dynamic_slice'),
        'dynamic_update_slice': content.count('stablehlo.dynamic_update_slice'),
        'slice': content.count('stablehlo.slice'),
        'add': content.count('stablehlo.add'),
        'multiply': content.count('stablehlo.multiply'),
        'divide': content.count('stablehlo.divide'),
        'subtract': content.count('stablehlo.subtract'),
        'power': content.count('stablehlo.power'),
        'exponential': content.count('stablehlo.exponential'),
        'constant': content.count('stablehlo.constant'),
        'broadcast_in_dim': content.count('stablehlo.broadcast_in_dim'),
        'transpose': content.count('stablehlo.transpose'),
        'reshape': content.count('stablehlo.reshape'),
        'concatenate': content.count('stablehlo.concatenate'),
        'select': content.count('stablehlo.select'),
        'compare': content.count('stablehlo.compare'),
        'func.call': content.count('func.call'),
    }

    print("\n" + "=" * 80)
    print("OPERATION COUNTS")
    print("=" * 80)
    for op, count in sorted(ops.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {op}: {count}")

    total_d2d = ops['dynamic_slice'] + ops['dynamic_update_slice']
    print(f"\n  TOTAL D2D OPERATIONS: {total_d2d}")

    # Find and analyze all while loops
    loops = find_all_while_loops(lines)

    print("\n" + "=" * 80)
    print(f"WHILE LOOP ANALYSIS ({len(loops)} loops found)")
    print("=" * 80)

    for loop in loops:
        print(f"\nLoop {loop.loop_id + 1}:")
        print(f"  Location: lines {loop.start_line + 1} - {loop.end_line + 1}")
        print(f"  Iterations: {loop.num_iterations}")
        print(f"  Result: {loop.result_name}")
        print(f"  Body lines: {loop.body_end - loop.body_start}")
        print(f"  Dynamic slices: {len(loop.dynamic_slices)}")
        print(f"  Dynamic updates: {len(loop.dynamic_updates)}")
        print(f"  Function calls: {len(loop.func_calls)}")

        if loop.dynamic_slices:
            print(f"  First dynamic_slice: {loop.dynamic_slices[0][1][:80]}...")
        if loop.dynamic_updates:
            print(f"  First dynamic_update: {loop.dynamic_updates[0][1][:80]}...")
        if loop.func_calls:
            print(f"  Function called: {loop.func_calls[0][:80]}...")

    # D2D analysis
    print("\n" + "=" * 80)
    print("D2D COPY ANALYSIS")
    print("=" * 80)

    total_d2d_per_iter = 0
    for loop in loops:
        d2d_per_iter = len(loop.dynamic_slices) + len(loop.dynamic_updates)
        total_d2d_per_iter += d2d_per_iter * loop.num_iterations
        print(f"  Loop {loop.loop_id + 1}: {d2d_per_iter} D2D ops/iter x {loop.num_iterations} iters = {d2d_per_iter * loop.num_iterations} total")

    print(f"\n  TOTAL D2D OPS ACROSS ALL ITERATIONS: {total_d2d_per_iter}")

    # Extract tensor shapes
    print("\n" + "=" * 80)
    print("TENSOR SHAPES")
    print("=" * 80)

    shape_pattern = r'tensor<([^>]+)>'
    shapes = {}
    for match in re.finditer(shape_pattern, content):
        shape = match.group(1)
        shapes[shape] = shapes.get(shape, 0) + 1

    # Print most common shapes
    sorted_shapes = sorted(shapes.items(), key=lambda x: -x[1])[:20]
    for shape, count in sorted_shapes:
        print(f"  tensor<{shape}>: {count} occurrences")

    return {
        'ops': ops,
        'loops': loops,
        'total_d2d': total_d2d,
        'content': content,
        'lines': lines,
    }


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
    Generate unrolled StableHLO IR with valid SSA form.

    Key rules for valid SSA:
    1. Each %var can only be defined once
    2. Use before def is invalid
    3. iterArg variables must be replaced with actual values per iteration
    """
    nlev, ncells = input_shape

    output_lines = []
    output_lines.append(f"    // Unrolled scan: {loop_bound} iterations")
    output_lines.append(f"    // D2D operations eliminated via static slicing")

    # Global SSA counter for unique names
    ssa_counter = [100]  # Use list for closure mutation

    def next_ssa():
        ssa_counter[0] += 1
        return f"%v{ssa_counter[0]}"

    # Track carry variables: maps iterArg name -> current SSA value
    # Initialize with function args (matching the while loop signature)
    carry_map = {
        '%iterArg': '%arg2',      # input tensor
        '%iterArg_0': None,       # loop counter - will be constant k
        '%iterArg_1': '%arg0',    # carry a
        '%iterArg_2': '%arg1',    # carry b
        '%iterArg_3': None,       # output a accumulator
        '%iterArg_4': None,       # output b accumulator
    }

    # Initialize output accumulators
    output_lines.append(f"    %out_init_a = stablehlo.constant dense<0.000000e+00> : tensor<{nlev}x{ncells}xf32>")
    output_lines.append(f"    %out_init_b = stablehlo.constant dense<0.000000e+00> : tensor<{nlev}x{ncells}xf32>")
    carry_map['%iterArg_3'] = '%out_init_a'
    carry_map['%iterArg_4'] = '%out_init_b'

    # Track outputs per iteration for final concatenation
    iter_outputs_a = []
    iter_outputs_b = []

    for k in range(loop_bound):
        output_lines.append(f"    // === Iteration {k} ===")

        # Map for this iteration: original var -> new unique SSA
        iter_map = {}

        # Loop counter is now constant k
        carry_map['%iterArg_0'] = f'%c_k{k}'
        output_lines.append(f"    %c_k{k} = stablehlo.constant dense<{k}> : tensor<i32>")

        for line in body_lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            # Skip loop counter increment (we handle it)
            if 'stablehlo.add %iterArg_0' in line or 'add %iterArg_0' in line:
                continue

            # Skip stablehlo.return
            if line.startswith('stablehlo.return'):
                continue

            # Parse the operation
            # Format: %result = op args : types
            assign_match = re.match(r'(%\w+)\s*=\s*(.+)', line)
            if not assign_match:
                continue

            result_var = assign_match.group(1)
            rest = assign_match.group(2)

            # Generate new unique SSA name for result
            new_result = next_ssa()
            iter_map[result_var] = new_result

            # Replace all variable references
            def replace_refs(text):
                def replacer(m):
                    var = m.group(0)
                    # Check iter_map first (this iteration's defs)
                    if var in iter_map:
                        return iter_map[var]
                    # Check carry_map (cross-iteration state)
                    if var in carry_map and carry_map[var]:
                        return carry_map[var]
                    # Function args stay as-is
                    if var.startswith('%arg'):
                        return var
                    # Unknown - keep as is (will cause error if truly undefined)
                    return var
                return re.sub(r'%\w+', replacer, text)

            # Handle dynamic_slice -> static slice
            if 'stablehlo.dynamic_slice' in rest:
                # Parse: stablehlo.dynamic_slice %input, %idx1, %idx2, sizes = [s1, s2] : (types) -> result_type
                ds_match = re.match(r'stablehlo\.dynamic_slice\s+(%\w+),\s*[^,]+,\s*[^,]+,\s*sizes\s*=\s*\[([^\]]+)\]\s*:\s*\(([^)]+)\)\s*->\s*(.+)', rest)
                if ds_match:
                    input_tensor = ds_match.group(1)
                    sizes = ds_match.group(2)
                    input_type = ds_match.group(3).split(',')[0].strip()
                    result_type = ds_match.group(4).strip()

                    mapped_input = replace_refs(input_tensor)
                    size_list = [s.strip() for s in sizes.split(',')]

                    # stablehlo.slice syntax: [start:limit:stride] per dim
                    # For 2D: [k:k+1, 0:ncells]
                    start_indices = [str(k), '0']
                    limit_indices = [str(k + int(size_list[0])), size_list[1]]

                    output_lines.append(
                        f"    {new_result} = stablehlo.slice {mapped_input} "
                        f"[{start_indices[0]}:{limit_indices[0]}, {start_indices[1]}:{limit_indices[1]}] : "
                        f"{input_type} -> {result_type}"
                    )
                    continue

            # Handle dynamic_update_slice -> track for output construction
            if 'stablehlo.dynamic_update_slice' in rest:
                # Extract the update value (2nd operand)
                dus_match = re.match(r'stablehlo\.dynamic_update_slice\s+(%\w+),\s*(%\w+),', rest)
                if dus_match:
                    update_val = replace_refs(dus_match.group(2))
                    # Track which output this is (iterArg_3 or iterArg_4)
                    if result_var == '%34' or 'iterArg_3' in line:
                        iter_outputs_a.append((k, update_val))
                    else:
                        iter_outputs_b.append((k, update_val))
                    # Update carry map to point to accumulated output
                    # (simplified - in real impl would build the tensor)
                output_lines.append(f"    // D2D write eliminated: {result_var} at index {k}")
                continue

            # All other ops: replace refs and emit
            transformed_rest = replace_refs(rest)
            output_lines.append(f"    {new_result} ={transformed_rest}")

        # Update carry variables for next iteration
        # %18 is new carry_a, %25 is new carry_b in the simple scan
        if '%18' in iter_map:
            carry_map['%iterArg_1'] = iter_map['%18']
        if '%25' in iter_map:
            carry_map['%iterArg_2'] = iter_map['%25']

        output_lines.append("")

    # Final return
    final_a = carry_map.get('%iterArg_1', '%arg0')
    final_b = carry_map.get('%iterArg_2', '%arg1')
    output_lines.append(f"    // Final carry values")
    output_lines.append(f"    // carry_a = {final_a}, carry_b = {final_b}")
    output_lines.append(f"    return {final_a}, {final_b} : tensor<{ncells}xf32>, tensor<{ncells}xf32>")

    return "\n".join(output_lines)


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
        f.write(unrolled)
        f.write("\n  }\n")
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


def generate_python_unrolled_impl():
    """Generate a Python/JAX unrolled implementation."""
    code = '''#!/usr/bin/env python3
"""
Unrolled precipitation_effects implementation without lax.scan.

This eliminates the while loops that cause D2D copies by manually
unrolling the scan operations into explicit Python for loops.

The key insight is that Python for loops become static sequences
of operations in JAX's traced representation, avoiding the need
for XLA's while loop construct which uses dynamic indexing.

Usage:
    from precipitation_unrolled import precipitation_effects_unrolled
"""

import jax
import jax.numpy as jnp
from functools import partial

# Number of vertical levels - must be known at trace time
NLEV = 90


def precip_scan_body(carry, x):
    """Single iteration of precipitation scan - extracted from lax.scan."""
    pflx, precip, summed, rate, active = carry
    alpha, beta, qmin, dt_dz, rho_factor, q, rho, kmin = x

    new_active = active | kmin

    # Compute sedimentation flux
    q_rho = q * rho
    flux_in = pflx / dt_dz
    available = flux_in + 2.0 * precip

    q_plus_qmin = q_rho + qmin
    q_power = jnp.power(q_plus_qmin, beta)
    sed_flux = alpha * q_power * rho_factor * q_rho
    sed_flux = jnp.minimum(sed_flux, available)

    # Update state
    new_pflx = jnp.where(new_active, sed_flux * dt_dz, pflx)
    new_precip = jnp.where(new_active, sed_flux * 0.5, precip)

    new_carry = (new_pflx, new_precip, summed, rate, new_active)
    outputs = (new_pflx, new_precip)

    return new_carry, outputs


def temp_scan_body(carry, x):
    """Single iteration of temperature scan - extracted from lax.scan."""
    eflx, active = carry
    t, t_next, cp_factor, pflx, ice_flux, qv, qc, qi, rho, dz, dt, kmin = x

    new_active = active | kmin

    # Temperature adjustment from precipitation
    # ... (simplified - actual computation is more complex)
    new_t = jnp.where(new_active, t + cp_factor * pflx, t)
    new_eflx = jnp.where(new_active, eflx + pflx * dz, eflx)

    new_carry = (new_eflx, new_active)
    outputs = (new_t, new_eflx)

    return new_carry, outputs


def precipitation_effects_unrolled(
    last_lev, kmin_r, kmin_i, kmin_s, kmin_g,
    q, t, rho, dz, dt
):
    """
    Unrolled version of precipitation_effects.

    Instead of using lax.scan which creates while loops,
    this uses Python for loops which unroll into static
    sequences of operations at trace time.
    """
    ncells = t.shape[0]

    # Prepare inputs as before (same setup code)
    # ... (initialization code for alpha, beta, qmin, etc.)

    # Initialize carries
    pflx = jnp.zeros((4, ncells))
    precip = jnp.zeros((4, ncells))
    summed = jnp.zeros(ncells)
    rate = jnp.zeros((4, ncells))
    active = jnp.zeros((4, ncells), dtype=bool)

    # UNROLLED PRECIPITATION SCAN
    # Instead of: carry, outputs = lax.scan(precip_scan_body, init_carry, xs)
    # We manually unroll all NLEV iterations

    out_pflx_list = []
    out_precip_list = []

    carry = (pflx, precip, summed, rate, active)

    for k in range(NLEV):
        # Get slice for this level (this becomes a static slice)
        x_k = (
            alpha[:, k],
            beta[:, k],
            qmin[:, k],
            dt_dz[k],
            rho_factor[:, k],
            q_stacked[:, k, :],
            rho[k],
            kmin_stacked[:, k, :],
        )

        carry, (pflx_k, precip_k) = precip_scan_body(carry, x_k)
        out_pflx_list.append(pflx_k)
        out_precip_list.append(precip_k)

    # Stack outputs
    out_pflx = jnp.stack(out_pflx_list, axis=1)
    out_precip = jnp.stack(out_precip_list, axis=1)

    # Similar unrolling for temperature scan...
    # ... (omitted for brevity)

    return qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx


@partial(jax.jit, static_argnames=['last_lev'])
def precipitation_effects_unrolled_jit(
    last_lev, kmin_r, kmin_i, kmin_s, kmin_g,
    q, t, rho, dz, dt
):
    """JIT-compiled entry point."""
    return precipitation_effects_unrolled(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g,
        q, t, rho, dz, dt
    )
'''
    return code


def main():
    parser = argparse.ArgumentParser(
        description="Transform StableHLO IR to eliminate D2D copies by unrolling while loops"
    )
    parser.add_argument('input_file', nargs='?', help='Input StableHLO MLIR file')
    parser.add_argument('output_file', nargs='?', help='Output transformed MLIR file')
    parser.add_argument('--input', '-i', dest='input_alt', help='Input StableHLO file (alternative)')
    parser.add_argument('--output', '-o', dest='output_alt', help='Output file (alternative)')
    parser.add_argument('--loop-bound', type=int, dest='loop_bound',
                       help='Manually specify loop bound (required if auto-detection fails)')
    parser.add_argument('--analyze-only', '-a', action='store_true', dest='analyze_only',
                       help='Only analyze the IR, do not transform')
    parser.add_argument('--input-shape', type=str, dest='input_shape', default='90,20480',
                       help='Input tensor shape as "nlev,ncells" (default: 90,20480)')
    parser.add_argument('--generate-python', action='store_true',
                       help='Generate Python unrolled implementation')

    args = parser.parse_args()

    # Handle alternative input argument
    input_file = args.input_file or args.input_alt
    output_file = args.output_file or args.output_alt

    print("=" * 80)
    print("StableHLO Transformation: Eliminate D2D Copies")
    print("=" * 80)
    print()

    if args.generate_python:
        print("Generating Python unrolled implementation...")
        print()
        print(generate_python_unrolled_impl())
        return

    if input_file is None:
        parser.print_help()
        sys.exit(1)

    if args.analyze_only:
        analyze_stablehlo_ir(input_file)
        return

    if output_file is None:
        base = os.path.splitext(input_file)[0]
        # Use .stablehlo extension for hlo-opt compatibility
        output_file = f"{base}_unrolled.stablehlo"

    # Parse input shape
    try:
        nlev, ncells = map(int, args.input_shape.split(','))
    except ValueError:
        print(f"ERROR: Invalid input shape '{args.input_shape}'. Use format 'nlev,ncells'")
        sys.exit(1)

    success = transform_stablehlo_ir(
        input_file,
        output_file,
        manual_bound=args.loop_bound,
        input_shape=(nlev, ncells)
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
