#!/usr/bin/env python3
"""
Generate optimized StableHLO for q_t_update with better kernel fusion.

The issue: JAX's q_t_update generates StableHLO with many separate operations
(~27 power, ~7 exp, ~169 multiply, etc.) that XLA compiles to ~80 separate
CUDA kernels, each with ~0.3ms launch overhead = ~24ms total.

Optimization strategy:
1. Replace expensive power(x, n) with multiplication chains where n is small integer
2. Combine repeated subexpressions (CSE - common subexpression elimination)
3. Use log/exp instead of power where appropriate: x^n = exp(n * log(x))
4. Group operations into larger fused blocks

The goal is to reduce from ~24ms to ~5ms (or better).

Usage:
    python generate_optimized_q_t_update.py --input stablehlo/q_t_update_transposed.stablehlo \\
        --output stablehlo/q_t_update_optimized.stablehlo
"""

import argparse
import re
from pathlib import Path


def analyze_stablehlo(text: str) -> dict:
    """Analyze StableHLO to understand the operation breakdown."""
    return {
        'total_chars': len(text),
        'lines': text.count('\n'),
        'power_ops': text.count('stablehlo.power'),
        'exp_ops': text.count('stablehlo.exponential'),
        'log_ops': text.count('stablehlo.log'),
        'multiply_ops': text.count('stablehlo.multiply'),
        'divide_ops': text.count('stablehlo.divide'),
        'add_ops': text.count('stablehlo.add'),
        'subtract_ops': text.count('stablehlo.subtract'),
        'compare_ops': text.count('stablehlo.compare'),
        'select_ops': text.count('stablehlo.select'),
        'maximum_ops': text.count('stablehlo.maximum'),
        'minimum_ops': text.count('stablehlo.minimum'),
        'broadcast_ops': text.count('stablehlo.broadcast'),
        'func_calls': text.count('call @'),
        'while_loops': text.count('stablehlo.while'),
    }


def print_analysis(name: str, stats: dict):
    """Print analysis summary."""
    print(f"\n{name}:")
    print(f"  Size: {stats['total_chars']:,} chars, {stats['lines']} lines")
    print(f"  Arithmetic: {stats['multiply_ops']} mul, {stats['add_ops']} add, {stats['divide_ops']} div, {stats['subtract_ops']} sub")
    print(f"  Transcendental: {stats['power_ops']} pow, {stats['exp_ops']} exp, {stats['log_ops']} log")
    print(f"  Comparison: {stats['compare_ops']} cmp, {stats['select_ops']} select")
    print(f"  Min/Max: {stats['maximum_ops']} max, {stats['minimum_ops']} min")
    print(f"  Broadcast: {stats['broadcast_ops']}")
    print(f"  Function calls: {stats['func_calls']}")
    print(f"  While loops: {stats['while_loops']}")


def optimize_power_ops(text: str) -> str:
    """
    Optimize power operations.

    Strategy:
    1. x^2 -> x * x
    2. x^3 -> x * x * x
    3. x^0.5 -> sqrt(x)
    4. x^(-0.5) -> 1/sqrt(x)
    5. x^(-1) -> 1/x
    6. x^n where n is small integer -> multiplication chain
    7. x^n where n is arbitrary -> exp(n * log(x))

    Note: We can't easily do this transformation at StableHLO level because
    the power operation already exists. The optimization would need to be
    done at a higher level (JAX source) or via MLIR passes.
    """
    # For now, just return the text unchanged - we'll document what
    # optimizations would be beneficial
    return text


def inline_helper_functions(text: str) -> str:
    """
    Inline the helper _where functions to reduce function call overhead.

    The StableHLO has 4 helper functions (_where, _where_0, _where_1, _where_2)
    that are called 88 times. Inlining them eliminates call overhead.
    """
    # Parse the helper functions
    # Pattern: func.func private @_where... { ... }

    # This is complex to do correctly in text form. Would need proper MLIR parsing.
    # For now, document this as an optimization opportunity.
    return text


def add_fusion_hints(text: str) -> str:
    """
    Add fusion boundary hints to encourage XLA to fuse operations.

    Note: StableHLO doesn't have explicit fusion hints. This optimization
    would need to be done via:
    1. XLA_FLAGS environment variable
    2. Custom XLA passes
    3. Restructuring the computation graph
    """
    return text


def optimize_q_t_update(input_text: str) -> str:
    """
    Apply all optimizations to q_t_update StableHLO.

    Current optimizations:
    1. Analyze the structure
    2. (Future) Inline helper functions
    3. (Future) Replace power ops with multiplication chains
    4. (Future) Add fusion hints

    The main optimization opportunity is at the JAX source level:
    - Replace jnp.power(x, 2) with x * x
    - Replace jnp.power(x, 3) with x * x * x
    - Use lax.select instead of jnp.where for better fusion
    """

    # For now, the optimization is limited at the StableHLO level
    # The real optimization needs to happen in the JAX source code

    text = input_text

    # Add a header comment explaining the optimization
    header = """// OPTIMIZED Q_T_UPDATE StableHLO
//
// Optimization opportunities identified:
// 1. Power operations: 27 stablehlo.power ops
//    - Many could be replaced with multiplication chains
//    - x^2 -> x*x, x^3 -> x*x*x, etc.
//    - This should be done in JAX source, not here
//
// 2. Helper function calls: 88 calls to _where variants
//    - Could be inlined to reduce call overhead
//
// 3. Broadcast operations: Many scalars broadcast to tensors
//    - XLA should handle these efficiently
//
// 4. No while loops (good!) - purely element-wise
//
// To achieve better performance:
// - Modify muphys_jax/core/transitions.py to use x*x instead of power(x, 2)
// - Modify muphys_jax/core/properties.py similarly
// - Consider using lax.select directly instead of jnp.where
//
"""

    # Insert header after module declaration
    lines = text.split('\n')
    output_lines = []
    header_inserted = False

    for line in lines:
        output_lines.append(line)
        if not header_inserted and line.startswith('module @'):
            output_lines.append('')
            for header_line in header.strip().split('\n'):
                output_lines.append('  ' + header_line)
            output_lines.append('')
            header_inserted = True

    return '\n'.join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized StableHLO for q_t_update"
    )
    parser.add_argument("--input", "-i", required=True, help="Input StableHLO file")
    parser.add_argument("--output", "-o", required=True, help="Output StableHLO file")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze, don't generate output")
    args = parser.parse_args()

    print("=" * 70)
    print("Q_T_UPDATE StableHLO Optimizer")
    print("=" * 70)

    # Load input
    input_path = Path(args.input)
    print(f"\nLoading: {input_path}")

    with open(input_path, 'r') as f:
        input_text = f.read()

    # Analyze input
    input_stats = analyze_stablehlo(input_text)
    print_analysis("Input analysis", input_stats)

    if args.analyze_only:
        print("\n(Analysis only mode - no output generated)")
        return

    # Optimize
    print("\n" + "=" * 70)
    print("Applying optimizations...")
    print("=" * 70)

    output_text = optimize_q_t_update(input_text)

    # Analyze output
    output_stats = analyze_stablehlo(output_text)
    print_analysis("Output analysis", output_stats)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(output_text)

    print(f"\n✓ Saved optimized StableHLO to: {output_path}")
    print(f"  Input size:  {input_stats['total_chars']:,} chars")
    print(f"  Output size: {output_stats['total_chars']:,} chars")

    # Print recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR BETTER PERFORMANCE")
    print("=" * 70)
    print("""
The main optimization opportunity is at the JAX SOURCE level, not StableHLO.

1. In muphys_jax/core/transitions.py:
   - Replace: jnp.power(x, 2) -> x * x
   - Replace: jnp.power(x, 0.5) -> jnp.sqrt(x)
   - Replace: jnp.power(x, 0.16667) -> custom cube root

2. In muphys_jax/core/properties.py:
   - Similar power operation replacements
   - Use lax.select instead of jnp.where where possible

3. In JAX source (general):
   - Minimize number of operations
   - Group related computations
   - Avoid repeated computation of same values

4. XLA flags to try:
   - XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
   - XLA_FLAGS="--xla_gpu_enable_fast_min_max=true"

The current StableHLO is already element-wise with no loops.
The bottleneck is XLA kernel launch overhead (~0.3ms × ~80 kernels = 24ms).
""")


if __name__ == "__main__":
    main()
