#!/usr/bin/env python3
"""
Generate fused StableHLO that avoids per-level slicing.

Key insight: Instead of slicing into 90 separate tensors, we:
1. Compute all level-local operations on full tensor<ncells×nlev>
2. Use cumulative/scan operations for dependencies
3. Let XLA fuse everything into minimal kernels

This should match CUDA/Triton performance by:
- Avoiding 990 slice operations
- Enabling full memory coalescing
- Allowing XLA to generate a single fused kernel
"""

import argparse


def generate_fused_stablehlo(nlev: int = 90, ncells: int = 327680) -> str:
    """Generate fused StableHLO without per-level slicing.

    The key difference from the unrolled version:
    - All operations work on full tensor<ncells×nlev>
    - We express the scan dependency using slice+pad+concat for shifting
    - XLA can fuse all operations into a single kernel
    """

    lines = []

    # Module header
    lines.append(f'module @jit_precip_effect_fused attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{')
    lines.append('')
    lines.append('  // FUSED VERSION: No per-level slicing')
    lines.append('  // All operations on full tensor<ncells×nlev>')
    lines.append('  // Scan dependency expressed via slice+pad shifting')
    lines.append('')

    # Function signature - same as original
    args = ', '.join([
        f'%arg0: tensor<{ncells}x{nlev}xi1>',   # kmin_r
        f'%arg1: tensor<{ncells}x{nlev}xi1>',   # kmin_i
        f'%arg2: tensor<{ncells}x{nlev}xi1>',   # kmin_s
        f'%arg3: tensor<{ncells}x{nlev}xi1>',   # kmin_g
        f'%arg4: tensor<{ncells}x{nlev}xf64>',  # qv
        f'%arg5: tensor<{ncells}x{nlev}xf64>',  # qc
        f'%arg6: tensor<{ncells}x{nlev}xf64>',  # qr
        f'%arg7: tensor<{ncells}x{nlev}xf64>',  # qs
        f'%arg8: tensor<{ncells}x{nlev}xf64>',  # qi
        f'%arg9: tensor<{ncells}x{nlev}xf64>',  # qg
        f'%arg10: tensor<{ncells}x{nlev}xf64>', # t
        f'%arg11: tensor<{ncells}x{nlev}xf64>', # rho
        f'%arg12: tensor<{ncells}x{nlev}xf64>', # dz
    ])

    ret_types = ', '.join([f'tensor<{ncells}x{nlev}xf64>'] * 11)
    lines.append(f'  func.func public @main({args}) -> ({ret_types}) {{')

    # Constants
    lines.append('    // Scalar constants')
    lines.append('    %cst_0 = stablehlo.constant dense<0.0> : tensor<f64>')
    lines.append('    %cst_05 = stablehlo.constant dense<0.5> : tensor<f64>')
    lines.append('    %cst_1 = stablehlo.constant dense<1.0> : tensor<f64>')
    lines.append('    %cst_2 = stablehlo.constant dense<2.0> : tensor<f64>')
    lines.append('    %dt = stablehlo.constant dense<30.0> : tensor<f64>')
    lines.append('    %rho0 = stablehlo.constant dense<1.225> : tensor<f64>')
    lines.append('    %false = stablehlo.constant dense<false> : tensor<i1>')
    lines.append('')

    # Physics constants per species
    lines.append('    // Physics constants per species [r, s, i, g]')
    lines.append('    %vel_coeff_r = stablehlo.constant dense<14.58> : tensor<f64>')
    lines.append('    %vel_coeff_s = stablehlo.constant dense<57.8> : tensor<f64>')
    lines.append('    %vel_coeff_i = stablehlo.constant dense<1.25> : tensor<f64>')
    lines.append('    %vel_coeff_g = stablehlo.constant dense<12.24> : tensor<f64>')
    lines.append('    %vel_exp_r = stablehlo.constant dense<0.111> : tensor<f64>')
    lines.append('    %vel_exp_s = stablehlo.constant dense<0.16666666666666666> : tensor<f64>')
    lines.append('    %vel_exp_i = stablehlo.constant dense<0.16> : tensor<f64>')
    lines.append('    %vel_exp_g = stablehlo.constant dense<0.217> : tensor<f64>')
    lines.append('    %qmin_r = stablehlo.constant dense<1.0e-12> : tensor<f64>')
    lines.append('    %qmin_s = stablehlo.constant dense<1.0e-12> : tensor<f64>')
    lines.append('    %qmin_i = stablehlo.constant dense<1.0e-12> : tensor<f64>')
    lines.append('    %qmin_g = stablehlo.constant dense<1.0e-08> : tensor<f64>')
    lines.append('')

    # Broadcast all constants to full tensor size
    lines.append('    // Broadcast constants to full tensor size')
    for name in ['cst_0', 'cst_05', 'cst_1', 'cst_2', 'dt', 'rho0']:
        lines.append(f'    %{name}_full = stablehlo.broadcast_in_dim %{name}, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %false_full = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<{ncells}x{nlev}xi1>')
    for sp in ['r', 's', 'i', 'g']:
        lines.append(f'    %vel_coeff_{sp}_full = stablehlo.broadcast_in_dim %vel_coeff_{sp}, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %vel_exp_{sp}_full = stablehlo.broadcast_in_dim %vel_exp_{sp}, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %qmin_{sp}_full = stablehlo.broadcast_in_dim %qmin_{sp}, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
    lines.append('')

    # Precompute level-local values (no dependencies between levels)
    lines.append('    // ========== LEVEL-LOCAL COMPUTATIONS (parallelizable) ==========')
    lines.append('    // zeta = dt / (2 * dz)')
    lines.append(f'    %dz_times_2 = stablehlo.multiply %arg12, %cst_2_full : tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %zeta = stablehlo.divide %dt_full, %dz_times_2 : tensor<{ncells}x{nlev}xf64>')
    lines.append('')
    lines.append('    // rho_sqrt = sqrt(rho0 / rho)')
    lines.append(f'    %rho_ratio = stablehlo.divide %rho0_full, %arg11 : tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %rho_sqrt = stablehlo.sqrt %rho_ratio : tensor<{ncells}x{nlev}xf64>')
    lines.append('')

    # For each species, compute level-local terms
    for sp, sp_idx in [('r', 6), ('s', 7), ('i', 8), ('g', 9)]:
        q_arg = f'%arg{sp_idx}'
        kmin_arg = f'%arg{0 if sp == "r" else 1 if sp == "i" else 2 if sp == "s" else 3}'

        lines.append(f'    // Species {sp}: level-local terms')
        lines.append(f'    %rho_x_{sp} = stablehlo.multiply {q_arg}, %arg11 : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %term1_{sp} = stablehlo.divide %rho_x_{sp}, %zeta : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %rho_x_offset_{sp} = stablehlo.add %rho_x_{sp}, %qmin_{sp}_full : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %rho_x_pow_{sp} = stablehlo.power %rho_x_offset_{sp}, %vel_exp_{sp}_full : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %fall_speed_{sp} = stablehlo.multiply %vel_coeff_{sp}_full, %rho_x_pow_{sp} : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %vc_{sp} = stablehlo.multiply %rho_sqrt, %vel_coeff_{sp}_full : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %flux_raw_{sp} = stablehlo.multiply %rho_x_{sp}, %rho_sqrt : tensor<{ncells}x{nlev}xf64>')
        lines.append(f'    %flux_scaled_{sp} = stablehlo.multiply %flux_raw_{sp}, %fall_speed_{sp} : tensor<{ncells}x{nlev}xf64>')
        lines.append('')

    # Now we need to express the scan. The challenge is:
    # - pflx[k] depends on pflx[k-1]
    # - activated[k] depends on activated[k-1]
    #
    # Option 1: Unroll the dependency explicitly (current approach)
    # Option 2: Use a while loop (original JAX approach)
    # Option 3: Express as cumulative sum/product patterns
    #
    # For Option 3, the activation is: activated[k] = OR(activated[k-1], kmin[k])
    # This is a cumulative OR starting from false, which equals:
    #   activated[k] = ANY(kmin[0:k+1])
    #
    # We can compute this as: cummax along axis=1 of kmin (treating bool as 0/1)

    lines.append('    // ========== SCAN DEPENDENCIES (cumulative operations) ==========')
    lines.append('    // activated[k] = cumulative OR of kmin[0:k+1]')
    lines.append('    // This can be expressed as cummax of the boolean mask')
    lines.append('')

    # For activation: use a while loop to compute cumulative OR
    # Actually, let's use the XLA reduce_window approach
    lines.append('    // Unfortunately StableHLO does not have native cumulative ops.')
    lines.append('    // We must either:')
    lines.append('    // 1. Use while loop (creates D2D copies)')
    lines.append('    // 2. Unroll (current approach, many slices)')
    lines.append('    // 3. Use custom reduce_window with identity kernel')
    lines.append('')
    lines.append('    // For now, return pass-through as placeholder')
    lines.append('    // Real implementation needs proper scan handling')
    lines.append('')

    # Placeholder return
    lines.append(f'    return %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %cst_0_full, %cst_0_full, %cst_0_full, %cst_0_full : {ret_types}')
    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate fused StableHLO (experimental)")
    parser.add_argument("-o", "--output", default="shlo/precip_effect_fused.stablehlo")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENTAL: Fused StableHLO Generator")
    print("=" * 60)
    print()
    print("Challenge: StableHLO lacks native cumulative/scan operations")
    print("Options to express k→k+1 dependencies:")
    print("  1. while loop (D2D copies)")
    print("  2. unrolled slices (990 slice ops)")
    print("  3. reduce_window tricks (complex)")
    print()

    stablehlo_text = generate_fused_stablehlo(args.nlev, args.ncells)

    with open(args.output, 'w') as f:
        f.write(stablehlo_text)

    print(f"Written to: {args.output}")
    print(f"File size: {len(stablehlo_text) / 1024:.1f} KB")
    print()
    print("NOTE: This is a placeholder. The core challenge is expressing")
    print("the scan dependency without while loops or per-level slicing.")


if __name__ == "__main__":
    main()
