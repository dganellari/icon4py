#!/usr/bin/env python3
"""
Generate fully unrolled StableHLO with TRANSPOSED memory layout.

Key optimization: Use tensor<nlev×ncells> instead of tensor<ncells×nlev>

Why this matters for GPU:
- GPU memory is coalesced along the LAST dimension
- With tensor<ncells×nlev>, slicing column k reads scattered memory (stride=nlev)
- With tensor<nlev×ncells>, slicing row k reads CONTIGUOUS memory (stride=1)

This should match DaCe/CUDA performance by ensuring coalesced memory access.

Usage:
    python generate_unrolled_transposed.py --ncells 327680 --nlev 90
"""

import argparse


def generate_unrolled_transposed(nlev: int = 90, ncells: int = 327680) -> str:
    """Generate fully unrolled StableHLO with nlev×ncells layout."""

    lines = []

    # Module header
    lines.append(f'module @jit_precip_unrolled_transposed_{nlev} attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{')
    lines.append('')
    lines.append('  // TRANSPOSED LAYOUT: tensor<nlev×ncells> for coalesced GPU memory access')
    lines.append('  // Each level slice reads contiguous memory')
    lines.append('')

    # Function signature - inputs are nlev×ncells (transposed)
    args = ', '.join([
        f'%arg0: tensor<{nlev}x{ncells}xi1>',   # kmin_r (transposed)
        f'%arg1: tensor<{nlev}x{ncells}xi1>',   # kmin_i
        f'%arg2: tensor<{nlev}x{ncells}xi1>',   # kmin_s
        f'%arg3: tensor<{nlev}x{ncells}xi1>',   # kmin_g
        f'%arg4: tensor<{nlev}x{ncells}xf64>',  # qv
        f'%arg5: tensor<{nlev}x{ncells}xf64>',  # qc
        f'%arg6: tensor<{nlev}x{ncells}xf64>',  # qr
        f'%arg7: tensor<{nlev}x{ncells}xf64>',  # qs
        f'%arg8: tensor<{nlev}x{ncells}xf64>',  # qi
        f'%arg9: tensor<{nlev}x{ncells}xf64>',  # qg
        f'%arg10: tensor<{nlev}x{ncells}xf64>', # t
        f'%arg11: tensor<{nlev}x{ncells}xf64>', # rho
        f'%arg12: tensor<{nlev}x{ncells}xf64>', # dz
    ])

    ret_types = ', '.join([f'tensor<{nlev}x{ncells}xf64>'] * 11)
    lines.append(f'  func.func public @main({args}) -> ({ret_types}) {{')

    # Constants
    lines.append('    // Scalar constants')
    lines.append('    %cst_0 = stablehlo.constant dense<0.0> : tensor<f64>')
    lines.append('    %cst_05 = stablehlo.constant dense<0.5> : tensor<f64>')
    lines.append('    %cst_1 = stablehlo.constant dense<1.0> : tensor<f64>')
    lines.append('    %cst_2 = stablehlo.constant dense<2.0> : tensor<f64>')
    lines.append('    %cst_30 = stablehlo.constant dense<30.0> : tensor<f64>')
    lines.append('    %cst_rho0 = stablehlo.constant dense<1.225> : tensor<f64>')
    lines.append('    %false = stablehlo.constant dense<false> : tensor<i1>')
    lines.append('')

    # Physics constants
    lines.append('    // Velocity coefficients [rain, snow, ice, graupel]')
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

    # Broadcast constants to full size (nlev×ncells)
    lines.append('    // Broadcast constants to full tensor size')
    lines.append(f'    %bcast_2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<{nlev}x{ncells}xf64>')
    lines.append(f'    %bcast_30 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> tensor<{nlev}x{ncells}xf64>')
    lines.append(f'    %bcast_rho0 = stablehlo.broadcast_in_dim %cst_rho0, dims = [] : (tensor<f64>) -> tensor<{nlev}x{ncells}xf64>')
    lines.append('')

    # Precompute zeta and rho_sqrt on full tensors
    lines.append('    // Precompute zeta = dt / (2 * dz)')
    lines.append(f'    %dz_times_2 = stablehlo.multiply %arg12, %bcast_2 : tensor<{nlev}x{ncells}xf64>')
    lines.append(f'    %zeta_full = stablehlo.divide %bcast_30, %dz_times_2 : tensor<{nlev}x{ncells}xf64>')
    lines.append('')
    lines.append('    // Precompute rho_sqrt = sqrt(rho0 / rho)')
    lines.append(f'    %rho_ratio = stablehlo.divide %bcast_rho0, %arg11 : tensor<{nlev}x{ncells}xf64>')
    lines.append(f'    %rho_sqrt = stablehlo.sqrt %rho_ratio : tensor<{nlev}x{ncells}xf64>')
    lines.append('')

    # Slice all inputs - TRANSPOSED: slice along first dimension (rows)
    # This gives CONTIGUOUS memory access on GPU!
    lines.append('    // ========== SLICE ALL INPUTS (CONTIGUOUS MEMORY) ==========')
    lines.append('    // Slicing [k:k+1, 0:ncells] reads contiguous memory on GPU')
    for k in range(nlev):
        lines.append(f'    // Level {k}')
        # Slice pattern: [k:k+1, 0:ncells] -> tensor<1×ncells>
        lines.append(f'    %kmin_r_{k} = stablehlo.slice %arg0 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xi1>) -> tensor<1x{ncells}xi1>')
        lines.append(f'    %kmin_s_{k} = stablehlo.slice %arg2 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xi1>) -> tensor<1x{ncells}xi1>')
        lines.append(f'    %kmin_i_{k} = stablehlo.slice %arg1 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xi1>) -> tensor<1x{ncells}xi1>')
        lines.append(f'    %kmin_g_{k} = stablehlo.slice %arg3 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xi1>) -> tensor<1x{ncells}xi1>')
        lines.append(f'    %qr_{k} = stablehlo.slice %arg6 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %qs_{k} = stablehlo.slice %arg7 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %qi_{k} = stablehlo.slice %arg8 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %qg_{k} = stablehlo.slice %arg9 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %rho_{k} = stablehlo.slice %arg11 [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %zeta_{k} = stablehlo.slice %zeta_full [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %rho_sqrt_{k} = stablehlo.slice %rho_sqrt [{k}:{k+1}, 0:{ncells}] : (tensor<{nlev}x{ncells}xf64>) -> tensor<1x{ncells}xf64>')
        lines.append('')

    # Broadcast 1D constants to 1×ncells
    lines.append('    // Broadcast constants to 1×ncells')
    lines.append(f'    %bcast_0_1d = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
    lines.append(f'    %bcast_05_1d = stablehlo.broadcast_in_dim %cst_05, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
    lines.append(f'    %bcast_1_1d = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
    lines.append(f'    %bcast_2_1d = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
    lines.append(f'    %false_1d = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<1x{ncells}xi1>')
    for sp in ['r', 's', 'i', 'g']:
        lines.append(f'    %bcast_vel_coeff_{sp} = stablehlo.broadcast_in_dim %vel_coeff_{sp}, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %bcast_vel_exp_{sp} = stablehlo.broadcast_in_dim %vel_exp_{sp}, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %bcast_qmin_{sp} = stablehlo.broadcast_in_dim %qmin_{sp}, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
    lines.append('')

    # Initial carry state
    lines.append('    // ========== INITIAL CARRY STATE ==========')
    for sp in ['r', 's', 'i', 'g']:
        lines.append(f'    %pflx_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
        lines.append(f'    %activated_{sp}_init = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<1x{ncells}xi1>')
        lines.append(f'    %rhox_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<1x{ncells}xf64>')
    lines.append('')

    # Unrolled computation for each level
    for k in range(nlev):
        lines.append(f'    // ========== LEVEL {k} ==========')

        prev = '_init' if k == 0 else f'_out_{k-1}'
        out = f'_out_{k}'

        for sp in ['r', 's', 'i', 'g']:
            lines.append(f'    // Species {sp}')
            lines.append(f'    %activated_{sp}{out} = stablehlo.or %activated_{sp}{prev}, %kmin_{sp}_{k} : tensor<1x{ncells}xi1>')
            lines.append(f'    %rho_x_{sp}_{k} = stablehlo.multiply %q{sp}_{k}, %rho_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %term1_{sp}_{k} = stablehlo.divide %rho_x_{sp}_{k}, %zeta_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %pflx_2_{sp}_{k} = stablehlo.multiply %pflx_{sp}{prev}, %bcast_2_1d : tensor<1x{ncells}xf64>')
            lines.append(f'    %flx_eff_{sp}_{k} = stablehlo.add %term1_{sp}_{k}, %pflx_2_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %rho_x_offset_{sp}_{k} = stablehlo.add %rho_x_{sp}_{k}, %bcast_qmin_{sp} : tensor<1x{ncells}xf64>')
            lines.append(f'    %rho_x_pow_{sp}_{k} = stablehlo.power %rho_x_offset_{sp}_{k}, %bcast_vel_exp_{sp} : tensor<1x{ncells}xf64>')
            lines.append(f'    %fall_speed_{sp}_{k} = stablehlo.multiply %bcast_vel_coeff_{sp}, %rho_x_pow_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %flux_raw_{sp}_{k} = stablehlo.multiply %rho_x_{sp}_{k}, %rho_sqrt_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %flux_scaled_{sp}_{k} = stablehlo.multiply %flux_raw_{sp}_{k}, %fall_speed_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %flx_partial_{sp}_{k} = stablehlo.minimum %flux_scaled_{sp}_{k}, %flx_eff_{sp}_{k} : tensor<1x{ncells}xf64>')

            # Terminal velocity from previous rhox
            lines.append(f'    %rhox_prev_offset_{sp}_{k} = stablehlo.add %rhox_{sp}{prev}, %bcast_qmin_{sp} : tensor<1x{ncells}xf64>')
            lines.append(f'    %rhox_prev_pow_{sp}_{k} = stablehlo.power %rhox_prev_offset_{sp}_{k}, %bcast_vel_exp_{sp} : tensor<1x{ncells}xf64>')
            lines.append(f'    %vc_vel_{sp}_{k} = stablehlo.multiply %rho_sqrt_{k}, %bcast_vel_coeff_{sp} : tensor<1x{ncells}xf64>')
            lines.append(f'    %vt_active_{sp}_{k} = stablehlo.multiply %vc_vel_{sp}_{k}, %rhox_prev_pow_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %vt_{sp}_{k} = stablehlo.select %activated_{sp}{prev}, %vt_active_{sp}_{k}, %bcast_0_1d : tensor<1x{ncells}xi1>, tensor<1x{ncells}xf64>')

            # q_activated calculation
            lines.append(f'    %flx_diff_{sp}_{k} = stablehlo.subtract %flx_eff_{sp}_{k}, %flx_partial_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %num_{sp}_{k} = stablehlo.multiply %zeta_{k}, %flx_diff_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %zeta_vt_{sp}_{k} = stablehlo.multiply %zeta_{k}, %vt_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %denom_inner_{sp}_{k} = stablehlo.add %zeta_vt_{sp}_{k}, %bcast_1_1d : tensor<1x{ncells}xf64>')
            lines.append(f'    %denom_{sp}_{k} = stablehlo.multiply %denom_inner_{sp}_{k}, %rho_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %q_activated_{sp}_{k} = stablehlo.divide %num_{sp}_{k}, %denom_{sp}_{k} : tensor<1x{ncells}xf64>')

            # Flux calculation
            lines.append(f'    %q_rho_{sp}_{k} = stablehlo.multiply %q_activated_{sp}_{k}, %rho_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %q_rho_vt_{sp}_{k} = stablehlo.multiply %q_rho_{sp}_{k}, %vt_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %flx_sum_{sp}_{k} = stablehlo.add %q_rho_vt_{sp}_{k}, %flx_partial_{sp}_{k} : tensor<1x{ncells}xf64>')
            lines.append(f'    %flx_activated_{sp}_{k} = stablehlo.multiply %flx_sum_{sp}_{k}, %bcast_05_1d : tensor<1x{ncells}xf64>')

            # Select based on activation
            lines.append(f'    %q{sp}{out} = stablehlo.select %activated_{sp}{out}, %q_activated_{sp}_{k}, %q{sp}_{k} : tensor<1x{ncells}xi1>, tensor<1x{ncells}xf64>')
            lines.append(f'    %pflx_{sp}{out} = stablehlo.select %activated_{sp}{out}, %flx_activated_{sp}_{k}, %bcast_0_1d : tensor<1x{ncells}xi1>, tensor<1x{ncells}xf64>')
            lines.append(f'    %rhox_{sp}{out} = stablehlo.multiply %q{sp}{out}, %rho_{k} : tensor<1x{ncells}xf64>')
            lines.append('')

    # Concatenate outputs along first dimension (stack rows)
    lines.append('    // ========== CONCATENATE OUTPUTS (STACK ROWS) ==========')

    type_1d = f'tensor<1x{ncells}xf64>'
    type_tuple = ', '.join([type_1d] * nlev)

    for name, prefix in [('qr_out_full', 'qr_out_'), ('qs_out_full', 'qs_out_'),
                         ('qi_out_full', 'qi_out_'), ('qg_out_full', 'qg_out_')]:
        args_list = ', '.join([f'%{prefix}{k}' for k in range(nlev)])
        lines.append(f'    %{name} = stablehlo.concatenate {args_list}, dim = 0 : ({type_tuple}) -> tensor<{nlev}x{ncells}xf64>')

    for name, prefix in [('pflx_r_full', 'pflx_r_out_'), ('pflx_s_full', 'pflx_s_out_'),
                         ('pflx_i_full', 'pflx_i_out_'), ('pflx_g_full', 'pflx_g_out_')]:
        args_list = ', '.join([f'%{prefix}{k}' for k in range(nlev)])
        lines.append(f'    %{name} = stablehlo.concatenate {args_list}, dim = 0 : ({type_tuple}) -> tensor<{nlev}x{ncells}xf64>')

    # Return
    lines.append('')
    lines.append(f'    return %arg4, %arg5, %qr_out_full, %qs_out_full, %qi_out_full, %qg_out_full, %arg10, %pflx_r_full, %pflx_s_full, %pflx_i_full, %pflx_g_full : {ret_types}')
    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate unrolled StableHLO with transposed layout")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"shlo/precip_effect_transposed_{args.ncells}x{args.nlev}.stablehlo"

    print(f"Generating transposed unrolled StableHLO...")
    print(f"  Layout: tensor<{args.nlev}×{args.ncells}> (nlev×ncells)")
    print(f"  This ensures coalesced GPU memory access")

    stablehlo_text = generate_unrolled_transposed(args.nlev, args.ncells)

    with open(args.output, 'w') as f:
        f.write(stablehlo_text)

    print(f"Written to: {args.output}")
    print(f"File size: {len(stablehlo_text) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
