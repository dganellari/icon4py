#!/usr/bin/env python3
"""
Generate fully unrolled StableHLO for the precipitation_effects kernel.

This mirrors generate_unrolled_hlo.py but outputs StableHLO (MLIR) format
instead of XLA HLO format.

The unrolling eliminates:
- While loop overhead
- Dynamic slice/update operations
- Tuple passing between iterations
"""

import argparse


def generate_unrolled_stablehlo(nlev: int = 90, ncells: int = 20480) -> str:
    """Generate fully unrolled StableHLO."""

    lines = []

    # Module header
    lines.append(f'module @jit_precip_effect_unrolled_{nlev} attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{')
    lines.append('')
    lines.append('  // FULLY UNROLLED: All levels computed with static slicing')
    lines.append('  // No while loop, no dynamic indexing')
    lines.append('')

    # Function signature
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
    lines.append('    // Constants')
    lines.append('    %cst_0 = stablehlo.constant dense<0.0> : tensor<f64>')
    lines.append('    %cst_05 = stablehlo.constant dense<0.5> : tensor<f64>')
    lines.append('    %cst_1 = stablehlo.constant dense<1.0> : tensor<f64>')
    lines.append('    %cst_2 = stablehlo.constant dense<2.0> : tensor<f64>')
    lines.append('    %cst_30 = stablehlo.constant dense<30.0> : tensor<f64>')
    lines.append('    %cst_rho0 = stablehlo.constant dense<1.225> : tensor<f64>')
    lines.append('    %false = stablehlo.constant dense<false> : tensor<i1>')
    lines.append('')

    # Velocity coefficients
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

    # Broadcast constants to full size
    lines.append('    // Broadcast constants')
    lines.append(f'    %bcast_2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %bcast_30 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %bcast_rho0 = stablehlo.broadcast_in_dim %cst_rho0, dims = [] : (tensor<f64>) -> tensor<{ncells}x{nlev}xf64>')
    lines.append('')

    # Precompute zeta and rho_sqrt
    lines.append('    // Precompute zeta = dt / (2 * dz)')
    lines.append(f'    %dz_times_2 = stablehlo.multiply %arg12, %bcast_2 : tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %zeta_full = stablehlo.divide %bcast_30, %dz_times_2 : tensor<{ncells}x{nlev}xf64>')
    lines.append('')
    lines.append('    // Precompute rho_sqrt = sqrt(rho0 / rho)')
    lines.append(f'    %rho_ratio = stablehlo.divide %bcast_rho0, %arg11 : tensor<{ncells}x{nlev}xf64>')
    lines.append(f'    %rho_sqrt = stablehlo.sqrt %rho_ratio : tensor<{ncells}x{nlev}xf64>')
    lines.append('')

    # Slice all inputs for each level
    lines.append('    // ========== SLICE ALL INPUTS (STATIC) ==========')
    for k in range(nlev):
        lines.append(f'    // Level {k}')
        lines.append(f'    %kmin_r_{k} = stablehlo.slice %arg0 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xi1>) -> tensor<{ncells}x1xi1>')
        lines.append(f'    %kmin_s_{k} = stablehlo.slice %arg2 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xi1>) -> tensor<{ncells}x1xi1>')
        lines.append(f'    %kmin_i_{k} = stablehlo.slice %arg1 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xi1>) -> tensor<{ncells}x1xi1>')
        lines.append(f'    %kmin_g_{k} = stablehlo.slice %arg3 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xi1>) -> tensor<{ncells}x1xi1>')
        lines.append(f'    %qr_{k} = stablehlo.slice %arg6 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %qs_{k} = stablehlo.slice %arg7 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %qi_{k} = stablehlo.slice %arg8 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %qg_{k} = stablehlo.slice %arg9 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %rho_{k} = stablehlo.slice %arg11 [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %zeta_{k} = stablehlo.slice %zeta_full [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %rho_sqrt_{k} = stablehlo.slice %rho_sqrt [0:{ncells}, {k}:{k+1}] : (tensor<{ncells}x{nlev}xf64>) -> tensor<{ncells}x1xf64>')
        lines.append('')

    # Broadcast 1D constants
    lines.append('    // Broadcast 1D constants')
    lines.append(f'    %bcast_0_1d = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
    lines.append(f'    %bcast_05_1d = stablehlo.broadcast_in_dim %cst_05, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
    lines.append(f'    %bcast_1_1d = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
    lines.append(f'    %bcast_2_1d = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
    lines.append(f'    %false_1d = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<{ncells}x1xi1>')
    for sp in ['r', 's', 'i', 'g']:
        lines.append(f'    %bcast_vel_coeff_{sp} = stablehlo.broadcast_in_dim %vel_coeff_{sp}, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %bcast_vel_exp_{sp} = stablehlo.broadcast_in_dim %vel_exp_{sp}, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %bcast_qmin_{sp} = stablehlo.broadcast_in_dim %qmin_{sp}, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
    lines.append('')

    # Initial carry state
    lines.append('    // ========== INITIAL CARRY STATE ==========')
    for sp in ['r', 's', 'i', 'g']:
        lines.append(f'    %pflx_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
        lines.append(f'    %activated_{sp}_init = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<{ncells}x1xi1>')
        lines.append(f'    %rhox_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<{ncells}x1xf64>')
    lines.append('')

    # Unrolled computation for each level
    for k in range(nlev):
        lines.append(f'    // ========== LEVEL {k} ==========')

        prev = '_init' if k == 0 else f'_out_{k-1}'
        out = f'_out_{k}'

        for sp in ['r', 's', 'i', 'g']:
            lines.append(f'    // Species {sp}')
            lines.append(f'    %activated_{sp}{out} = stablehlo.or %activated_{sp}{prev}, %kmin_{sp}_{k} : tensor<{ncells}x1xi1>')
            lines.append(f'    %rho_x_{sp}_{k} = stablehlo.multiply %q{sp}_{k}, %rho_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %term1_{sp}_{k} = stablehlo.divide %rho_x_{sp}_{k}, %zeta_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %pflx_2_{sp}_{k} = stablehlo.multiply %pflx_{sp}{prev}, %bcast_2_1d : tensor<{ncells}x1xf64>')
            lines.append(f'    %flx_eff_{sp}_{k} = stablehlo.add %term1_{sp}_{k}, %pflx_2_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %rho_x_offset_{sp}_{k} = stablehlo.add %rho_x_{sp}_{k}, %bcast_qmin_{sp} : tensor<{ncells}x1xf64>')
            lines.append(f'    %rho_x_pow_{sp}_{k} = stablehlo.power %rho_x_offset_{sp}_{k}, %bcast_vel_exp_{sp} : tensor<{ncells}x1xf64>')
            lines.append(f'    %fall_speed_{sp}_{k} = stablehlo.multiply %bcast_vel_coeff_{sp}, %rho_x_pow_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %flux_raw_{sp}_{k} = stablehlo.multiply %rho_x_{sp}_{k}, %rho_sqrt_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %flux_scaled_{sp}_{k} = stablehlo.multiply %flux_raw_{sp}_{k}, %fall_speed_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %flx_partial_{sp}_{k} = stablehlo.minimum %flux_scaled_{sp}_{k}, %flx_eff_{sp}_{k} : tensor<{ncells}x1xf64>')

            # Terminal velocity from previous rhox
            lines.append(f'    %rhox_prev_offset_{sp}_{k} = stablehlo.add %rhox_{sp}{prev}, %bcast_qmin_{sp} : tensor<{ncells}x1xf64>')
            lines.append(f'    %rhox_prev_pow_{sp}_{k} = stablehlo.power %rhox_prev_offset_{sp}_{k}, %bcast_vel_exp_{sp} : tensor<{ncells}x1xf64>')
            lines.append(f'    %vc_vel_{sp}_{k} = stablehlo.multiply %rho_sqrt_{k}, %bcast_vel_coeff_{sp} : tensor<{ncells}x1xf64>')
            lines.append(f'    %vt_active_{sp}_{k} = stablehlo.multiply %vc_vel_{sp}_{k}, %rhox_prev_pow_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %vt_{sp}_{k} = stablehlo.select %activated_{sp}{prev}, %vt_active_{sp}_{k}, %bcast_0_1d : tensor<{ncells}x1xi1>, tensor<{ncells}x1xf64>')

            # q_activated calculation
            lines.append(f'    %flx_diff_{sp}_{k} = stablehlo.subtract %flx_eff_{sp}_{k}, %flx_partial_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %num_{sp}_{k} = stablehlo.multiply %zeta_{k}, %flx_diff_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %zeta_vt_{sp}_{k} = stablehlo.multiply %zeta_{k}, %vt_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %denom_inner_{sp}_{k} = stablehlo.add %zeta_vt_{sp}_{k}, %bcast_1_1d : tensor<{ncells}x1xf64>')
            lines.append(f'    %denom_{sp}_{k} = stablehlo.multiply %denom_inner_{sp}_{k}, %rho_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %q_activated_{sp}_{k} = stablehlo.divide %num_{sp}_{k}, %denom_{sp}_{k} : tensor<{ncells}x1xf64>')

            # Flux calculation
            lines.append(f'    %q_rho_{sp}_{k} = stablehlo.multiply %q_activated_{sp}_{k}, %rho_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %q_rho_vt_{sp}_{k} = stablehlo.multiply %q_rho_{sp}_{k}, %vt_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %flx_sum_{sp}_{k} = stablehlo.add %q_rho_vt_{sp}_{k}, %flx_partial_{sp}_{k} : tensor<{ncells}x1xf64>')
            lines.append(f'    %flx_activated_{sp}_{k} = stablehlo.multiply %flx_sum_{sp}_{k}, %bcast_05_1d : tensor<{ncells}x1xf64>')

            # Select based on activation
            lines.append(f'    %q{sp}{out} = stablehlo.select %activated_{sp}{out}, %q_activated_{sp}_{k}, %q{sp}_{k} : tensor<{ncells}x1xi1>, tensor<{ncells}x1xf64>')
            lines.append(f'    %pflx_{sp}{out} = stablehlo.select %activated_{sp}{out}, %flx_activated_{sp}_{k}, %bcast_0_1d : tensor<{ncells}x1xi1>, tensor<{ncells}x1xf64>')
            lines.append(f'    %rhox_{sp}{out} = stablehlo.multiply %q{sp}{out}, %rho_{k} : tensor<{ncells}x1xf64>')
            lines.append('')

    # Concatenate outputs
    lines.append('    // ========== CONCATENATE OUTPUTS ==========')

    # Build proper type tuple string (90 types separated by commas)
    type_1d = f'tensor<{ncells}x1xf64>'
    type_tuple = ', '.join([type_1d] * nlev)

    for name, prefix in [('qr_out_full', 'qr_out_'), ('qs_out_full', 'qs_out_'),
                         ('qi_out_full', 'qi_out_'), ('qg_out_full', 'qg_out_')]:
        args = ', '.join([f'%{prefix}{k}' for k in range(nlev)])
        lines.append(f'    %{name} = stablehlo.concatenate {args}, dim = 1 : ({type_tuple}) -> tensor<{ncells}x{nlev}xf64>')

    for name, prefix in [('pflx_r_full', 'pflx_r_out_'), ('pflx_s_full', 'pflx_s_out_'),
                         ('pflx_i_full', 'pflx_i_out_'), ('pflx_g_full', 'pflx_g_out_')]:
        args = ', '.join([f'%{prefix}{k}' for k in range(nlev)])
        lines.append(f'    %{name} = stablehlo.concatenate {args}, dim = 1 : ({type_tuple}) -> tensor<{ncells}x{nlev}xf64>')

    # Return (simplified - just return q outputs and pass-through others)
    lines.append('')
    lines.append(f'    return %arg4, %arg5, %qr_out_full, %qs_out_full, %qi_out_full, %qg_out_full, %arg10, %pflx_r_full, %pflx_s_full, %pflx_i_full, %pflx_g_full : {ret_types}')
    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate fully unrolled StableHLO")
    parser.add_argument("-o", "--output", default="shlo/precip_effect_x64_unrolled.stablehlo")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=20480)
    args = parser.parse_args()

    print(f"Generating unrolled StableHLO with {args.nlev} levels...")
    stablehlo_text = generate_unrolled_stablehlo(args.nlev, args.ncells)

    with open(args.output, 'w') as f:
        f.write(stablehlo_text)

    print(f"Written to: {args.output}")
    print(f"File size: {len(stablehlo_text) / 1024:.1f} KB")

    # Check for while loops
    if 'stablehlo.while' in stablehlo_text:
        print("WARNING: StableHLO contains while loops!")
    else:
        print("SUCCESS: No while loops (fully unrolled)")


if __name__ == "__main__":
    main()
