#!/usr/bin/env python3
"""
Generate fully unrolled HLO for the precipitation_effects kernel.

This completely unrolls the 90-iteration loop, eliminating:
- While loop overhead
- Dynamic slice/update operations
- Tuple passing between iterations

The resulting HLO is large but should be faster due to:
- Static memory access patterns
- Better instruction scheduling
- No loop control overhead
"""

import argparse


def generate_unrolled_hlo(nlev: int = 90, ncells: int = 20480) -> str:
    """Generate fully unrolled HLO."""

    lines = []

    # Module header
    lines.append(f"HloModule jit_precip_effect_unrolled_{nlev}, entry_computation_layout={{(pred[{ncells},{nlev}]{{1,0}}, pred[{ncells},{nlev}]{{1,0}}, pred[{ncells},{nlev}]{{1,0}}, pred[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, /*index=5*/f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, /*index=10*/f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}})->(f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{0,1}}, /*index=5*/f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, /*index=10*/f64[{ncells},{nlev}]{{0,1}})}}")
    lines.append("")
    lines.append("// FULLY UNROLLED: All 90 levels computed with static slicing")
    lines.append("// No while loop, no dynamic indexing")
    lines.append("")

    # Entry computation
    lines.append("ENTRY main.unrolled {")
    lines.append("  // Input parameters")
    lines.append(f"  Arg_kmin_r = pred[{ncells},{nlev}]{{1,0}} parameter(0)")
    lines.append(f"  Arg_kmin_i = pred[{ncells},{nlev}]{{1,0}} parameter(1)")
    lines.append(f"  Arg_kmin_s = pred[{ncells},{nlev}]{{1,0}} parameter(2)")
    lines.append(f"  Arg_kmin_g = pred[{ncells},{nlev}]{{1,0}} parameter(3)")
    lines.append(f"  Arg_qv = f64[{ncells},{nlev}]{{1,0}} parameter(4)")
    lines.append(f"  Arg_qc = f64[{ncells},{nlev}]{{1,0}} parameter(5)")
    lines.append(f"  Arg_qr = f64[{ncells},{nlev}]{{1,0}} parameter(6)")
    lines.append(f"  Arg_qs = f64[{ncells},{nlev}]{{1,0}} parameter(7)")
    lines.append(f"  Arg_qi = f64[{ncells},{nlev}]{{1,0}} parameter(8)")
    lines.append(f"  Arg_qg = f64[{ncells},{nlev}]{{1,0}} parameter(9)")
    lines.append(f"  Arg_t = f64[{ncells},{nlev}]{{1,0}} parameter(10)")
    lines.append(f"  Arg_rho = f64[{ncells},{nlev}]{{1,0}} parameter(11)")
    lines.append(f"  Arg_dz = f64[{ncells},{nlev}]{{1,0}} parameter(12)")
    lines.append("")

    # Constants
    lines.append("  // Constants")
    lines.append("  const_0 = f64[] constant(0)")
    lines.append("  const_05 = f64[] constant(0.5)")
    lines.append("  const_1 = f64[] constant(1)")
    lines.append("  const_2 = f64[] constant(2)")
    lines.append("  const_30 = f64[] constant(30)")
    lines.append("  const_rho0 = f64[] constant(1.225)")
    lines.append("  const_cvd = f64[] constant(717.6)")
    lines.append("  const_clw = f64[] constant(4192.6641119999995)")
    lines.append("  const_ci = f64[] constant(2108)")
    lines.append("  const_cvv = f64[] constant(1407.95)")
    lines.append("  const_lvc = f64[] constant(3135383.2031928)")
    lines.append("  const_lsc = f64[] constant(2899657.201)")
    lines.append("  false_scalar = pred[] constant(false)")
    lines.append("")

    # Velocity coefficients
    lines.append("  // Velocity coefficients")
    lines.append("  vel_coeff_r = f64[] constant(14.58)")
    lines.append("  vel_coeff_s = f64[] constant(57.8)")
    lines.append("  vel_coeff_i = f64[] constant(1.25)")
    lines.append("  vel_coeff_g = f64[] constant(12.24)")
    lines.append("  vel_exp_r = f64[] constant(0.111)")
    lines.append("  vel_exp_s = f64[] constant(0.16666666666666666)")
    lines.append("  vel_exp_i = f64[] constant(0.16)")
    lines.append("  vel_exp_g = f64[] constant(0.217)")
    lines.append("  qmin_r = f64[] constant(1e-12)")
    lines.append("  qmin_s = f64[] constant(1e-12)")
    lines.append("  qmin_i = f64[] constant(1e-12)")
    lines.append("  qmin_g = f64[] constant(1e-08)")
    lines.append("")

    # Precompute
    lines.append("  // Precompute zeta = dt / (2 * dz)")
    lines.append(f"  bcast_30 = f64[{ncells},{nlev}]{{1,0}} broadcast(const_30), dimensions={{}}")
    lines.append(f"  bcast_2 = f64[{ncells},{nlev}]{{1,0}} broadcast(const_2), dimensions={{}}")
    lines.append(f"  dz_times_2 = f64[{ncells},{nlev}]{{1,0}} multiply(Arg_dz, bcast_2)")
    lines.append(f"  zeta_full = f64[{ncells},{nlev}]{{1,0}} divide(bcast_30, dz_times_2)")
    lines.append("")

    lines.append("  // Precompute rho_sqrt = sqrt(rho0 / rho)")
    lines.append(f"  bcast_rho0 = f64[{ncells},{nlev}]{{1,0}} broadcast(const_rho0), dimensions={{}}")
    lines.append(f"  rho_ratio = f64[{ncells},{nlev}]{{1,0}} divide(bcast_rho0, Arg_rho)")
    lines.append(f"  rho_sqrt = f64[{ncells},{nlev}]{{1,0}} sqrt(rho_ratio)")
    lines.append("")

    # Slice all inputs
    lines.append("  // ========== SLICE ALL INPUTS (STATIC) ==========")
    for k in range(nlev):
        lines.append(f"  // Level {k}")
        lines.append(f"  kmin_r_{k} = pred[{ncells},1]{{1,0}} slice(Arg_kmin_r), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  kmin_s_{k} = pred[{ncells},1]{{1,0}} slice(Arg_kmin_s), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  kmin_i_{k} = pred[{ncells},1]{{1,0}} slice(Arg_kmin_i), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  kmin_g_{k} = pred[{ncells},1]{{1,0}} slice(Arg_kmin_g), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  qr_{k} = f64[{ncells},1]{{1,0}} slice(Arg_qr), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  qs_{k} = f64[{ncells},1]{{1,0}} slice(Arg_qs), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  qi_{k} = f64[{ncells},1]{{1,0}} slice(Arg_qi), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  qg_{k} = f64[{ncells},1]{{1,0}} slice(Arg_qg), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  qv_{k} = f64[{ncells},1]{{1,0}} slice(Arg_qv), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  qc_{k} = f64[{ncells},1]{{1,0}} slice(Arg_qc), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  t_{k} = f64[{ncells},1]{{1,0}} slice(Arg_t), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  rho_{k} = f64[{ncells},1]{{1,0}} slice(Arg_rho), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  dz_{k} = f64[{ncells},1]{{1,0}} slice(Arg_dz), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  zeta_{k} = f64[{ncells},1]{{1,0}} slice(zeta_full), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append(f"  rho_sqrt_{k} = f64[{ncells},1]{{1,0}} slice(rho_sqrt), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        if k < nlev - 1:
            lines.append(f"  t_kp1_{k} = f64[{ncells},1]{{1,0}} slice(Arg_t), slice={{[0:{ncells}], [{k+1}:{k+2}]}}")
        else:
            lines.append(f"  t_kp1_{k} = f64[{ncells},1]{{1,0}} slice(Arg_t), slice={{[0:{ncells}], [{k}:{k+1}]}}")
        lines.append("")

    # Broadcast constants
    lines.append("  // Broadcast constants")
    lines.append(f"  bcast_0_1d = f64[{ncells},1]{{1,0}} broadcast(const_0), dimensions={{}}")
    lines.append(f"  bcast_05_1d = f64[{ncells},1]{{1,0}} broadcast(const_05), dimensions={{}}")
    lines.append(f"  bcast_1_1d = f64[{ncells},1]{{1,0}} broadcast(const_1), dimensions={{}}")
    lines.append(f"  bcast_2_1d = f64[{ncells},1]{{1,0}} broadcast(const_2), dimensions={{}}")
    lines.append(f"  bcast_30_1d = f64[{ncells},1]{{1,0}} broadcast(const_30), dimensions={{}}")
    lines.append(f"  bcast_cvd_1d = f64[{ncells},1]{{1,0}} broadcast(const_cvd), dimensions={{}}")
    lines.append(f"  bcast_clw_1d = f64[{ncells},1]{{1,0}} broadcast(const_clw), dimensions={{}}")
    lines.append(f"  bcast_ci_1d = f64[{ncells},1]{{1,0}} broadcast(const_ci), dimensions={{}}")
    lines.append(f"  bcast_cvv_1d = f64[{ncells},1]{{1,0}} broadcast(const_cvv), dimensions={{}}")
    lines.append(f"  bcast_lvc_1d = f64[{ncells},1]{{1,0}} broadcast(const_lvc), dimensions={{}}")
    lines.append(f"  bcast_lsc_1d = f64[{ncells},1]{{1,0}} broadcast(const_lsc), dimensions={{}}")
    lines.append(f"  bcast_vel_coeff_r = f64[{ncells},1]{{1,0}} broadcast(vel_coeff_r), dimensions={{}}")
    lines.append(f"  bcast_vel_coeff_s = f64[{ncells},1]{{1,0}} broadcast(vel_coeff_s), dimensions={{}}")
    lines.append(f"  bcast_vel_coeff_i = f64[{ncells},1]{{1,0}} broadcast(vel_coeff_i), dimensions={{}}")
    lines.append(f"  bcast_vel_coeff_g = f64[{ncells},1]{{1,0}} broadcast(vel_coeff_g), dimensions={{}}")
    lines.append(f"  bcast_vel_exp_r = f64[{ncells},1]{{1,0}} broadcast(vel_exp_r), dimensions={{}}")
    lines.append(f"  bcast_vel_exp_s = f64[{ncells},1]{{1,0}} broadcast(vel_exp_s), dimensions={{}}")
    lines.append(f"  bcast_vel_exp_i = f64[{ncells},1]{{1,0}} broadcast(vel_exp_i), dimensions={{}}")
    lines.append(f"  bcast_vel_exp_g = f64[{ncells},1]{{1,0}} broadcast(vel_exp_g), dimensions={{}}")
    lines.append(f"  bcast_qmin_r = f64[{ncells},1]{{1,0}} broadcast(qmin_r), dimensions={{}}")
    lines.append(f"  bcast_qmin_s = f64[{ncells},1]{{1,0}} broadcast(qmin_s), dimensions={{}}")
    lines.append(f"  bcast_qmin_i = f64[{ncells},1]{{1,0}} broadcast(qmin_i), dimensions={{}}")
    lines.append(f"  bcast_qmin_g = f64[{ncells},1]{{1,0}} broadcast(qmin_g), dimensions={{}}")
    lines.append(f"  false_1d = pred[{ncells},1]{{1,0}} broadcast(false_scalar), dimensions={{}}")
    lines.append("")

    # Initial carry state
    lines.append("  // ========== INITIAL CARRY STATE ==========")
    for sp in ['r', 's', 'i', 'g']:
        lines.append(f"  q{sp}_init = f64[{ncells},1]{{1,0}} broadcast(const_0), dimensions={{}}")
        lines.append(f"  pflx_{sp}_init = f64[{ncells},1]{{1,0}} broadcast(const_0), dimensions={{}}")
        lines.append(f"  vc_{sp}_init = f64[{ncells},1]{{1,0}} broadcast(const_0), dimensions={{}}")
        lines.append(f"  activated_{sp}_init = pred[{ncells},1]{{1,0}} broadcast(false_scalar), dimensions={{}}")
    lines.append(f"  rho_prev_init = f64[{ncells},1]{{1,0}} broadcast(const_0), dimensions={{}}")
    lines.append(f"  eflx_prev_init = f64[{ncells},1]{{1,0}} broadcast(const_0), dimensions={{}}")
    lines.append(f"  temp_activated_init = pred[{ncells},1]{{1,0}} broadcast(false_scalar), dimensions={{}}")
    lines.append("")

    # Unrolled computation for each level
    for k in range(nlev):
        lines.append(f"  // ========== LEVEL {k} ==========")

        prev = "_init" if k == 0 else f"_out_{k-1}"
        out = f"_out_{k}"

        for sp in ['r', 's', 'i', 'g']:
            lines.append(f"  // Species {sp}")
            lines.append(f"  activated_{sp}{out} = pred[{ncells},1]{{1,0}} or(activated_{sp}{prev}, kmin_{sp}_{k})")
            lines.append(f"  rho_x_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(q{sp}_{k}, rho_{k})")
            lines.append(f"  term1_{sp}_{k} = f64[{ncells},1]{{1,0}} divide(rho_x_{sp}_{k}, zeta_{k})")
            lines.append(f"  pflx_2_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(pflx_{sp}{prev}, bcast_2_1d)")
            lines.append(f"  flx_eff_{sp}_{k} = f64[{ncells},1]{{1,0}} add(term1_{sp}_{k}, pflx_2_{sp}_{k})")
            lines.append(f"  rho_x_offset_{sp}_{k} = f64[{ncells},1]{{1,0}} add(rho_x_{sp}_{k}, bcast_qmin_{sp})")
            lines.append(f"  rho_x_pow_{sp}_{k} = f64[{ncells},1]{{1,0}} power(rho_x_offset_{sp}_{k}, bcast_vel_exp_{sp})")
            lines.append(f"  fall_speed_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(bcast_vel_coeff_{sp}, rho_x_pow_{sp}_{k})")
            lines.append(f"  flux_raw_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(rho_x_{sp}_{k}, rho_sqrt_{k})")
            lines.append(f"  flux_scaled_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(flux_raw_{sp}_{k}, fall_speed_{sp}_{k})")
            lines.append(f"  flx_partial_{sp}_{k} = f64[{ncells},1]{{1,0}} minimum(flux_scaled_{sp}_{k}, flx_eff_{sp}_{k})")
            lines.append(f"  q_sum_{sp}_{k} = f64[{ncells},1]{{1,0}} add(q{sp}{prev}, q{sp}_{k})")
            lines.append(f"  q_mid_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(q_sum_{sp}_{k}, bcast_05_1d)")
            lines.append(f"  rhox_mid_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(q_mid_{sp}_{k}, rho_prev{prev})")
            lines.append(f"  rhox_mid_offset_{sp}_{k} = f64[{ncells},1]{{1,0}} add(rhox_mid_{sp}_{k}, bcast_qmin_{sp})")
            lines.append(f"  rhox_mid_pow_{sp}_{k} = f64[{ncells},1]{{1,0}} power(rhox_mid_offset_{sp}_{k}, bcast_vel_exp_{sp})")
            lines.append(f"  vc_prev_vel_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(vc_{sp}{prev}, bcast_vel_coeff_{sp})")
            lines.append(f"  vt_active_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(vc_prev_vel_{sp}_{k}, rhox_mid_pow_{sp}_{k})")
            lines.append(f"  vt_{sp}_{k} = f64[{ncells},1]{{1,0}} select(activated_{sp}{prev}, vt_active_{sp}_{k}, bcast_0_1d)")
            lines.append(f"  flx_diff_{sp}_{k} = f64[{ncells},1]{{1,0}} subtract(flx_eff_{sp}_{k}, flx_partial_{sp}_{k})")
            lines.append(f"  num_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(zeta_{k}, flx_diff_{sp}_{k})")
            lines.append(f"  zeta_vt_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(zeta_{k}, vt_{sp}_{k})")
            lines.append(f"  denom_inner_{sp}_{k} = f64[{ncells},1]{{1,0}} add(zeta_vt_{sp}_{k}, bcast_1_1d)")
            lines.append(f"  denom_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(denom_inner_{sp}_{k}, rho_{k})")
            lines.append(f"  q_activated_{sp}_{k} = f64[{ncells},1]{{1,0}} divide(num_{sp}_{k}, denom_{sp}_{k})")
            lines.append(f"  q_rho_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(q_activated_{sp}_{k}, rho_{k})")
            lines.append(f"  q_rho_vt_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(q_rho_{sp}_{k}, vt_{sp}_{k})")
            lines.append(f"  flx_sum_{sp}_{k} = f64[{ncells},1]{{1,0}} add(q_rho_vt_{sp}_{k}, flx_partial_{sp}_{k})")
            lines.append(f"  flx_activated_{sp}_{k} = f64[{ncells},1]{{1,0}} multiply(flx_sum_{sp}_{k}, bcast_05_1d)")
            lines.append(f"  q{sp}{out} = f64[{ncells},1]{{1,0}} select(activated_{sp}{out}, q_activated_{sp}_{k}, q{sp}_{k})")
            lines.append(f"  pflx_{sp}{out} = f64[{ncells},1]{{1,0}} select(activated_{sp}{out}, flx_activated_{sp}_{k}, bcast_0_1d)")
            lines.append(f"  vc_{sp}{out} = f64[{ncells},1]{{1,0}} copy(rho_sqrt_{k})")
            lines.append("")

        lines.append(f"  rho_prev{out} = f64[{ncells},1]{{1,0}} copy(rho_{k})")

        # Temperature
        lines.append(f"  // Temperature")
        lines.append(f"  kmin_or_1_{k} = pred[{ncells},1]{{1,0}} or(kmin_r_{k}, kmin_s_{k})")
        lines.append(f"  kmin_or_2_{k} = pred[{ncells},1]{{1,0}} or(kmin_or_1_{k}, kmin_i_{k})")
        lines.append(f"  temp_mask_{k} = pred[{ncells},1]{{1,0}} or(kmin_or_2_{k}, kmin_g_{k})")
        lines.append(f"  temp_activated{out} = pred[{ncells},1]{{1,0}} or(temp_activated{prev}, temp_mask_{k})")
        lines.append(f"  qliq_{k} = f64[{ncells},1]{{1,0}} add(qc_{k}, qr{out})")
        lines.append(f"  qice_1_{k} = f64[{ncells},1]{{1,0}} add(qs{out}, qi{out})")
        lines.append(f"  qice_{k} = f64[{ncells},1]{{1,0}} add(qice_1_{k}, qg{out})")
        lines.append(f"  pflx_si_{k} = f64[{ncells},1]{{1,0}} add(pflx_s{out}, pflx_i{out})")
        lines.append(f"  pflx_ice_{k} = f64[{ncells},1]{{1,0}} add(pflx_si_{k}, pflx_g{out})")
        lines.append(f"  cvd_t_kp1_{k} = f64[{ncells},1]{{1,0}} multiply(t_kp1_{k}, bcast_cvd_1d)")
        lines.append(f"  clw_t_{k} = f64[{ncells},1]{{1,0}} multiply(t_{k}, bcast_clw_1d)")
        lines.append(f"  rain_term_{k} = f64[{ncells},1]{{1,0}} subtract(clw_t_{k}, cvd_t_kp1_{k})")
        lines.append(f"  rain_term2_{k} = f64[{ncells},1]{{1,0}} subtract(rain_term_{k}, bcast_lvc_1d)")
        lines.append(f"  eflx_rain_{k} = f64[{ncells},1]{{1,0}} multiply(pflx_r{out}, rain_term2_{k})")
        lines.append(f"  ci_t_{k} = f64[{ncells},1]{{1,0}} multiply(t_{k}, bcast_ci_1d)")
        lines.append(f"  ice_term_{k} = f64[{ncells},1]{{1,0}} subtract(ci_t_{k}, cvd_t_kp1_{k})")
        lines.append(f"  ice_term2_{k} = f64[{ncells},1]{{1,0}} subtract(ice_term_{k}, bcast_lsc_1d)")
        lines.append(f"  eflx_ice_{k} = f64[{ncells},1]{{1,0}} multiply(pflx_ice_{k}, ice_term2_{k})")
        lines.append(f"  eflx_sum_{k} = f64[{ncells},1]{{1,0}} add(eflx_rain_{k}, eflx_ice_{k})")
        lines.append(f"  eflx_new_raw_{k} = f64[{ncells},1]{{1,0}} multiply(bcast_30_1d, eflx_sum_{k})")
        lines.append(f"  ei_{k} = f64[{ncells},1]{{1,0}} multiply(t_{k}, bcast_cvd_1d)")
        lines.append(f"  e_int_1_{k} = f64[{ncells},1]{{1,0}} add(ei_{k}, eflx_prev{prev})")
        lines.append(f"  e_int_{k} = f64[{ncells},1]{{1,0}} subtract(e_int_1_{k}, eflx_new_raw_{k})")
        lines.append(f"  qtot_{k} = f64[{ncells},1]{{1,0}} add(qliq_{k}, qice_{k})")
        lines.append(f"  qtot2_{k} = f64[{ncells},1]{{1,0}} add(qtot_{k}, qv_{k})")
        lines.append(f"  one_minus_qtot_{k} = f64[{ncells},1]{{1,0}} subtract(bcast_1_1d, qtot2_{k})")
        lines.append(f"  cv_dry_{k} = f64[{ncells},1]{{1,0}} multiply(one_minus_qtot_{k}, bcast_cvd_1d)")
        lines.append(f"  cv_vapor_{k} = f64[{ncells},1]{{1,0}} multiply(qv_{k}, bcast_cvv_1d)")
        lines.append(f"  cv_liq_{k} = f64[{ncells},1]{{1,0}} multiply(qliq_{k}, bcast_clw_1d)")
        lines.append(f"  cv_ice_{k} = f64[{ncells},1]{{1,0}} multiply(qice_{k}, bcast_ci_1d)")
        lines.append(f"  cv_sum1_{k} = f64[{ncells},1]{{1,0}} add(cv_dry_{k}, cv_vapor_{k})")
        lines.append(f"  cv_sum2_{k} = f64[{ncells},1]{{1,0}} add(cv_sum1_{k}, cv_liq_{k})")
        lines.append(f"  cv_sum3_{k} = f64[{ncells},1]{{1,0}} add(cv_sum2_{k}, cv_ice_{k})")
        lines.append(f"  rho_dz_{k} = f64[{ncells},1]{{1,0}} multiply(rho_{k}, dz_{k})")
        lines.append(f"  cv_{k} = f64[{ncells},1]{{1,0}} multiply(cv_sum3_{k}, rho_dz_{k})")
        lines.append(f"  lh_liq_{k} = f64[{ncells},1]{{1,0}} multiply(qliq_{k}, bcast_lvc_1d)")
        lines.append(f"  lh_ice_{k} = f64[{ncells},1]{{1,0}} multiply(qice_{k}, bcast_lsc_1d)")
        lines.append(f"  lh_sum_{k} = f64[{ncells},1]{{1,0}} add(lh_liq_{k}, lh_ice_{k})")
        lines.append(f"  lh_corr_{k} = f64[{ncells},1]{{1,0}} multiply(rho_dz_{k}, lh_sum_{k})")
        lines.append(f"  t_num_{k} = f64[{ncells},1]{{1,0}} add(e_int_{k}, lh_corr_{k})")
        lines.append(f"  t_new_raw_{k} = f64[{ncells},1]{{1,0}} divide(t_num_{k}, cv_{k})")
        lines.append(f"  eflx_prev{out} = f64[{ncells},1]{{1,0}} select(temp_activated{out}, eflx_new_raw_{k}, eflx_prev{prev})")
        lines.append(f"  t{out} = f64[{ncells},1]{{1,0}} select(temp_activated{out}, t_new_raw_{k}, t_{k})")
        lines.append("")

    # Concatenate outputs
    lines.append("  // ========== CONCATENATE OUTPUTS ==========")
    for name, prefix in [("qr_out_full", "qr_out_"), ("qs_out_full", "qs_out_"), 
                         ("qi_out_full", "qi_out_"), ("qg_out_full", "qg_out_")]:
        args = ", ".join([f"{prefix}{k}" for k in range(nlev)])
        lines.append(f"  {name} = f64[{ncells},{nlev}]{{1,0}} concatenate({args}), dimensions={{1}}")

    t_args = ", ".join([f"t_out_{k}" for k in range(nlev)])
    lines.append(f"  t_out_concat = f64[{ncells},{nlev}]{{1,0}} concatenate({t_args}), dimensions={{1}}")
    lines.append(f"  t_out_full = f64[{ncells},{nlev}]{{0,1}} transpose(t_out_concat), dimensions={{0,1}}")

    for name, prefix in [("pflx_r_full", "pflx_r_out_"), ("pflx_s_full", "pflx_s_out_"),
                         ("pflx_i_full", "pflx_i_out_"), ("pflx_g_full", "pflx_g_out_")]:
        args = ", ".join([f"{prefix}{k}" for k in range(nlev)])
        lines.append(f"  {name} = f64[{ncells},{nlev}]{{1,0}} concatenate({args}), dimensions={{1}}")

    lines.append(f"  pflx_si_full = f64[{ncells},{nlev}]{{1,0}} add(pflx_s_full, pflx_i_full)")
    lines.append(f"  pflx_sig_full = f64[{ncells},{nlev}]{{1,0}} add(pflx_si_full, pflx_g_full)")
    lines.append(f"  pflx_tot_full = f64[{ncells},{nlev}]{{1,0}} add(pflx_r_full, pflx_sig_full)")
    lines.append(f"  bcast_30_full = f64[{ncells},{nlev}]{{1,0}} broadcast(const_30), dimensions={{}}")
    lines.append(f"  pr_out = f64[{ncells},{nlev}]{{1,0}} divide(pflx_r_full, bcast_30_full)")
    lines.append(f"  ps_out = f64[{ncells},{nlev}]{{1,0}} divide(pflx_s_full, bcast_30_full)")
    lines.append(f"  pi_out = f64[{ncells},{nlev}]{{1,0}} divide(pflx_i_full, bcast_30_full)")
    lines.append(f"  pg_out = f64[{ncells},{nlev}]{{1,0}} divide(pflx_g_full, bcast_30_full)")

    eflx_args = ", ".join([f"eflx_prev_out_{k}" for k in range(nlev)])
    lines.append(f"  eflx_concat = f64[{ncells},{nlev}]{{1,0}} concatenate({eflx_args}), dimensions={{1}}")
    lines.append(f"  eflx_out_transpose = f64[{ncells},{nlev}]{{0,1}} transpose(eflx_concat), dimensions={{0,1}}")
    lines.append(f"  eflx_out = f64[{ncells},{nlev}]{{0,1}} divide(eflx_out_transpose, bcast_30_full)")
    lines.append("")

    lines.append(f"  ROOT result = (f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{0,1}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{1,0}}, f64[{ncells},{nlev}]{{0,1}}) tuple(qr_out_full, qs_out_full, qi_out_full, qg_out_full, t_out_full, pflx_tot_full, pr_out, ps_out, pi_out, pg_out, eflx_out)")
    lines.append("}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate fully unrolled HLO")
    parser.add_argument("-o", "--output", default="shlo/precip_effect_x64_unrolled.hlo")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=20480)
    args = parser.parse_args()

    print(f"Generating unrolled HLO with {args.nlev} levels...")
    hlo_text = generate_unrolled_hlo(args.nlev, args.ncells)

    with open(args.output, 'w') as f:
        f.write(hlo_text)

    print(f"Written to: {args.output}")
    print(f"File size: {len(hlo_text) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
