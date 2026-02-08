#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate fully unrolled StableHLO with TRANSPOSED memory layout.

Generates the COMPLETE precipitation_effects computation:
1. Internal energy computation (ei_old)
2. Velocity scale factors (xrho, snow_number for vc_s, ice power for vc_i)
3. Interleaved precipitation scan + temperature update (per-level)
4. Correct 11-output return values

Optimizations vs previous version:
- Interleaved: precip and temp scans fused per-level (no concat-then-reslice)
- Reduced carry: rho_prev/vc_prev eliminated (direct reference to level k-1)
- Lazy slicing: inputs sliced at point of use, not upfront

Usage:
    python generate_unrolled_transposed.py --ncells 327680 --nlev 90
"""

import argparse


def generate_unrolled_transposed(nlev: int = 90, ncells: int = 327680) -> str:
    """Generate fully unrolled StableHLO with nlev x ncells layout."""

    lines = []

    # Shorthand types
    tf = f"tensor<{nlev}x{ncells}xf64>"
    tb = f"tensor<{nlev}x{ncells}xi1>"
    tf1 = f"tensor<1x{ncells}xf64>"
    tb1 = f"tensor<1x{ncells}xi1>"

    last_lev = nlev - 1

    # Module header
    lines.append(
        f"module @jit_precip_unrolled_transposed_{nlev} attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{"
    )
    lines.append("")
    lines.append("  // TRANSPOSED LAYOUT: tensor<nlev x ncells> for coalesced GPU memory access")
    lines.append("  // Interleaved precip + temp scans, reduced carry state")
    lines.append("")

    # Function signature - inputs are nlev x ncells (transposed)
    args = ", ".join(
        [
            f"%arg0: {tb}",  # kmin_r
            f"%arg1: {tb}",  # kmin_i
            f"%arg2: {tb}",  # kmin_s
            f"%arg3: {tb}",  # kmin_g
            f"%arg4: {tf}",  # qv
            f"%arg5: {tf}",  # qc
            f"%arg6: {tf}",  # qr
            f"%arg7: {tf}",  # qs
            f"%arg8: {tf}",  # qi
            f"%arg9: {tf}",  # qg
            f"%arg10: {tf}",  # t
            f"%arg11: {tf}",  # rho
            f"%arg12: {tf}",  # dz
        ]
    )

    ret_types = ", ".join([tf] * 11)
    lines.append(f"  func.func public @main({args}) -> ({ret_types}) {{")

    # ================================================================
    # Scalar constants
    # ================================================================
    lines.append("    // ========== SCALAR CONSTANTS ==========")
    lines.append("    %cst_0 = stablehlo.constant dense<0.0> : tensor<f64>")
    lines.append("    %cst_05 = stablehlo.constant dense<0.5> : tensor<f64>")
    lines.append("    %cst_1 = stablehlo.constant dense<1.0> : tensor<f64>")
    lines.append("    %cst_2 = stablehlo.constant dense<2.0> : tensor<f64>")
    lines.append("    %cst_3 = stablehlo.constant dense<3.0> : tensor<f64>")
    lines.append("    %cst_4 = stablehlo.constant dense<4.0> : tensor<f64>")
    lines.append("    %cst_10 = stablehlo.constant dense<10.0> : tensor<f64>")
    lines.append("    %cst_30 = stablehlo.constant dense<30.0> : tensor<f64>")
    lines.append("    %false = stablehlo.constant dense<false> : tensor<i1>")
    lines.append("")

    # Physics constants
    lines.append("    // Physics constants")
    lines.append("    %cst_rho0 = stablehlo.constant dense<1.225> : tensor<f64>")
    lines.append("    %cst_cvd = stablehlo.constant dense<717.60> : tensor<f64>")
    lines.append("    %cst_cvv = stablehlo.constant dense<1407.95> : tensor<f64>")
    lines.append("    %cst_clw = stablehlo.constant dense<4192.6641119999995> : tensor<f64>")
    lines.append("    %cst_ci = stablehlo.constant dense<2108.0> : tensor<f64>")
    lines.append("    %cst_lvc = stablehlo.constant dense<3135383.2031927998> : tensor<f64>")
    lines.append("    %cst_lsc = stablehlo.constant dense<2899657.2009999999> : tensor<f64>")
    lines.append("    %cst_tmelt = stablehlo.constant dense<273.15> : tensor<f64>")
    lines.append("    %cst_qmin = stablehlo.constant dense<1.0e-15> : tensor<f64>")
    lines.append("")

    # Snow number constants
    lines.append("    // Snow number constants")
    lines.append("    %cst_tmin = stablehlo.constant dense<233.15> : tensor<f64>")  # tmelt - 40
    lines.append("    %cst_tmax = stablehlo.constant dense<273.15> : tensor<f64>")  # tmelt
    lines.append("    %cst_qsmin = stablehlo.constant dense<2.0e-6> : tensor<f64>")
    lines.append("    %cst_xa1 = stablehlo.constant dense<-1.65> : tensor<f64>")
    lines.append("    %cst_xa2 = stablehlo.constant dense<5.45e-2> : tensor<f64>")
    lines.append("    %cst_xa3 = stablehlo.constant dense<3.27e-4> : tensor<f64>")
    lines.append("    %cst_xb1 = stablehlo.constant dense<1.42> : tensor<f64>")
    lines.append("    %cst_xb2 = stablehlo.constant dense<1.19e-2> : tensor<f64>")
    lines.append("    %cst_xb3 = stablehlo.constant dense<9.60e-5> : tensor<f64>")
    lines.append("    %cst_n0s0 = stablehlo.constant dense<8.00e5> : tensor<f64>")
    n0s1_val = 13.5 * 5.65e05
    lines.append(f"    %cst_n0s1 = stablehlo.constant dense<{n0s1_val}> : tensor<f64>")
    lines.append("    %cst_n0s2 = stablehlo.constant dense<-0.107> : tensor<f64>")
    lines.append("    %cst_n0s3 = stablehlo.constant dense<13.5> : tensor<f64>")
    n0s4_val = 0.5 * n0s1_val
    lines.append(f"    %cst_n0s4 = stablehlo.constant dense<{n0s4_val}> : tensor<f64>")
    lines.append("    %cst_n0s5 = stablehlo.constant dense<1.0e6> : tensor<f64>")
    n0s6_val = 1.0e2 * n0s1_val
    lines.append(f"    %cst_n0s6 = stablehlo.constant dense<{n0s6_val}> : tensor<f64>")
    lines.append("    %cst_n0s7 = stablehlo.constant dense<1.0e9> : tensor<f64>")
    lines.append("    %cst_ams = stablehlo.constant dense<0.069> : tensor<f64>")
    lines.append(
        "    %cst_neg_one_sixth = stablehlo.constant dense<-0.16666666666666666> : tensor<f64>"
    )
    lines.append(
        "    %cst_two_thirds = stablehlo.constant dense<0.66666666666666667> : tensor<f64>"
    )
    lines.append("")

    # Velocity coefficients
    lines.append("    // Velocity coefficients [rain, snow, ice, graupel]")
    lines.append("    %vel_coeff_r = stablehlo.constant dense<14.58> : tensor<f64>")
    lines.append("    %vel_coeff_s = stablehlo.constant dense<57.8> : tensor<f64>")
    lines.append("    %vel_coeff_i = stablehlo.constant dense<1.25> : tensor<f64>")
    lines.append("    %vel_coeff_g = stablehlo.constant dense<12.24> : tensor<f64>")
    lines.append("    %vel_exp_r = stablehlo.constant dense<0.111> : tensor<f64>")
    lines.append("    %vel_exp_s = stablehlo.constant dense<0.16666666666666666> : tensor<f64>")
    lines.append("    %vel_exp_i = stablehlo.constant dense<0.16> : tensor<f64>")
    lines.append("    %vel_exp_g = stablehlo.constant dense<0.217> : tensor<f64>")
    lines.append("    %qmin_r = stablehlo.constant dense<1.0e-12> : tensor<f64>")
    lines.append("    %qmin_s = stablehlo.constant dense<1.0e-12> : tensor<f64>")
    lines.append("    %qmin_i = stablehlo.constant dense<1.0e-12> : tensor<f64>")
    lines.append("    %qmin_g = stablehlo.constant dense<1.0e-08> : tensor<f64>")
    lines.append("")

    # ================================================================
    # Broadcast constants to full size (only for Phase 1/2 on full tensors)
    # ================================================================
    lines.append("    // ========== FULL-SIZE BROADCASTS (Phase 1/2 only) ==========")
    for name in ["0", "1", "2", "3", "4", "10", "30"]:
        lines.append(
            f"    %bcast_{name} = stablehlo.broadcast_in_dim %cst_{name}, dims = [] : (tensor<f64>) -> {tf}"
        )
    lines.append(
        f"    %bcast_rho0 = stablehlo.broadcast_in_dim %cst_rho0, dims = [] : (tensor<f64>) -> {tf}"
    )
    for name in [
        "cvd",
        "cvv",
        "clw",
        "ci",
        "lvc",
        "lsc",
        "tmelt",
        "qmin",
        "tmin",
        "tmax",
        "qsmin",
        "xa1",
        "xa2",
        "xa3",
        "xb1",
        "xb2",
        "xb3",
        "n0s0",
        "n0s1",
        "n0s2",
        "n0s3",
        "n0s4",
        "n0s5",
        "n0s6",
        "n0s7",
        "ams",
        "neg_one_sixth",
        "two_thirds",
    ]:
        lines.append(
            f"    %bcast_{name} = stablehlo.broadcast_in_dim %cst_{name}, dims = [] : (tensor<f64>) -> {tf}"
        )
    lines.append("")

    # ================================================================
    # PHASE 1: Internal energy computation on full tensors
    # ================================================================
    lines.append("    // ========== PHASE 1: INTERNAL ENERGY (ei_old) ==========")
    lines.append(f"    %qliq_old = stablehlo.add %arg5, %arg6 : {tf}")
    lines.append(f"    %qice_old_tmp1 = stablehlo.add %arg7, %arg8 : {tf}")
    lines.append(f"    %qice_old = stablehlo.add %qice_old_tmp1, %arg9 : {tf}")
    lines.append(f"    %qtot_old_tmp = stablehlo.add %arg4, %qliq_old : {tf}")
    lines.append(f"    %qtot_old = stablehlo.add %qtot_old_tmp, %qice_old : {tf}")
    lines.append(f"    %one_m_qtot = stablehlo.subtract %bcast_1, %qtot_old : {tf}")
    lines.append(f"    %cv_term1 = stablehlo.multiply %bcast_cvd, %one_m_qtot : {tf}")
    lines.append(f"    %cv_term2 = stablehlo.multiply %bcast_cvv, %arg4 : {tf}")
    lines.append(f"    %cv_term3 = stablehlo.multiply %bcast_clw, %qliq_old : {tf}")
    lines.append(f"    %cv_term4 = stablehlo.multiply %bcast_ci, %qice_old : {tf}")
    lines.append(f"    %cv_sum1 = stablehlo.add %cv_term1, %cv_term2 : {tf}")
    lines.append(f"    %cv_sum2 = stablehlo.add %cv_sum1, %cv_term3 : {tf}")
    lines.append(f"    %cv_old = stablehlo.add %cv_sum2, %cv_term4 : {tf}")
    lines.append(f"    %cv_t = stablehlo.multiply %cv_old, %arg10 : {tf}")
    lines.append(f"    %qliq_lvc = stablehlo.multiply %qliq_old, %bcast_lvc : {tf}")
    lines.append(f"    %qice_lsc = stablehlo.multiply %qice_old, %bcast_lsc : {tf}")
    lines.append(f"    %ei_inner1 = stablehlo.subtract %cv_t, %qliq_lvc : {tf}")
    lines.append(f"    %ei_inner = stablehlo.subtract %ei_inner1, %qice_lsc : {tf}")
    lines.append(f"    %rho_dz_full = stablehlo.multiply %arg11, %arg12 : {tf}")
    lines.append(f"    %ei_old_full = stablehlo.multiply %rho_dz_full, %ei_inner : {tf}")
    lines.append("")

    # ================================================================
    # PHASE 2: Velocity scale factors on full tensors
    # ================================================================
    lines.append("    // ========== PHASE 2: VELOCITY SCALE FACTORS ==========")
    lines.append(f"    %dz_times_2 = stablehlo.multiply %arg12, %bcast_2 : {tf}")
    lines.append(f"    %zeta_full = stablehlo.divide %bcast_30, %dz_times_2 : {tf}")
    lines.append("")
    lines.append(f"    %rho_ratio = stablehlo.divide %bcast_rho0, %arg11 : {tf}")
    lines.append(f"    %xrho = stablehlo.sqrt %rho_ratio : {tf}")
    lines.append("")

    # vc_r = xrho (default) — just use xrho directly, no copy
    # vc_g = xrho (default) — just use xrho directly, no copy
    # vc_i = xrho^(2/3)
    lines.append(f"    %vc_i_full = stablehlo.power %xrho, %bcast_two_thirds : {tf}")
    lines.append("")

    # vc_s = xrho * snow_number(t, rho, qs)^(-1/6)
    lines.append(f"    %sn_clamped_hi = stablehlo.minimum %arg10, %bcast_tmax : {tf}")
    lines.append(f"    %sn_clamped = stablehlo.maximum %sn_clamped_hi, %bcast_tmin : {tf}")
    lines.append(f"    %sn_tc = stablehlo.subtract %sn_clamped, %bcast_tmelt : {tf}")
    lines.append(f"    %sn_xa3_tc = stablehlo.multiply %bcast_xa3, %sn_tc : {tf}")
    lines.append(f"    %sn_xa2_sum = stablehlo.add %bcast_xa2, %sn_xa3_tc : {tf}")
    lines.append(f"    %sn_xa2_tc = stablehlo.multiply %sn_xa2_sum, %sn_tc : {tf}")
    lines.append(f"    %sn_exp_arg = stablehlo.add %bcast_xa1, %sn_xa2_tc : {tf}")
    lines.append(f"    %sn_alf = stablehlo.power %bcast_10, %sn_exp_arg : {tf}")
    lines.append(f"    %sn_xb3_tc = stablehlo.multiply %bcast_xb3, %sn_tc : {tf}")
    lines.append(f"    %sn_xb2_sum = stablehlo.add %bcast_xb2, %sn_xb3_tc : {tf}")
    lines.append(f"    %sn_xb2_tc = stablehlo.multiply %sn_xb2_sum, %sn_tc : {tf}")
    lines.append(f"    %sn_bet = stablehlo.add %bcast_xb1, %sn_xb2_tc : {tf}")
    lines.append(f"    %sn_qs_qsmin = stablehlo.add %arg7, %bcast_qsmin : {tf}")
    lines.append(f"    %sn_qs_rho = stablehlo.multiply %sn_qs_qsmin, %arg11 : {tf}")
    lines.append(f"    %sn_qs_rho_ams = stablehlo.divide %sn_qs_rho, %bcast_ams : {tf}")
    lines.append(f"    %sn_3bet = stablehlo.multiply %bcast_3, %sn_bet : {tf}")
    lines.append(f"    %sn_exponent = stablehlo.subtract %bcast_4, %sn_3bet : {tf}")
    lines.append(f"    %sn_base_pow = stablehlo.power %sn_qs_rho_ams, %sn_exponent : {tf}")
    lines.append(f"    %sn_alf3 = stablehlo.multiply %sn_alf, %sn_alf : {tf}")
    lines.append(f"    %sn_alf_cubed = stablehlo.multiply %sn_alf3, %sn_alf : {tf}")
    lines.append(f"    %sn_n0s_num = stablehlo.multiply %bcast_n0s3, %sn_base_pow : {tf}")
    lines.append(f"    %sn_n0s = stablehlo.divide %sn_n0s_num, %sn_alf_cubed : {tf}")
    lines.append(f"    %sn_n0s2_tc = stablehlo.multiply %bcast_n0s2, %sn_tc : {tf}")
    lines.append(f"    %sn_y = stablehlo.exponential %sn_n0s2_tc : {tf}")
    lines.append(f"    %sn_n0s4y = stablehlo.multiply %bcast_n0s4, %sn_y : {tf}")
    lines.append(f"    %sn_n0smn = stablehlo.maximum %sn_n0s4y, %bcast_n0s5 : {tf}")
    lines.append(f"    %sn_n0s6y = stablehlo.multiply %bcast_n0s6, %sn_y : {tf}")
    lines.append(f"    %sn_n0smx = stablehlo.minimum %sn_n0s6y, %bcast_n0s7 : {tf}")
    lines.append(f"    %sn_clamped_lo = stablehlo.maximum %sn_n0smn, %sn_n0s : {tf}")
    lines.append(f"    %sn_clamped_result = stablehlo.minimum %sn_n0smx, %sn_clamped_lo : {tf}")
    lines.append(
        f"    %sn_qs_gt_qmin = stablehlo.compare  GT, %arg7, %bcast_qmin,  FLOAT : ({tf}, {tf}) -> {tb}"
    )
    lines.append(
        f"    %snow_number = stablehlo.select %sn_qs_gt_qmin, %sn_clamped_result, %bcast_n0s0 : {tb}, {tf}"
    )
    lines.append(f"    %sn_pow = stablehlo.power %snow_number, %bcast_neg_one_sixth : {tf}")
    lines.append(f"    %vc_s_full = stablehlo.multiply %xrho, %sn_pow : {tf}")
    lines.append("")

    # ================================================================
    # Per-level broadcasts (1 x ncells constants)
    # ================================================================
    lines.append("    // ========== PER-LEVEL BROADCASTS ==========")
    lines.append(
        f"    %bcast_0_1d = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_05_1d = stablehlo.broadcast_in_dim %cst_05, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_1_1d = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_2_1d = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_30_1d = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_cvd_1d = stablehlo.broadcast_in_dim %cst_cvd, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_cvv_1d = stablehlo.broadcast_in_dim %cst_cvv, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_clw_1d = stablehlo.broadcast_in_dim %cst_clw, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_ci_1d = stablehlo.broadcast_in_dim %cst_ci, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_lvc_1d = stablehlo.broadcast_in_dim %cst_lvc, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append(
        f"    %bcast_lsc_1d = stablehlo.broadcast_in_dim %cst_lsc, dims = [] : (tensor<f64>) -> {tf1}"
    )
    for sp in ["r", "s", "i", "g"]:
        lines.append(
            f"    %bcast_vel_coeff_{sp} = stablehlo.broadcast_in_dim %vel_coeff_{sp}, dims = [] : (tensor<f64>) -> {tf1}"
        )
        lines.append(
            f"    %bcast_vel_exp_{sp} = stablehlo.broadcast_in_dim %vel_exp_{sp}, dims = [] : (tensor<f64>) -> {tf1}"
        )
        lines.append(
            f"    %bcast_qmin_{sp} = stablehlo.broadcast_in_dim %qmin_{sp}, dims = [] : (tensor<f64>) -> {tf1}"
        )
    lines.append("")

    # ================================================================
    # Initial precip carry state (reduced: no rho_prev, no vc_prev)
    # ================================================================
    lines.append("    // ========== INITIAL CARRY STATE ==========")
    for sp in ["r", "s", "i", "g"]:
        lines.append(
            f"    %q_{sp}_prev_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}"
        )
        lines.append(
            f"    %pflx_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}"
        )
        lines.append(
            f"    %activated_{sp}_init = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> {tb1}"
        )
    # Temperature scan carry
    lines.append(
        f"    %t_scan_activated_init = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> {tb1}"
    )
    lines.append(
        f"    %eflx_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}"
    )
    lines.append("")

    # Map species to vc source (full tensor name for slicing)
    # vc_r and vc_g both use xrho directly
    vc_full_map = {"r": "xrho", "s": "vc_s_full", "i": "vc_i_full", "g": "xrho"}

    # ================================================================
    # INTERLEAVED PER-LEVEL LOOP: precip scan + temp scan
    # ================================================================
    for k in range(nlev):
        lines.append(f"    // ========== LEVEL {k} ==========")

        # --- Slice inputs for this level (lazy slicing) ---
        lines.append(
            f"    %kmin_r_{k} = stablehlo.slice %arg0 [{k}:{k+1}, 0:{ncells}] : ({tb}) -> {tb1}"
        )
        lines.append(
            f"    %kmin_s_{k} = stablehlo.slice %arg2 [{k}:{k+1}, 0:{ncells}] : ({tb}) -> {tb1}"
        )
        lines.append(
            f"    %kmin_i_{k} = stablehlo.slice %arg1 [{k}:{k+1}, 0:{ncells}] : ({tb}) -> {tb1}"
        )
        lines.append(
            f"    %kmin_g_{k} = stablehlo.slice %arg3 [{k}:{k+1}, 0:{ncells}] : ({tb}) -> {tb1}"
        )
        lines.append(
            f"    %qv_{k} = stablehlo.slice %arg4 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %qc_{k} = stablehlo.slice %arg5 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %qr_{k} = stablehlo.slice %arg6 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %qs_{k} = stablehlo.slice %arg7 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %qi_{k} = stablehlo.slice %arg8 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %qg_{k} = stablehlo.slice %arg9 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %t_{k} = stablehlo.slice %arg10 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %rho_{k} = stablehlo.slice %arg11 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %dz_{k} = stablehlo.slice %arg12 [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %zeta_{k} = stablehlo.slice %zeta_full [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        lines.append(
            f"    %ei_old_{k} = stablehlo.slice %ei_old_full [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
        )
        # Slice velocity scale factors for this level
        for sp in ["r", "s", "i", "g"]:
            vc_src = vc_full_map[sp]
            lines.append(
                f"    %vc_{sp}_{k} = stablehlo.slice %{vc_src} [{k}:{k+1}, 0:{ncells}] : ({tf}) -> {tf1}"
            )

        prev = "_init" if k == 0 else f"_out_{k-1}"
        out = f"_out_{k}"

        # --- Precipitation scan for all 4 species ---
        q_input_map = {"r": "qr", "s": "qs", "i": "qi", "g": "qg"}

        for sp in ["r", "s", "i", "g"]:
            q_name = q_input_map[sp]
            lines.append(f"    // Species {sp}")

            # activated = activated_prev | mask
            lines.append(
                f"    %activated_{sp}{out} = stablehlo.or %activated_{sp}{prev}, %kmin_{sp}_{k} : {tb1}"
            )

            # rho_x = q * rho
            lines.append(
                f"    %rho_x_{sp}_{k} = stablehlo.multiply %{q_name}_{k}, %rho_{k} : {tf1}"
            )

            # flx_eff = (rho_x / zeta) + 2 * flx_prev
            lines.append(
                f"    %term1_{sp}_{k} = stablehlo.divide %rho_x_{sp}_{k}, %zeta_{k} : {tf1}"
            )
            lines.append(
                f"    %pflx_2_{sp}_{k} = stablehlo.multiply %pflx_{sp}{prev}, %bcast_2_1d : {tf1}"
            )
            lines.append(
                f"    %flx_eff_{sp}_{k} = stablehlo.add %term1_{sp}_{k}, %pflx_2_{sp}_{k} : {tf1}"
            )

            # fall_speed = prefactor * pow(rho_x + offset, exponent)
            lines.append(
                f"    %rho_x_offset_{sp}_{k} = stablehlo.add %rho_x_{sp}_{k}, %bcast_qmin_{sp} : {tf1}"
            )
            lines.append(
                f"    %rho_x_pow_{sp}_{k} = stablehlo.power %rho_x_offset_{sp}_{k}, %bcast_vel_exp_{sp} : {tf1}"
            )
            lines.append(
                f"    %fall_speed_{sp}_{k} = stablehlo.multiply %bcast_vel_coeff_{sp}, %rho_x_pow_{sp}_{k} : {tf1}"
            )

            # flx_partial = min(rho_x * vc * fall_speed, flx_eff)
            lines.append(
                f"    %flux_raw_{sp}_{k} = stablehlo.multiply %rho_x_{sp}_{k}, %vc_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %flux_scaled_{sp}_{k} = stablehlo.multiply %flux_raw_{sp}_{k}, %fall_speed_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %flx_partial_{sp}_{k} = stablehlo.minimum %flux_scaled_{sp}_{k}, %flx_eff_{sp}_{k} : {tf1}"
            )

            # rhox_prev = (q_prev + q) * 0.5 * rho_prev
            # rho_prev is rho[k-1] (direct reference, not carry)
            if k == 0:
                # At level 0, rho_prev=0 so rhox_prev=0 regardless
                lines.append(
                    f"    %rhox_prev_{sp}_{k} = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}"
                )
            else:
                lines.append(
                    f"    %q_sum_{sp}_{k} = stablehlo.add %q_{sp}_prev{prev}, %{q_name}_{k} : {tf1}"
                )
                lines.append(
                    f"    %q_avg_{sp}_{k} = stablehlo.multiply %q_sum_{sp}_{k}, %bcast_05_1d : {tf1}"
                )
                lines.append(
                    f"    %rhox_prev_{sp}_{k} = stablehlo.multiply %q_avg_{sp}_{k}, %rho_{k-1} : {tf1}"
                )

            # vt_active = vc_prev * prefactor * pow(rhox_prev + offset, exponent)
            # vc_prev is vc[k-1] (direct reference, not carry)
            if k == 0:
                # At level 0, activated_prev=false so vt = 0 regardless
                lines.append(
                    f"    %vt_{sp}_{k} = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}"
                )
            else:
                lines.append(
                    f"    %rhox_prev_offset_{sp}_{k} = stablehlo.add %rhox_prev_{sp}_{k}, %bcast_qmin_{sp} : {tf1}"
                )
                lines.append(
                    f"    %rhox_prev_pow_{sp}_{k} = stablehlo.power %rhox_prev_offset_{sp}_{k}, %bcast_vel_exp_{sp} : {tf1}"
                )
                lines.append(
                    f"    %vel_prefactor_{sp}_{k} = stablehlo.multiply %vc_{sp}_{k-1}, %bcast_vel_coeff_{sp} : {tf1}"
                )
                lines.append(
                    f"    %vt_active_{sp}_{k} = stablehlo.multiply %vel_prefactor_{sp}_{k}, %rhox_prev_pow_{sp}_{k} : {tf1}"
                )
                lines.append(
                    f"    %vt_{sp}_{k} = stablehlo.select %activated_{sp}{prev}, %vt_active_{sp}_{k}, %bcast_0_1d : {tb1}, {tf1}"
                )

            # q_activated = (zeta * (flx_eff - flx_partial)) / ((1 + zeta * vt) * rho)
            lines.append(
                f"    %flx_diff_{sp}_{k} = stablehlo.subtract %flx_eff_{sp}_{k}, %flx_partial_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %num_{sp}_{k} = stablehlo.multiply %zeta_{k}, %flx_diff_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %zeta_vt_{sp}_{k} = stablehlo.multiply %zeta_{k}, %vt_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %denom_inner_{sp}_{k} = stablehlo.add %zeta_vt_{sp}_{k}, %bcast_1_1d : {tf1}"
            )
            lines.append(
                f"    %denom_{sp}_{k} = stablehlo.multiply %denom_inner_{sp}_{k}, %rho_{k} : {tf1}"
            )
            lines.append(
                f"    %q_activated_{sp}_{k} = stablehlo.divide %num_{sp}_{k}, %denom_{sp}_{k} : {tf1}"
            )

            # flx_activated = (q_activated * rho * vt + flx_partial) * 0.5
            lines.append(
                f"    %q_rho_{sp}_{k} = stablehlo.multiply %q_activated_{sp}_{k}, %rho_{k} : {tf1}"
            )
            lines.append(
                f"    %q_rho_vt_{sp}_{k} = stablehlo.multiply %q_rho_{sp}_{k}, %vt_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %flx_sum_{sp}_{k} = stablehlo.add %q_rho_vt_{sp}_{k}, %flx_partial_{sp}_{k} : {tf1}"
            )
            lines.append(
                f"    %flx_activated_{sp}_{k} = stablehlo.multiply %flx_sum_{sp}_{k}, %bcast_05_1d : {tf1}"
            )

            # Select based on activation (q_out, flx_out)
            lines.append(
                f"    %{q_name}{out} = stablehlo.select %activated_{sp}{out}, %q_activated_{sp}_{k}, %{q_name}_{k} : {tb1}, {tf1}"
            )
            lines.append(
                f"    %pflx_{sp}{out} = stablehlo.select %activated_{sp}{out}, %flx_activated_{sp}_{k}, %bcast_0_1d : {tb1}, {tf1}"
            )

            # Carry update: only q_prev (no rho_prev, vc_prev needed)
            lines.append(
                f"    %q_{sp}_prev{out} = stablehlo.add %{q_name}{out}, %bcast_0_1d : {tf1}"
            )
            lines.append("")

        # --- Post-precip intermediates (using precip outputs directly) ---
        lines.append(f"    // Post-precip intermediates for level {k}")
        # qliq_new = qc + qr_out
        lines.append(f"    %qliq_new_{k} = stablehlo.add %qc_{k}, %qr{out} : {tf1}")
        # qice_new = qs_out + qi_out + qg_out
        lines.append(f"    %qice_new_tmp_{k} = stablehlo.add %qs{out}, %qi{out} : {tf1}")
        lines.append(f"    %qice_new_{k} = stablehlo.add %qice_new_tmp_{k}, %qg{out} : {tf1}")
        # pflx_tot = ps + pi + pg
        lines.append(f"    %pflx_tot_tmp_{k} = stablehlo.add %pflx_s{out}, %pflx_i{out} : {tf1}")
        lines.append(f"    %pflx_tot_{k} = stablehlo.add %pflx_tot_tmp_{k}, %pflx_g{out} : {tf1}")
        # t_kp1: t[k+1] for k < last_lev, else t[k]
        if k < last_lev:
            # Slice t[k+1] (lazy — only sliced when needed)
            lines.append(
                f"    %t_kp1_{k} = stablehlo.slice %arg10 [{k+1}:{k+2}, 0:{ncells}] : ({tf}) -> {tf1}"
            )
        else:
            lines.append(f"    %t_kp1_{k} = stablehlo.add %t_{k}, %bcast_0_1d : {tf1}")
        # kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g
        lines.append(f"    %kmin_rsig_tmp1_{k} = stablehlo.or %kmin_r_{k}, %kmin_s_{k} : {tb1}")
        lines.append(
            f"    %kmin_rsig_tmp2_{k} = stablehlo.or %kmin_rsig_tmp1_{k}, %kmin_i_{k} : {tb1}"
        )
        lines.append(f"    %kmin_rsig_{k} = stablehlo.or %kmin_rsig_tmp2_{k}, %kmin_g_{k} : {tb1}")
        lines.append("")

        # --- Temperature update scan for this level ---
        lines.append(f"    // Temperature update level {k}")
        prev_act = "t_scan_activated_init" if k == 0 else f"t_scan_activated_{k-1}"
        prev_eflx = "eflx_init" if k == 0 else f"eflx_out_{k-1}"

        lines.append(
            f"    %t_scan_activated_{k} = stablehlo.or %{prev_act}, %kmin_rsig_{k} : {tb1}"
        )

        # cvd_t_kp1 = cvd * t_kp1[k]
        lines.append(f"    %cvd_t_kp1_{k} = stablehlo.multiply %bcast_cvd_1d, %t_kp1_{k} : {tf1}")

        # eflx_new = dt * (pr * (clw*t - cvd_t_kp1 - lvc) + pflx_tot * (ci*t - cvd_t_kp1 - lsc))
        lines.append(f"    %clw_t_{k} = stablehlo.multiply %bcast_clw_1d, %t_{k} : {tf1}")
        lines.append(f"    %pr_term1_{k} = stablehlo.subtract %clw_t_{k}, %cvd_t_kp1_{k} : {tf1}")
        lines.append(f"    %pr_term2_{k} = stablehlo.subtract %pr_term1_{k}, %bcast_lvc_1d : {tf1}")
        lines.append(
            f"    %pr_contrib_{k} = stablehlo.multiply %pflx_r{out}, %pr_term2_{k} : {tf1}"
        )

        lines.append(f"    %ci_t_{k} = stablehlo.multiply %bcast_ci_1d, %t_{k} : {tf1}")
        lines.append(f"    %pf_term1_{k} = stablehlo.subtract %ci_t_{k}, %cvd_t_kp1_{k} : {tf1}")
        lines.append(f"    %pf_term2_{k} = stablehlo.subtract %pf_term1_{k}, %bcast_lsc_1d : {tf1}")
        lines.append(
            f"    %pf_contrib_{k} = stablehlo.multiply %pflx_tot_{k}, %pf_term2_{k} : {tf1}"
        )

        lines.append(f"    %eflx_sum_{k} = stablehlo.add %pr_contrib_{k}, %pf_contrib_{k} : {tf1}")
        lines.append(f"    %eflx_new_{k} = stablehlo.multiply %bcast_30_1d, %eflx_sum_{k} : {tf1}")

        # e_int = ei_old[k] + prev_eflx - eflx_new
        lines.append(f"    %e_int_tmp_{k} = stablehlo.add %ei_old_{k}, %{prev_eflx} : {tf1}")
        lines.append(f"    %e_int_{k} = stablehlo.subtract %e_int_tmp_{k}, %eflx_new_{k} : {tf1}")

        # qtot = qliq_new + qice_new + qv
        lines.append(f"    %qtot_tmp_{k} = stablehlo.add %qliq_new_{k}, %qice_new_{k} : {tf1}")
        lines.append(f"    %qtot_{k} = stablehlo.add %qtot_tmp_{k}, %qv_{k} : {tf1}")

        # rho_dz = rho * dz
        lines.append(f"    %rho_dz_{k} = stablehlo.multiply %rho_{k}, %dz_{k} : {tf1}")

        # cv = (cvd*(1-qtot) + cvv*qv + clw*qliq + ci*qice) * rho_dz
        lines.append(f"    %one_m_qt_{k} = stablehlo.subtract %bcast_1_1d, %qtot_{k} : {tf1}")
        lines.append(f"    %cv_t1_{k} = stablehlo.multiply %bcast_cvd_1d, %one_m_qt_{k} : {tf1}")
        lines.append(f"    %cv_t2_{k} = stablehlo.multiply %bcast_cvv_1d, %qv_{k} : {tf1}")
        lines.append(f"    %cv_t3_{k} = stablehlo.multiply %bcast_clw_1d, %qliq_new_{k} : {tf1}")
        lines.append(f"    %cv_t4_{k} = stablehlo.multiply %bcast_ci_1d, %qice_new_{k} : {tf1}")
        lines.append(f"    %cv_s1_{k} = stablehlo.add %cv_t1_{k}, %cv_t2_{k} : {tf1}")
        lines.append(f"    %cv_s2_{k} = stablehlo.add %cv_s1_{k}, %cv_t3_{k} : {tf1}")
        lines.append(f"    %cv_s3_{k} = stablehlo.add %cv_s2_{k}, %cv_t4_{k} : {tf1}")
        lines.append(f"    %cv_{k} = stablehlo.multiply %cv_s3_{k}, %rho_dz_{k} : {tf1}")

        # t_new = (e_int + rho_dz * (qliq*lvc + qice*lsc)) / cv
        lines.append(f"    %qliq_lvc_{k} = stablehlo.multiply %qliq_new_{k}, %bcast_lvc_1d : {tf1}")
        lines.append(f"    %qice_lsc_{k} = stablehlo.multiply %qice_new_{k}, %bcast_lsc_1d : {tf1}")
        lines.append(f"    %lsum_{k} = stablehlo.add %qliq_lvc_{k}, %qice_lsc_{k} : {tf1}")
        lines.append(f"    %rdz_l_{k} = stablehlo.multiply %rho_dz_{k}, %lsum_{k} : {tf1}")
        lines.append(f"    %t_num_{k} = stablehlo.add %e_int_{k}, %rdz_l_{k} : {tf1}")
        lines.append(f"    %t_new_{k} = stablehlo.divide %t_num_{k}, %cv_{k} : {tf1}")

        # eflx_out = select(activated, eflx_new, prev_eflx)
        lines.append(
            f"    %eflx_out_{k} = stablehlo.select %t_scan_activated_{k}, %eflx_new_{k}, %{prev_eflx} : {tb1}, {tf1}"
        )
        # t_out = select(activated, t_new, t[k])
        lines.append(
            f"    %t_out_{k} = stablehlo.select %t_scan_activated_{k}, %t_new_{k}, %t_{k} : {tb1}, {tf1}"
        )
        lines.append("")

    # ================================================================
    # Concatenate all per-level outputs
    # ================================================================
    lines.append("    // ========== CONCATENATE OUTPUTS ==========")

    type_tuple = ", ".join([tf1] * nlev)

    # Precip species outputs
    for name, prefix in [
        ("qr_out_full", "qr_out_"),
        ("qs_out_full", "qs_out_"),
        ("qi_out_full", "qi_out_"),
        ("qg_out_full", "qg_out_"),
    ]:
        args_list = ", ".join([f"%{prefix}{k}" for k in range(nlev)])
        lines.append(
            f"    %{name} = stablehlo.concatenate {args_list}, dim = 0 : ({type_tuple}) -> {tf}"
        )

    # Precip flux outputs
    for name, prefix in [
        ("pflx_r_full", "pflx_r_out_"),
        ("pflx_s_full", "pflx_s_out_"),
        ("pflx_i_full", "pflx_i_out_"),
        ("pflx_g_full", "pflx_g_out_"),
    ]:
        args_list = ", ".join([f"%{prefix}{k}" for k in range(nlev)])
        lines.append(
            f"    %{name} = stablehlo.concatenate {args_list}, dim = 0 : ({type_tuple}) -> {tf}"
        )

    # Temperature output
    t_args = ", ".join([f"%t_out_{k}" for k in range(nlev)])
    lines.append(
        f"    %t_new_full = stablehlo.concatenate {t_args}, dim = 0 : ({type_tuple}) -> {tf}"
    )

    # eflx output
    eflx_args = ", ".join([f"%eflx_out_{k}" for k in range(nlev)])
    lines.append(
        f"    %eflx_full = stablehlo.concatenate {eflx_args}, dim = 0 : ({type_tuple}) -> {tf}"
    )

    # pflx_tot output
    pflx_tot_args = ", ".join([f"%pflx_tot_{k}" for k in range(nlev)])
    lines.append(
        f"    %pflx_tot_full = stablehlo.concatenate {pflx_tot_args}, dim = 0 : ({type_tuple}) -> {tf}"
    )
    lines.append("")

    # ================================================================
    # Final outputs
    # ================================================================
    lines.append("    // ========== FINAL OUTPUTS ==========")

    # pflx_tot_plus_pr = pflx_tot + pr
    lines.append(f"    %pflx_tot_plus_pr = stablehlo.add %pflx_tot_full, %pflx_r_full : {tf}")

    # eflx / dt
    lines.append(f"    %eflx_over_dt = stablehlo.divide %eflx_full, %bcast_30 : {tf}")
    lines.append("")

    # Return: qr, qs, qi, qg, t_new, pflx_tot+pr, pr, ps, pi, pg, eflx/dt
    lines.append(
        f"    return %qr_out_full, %qs_out_full, %qi_out_full, %qg_out_full, %t_new_full, %pflx_tot_plus_pr, %pflx_r_full, %pflx_s_full, %pflx_i_full, %pflx_g_full, %eflx_over_dt : {ret_types}"
    )
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate unrolled StableHLO with transposed layout"
    )
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    if args.output is None:
        args.output = "stablehlo/precip_transposed.stablehlo"

    print("Generating interleaved transposed unrolled StableHLO...")
    print(f"  Layout: tensor<{args.nlev}x{args.ncells}> (nlev x ncells)")
    print("  Interleaved: precip + temp scan fused per-level")
    print("  Reduced carry: no rho_prev/vc_prev (direct references)")
    print("  Outputs: 11 (qr, qs, qi, qg, t_new, pflx_tot+pr, pr, ps, pi, pg, eflx/dt)")

    stablehlo_text = generate_unrolled_transposed(args.nlev, args.ncells)

    import pathlib

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(stablehlo_text)

    print(f"Written to: {args.output}")
    print(f"File size: {len(stablehlo_text) / 1024:.1f} KB")
    print(f"Lines: {stablehlo_text.count(chr(10))}")


if __name__ == "__main__":
    main()
