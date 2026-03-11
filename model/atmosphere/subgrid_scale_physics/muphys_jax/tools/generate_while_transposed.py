#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate StableHLO with a while loop for precipitation + temperature.

Emits a single while loop body processing 4 species + temperature per
iteration. Powers decomposed into exp/log/cbrt/sqrt to avoid XLA
fusion barriers. Phase 1 (ei_old) and Phase 2 (velocity) precomputed
on full tensors, then sliced per level in the loop.

Usage: python generate_while_transposed.py --ncells 327680 --nlev 90
"""

import argparse


def generate_while_transposed(nlev: int = 90, ncells: int = 327680) -> str:
    """Generate StableHLO with while loop for nlev x ncells layout."""

    lines = []

    # Types
    tf = f"tensor<{nlev}x{ncells}xf64>"
    tb = f"tensor<{nlev}x{ncells}xi1>"
    t1f = f"tensor<{ncells}xf64>"
    t1b = f"tensor<{ncells}xi1>"
    ti = "tensor<i64>"

    # Module header
    lines.append(
        f"module @jit_precip_while_transposed_{nlev} attributes "
        f"{{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{"
    )
    lines.append("")

    # ================================================================
    # Function signature
    # ================================================================
    args = ", ".join([
        f"%arg0: {tb}",   # kmin_r
        f"%arg1: {tb}",   # kmin_i
        f"%arg2: {tb}",   # kmin_s
        f"%arg3: {tb}",   # kmin_g
        f"%arg4: {tf}",   # qv
        f"%arg5: {tf}",   # qc
        f"%arg6: {tf}",   # qr
        f"%arg7: {tf}",   # qs
        f"%arg8: {tf}",   # qi
        f"%arg9: {tf}",   # qg
        f"%arg10: {tf}",  # t
        f"%arg11: {tf}",  # rho
        f"%arg12: {tf}",  # dz
    ])
    ret_types = ", ".join([tf] * 11)
    lines.append(f"  func.func public @main({args}) -> ({ret_types}) {{")

    # ================================================================
    # Scalar + broadcast constants for Phase 1/2 (outer scope)
    # ================================================================
    lines.append("    // ========== CONSTANTS FOR PHASE 1/2 ==========")

    scalar_consts = {
        "s_0": "0.0", "s_05": "0.5", "s_1": "1.0", "s_2": "2.0",
        "s_3": "3.0", "s_4": "4.0", "s_10": "10.0", "s_30": "30.0",
        "s_rho0": "1.225",
        "s_cvd": "717.60", "s_cvv": "1407.95",
        "s_clw": "4192.6641119999995", "s_ci": "2108.0",
        "s_lvc": "3135383.2031927998", "s_lsc": "2899657.2009999999",
        "s_tmelt": "273.15", "s_qmin": "1.0e-15",
        "s_tmin": "233.15", "s_tmax": "273.15",
        "s_qsmin": "2.0e-6",
        "s_xa1": "-1.65", "s_xa2": "5.45e-2", "s_xa3": "3.27e-4",
        "s_xb1": "1.42", "s_xb2": "1.19e-2", "s_xb3": "9.60e-5",
        "s_n0s0": "8.00e5", "s_n0s2": "-0.107", "s_n0s3": "13.5",
        "s_n0s5": "1.0e6", "s_n0s7": "1.0e9",
        "s_ams": "0.069",
        "s_neg_one_sixth": "-0.16666666666666666",
        "s_two_thirds": "0.66666666666666667",
    }
    n0s1_val = 13.5 * 5.65e05
    scalar_consts["s_n0s1"] = str(n0s1_val)
    scalar_consts["s_n0s4"] = str(0.5 * n0s1_val)
    scalar_consts["s_n0s6"] = str(1.0e2 * n0s1_val)

    for name, val in scalar_consts.items():
        lines.append(f"    %{name} = stablehlo.constant dense<{val}> : tensor<f64>")
    lines.append("")

    # Broadcast scalars to full (nlev, ncells) for Phase 1/2
    for name in scalar_consts:
        bname = name.replace("s_", "b_")
        lines.append(
            f"    %{bname} = stablehlo.broadcast_in_dim %{name}, dims = [] "
            f": (tensor<f64>) -> {tf}"
        )
    lines.append("")

    # ================================================================
    # PHASE 1: Internal energy (full tensor)
    # ================================================================
    lines.append("    // ========== PHASE 1: INTERNAL ENERGY ==========")
    lines.append(f"    %qliq_old = stablehlo.add %arg5, %arg6 : {tf}")
    lines.append(f"    %qice_t1 = stablehlo.add %arg7, %arg8 : {tf}")
    lines.append(f"    %qice_old = stablehlo.add %qice_t1, %arg9 : {tf}")
    lines.append(f"    %qtot_t1 = stablehlo.add %arg4, %qliq_old : {tf}")
    lines.append(f"    %qtot_old = stablehlo.add %qtot_t1, %qice_old : {tf}")
    lines.append(f"    %one_m_qt = stablehlo.subtract %b_1, %qtot_old : {tf}")
    lines.append(f"    %cv1 = stablehlo.multiply %b_cvd, %one_m_qt : {tf}")
    lines.append(f"    %cv2 = stablehlo.multiply %b_cvv, %arg4 : {tf}")
    lines.append(f"    %cv3 = stablehlo.multiply %b_clw, %qliq_old : {tf}")
    lines.append(f"    %cv4 = stablehlo.multiply %b_ci, %qice_old : {tf}")
    lines.append(f"    %cvs1 = stablehlo.add %cv1, %cv2 : {tf}")
    lines.append(f"    %cvs2 = stablehlo.add %cvs1, %cv3 : {tf}")
    lines.append(f"    %cv_old = stablehlo.add %cvs2, %cv4 : {tf}")
    lines.append(f"    %cv_t = stablehlo.multiply %cv_old, %arg10 : {tf}")
    lines.append(f"    %ql_lvc = stablehlo.multiply %qliq_old, %b_lvc : {tf}")
    lines.append(f"    %qi_lsc = stablehlo.multiply %qice_old, %b_lsc : {tf}")
    lines.append(f"    %ei_t1 = stablehlo.subtract %cv_t, %ql_lvc : {tf}")
    lines.append(f"    %ei_inner = stablehlo.subtract %ei_t1, %qi_lsc : {tf}")
    lines.append(f"    %rho_dz_f = stablehlo.multiply %arg11, %arg12 : {tf}")
    lines.append(f"    %ei_old = stablehlo.multiply %rho_dz_f, %ei_inner : {tf}")
    lines.append("")

    # ================================================================
    # PHASE 2: Velocity scale factors (full tensor)
    # ================================================================
    lines.append("    // ========== PHASE 2: VELOCITY SCALE FACTORS ==========")
    lines.append(f"    %dz2 = stablehlo.multiply %arg12, %b_2 : {tf}")
    lines.append(f"    %zeta_f = stablehlo.divide %b_30, %dz2 : {tf}")
    lines.append(f"    %rr = stablehlo.divide %b_rho0, %arg11 : {tf}")
    lines.append(f"    %xrho = stablehlo.sqrt %rr : {tf}")
    # xrho^(2/3) = cbrt(xrho)^2  [FAST: cbrt+mul vs power fusion barrier]
    lines.append(f"    %xrho_cbrt = stablehlo.cbrt %xrho : {tf}")
    lines.append(f"    %vc_i_f = stablehlo.multiply %xrho_cbrt, %xrho_cbrt : {tf}")
    lines.append("")

    # Snow velocity scale factor
    lines.append(f"    %sn_hi = stablehlo.minimum %arg10, %b_tmax : {tf}")
    lines.append(f"    %sn_cl = stablehlo.maximum %sn_hi, %b_tmin : {tf}")
    lines.append(f"    %sn_tc = stablehlo.subtract %sn_cl, %b_tmelt : {tf}")
    lines.append(f"    %xa3tc = stablehlo.multiply %b_xa3, %sn_tc : {tf}")
    lines.append(f"    %xa2s = stablehlo.add %b_xa2, %xa3tc : {tf}")
    lines.append(f"    %xa2tc = stablehlo.multiply %xa2s, %sn_tc : {tf}")
    lines.append(f"    %exparg = stablehlo.add %b_xa1, %xa2tc : {tf}")
    # 10^exparg = exp(exparg * ln10)  [FAST: exp+mul vs power fusion barrier]
    import math
    lines.append(f"    %s_ln10 = stablehlo.constant dense<{math.log(10.0):.17e}> : tensor<f64>")
    lines.append(f"    %b_ln10 = stablehlo.broadcast_in_dim %s_ln10, dims = [] : (tensor<f64>) -> {tf}")
    lines.append(f"    %exparg_ln10 = stablehlo.multiply %exparg, %b_ln10 : {tf}")
    lines.append(f"    %alf = stablehlo.exponential %exparg_ln10 : {tf}")
    lines.append(f"    %xb3tc = stablehlo.multiply %b_xb3, %sn_tc : {tf}")
    lines.append(f"    %xb2s = stablehlo.add %b_xb2, %xb3tc : {tf}")
    lines.append(f"    %xb2tc = stablehlo.multiply %xb2s, %sn_tc : {tf}")
    lines.append(f"    %bet = stablehlo.add %b_xb1, %xb2tc : {tf}")
    lines.append(f"    %qs_qsm = stablehlo.add %arg7, %b_qsmin : {tf}")
    lines.append(f"    %qs_rho = stablehlo.multiply %qs_qsm, %arg11 : {tf}")
    lines.append(f"    %qs_ra = stablehlo.divide %qs_rho, %b_ams : {tf}")
    lines.append(f"    %bet3 = stablehlo.multiply %b_3, %bet : {tf}")
    lines.append(f"    %snexp = stablehlo.subtract %b_4, %bet3 : {tf}")
    # base^variable_exp = exp(variable_exp * log(base))  [FAST: exp+log+mul vs power]
    lines.append(f"    %log_qs_ra = stablehlo.log %qs_ra : {tf}")
    lines.append(f"    %snexp_log = stablehlo.multiply %snexp, %log_qs_ra : {tf}")
    lines.append(f"    %bpow = stablehlo.exponential %snexp_log : {tf}")
    lines.append(f"    %alf2 = stablehlo.multiply %alf, %alf : {tf}")
    lines.append(f"    %alf3 = stablehlo.multiply %alf2, %alf : {tf}")
    lines.append(f"    %n0sn = stablehlo.multiply %b_n0s3, %bpow : {tf}")
    lines.append(f"    %n0s = stablehlo.divide %n0sn, %alf3 : {tf}")
    lines.append(f"    %n2tc = stablehlo.multiply %b_n0s2, %sn_tc : {tf}")
    lines.append(f"    %y = stablehlo.exponential %n2tc : {tf}")
    lines.append(f"    %n4y = stablehlo.multiply %b_n0s4, %y : {tf}")
    lines.append(f"    %nmn = stablehlo.maximum %n4y, %b_n0s5 : {tf}")
    lines.append(f"    %n6y = stablehlo.multiply %b_n0s6, %y : {tf}")
    lines.append(f"    %nmx = stablehlo.minimum %n6y, %b_n0s7 : {tf}")
    lines.append(f"    %ncl = stablehlo.maximum %nmn, %n0s : {tf}")
    lines.append(f"    %ncr = stablehlo.minimum %nmx, %ncl : {tf}")
    lines.append(
        f"    %qs_gt = stablehlo.compare GT, %arg7, %b_qmin, FLOAT : ({tf}, {tf}) -> {tb}"
    )
    lines.append(f"    %sn_num = stablehlo.select %qs_gt, %ncr, %b_n0s0 : {tb}, {tf}")
    # snow_number^(-1/6) = 1/cbrt(sqrt(snow_number))  [FAST: sqrt+cbrt+div vs power]
    lines.append(f"    %sn_sqrt = stablehlo.sqrt %sn_num : {tf}")
    lines.append(f"    %sn_cbrt_sqrt = stablehlo.cbrt %sn_sqrt : {tf}")
    lines.append(f"    %sn_pw = stablehlo.divide %b_1, %sn_cbrt_sqrt : {tf}")
    lines.append(f"    %vc_s_f = stablehlo.multiply %xrho, %sn_pw : {tf}")
    lines.append("")

    # t_kp1 full tensor: shift t by one level
    lines.append("    // t[k+1] for temperature scan")
    lines.append(
        f"    %t_tail = stablehlo.slice %arg10 [1:{nlev}, 0:{ncells}] "
        f": ({tf}) -> tensor<{nlev-1}x{ncells}xf64>"
    )
    lines.append(
        f"    %t_last = stablehlo.slice %arg10 [{nlev-1}:{nlev}, 0:{ncells}] "
        f": ({tf}) -> tensor<1x{ncells}xf64>"
    )
    lines.append(
        f"    %t_kp1_f = stablehlo.concatenate %t_tail, %t_last, dim = 0 "
        f": (tensor<{nlev-1}x{ncells}xf64>, tensor<1x{ncells}xf64>) -> {tf}"
    )
    # kmin_rsig full tensor
    lines.append(f"    %kr_or_ks = stablehlo.or %arg0, %arg2 : {tb}")
    lines.append(f"    %kr_ks_ki = stablehlo.or %kr_or_ks, %arg1 : {tb}")
    lines.append(f"    %krsig_f = stablehlo.or %kr_ks_ki, %arg3 : {tb}")
    lines.append("")

    # ================================================================
    # Build iterargs list for while loop
    # ================================================================
    # The while body is a separate MLIR region that can only reference:
    # 1. Its own iterarg names
    # 2. Constants defined within the body
    # So ALL outer-scope tensors must be passed as iterargs.

    lines.append("    // ========== INITIAL CARRY ==========")
    lines.append(f"    %k_init = stablehlo.constant dense<0> : {ti}")
    for sp in ["r", "s", "i", "g"]:
        lines.append(f"    %qp_{sp}_init = stablehlo.constant dense<0.0> : {t1f}")
        lines.append(f"    %fp_{sp}_init = stablehlo.constant dense<0.0> : {t1f}")
        lines.append(f"    %ap_{sp}_init = stablehlo.constant dense<false> : {t1b}")
    lines.append(f"    %eflx_init = stablehlo.constant dense<0.0> : {t1f}")
    lines.append(f"    %tact_init = stablehlo.constant dense<false> : {t1b}")
    lines.append(f"    %rho_p_init = stablehlo.constant dense<0.0> : {t1f}")
    for sp in ["r", "s", "i", "g"]:
        lines.append(f"    %vcp_{sp}_init = stablehlo.constant dense<0.0> : {t1f}")
    out_names = ["qr_acc", "qs_acc", "qi_acc", "qg_acc",
                 "fr_acc", "fs_acc", "fi_acc", "fg_acc",
                 "t_acc", "pt_acc", "ef_acc"]
    for name in out_names:
        lines.append(f"    %{name}_init = stablehlo.constant dense<0.0> : {tf}")
    lines.append("")

    # Build iterargs: (name, init_value, type, friendly_name)
    # Group 1: mutable carry (20 items)
    iterargs = []
    iterargs.append(("iterArg", "k_init", ti, "k"))
    idx = 0
    for sp in ["r", "s", "i", "g"]:
        iterargs.append((f"iterArg_{idx+1}", f"qp_{sp}_init", t1f, f"qp_{sp}"))
        iterargs.append((f"iterArg_{idx+2}", f"fp_{sp}_init", t1f, f"fp_{sp}"))
        iterargs.append((f"iterArg_{idx+3}", f"ap_{sp}_init", t1b, f"ap_{sp}"))
        idx += 3
    # idx = 12
    iterargs.append((f"iterArg_{idx+1}", "eflx_init", t1f, "eflx"))      # 13
    iterargs.append((f"iterArg_{idx+2}", "tact_init", t1b, "tact"))       # 14
    iterargs.append((f"iterArg_{idx+3}", "rho_p_init", t1f, "rho_p"))     # 15
    idx += 3
    for sp in ["r", "s", "i", "g"]:
        iterargs.append((f"iterArg_{idx+1}", f"vcp_{sp}_init", t1f, f"vcp_{sp}"))
        idx += 1
    # idx = 19, output accumulators at 20-30
    acc_start_idx = idx + 1  # 20
    for name in out_names:
        iterargs.append((f"iterArg_{idx+1}", f"{name}_init", tf, name))
        idx += 1
    # idx = 30

    # Group 2: read-only full tensors (passed through unchanged)
    # These are needed inside the body for dynamic_slice
    ro_start_idx = idx + 1  # 31
    read_only_tensors = [
        ("kmin_r", "arg0", tb),
        ("kmin_i", "arg1", tb),
        ("kmin_s", "arg2", tb),
        ("kmin_g", "arg3", tb),
        ("qv_f", "arg4", tf),
        ("qc_f", "arg5", tf),
        ("qr_f", "arg6", tf),
        ("qs_f", "arg7", tf),
        ("qi_f", "arg8", tf),
        ("qg_f", "arg9", tf),
        ("t_f", "arg10", tf),
        ("rho_f", "arg11", tf),
        ("dz_f", "arg12", tf),
        ("zeta_ro", "zeta_f", tf),
        ("ei_ro", "ei_old", tf),
        ("xrho_ro", "xrho", tf),
        ("vcs_ro", "vc_s_f", tf),
        ("vci_ro", "vc_i_f", tf),
        ("tkp1_ro", "t_kp1_f", tf),
        ("krsig_ro", "krsig_f", tb),
    ]
    for friendly, init_name, rotype in read_only_tensors:
        iterargs.append((f"iterArg_{idx+1}", init_name, rotype, friendly))
        idx += 1

    n_iterargs = len(iterargs)  # 51

    # Build name_map: friendly_name -> iterarg_name
    name_map = {ia[3]: ia[0] for ia in iterargs}

    # Build while header
    while_args = ", ".join(f"%{ia[0]} = %{ia[1]}" for ia in iterargs)
    while_types = ", ".join(ia[2] for ia in iterargs)

    lines.append("    // ========== WHILE LOOP ==========")
    lines.append(
        f"    %result:{n_iterargs} = stablehlo.while({while_args}) : {while_types}"
    )

    # ---- Condition ----
    k_ia = name_map["k"]
    lines.append(f"     cond {{")
    lines.append(f"      %nlev_c = stablehlo.constant dense<{nlev}> : {ti}")
    lines.append(
        f"      %cmp = stablehlo.compare  LT, %{k_ia}, %nlev_c, "
        f" SIGNED : ({ti}, {ti}) -> tensor<i1>"
    )
    lines.append(f"      stablehlo.return %cmp : tensor<i1>")
    lines.append(f"    }} do {{")
    lines.append("")

    # ---- Body ----
    # All constants used in the body are defined here (inside the body region)
    lines.append("      // Body constants")
    lines.append(f"      %cst_0 = stablehlo.constant dense<0.0> : {t1f}")
    lines.append(f"      %cst_05 = stablehlo.constant dense<0.5> : {t1f}")
    lines.append(f"      %cst_1 = stablehlo.constant dense<1.0> : {t1f}")
    lines.append(f"      %cst_2 = stablehlo.constant dense<2.0> : {t1f}")
    lines.append(f"      %cst_30 = stablehlo.constant dense<30.0> : {t1f}")
    lines.append(f"      %cst_cvd = stablehlo.constant dense<717.60> : {t1f}")
    lines.append(f"      %cst_cvv = stablehlo.constant dense<1407.95> : {t1f}")
    lines.append(f"      %cst_clw = stablehlo.constant dense<4192.6641119999995> : {t1f}")
    lines.append(f"      %cst_ci = stablehlo.constant dense<2108.0> : {t1f}")
    lines.append(f"      %cst_lvc = stablehlo.constant dense<3135383.2031927998> : {t1f}")
    lines.append(f"      %cst_lsc = stablehlo.constant dense<2899657.2009999999> : {t1f}")
    vel_params = {
        "r": ("14.58", "0.111", "1.0e-12"),
        "s": ("57.8", "0.16666666666666666", "1.0e-12"),
        "i": ("1.25", "0.16", "1.0e-12"),
        "g": ("12.24", "0.217", "1.0e-08"),
    }
    for sp, (coeff, exp, qmin) in vel_params.items():
        lines.append(f"      %vel_coeff_{sp} = stablehlo.constant dense<{coeff}> : {t1f}")
        lines.append(f"      %vel_exp_{sp} = stablehlo.constant dense<{exp}> : {t1f}")
        lines.append(f"      %qmin_{sp} = stablehlo.constant dense<{qmin}> : {t1f}")
    lines.append(f"      %zero_idx = stablehlo.constant dense<0> : {ti}")
    lines.append("")

    # Dynamic slice inputs at position k from read-only iterargs
    lines.append("      // Slice inputs at level k")
    # Map from slice output name -> (read-only friendly name, src_type, dst_type, dtype)
    slice_inputs = {
        "kr_k": ("kmin_r", tb, t1b, "i1"),
        "ks_k": ("kmin_s", tb, t1b, "i1"),
        "ki_k": ("kmin_i", tb, t1b, "i1"),
        "kg_k": ("kmin_g", tb, t1b, "i1"),
        "qv_k": ("qv_f", tf, t1f, "f64"),
        "qc_k": ("qc_f", tf, t1f, "f64"),
        "qr_k": ("qr_f", tf, t1f, "f64"),
        "qs_k": ("qs_f", tf, t1f, "f64"),
        "qi_k": ("qi_f", tf, t1f, "f64"),
        "qg_k": ("qg_f", tf, t1f, "f64"),
        "t_k": ("t_f", tf, t1f, "f64"),
        "rho_k": ("rho_f", tf, t1f, "f64"),
        "dz_k": ("dz_f", tf, t1f, "f64"),
        "zeta_k": ("zeta_ro", tf, t1f, "f64"),
        "ei_k": ("ei_ro", tf, t1f, "f64"),
        "vcr_k": ("xrho_ro", tf, t1f, "f64"),
        "vcs_k": ("vcs_ro", tf, t1f, "f64"),
        "vci_k": ("vci_ro", tf, t1f, "f64"),
        "vcg_k": ("xrho_ro", tf, t1f, "f64"),
        "tkp1_k": ("tkp1_ro", tf, t1f, "f64"),
        "tmsk_k": ("krsig_ro", tb, t1b, "i1"),
    }

    for out_name, (ro_friendly, src_type, dst_type, dtype) in slice_inputs.items():
        src_ia = name_map[ro_friendly]
        lines.append(
            f"      %{out_name}_2d = stablehlo.dynamic_slice %{src_ia}, %{k_ia}, %zero_idx, "
            f"sizes = [1, {ncells}] : ({src_type}, {ti}, {ti}) -> tensor<1x{ncells}x{dtype}>"
        )
        lines.append(
            f"      %{out_name} = stablehlo.reshape %{out_name}_2d "
            f": (tensor<1x{ncells}x{dtype}>) -> tensor<{ncells}x{dtype}>"
        )
    lines.append("")

    # ---- 4 species precipitation ----
    species = [
        ("r", "qr_k", "kr_k", "vcr_k"),
        ("s", "qs_k", "ks_k", "vcs_k"),
        ("i", "qi_k", "ki_k", "vci_k"),
        ("g", "qg_k", "kg_k", "vcg_k"),
    ]

    for sp, q_in, mask_in, vc_in in species:
        ap_name = name_map[f"ap_{sp}"]
        qp_name = name_map[f"qp_{sp}"]
        fp_name = name_map[f"fp_{sp}"]
        vcp_name = name_map[f"vcp_{sp}"]
        rho_p_name = name_map["rho_p"]

        lines.append(f"      // ---- Species {sp} ----")
        lines.append(f"      %act_{sp} = stablehlo.or %{ap_name}, %{mask_in} : {t1b}")
        lines.append(f"      %rx_{sp} = stablehlo.multiply %{q_in}, %rho_k : {t1f}")
        lines.append(f"      %t1_{sp} = stablehlo.divide %rx_{sp}, %zeta_k : {t1f}")
        lines.append(f"      %t2_{sp} = stablehlo.multiply %{fp_name}, %cst_2 : {t1f}")
        lines.append(f"      %fe_{sp} = stablehlo.add %t1_{sp}, %t2_{sp} : {t1f}")
        lines.append(f"      %rxo_{sp} = stablehlo.add %rx_{sp}, %qmin_{sp} : {t1f}")
        # FAST POWER: x^exp -> exp(exp*log(x)) for r/i/g, cbrt(sqrt(x)) for s
        if sp == "s":
            # Snow: exponent = 1/6 -> cbrt(sqrt(x)) [exact, ~46 cycles, fuses well]
            lines.append(f"      %rxp_sqrt_{sp} = stablehlo.sqrt %rxo_{sp} : {t1f}")
            lines.append(f"      %rxp_{sp} = stablehlo.cbrt %rxp_sqrt_{sp} : {t1f}")
        else:
            # Rain/ice/graupel: exp(c*log(x)) — power blocks XLA fusion!
            lines.append(f"      %rxp_log_{sp} = stablehlo.log %rxo_{sp} : {t1f}")
            lines.append(f"      %rxp_exparg_{sp} = stablehlo.multiply %vel_exp_{sp}, %rxp_log_{sp} : {t1f}")
            lines.append(f"      %rxp_{sp} = stablehlo.exponential %rxp_exparg_{sp} : {t1f}")
        lines.append(f"      %fs_{sp} = stablehlo.multiply %vel_coeff_{sp}, %rxp_{sp} : {t1f}")
        lines.append(f"      %fraw_{sp} = stablehlo.multiply %rx_{sp}, %{vc_in} : {t1f}")
        lines.append(f"      %fscl_{sp} = stablehlo.multiply %fraw_{sp}, %fs_{sp} : {t1f}")
        lines.append(f"      %fp_{sp}_new = stablehlo.minimum %fscl_{sp}, %fe_{sp} : {t1f}")
        lines.append(f"      %qsum_{sp} = stablehlo.add %{qp_name}, %{q_in} : {t1f}")
        lines.append(f"      %qa_{sp} = stablehlo.multiply %qsum_{sp}, %cst_05 : {t1f}")
        lines.append(f"      %rxp2_{sp} = stablehlo.multiply %qa_{sp}, %{rho_p_name} : {t1f}")
        lines.append(f"      %rxpo_{sp} = stablehlo.add %rxp2_{sp}, %qmin_{sp} : {t1f}")
        # FAST POWER: same decomposition for rhox_prev
        if sp == "s":
            lines.append(f"      %rxpp_sqrt_{sp} = stablehlo.sqrt %rxpo_{sp} : {t1f}")
            lines.append(f"      %rxpp_{sp} = stablehlo.cbrt %rxpp_sqrt_{sp} : {t1f}")
        else:
            lines.append(f"      %rxpp_log_{sp} = stablehlo.log %rxpo_{sp} : {t1f}")
            lines.append(f"      %rxpp_exparg_{sp} = stablehlo.multiply %vel_exp_{sp}, %rxpp_log_{sp} : {t1f}")
            lines.append(f"      %rxpp_{sp} = stablehlo.exponential %rxpp_exparg_{sp} : {t1f}")
        lines.append(f"      %vpf_{sp} = stablehlo.multiply %{vcp_name}, %vel_coeff_{sp} : {t1f}")
        lines.append(f"      %vta_{sp} = stablehlo.multiply %vpf_{sp}, %rxpp_{sp} : {t1f}")
        lines.append(f"      %vt_{sp} = stablehlo.select %{ap_name}, %vta_{sp}, %cst_0 : {t1b}, {t1f}")
        lines.append(f"      %fd_{sp} = stablehlo.subtract %fe_{sp}, %fp_{sp}_new : {t1f}")
        lines.append(f"      %nm_{sp} = stablehlo.multiply %zeta_k, %fd_{sp} : {t1f}")
        lines.append(f"      %zv_{sp} = stablehlo.multiply %zeta_k, %vt_{sp} : {t1f}")
        lines.append(f"      %di_{sp} = stablehlo.add %zv_{sp}, %cst_1 : {t1f}")
        lines.append(f"      %dn_{sp} = stablehlo.multiply %di_{sp}, %rho_k : {t1f}")
        lines.append(f"      %qact_{sp} = stablehlo.divide %nm_{sp}, %dn_{sp} : {t1f}")
        lines.append(f"      %qrho_{sp} = stablehlo.multiply %qact_{sp}, %rho_k : {t1f}")
        lines.append(f"      %qrv_{sp} = stablehlo.multiply %qrho_{sp}, %vt_{sp} : {t1f}")
        lines.append(f"      %fsa_{sp} = stablehlo.add %qrv_{sp}, %fp_{sp}_new : {t1f}")
        lines.append(f"      %fact_{sp} = stablehlo.multiply %fsa_{sp}, %cst_05 : {t1f}")
        lines.append(f"      %qo_{sp} = stablehlo.select %act_{sp}, %qact_{sp}, %{q_in} : {t1b}, {t1f}")
        lines.append(f"      %fo_{sp} = stablehlo.select %act_{sp}, %fact_{sp}, %cst_0 : {t1b}, {t1f}")
        lines.append("")

    # ---- Post-precip intermediates ----
    eflx_ia = name_map["eflx"]
    tact_ia = name_map["tact"]

    lines.append("      // Post-precip intermediates")
    lines.append(f"      %qliq_n = stablehlo.add %qc_k, %qo_r : {t1f}")
    lines.append(f"      %qit1 = stablehlo.add %qo_s, %qo_i : {t1f}")
    lines.append(f"      %qice_n = stablehlo.add %qit1, %qo_g : {t1f}")
    lines.append(f"      %ptt1 = stablehlo.add %fo_s, %fo_i : {t1f}")
    lines.append(f"      %ptot_k = stablehlo.add %ptt1, %fo_g : {t1f}")
    lines.append("")

    # ---- Temperature update ----
    lines.append("      // Temperature update")
    lines.append(f"      %t_act_n = stablehlo.or %{tact_ia}, %tmsk_k : {t1b}")
    lines.append(f"      %cvd_tkp1 = stablehlo.multiply %cst_cvd, %tkp1_k : {t1f}")
    lines.append(f"      %clw_t = stablehlo.multiply %cst_clw, %t_k : {t1f}")
    lines.append(f"      %prt1 = stablehlo.subtract %clw_t, %cvd_tkp1 : {t1f}")
    lines.append(f"      %prt2 = stablehlo.subtract %prt1, %cst_lvc : {t1f}")
    lines.append(f"      %pr_c = stablehlo.multiply %fo_r, %prt2 : {t1f}")
    lines.append(f"      %ci_t = stablehlo.multiply %cst_ci, %t_k : {t1f}")
    lines.append(f"      %pft1 = stablehlo.subtract %ci_t, %cvd_tkp1 : {t1f}")
    lines.append(f"      %pft2 = stablehlo.subtract %pft1, %cst_lsc : {t1f}")
    lines.append(f"      %pf_c = stablehlo.multiply %ptot_k, %pft2 : {t1f}")
    lines.append(f"      %efsum = stablehlo.add %pr_c, %pf_c : {t1f}")
    lines.append(f"      %ef_new = stablehlo.multiply %cst_30, %efsum : {t1f}")
    lines.append(f"      %eit1 = stablehlo.add %ei_k, %{eflx_ia} : {t1f}")
    lines.append(f"      %e_int = stablehlo.subtract %eit1, %ef_new : {t1f}")
    lines.append(f"      %qtt1 = stablehlo.add %qliq_n, %qice_n : {t1f}")
    lines.append(f"      %qtot = stablehlo.add %qtt1, %qv_k : {t1f}")
    lines.append(f"      %rdz = stablehlo.multiply %rho_k, %dz_k : {t1f}")
    lines.append(f"      %omq = stablehlo.subtract %cst_1, %qtot : {t1f}")
    lines.append(f"      %cvt1 = stablehlo.multiply %cst_cvd, %omq : {t1f}")
    lines.append(f"      %cvt2 = stablehlo.multiply %cst_cvv, %qv_k : {t1f}")
    lines.append(f"      %cvt3 = stablehlo.multiply %cst_clw, %qliq_n : {t1f}")
    lines.append(f"      %cvt4 = stablehlo.multiply %cst_ci, %qice_n : {t1f}")
    lines.append(f"      %cvs1b = stablehlo.add %cvt1, %cvt2 : {t1f}")
    lines.append(f"      %cvs2b = stablehlo.add %cvs1b, %cvt3 : {t1f}")
    lines.append(f"      %cvs3b = stablehlo.add %cvs2b, %cvt4 : {t1f}")
    lines.append(f"      %cv_k = stablehlo.multiply %cvs3b, %rdz : {t1f}")
    lines.append(f"      %ql_l = stablehlo.multiply %qliq_n, %cst_lvc : {t1f}")
    lines.append(f"      %qi_l = stablehlo.multiply %qice_n, %cst_lsc : {t1f}")
    lines.append(f"      %lsum = stablehlo.add %ql_l, %qi_l : {t1f}")
    lines.append(f"      %rdl = stablehlo.multiply %rdz, %lsum : {t1f}")
    lines.append(f"      %tnum = stablehlo.add %e_int, %rdl : {t1f}")
    lines.append(f"      %t_new = stablehlo.divide %tnum, %cv_k : {t1f}")
    lines.append(f"      %ef_out = stablehlo.select %t_act_n, %ef_new, %{eflx_ia} : {t1b}, {t1f}")
    lines.append(f"      %t_out = stablehlo.select %t_act_n, %t_new, %t_k : {t1b}, {t1f}")
    lines.append("")

    # ---- Update output accumulators via dynamic_update_slice ----
    lines.append("      // Update output accumulators")
    acc_updates = [
        ("qr_acc", "qo_r"), ("qs_acc", "qo_s"), ("qi_acc", "qo_i"), ("qg_acc", "qo_g"),
        ("fr_acc", "fo_r"), ("fs_acc", "fo_s"), ("fi_acc", "fo_i"), ("fg_acc", "fo_g"),
        ("t_acc", "t_out"), ("pt_acc", "ptot_k"), ("ef_acc", "ef_out"),
    ]
    for acc_friendly, val_name in acc_updates:
        acc_ia = name_map[acc_friendly]
        lines.append(
            f"      %{val_name}_2d = stablehlo.reshape %{val_name} "
            f": ({t1f}) -> tensor<1x{ncells}xf64>"
        )
        lines.append(
            f"      %{acc_friendly}_new = stablehlo.dynamic_update_slice %{acc_ia}, %{val_name}_2d, "
            f"%{k_ia}, %zero_idx : ({tf}, tensor<1x{ncells}xf64>, {ti}, {ti}) -> {tf}"
        )
    lines.append("")

    # Increment counter
    lines.append(f"      %one = stablehlo.constant dense<1> : {ti}")
    lines.append(f"      %k_next = stablehlo.add %{k_ia}, %one : {ti}")
    lines.append("")

    # Build return values in the same order as iterargs
    # Group 1 (mutable carry): k_next, species carry, temp carry, rho/vc_prev, accumulators
    ret_vals = ["%k_next"]
    for sp in ["r", "s", "i", "g"]:
        ret_vals.extend([f"%qo_{sp}", f"%fo_{sp}", f"%act_{sp}"])
    ret_vals.extend(["%ef_out", "%t_act_n", "%rho_k"])
    vc_k_names = {"r": "vcr_k", "s": "vcs_k", "i": "vci_k", "g": "vcg_k"}
    for sp in ["r", "s", "i", "g"]:
        ret_vals.append(f"%{vc_k_names[sp]}")
    for acc_friendly, _ in acc_updates:
        ret_vals.append(f"%{acc_friendly}_new")

    # Group 2 (read-only): pass through unchanged using iterarg names
    for friendly, _, _ in read_only_tensors:
        ret_vals.append(f"%{name_map[friendly]}")

    lines.append(
        f"      stablehlo.return {', '.join(ret_vals)} : {while_types}"
    )
    lines.append(f"    }}")
    lines.append("")

    # ================================================================
    # Extract results from while loop
    # ================================================================
    lines.append("    // ========== COMPUTE FINAL OUTPUTS ==========")

    # Output accumulator indices
    acc_start = acc_start_idx  # 20
    pt_idx = acc_start + out_names.index("pt_acc")
    fr_idx = acc_start + out_names.index("fr_acc")
    lines.append(f"    %pflx_plus_pr = stablehlo.add %result#{pt_idx}, %result#{fr_idx} : {tf}")

    ef_idx = acc_start + out_names.index("ef_acc")
    lines.append(f"    %bcast_30_full = stablehlo.broadcast_in_dim %s_30, dims = [] : (tensor<f64>) -> {tf}")
    lines.append(f"    %eflx_div_dt = stablehlo.divide %result#{ef_idx}, %bcast_30_full : {tf}")
    lines.append("")

    qr_idx = acc_start + out_names.index("qr_acc")
    qs_idx = acc_start + out_names.index("qs_acc")
    qi_idx = acc_start + out_names.index("qi_acc")
    qg_idx = acc_start + out_names.index("qg_acc")
    t_idx = acc_start + out_names.index("t_acc")
    fs_idx = acc_start + out_names.index("fs_acc")
    fi_idx = acc_start + out_names.index("fi_acc")
    fg_idx = acc_start + out_names.index("fg_acc")

    ret_vals_str = ", ".join([
        f"%result#{qr_idx}", f"%result#{qs_idx}", f"%result#{qi_idx}", f"%result#{qg_idx}",
        f"%result#{t_idx}", "%pflx_plus_pr",
        f"%result#{fr_idx}", f"%result#{fs_idx}", f"%result#{fi_idx}", f"%result#{fg_idx}",
        "%eflx_div_dt",
    ])
    lines.append(f"    return {ret_vals_str} : {ret_types}")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate while-loop StableHLO with transposed layout"
    )
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    if args.output is None:
        args.output = "stablehlo/precip_while_transposed.stablehlo"

    print("Generating while-loop transposed StableHLO...")
    print(f"  Layout: tensor<{args.nlev}x{args.ncells}> (nlev x ncells)")
    print("  Single while loop: 4 species precip + temp update per iteration")
    print("  Fast power: zero stablehlo.power (exp+log, cbrt+sqrt decompositions)")

    stablehlo_text = generate_while_transposed(args.nlev, args.ncells)

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(stablehlo_text)

    print(f"Written to: {args.output}")
    print(f"File size: {len(stablehlo_text) / 1024:.1f} KB")
    print(f"Lines: {stablehlo_text.count(chr(10))}")
    print(f"Total iterargs: {51} (20 mutable carry + 11 accumulators + 20 read-only)")


if __name__ == "__main__":
    main()
