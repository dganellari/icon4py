#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate StableHLO with a while loop (not unrolled) for the precipitation scan.

Same physics as generate_unrolled_transposed.py, but the 90-level vertical scan
is expressed as a stablehlo.while loop instead of being unrolled. This should:
1. Eliminate ~1800 slice ops and ~11 concatenate ops
2. Reduce the IR from ~20K lines to ~500 lines
3. Let XLA compile the scan as fewer kernels instead of ~186

Usage:
    python generate_while_transposed.py --ncells 327680 --nlev 90
    python generate_while_transposed.py -o stablehlo/precip_while.stablehlo

Then combine with q_t_update:
    python generate_combined_graupel.py \
        --qt-update stablehlo/qt_update.stablehlo \
        --precip stablehlo/precip_while.stablehlo \
        -o stablehlo/graupel_while_combined.stablehlo
"""

import argparse
import math


def generate_while_transposed(nlev: int = 90, ncells: int = 327680) -> str:
    lines = []
    tf = f"tensor<{nlev}x{ncells}xf64>"
    tb = f"tensor<{nlev}x{ncells}xi1>"
    tf1 = f"tensor<1x{ncells}xf64>"
    tb1 = f"tensor<1x{ncells}xi1>"
    ti = "tensor<i64>"

    # While loop tuple layout:
    # 0: counter, 1-4: pflx_r/s/i/g, 5-8: activated_r/s/i/g,
    # 9-12: q_prev_r/s/i/g, 13: t_scan_activated, 14: eflx,
    # 15-25: accumulators [qr,qs,qi,qg,pflx_r,pflx_s,pflx_i,pflx_g,t,eflx,pflx_tot]
    ct = [ti] + [tf1]*4 + [tb1]*4 + [tf1]*4 + [tb1] + [tf1] + [tf]*11
    tt = f"tuple<{', '.join(ct)}>"

    lines.append(f"module @jit_precip_while_transposed_{nlev} attributes "
                 f"{{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{")
    args = ", ".join([f"%arg0: {tb}", f"%arg1: {tb}", f"%arg2: {tb}", f"%arg3: {tb}",
                      f"%arg4: {tf}", f"%arg5: {tf}", f"%arg6: {tf}", f"%arg7: {tf}",
                      f"%arg8: {tf}", f"%arg9: {tf}", f"%arg10: {tf}", f"%arg11: {tf}",
                      f"%arg12: {tf}"])
    ret_types = ", ".join([tf] * 11)
    lines.append(f"  func.func public @main({args}) -> ({ret_types}) {{")

    # Constants
    lines.append("    %cst_0 = stablehlo.constant dense<0.0> : tensor<f64>")
    lines.append("    %cst_05 = stablehlo.constant dense<0.5> : tensor<f64>")
    lines.append("    %cst_1 = stablehlo.constant dense<1.0> : tensor<f64>")
    lines.append("    %cst_2 = stablehlo.constant dense<2.0> : tensor<f64>")
    lines.append("    %cst_3 = stablehlo.constant dense<3.0> : tensor<f64>")
    lines.append("    %cst_4 = stablehlo.constant dense<4.0> : tensor<f64>")
    lines.append("    %cst_30 = stablehlo.constant dense<30.0> : tensor<f64>")
    lines.append("    %false = stablehlo.constant dense<false> : tensor<i1>")
    lines.append(f"    %c0 = stablehlo.constant dense<0> : {ti}")
    lines.append(f"    %c1 = stablehlo.constant dense<1> : {ti}")
    lines.append(f"    %c_nlev = stablehlo.constant dense<{nlev}> : {ti}")
    lines.append(f"    %c_nlev_m1 = stablehlo.constant dense<{nlev-1}> : {ti}")
    lines.append("    %cst_rho0 = stablehlo.constant dense<1.225> : tensor<f64>")
    lines.append("    %cst_cvd = stablehlo.constant dense<717.60> : tensor<f64>")
    lines.append("    %cst_cvv = stablehlo.constant dense<1407.95> : tensor<f64>")
    lines.append("    %cst_clw = stablehlo.constant dense<4192.6641119999995> : tensor<f64>")
    lines.append("    %cst_ci = stablehlo.constant dense<2108.0> : tensor<f64>")
    lines.append("    %cst_lvc = stablehlo.constant dense<3135383.2031927998> : tensor<f64>")
    lines.append("    %cst_lsc = stablehlo.constant dense<2899657.2009999999> : tensor<f64>")
    lines.append("    %cst_tmelt = stablehlo.constant dense<273.15> : tensor<f64>")
    lines.append("    %cst_qmin = stablehlo.constant dense<1.0e-15> : tensor<f64>")
    lines.append("    %cst_tmin = stablehlo.constant dense<233.15> : tensor<f64>")
    lines.append("    %cst_tmax = stablehlo.constant dense<273.15> : tensor<f64>")
    lines.append("    %cst_qsmin = stablehlo.constant dense<2.0e-6> : tensor<f64>")
    lines.append("    %cst_xa1 = stablehlo.constant dense<-1.65> : tensor<f64>")
    lines.append("    %cst_xa2 = stablehlo.constant dense<5.45e-2> : tensor<f64>")
    lines.append("    %cst_xa3 = stablehlo.constant dense<3.27e-4> : tensor<f64>")
    lines.append("    %cst_xb1 = stablehlo.constant dense<1.42> : tensor<f64>")
    lines.append("    %cst_xb2 = stablehlo.constant dense<1.19e-2> : tensor<f64>")
    lines.append("    %cst_xb3 = stablehlo.constant dense<9.60e-5> : tensor<f64>")
    lines.append("    %cst_n0s0 = stablehlo.constant dense<8.00e5> : tensor<f64>")
    n0s1 = 13.5 * 5.65e05
    lines.append(f"    %cst_n0s1 = stablehlo.constant dense<{n0s1}> : tensor<f64>")
    lines.append("    %cst_n0s2 = stablehlo.constant dense<-0.107> : tensor<f64>")
    lines.append("    %cst_n0s3 = stablehlo.constant dense<13.5> : tensor<f64>")
    lines.append(f"    %cst_n0s4 = stablehlo.constant dense<{0.5*n0s1}> : tensor<f64>")
    lines.append("    %cst_n0s5 = stablehlo.constant dense<1.0e6> : tensor<f64>")
    lines.append(f"    %cst_n0s6 = stablehlo.constant dense<{1e2*n0s1}> : tensor<f64>")
    lines.append("    %cst_n0s7 = stablehlo.constant dense<1.0e9> : tensor<f64>")
    lines.append("    %cst_ams = stablehlo.constant dense<0.069> : tensor<f64>")
    lines.append(f"    %cst_ln10 = stablehlo.constant dense<{math.log(10.0):.17e}> : tensor<f64>")
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

    # Full-size broadcasts
    for n in ["0","1","2","3","4","30"]:
        lines.append(f"    %bcast_{n} = stablehlo.broadcast_in_dim %cst_{n}, dims = [] : (tensor<f64>) -> {tf}")
    lines.append(f"    %bcast_rho0 = stablehlo.broadcast_in_dim %cst_rho0, dims = [] : (tensor<f64>) -> {tf}")
    for n in ["cvd","cvv","clw","ci","lvc","lsc","tmelt","qmin","tmin","tmax","qsmin",
              "xa1","xa2","xa3","xb1","xb2","xb3","n0s0","n0s1","n0s2","n0s3",
              "n0s4","n0s5","n0s6","n0s7","ams","ln10"]:
        lines.append(f"    %bcast_{n} = stablehlo.broadcast_in_dim %cst_{n}, dims = [] : (tensor<f64>) -> {tf}")

    # Phase 1: internal energy
    lines.append(f"    %qliq_old = stablehlo.add %arg5, %arg6 : {tf}")
    lines.append(f"    %qice_old_t = stablehlo.add %arg7, %arg8 : {tf}")
    lines.append(f"    %qice_old = stablehlo.add %qice_old_t, %arg9 : {tf}")
    lines.append(f"    %qtot_old_t = stablehlo.add %arg4, %qliq_old : {tf}")
    lines.append(f"    %qtot_old = stablehlo.add %qtot_old_t, %qice_old : {tf}")
    lines.append(f"    %one_m_qtot = stablehlo.subtract %bcast_1, %qtot_old : {tf}")
    lines.append(f"    %cv1 = stablehlo.multiply %bcast_cvd, %one_m_qtot : {tf}")
    lines.append(f"    %cv2 = stablehlo.multiply %bcast_cvv, %arg4 : {tf}")
    lines.append(f"    %cv3 = stablehlo.multiply %bcast_clw, %qliq_old : {tf}")
    lines.append(f"    %cv4 = stablehlo.multiply %bcast_ci, %qice_old : {tf}")
    lines.append(f"    %cvs1 = stablehlo.add %cv1, %cv2 : {tf}")
    lines.append(f"    %cvs2 = stablehlo.add %cvs1, %cv3 : {tf}")
    lines.append(f"    %cv_old = stablehlo.add %cvs2, %cv4 : {tf}")
    lines.append(f"    %cv_t = stablehlo.multiply %cv_old, %arg10 : {tf}")
    lines.append(f"    %ql_lvc = stablehlo.multiply %qliq_old, %bcast_lvc : {tf}")
    lines.append(f"    %qi_lsc = stablehlo.multiply %qice_old, %bcast_lsc : {tf}")
    lines.append(f"    %ei1 = stablehlo.subtract %cv_t, %ql_lvc : {tf}")
    lines.append(f"    %ei2 = stablehlo.subtract %ei1, %qi_lsc : {tf}")
    lines.append(f"    %rho_dz_full = stablehlo.multiply %arg11, %arg12 : {tf}")
    lines.append(f"    %ei_old_full = stablehlo.multiply %rho_dz_full, %ei2 : {tf}")

    # Phase 2: velocity scale factors
    lines.append(f"    %dz2 = stablehlo.multiply %arg12, %bcast_2 : {tf}")
    lines.append(f"    %zeta_full = stablehlo.divide %bcast_30, %dz2 : {tf}")
    lines.append(f"    %rho_ratio = stablehlo.divide %bcast_rho0, %arg11 : {tf}")
    lines.append(f"    %xrho = stablehlo.sqrt %rho_ratio : {tf}")
    lines.append(f"    %xrho_cbrt = stablehlo.cbrt %xrho : {tf}")
    lines.append(f"    %vc_i_full = stablehlo.multiply %xrho_cbrt, %xrho_cbrt : {tf}")
    lines.append(f"    %sn_chi = stablehlo.minimum %arg10, %bcast_tmax : {tf}")
    lines.append(f"    %sn_clo = stablehlo.maximum %sn_chi, %bcast_tmin : {tf}")
    lines.append(f"    %sn_tc = stablehlo.subtract %sn_clo, %bcast_tmelt : {tf}")
    lines.append(f"    %sn_a3t = stablehlo.multiply %bcast_xa3, %sn_tc : {tf}")
    lines.append(f"    %sn_a2s = stablehlo.add %bcast_xa2, %sn_a3t : {tf}")
    lines.append(f"    %sn_a2t = stablehlo.multiply %sn_a2s, %sn_tc : {tf}")
    lines.append(f"    %sn_ea = stablehlo.add %bcast_xa1, %sn_a2t : {tf}")
    lines.append(f"    %sn_ealn = stablehlo.multiply %sn_ea, %bcast_ln10 : {tf}")
    lines.append(f"    %sn_alf = stablehlo.exponential %sn_ealn : {tf}")
    lines.append(f"    %sn_b3t = stablehlo.multiply %bcast_xb3, %sn_tc : {tf}")
    lines.append(f"    %sn_b2s = stablehlo.add %bcast_xb2, %sn_b3t : {tf}")
    lines.append(f"    %sn_b2t = stablehlo.multiply %sn_b2s, %sn_tc : {tf}")
    lines.append(f"    %sn_bet = stablehlo.add %bcast_xb1, %sn_b2t : {tf}")
    lines.append(f"    %sn_qq = stablehlo.add %arg7, %bcast_qsmin : {tf}")
    lines.append(f"    %sn_qr = stablehlo.multiply %sn_qq, %arg11 : {tf}")
    lines.append(f"    %sn_qra = stablehlo.divide %sn_qr, %bcast_ams : {tf}")
    lines.append(f"    %sn_3b = stablehlo.multiply %bcast_3, %sn_bet : {tf}")
    lines.append(f"    %sn_exp = stablehlo.subtract %bcast_4, %sn_3b : {tf}")
    lines.append(f"    %sn_lb = stablehlo.log %sn_qra : {tf}")
    lines.append(f"    %sn_el = stablehlo.multiply %sn_exp, %sn_lb : {tf}")
    lines.append(f"    %sn_bp = stablehlo.exponential %sn_el : {tf}")
    lines.append(f"    %sn_a2 = stablehlo.multiply %sn_alf, %sn_alf : {tf}")
    lines.append(f"    %sn_a3 = stablehlo.multiply %sn_a2, %sn_alf : {tf}")
    lines.append(f"    %sn_nn = stablehlo.multiply %bcast_n0s3, %sn_bp : {tf}")
    lines.append(f"    %sn_n0s = stablehlo.divide %sn_nn, %sn_a3 : {tf}")
    lines.append(f"    %sn_n2t = stablehlo.multiply %bcast_n0s2, %sn_tc : {tf}")
    lines.append(f"    %sn_y = stablehlo.exponential %sn_n2t : {tf}")
    lines.append(f"    %sn_n4y = stablehlo.multiply %bcast_n0s4, %sn_y : {tf}")
    lines.append(f"    %sn_mn = stablehlo.maximum %sn_n4y, %bcast_n0s5 : {tf}")
    lines.append(f"    %sn_n6y = stablehlo.multiply %bcast_n0s6, %sn_y : {tf}")
    lines.append(f"    %sn_mx = stablehlo.minimum %sn_n6y, %bcast_n0s7 : {tf}")
    lines.append(f"    %sn_cl = stablehlo.maximum %sn_mn, %sn_n0s : {tf}")
    lines.append(f"    %sn_cr = stablehlo.minimum %sn_mx, %sn_cl : {tf}")
    lines.append(f"    %sn_gt = stablehlo.compare  GT, %arg7, %bcast_qmin,  FLOAT : ({tf}, {tf}) -> {tb}")
    lines.append(f"    %snow_number = stablehlo.select %sn_gt, %sn_cr, %bcast_n0s0 : {tb}, {tf}")
    lines.append(f"    %sn_sq = stablehlo.sqrt %snow_number : {tf}")
    lines.append(f"    %sn_cs = stablehlo.cbrt %sn_sq : {tf}")
    lines.append(f"    %sn_pw = stablehlo.divide %bcast_1, %sn_cs : {tf}")
    lines.append(f"    %vc_s_full = stablehlo.multiply %xrho, %sn_pw : {tf}")

    # Per-level broadcasts
    for n, c in [("0","cst_0"),("05","cst_05"),("1","cst_1"),("2","cst_2"),("30","cst_30"),
                 ("cvd","cst_cvd"),("cvv","cst_cvv"),("clw","cst_clw"),
                 ("ci","cst_ci"),("lvc","cst_lvc"),("lsc","cst_lsc")]:
        lines.append(f"    %b_{n} = stablehlo.broadcast_in_dim %{c}, dims = [] : (tensor<f64>) -> {tf1}")
    for sp in ["r","s","i","g"]:
        lines.append(f"    %b_vc_{sp} = stablehlo.broadcast_in_dim %vel_coeff_{sp}, dims = [] : (tensor<f64>) -> {tf1}")
        lines.append(f"    %b_ve_{sp} = stablehlo.broadcast_in_dim %vel_exp_{sp}, dims = [] : (tensor<f64>) -> {tf1}")
        lines.append(f"    %b_qm_{sp} = stablehlo.broadcast_in_dim %qmin_{sp}, dims = [] : (tensor<f64>) -> {tf1}")

    # Initial carry + accumulators
    for sp in ["r","s","i","g"]:
        lines.append(f"    %pf_{sp}_i = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}")
        lines.append(f"    %ac_{sp}_i = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> {tb1}")
        lines.append(f"    %qp_{sp}_i = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}")
    lines.append(f"    %tsa_i = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> {tb1}")
    lines.append(f"    %ef_i = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf1}")
    for n in ["qr","qs","qi","qg","pfr","pfs","pfi","pfg","ta","ea","pt"]:
        lines.append(f"    %{n}_z = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> {tf}")

    # While loop argument names and types (variadic, no tuple)
    sps = ["r","s","i","g"]
    an = ["qr_a","qs_a","qi_a","qg_a","pfr_a","pfs_a","pfi_a","pfg_a","ta_a","ea_a","pt_a"]
    w_names = (["k"] +
               [f"pf_{s}" for s in sps] +
               [f"ac_{s}" for s in sps] +
               [f"qp_{s}" for s in sps] +
               ["tsa", "efp"] + an)
    w_inits = (["%c0"] +
               [f"%pf_{s}_i" for s in sps] +
               [f"%ac_{s}_i" for s in sps] +
               [f"%qp_{s}_i" for s in sps] +
               ["%tsa_i", "%ef_i"] +
               [f"%{n}_z" if n in ["qr","qs","qi","qg"] else f"%{n}_z"
                for n in ["qr","qs","qi","qg","pfr","pfs","pfi","pfg","ta","ea","pt"]])
    w_types = ([ti] + [tf1]*4 + [tb1]*4 + [tf1]*4 + [tb1] + [tf1] + [tf]*11)
    n_args = len(w_names)

    # Build while op with variadic syntax
    while_args = ", ".join(f"%{w_names[i]} = {w_inits[i]}" for i in range(n_args))
    while_types = ", ".join(w_types)
    lines.append(f"    %res:{n_args} = stablehlo.while({while_args}) : {while_types}")

    # Condition (block args automatically named %k, %pf_r, etc.)
    lines.append(f"     cond {{")
    lines.append(f"      %cc = stablehlo.compare  LT, %k, %c_nlev : ({ti}, {ti}) -> tensor<i1>")
    lines.append(f"      stablehlo.return %cc : tensor<i1>")
    lines.append(f"    }} do {{")

    # Index math
    lines.append(f"      %km1r = stablehlo.subtract %k, %c1 : {ti}")
    lines.append(f"      %km1 = stablehlo.maximum %km1r, %c0 : {ti}")
    lines.append(f"      %kp1r = stablehlo.add %k, %c1 : {ti}")
    lines.append(f"      %kp1 = stablehlo.minimum %kp1r, %c_nlev_m1 : {ti}")
    lines.append(f"      %isk0 = stablehlo.compare  EQ, %k, %c0 : ({ti}, {ti}) -> tensor<i1>")
    lines.append(f"      %isk0b = stablehlo.broadcast_in_dim %isk0, dims = [] : (tensor<i1>) -> {tb1}")
    lines.append(f"      %z = stablehlo.constant dense<0> : {ti}")

    # Dynamic slice inputs
    km = {"r":"%arg0","s":"%arg2","i":"%arg1","g":"%arg3"}
    qa = {"r":"%arg6","s":"%arg7","i":"%arg8","g":"%arg9"}
    vc = {"r":"%xrho","s":"%vc_s_full","i":"%vc_i_full","g":"%xrho"}
    for sp in sps:
        lines.append(f"      %km_{sp} = stablehlo.dynamic_slice {km[sp]}, %k, %z, sizes = [1, {ncells}] : ({tb}, {ti}, {ti}) -> {tb1}")
    for n,a in [("qv","%arg4"),("qc","%arg5")]:
        lines.append(f"      %{n}_k = stablehlo.dynamic_slice {a}, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    for sp in sps:
        lines.append(f"      %q{sp}_k = stablehlo.dynamic_slice {qa[sp]}, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %t_k = stablehlo.dynamic_slice %arg10, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %rho_k = stablehlo.dynamic_slice %arg11, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %dz_k = stablehlo.dynamic_slice %arg12, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %ze_k = stablehlo.dynamic_slice %zeta_full, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %ei_k = stablehlo.dynamic_slice %ei_old_full, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    for sp in sps:
        lines.append(f"      %vc_{sp}_k = stablehlo.dynamic_slice {vc[sp]}, %k, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %rho_km = stablehlo.dynamic_slice %arg11, %km1, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    for sp in sps:
        lines.append(f"      %vc_{sp}_km = stablehlo.dynamic_slice {vc[sp]}, %km1, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")
    lines.append(f"      %t_kp = stablehlo.dynamic_slice %arg10, %kp1, %z, sizes = [1, {ncells}] : ({tf}, {ti}, {ti}) -> {tf1}")

    # Precip scan body for 4 species
    qn = {"r":"qr","s":"qs","i":"qi","g":"qg"}
    for sp in sps:
        q = qn[sp]
        lines.append(f"      %ac_{sp}_n = stablehlo.or %ac_{sp}, %km_{sp} : {tb1}")
        lines.append(f"      %rx_{sp} = stablehlo.multiply %{q}_k, %rho_k : {tf1}")
        lines.append(f"      %t1_{sp} = stablehlo.divide %rx_{sp}, %ze_k : {tf1}")
        lines.append(f"      %p2_{sp} = stablehlo.multiply %pf_{sp}, %b_2 : {tf1}")
        lines.append(f"      %fe_{sp} = stablehlo.add %t1_{sp}, %p2_{sp} : {tf1}")
        lines.append(f"      %rxo_{sp} = stablehlo.add %rx_{sp}, %b_qm_{sp} : {tf1}")
        if sp == "s":
            lines.append(f"      %rxs_{sp} = stablehlo.sqrt %rxo_{sp} : {tf1}")
            lines.append(f"      %rxp_{sp} = stablehlo.cbrt %rxs_{sp} : {tf1}")
        else:
            lines.append(f"      %rxl_{sp} = stablehlo.log %rxo_{sp} : {tf1}")
            lines.append(f"      %rxe_{sp} = stablehlo.multiply %b_ve_{sp}, %rxl_{sp} : {tf1}")
            lines.append(f"      %rxp_{sp} = stablehlo.exponential %rxe_{sp} : {tf1}")
        lines.append(f"      %fs_{sp} = stablehlo.multiply %b_vc_{sp}, %rxp_{sp} : {tf1}")
        lines.append(f"      %fr_{sp} = stablehlo.multiply %rx_{sp}, %vc_{sp}_k : {tf1}")
        lines.append(f"      %fsc_{sp} = stablehlo.multiply %fr_{sp}, %fs_{sp} : {tf1}")
        lines.append(f"      %fp_{sp} = stablehlo.minimum %fsc_{sp}, %fe_{sp} : {tf1}")
        lines.append(f"      %qs_{sp} = stablehlo.add %qp_{sp}, %{q}_k : {tf1}")
        lines.append(f"      %qa_{sp} = stablehlo.multiply %qs_{sp}, %b_05 : {tf1}")
        lines.append(f"      %rpg_{sp} = stablehlo.multiply %qa_{sp}, %rho_km : {tf1}")
        lines.append(f"      %rp_{sp} = stablehlo.select %isk0b, %b_0, %rpg_{sp} : {tb1}, {tf1}")
        lines.append(f"      %rpo_{sp} = stablehlo.add %rp_{sp}, %b_qm_{sp} : {tf1}")
        if sp == "s":
            lines.append(f"      %rps_{sp} = stablehlo.sqrt %rpo_{sp} : {tf1}")
            lines.append(f"      %rpp_{sp} = stablehlo.cbrt %rps_{sp} : {tf1}")
        else:
            lines.append(f"      %rpl_{sp} = stablehlo.log %rpo_{sp} : {tf1}")
            lines.append(f"      %rpe_{sp} = stablehlo.multiply %b_ve_{sp}, %rpl_{sp} : {tf1}")
            lines.append(f"      %rpp_{sp} = stablehlo.exponential %rpe_{sp} : {tf1}")
        lines.append(f"      %vp_{sp} = stablehlo.multiply %vc_{sp}_km, %b_vc_{sp} : {tf1}")
        lines.append(f"      %va_{sp} = stablehlo.multiply %vp_{sp}, %rpp_{sp} : {tf1}")
        lines.append(f"      %vg_{sp} = stablehlo.select %ac_{sp}, %va_{sp}, %b_0 : {tb1}, {tf1}")
        lines.append(f"      %vt_{sp} = stablehlo.select %isk0b, %b_0, %vg_{sp} : {tb1}, {tf1}")
        lines.append(f"      %fd_{sp} = stablehlo.subtract %fe_{sp}, %fp_{sp} : {tf1}")
        lines.append(f"      %nm_{sp} = stablehlo.multiply %ze_k, %fd_{sp} : {tf1}")
        lines.append(f"      %zv_{sp} = stablehlo.multiply %ze_k, %vt_{sp} : {tf1}")
        lines.append(f"      %di_{sp} = stablehlo.add %zv_{sp}, %b_1 : {tf1}")
        lines.append(f"      %dn_{sp} = stablehlo.multiply %di_{sp}, %rho_k : {tf1}")
        lines.append(f"      %qact_{sp} = stablehlo.divide %nm_{sp}, %dn_{sp} : {tf1}")
        lines.append(f"      %qr_{sp} = stablehlo.multiply %qact_{sp}, %rho_k : {tf1}")
        lines.append(f"      %qrv_{sp} = stablehlo.multiply %qr_{sp}, %vt_{sp} : {tf1}")
        lines.append(f"      %fls_{sp} = stablehlo.add %qrv_{sp}, %fp_{sp} : {tf1}")
        lines.append(f"      %fla_{sp} = stablehlo.multiply %fls_{sp}, %b_05 : {tf1}")
        lines.append(f"      %{q}_o = stablehlo.select %ac_{sp}_n, %qact_{sp}, %{q}_k : {tb1}, {tf1}")
        lines.append(f"      %pf_{sp}_o = stablehlo.select %ac_{sp}_n, %fla_{sp}, %b_0 : {tb1}, {tf1}")
        lines.append(f"      %qp_{sp}_o = stablehlo.add %{q}_o, %b_0 : {tf1}")

    # Post-precip intermediates
    lines.append(f"      %qln = stablehlo.add %qc_k, %qr_o : {tf1}")
    lines.append(f"      %qint = stablehlo.add %qs_o, %qi_o : {tf1}")
    lines.append(f"      %qin = stablehlo.add %qint, %qg_o : {tf1}")
    lines.append(f"      %ptt = stablehlo.add %pf_s_o, %pf_i_o : {tf1}")
    lines.append(f"      %ptot = stablehlo.add %ptt, %pf_g_o : {tf1}")
    lines.append(f"      %kr1 = stablehlo.or %km_r, %km_s : {tb1}")
    lines.append(f"      %kr2 = stablehlo.or %kr1, %km_i : {tb1}")
    lines.append(f"      %krs = stablehlo.or %kr2, %km_g : {tb1}")

    # Temperature update
    lines.append(f"      %tsa_n = stablehlo.or %tsa, %krs : {tb1}")
    lines.append(f"      %cvt1 = stablehlo.multiply %b_cvd, %t_kp : {tf1}")
    lines.append(f"      %clt = stablehlo.multiply %b_clw, %t_k : {tf1}")
    lines.append(f"      %pt1 = stablehlo.subtract %clt, %cvt1 : {tf1}")
    lines.append(f"      %pt2 = stablehlo.subtract %pt1, %b_lvc : {tf1}")
    lines.append(f"      %prc = stablehlo.multiply %pf_r_o, %pt2 : {tf1}")
    lines.append(f"      %cit = stablehlo.multiply %b_ci, %t_k : {tf1}")
    lines.append(f"      %pf1 = stablehlo.subtract %cit, %cvt1 : {tf1}")
    lines.append(f"      %pf2 = stablehlo.subtract %pf1, %b_lsc : {tf1}")
    lines.append(f"      %pfc = stablehlo.multiply %ptot, %pf2 : {tf1}")
    lines.append(f"      %efs = stablehlo.add %prc, %pfc : {tf1}")
    lines.append(f"      %efn = stablehlo.multiply %b_30, %efs : {tf1}")
    lines.append(f"      %eit = stablehlo.add %ei_k, %efp : {tf1}")
    lines.append(f"      %eint = stablehlo.subtract %eit, %efn : {tf1}")
    lines.append(f"      %qtt = stablehlo.add %qln, %qin : {tf1}")
    lines.append(f"      %qt = stablehlo.add %qtt, %qv_k : {tf1}")
    lines.append(f"      %rdz = stablehlo.multiply %rho_k, %dz_k : {tf1}")
    lines.append(f"      %omq = stablehlo.subtract %b_1, %qt : {tf1}")
    lines.append(f"      %c1v = stablehlo.multiply %b_cvd, %omq : {tf1}")
    lines.append(f"      %c2v = stablehlo.multiply %b_cvv, %qv_k : {tf1}")
    lines.append(f"      %c3v = stablehlo.multiply %b_clw, %qln : {tf1}")
    lines.append(f"      %c4v = stablehlo.multiply %b_ci, %qin : {tf1}")
    lines.append(f"      %cs1 = stablehlo.add %c1v, %c2v : {tf1}")
    lines.append(f"      %cs2 = stablehlo.add %cs1, %c3v : {tf1}")
    lines.append(f"      %cs3 = stablehlo.add %cs2, %c4v : {tf1}")
    lines.append(f"      %cvk = stablehlo.multiply %cs3, %rdz : {tf1}")
    lines.append(f"      %qlv = stablehlo.multiply %qln, %b_lvc : {tf1}")
    lines.append(f"      %qil = stablehlo.multiply %qin, %b_lsc : {tf1}")
    lines.append(f"      %lsm = stablehlo.add %qlv, %qil : {tf1}")
    lines.append(f"      %rl = stablehlo.multiply %rdz, %lsm : {tf1}")
    lines.append(f"      %tn = stablehlo.add %eint, %rl : {tf1}")
    lines.append(f"      %tnew = stablehlo.divide %tn, %cvk : {tf1}")
    lines.append(f"      %efo = stablehlo.select %tsa_n, %efn, %efp : {tb1}, {tf1}")
    lines.append(f"      %tout = stablehlo.select %tsa_n, %tnew, %t_k : {tb1}, {tf1}")

    # Write outputs into accumulators
    au = [("qr_a","qr_o"),("qs_a","qs_o"),("qi_a","qi_o"),("qg_a","qg_o"),
          ("pfr_a","pf_r_o"),("pfs_a","pf_s_o"),("pfi_a","pf_i_o"),("pfg_a","pf_g_o"),
          ("ta_a","tout"),("ea_a","efo"),("pt_a","ptot")]
    for a,v in au:
        lines.append(f"      %{a}_n = stablehlo.dynamic_update_slice %{a}, %{v}, %k, %z : ({tf}, {tf1}, {ti}, {ti}) -> {tf}")

    # Next counter + variadic return
    lines.append(f"      %kn = stablehlo.add %k, %c1 : {ti}")
    oe = ["%kn",
          "%pf_r_o","%pf_s_o","%pf_i_o","%pf_g_o",
          "%ac_r_n","%ac_s_n","%ac_i_n","%ac_g_n",
          "%qp_r_o","%qp_s_o","%qp_i_o","%qp_g_o",
          "%tsa_n","%efo",
          "%qr_a_n","%qs_a_n","%qi_a_n","%qg_a_n",
          "%pfr_a_n","%pfs_a_n","%pfi_a_n","%pfg_a_n",
          "%ta_a_n","%ea_a_n","%pt_a_n"]
    lines.append(f"      stablehlo.return {', '.join(oe)} : {', '.join(w_types)}")
    lines.append(f"    }}")

    # Results: %res#15=qr, #16=qs, #17=qi, #18=qg, #19=pfr, #20=pfs,
    #          #21=pfi, #22=pfg, #23=t, #24=eflx, #25=pflx_tot
    lines.append(f"    %ptpr = stablehlo.add %res#25, %res#19 : {tf}")
    lines.append(f"    %b30f = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> {tf}")
    lines.append(f"    %efdt = stablehlo.divide %res#24, %b30f : {tf}")
    lines.append(f"    return %res#15, %res#16, %res#17, %res#18, %res#23, %ptpr, %res#19, %res#20, %res#21, %res#22, %efdt : {ret_types}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate while-loop StableHLO")
    parser.add_argument("-o", "--output", default="stablehlo/precip_while.stablehlo")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    print(f"Generating while-loop StableHLO ({args.nlev} levels, {args.ncells} cells)...")
    text = generate_while_transposed(args.nlev, args.ncells)

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(text)

    print(f"Written to: {args.output} ({len(text)/1024:.1f} KB, {text.count(chr(10))} lines)")
    for op in ["dynamic_slice","dynamic_update_slice","multiply","add","subtract",
               "divide","select","compare","or","broadcast_in_dim","log",
               "exponential","sqrt","cbrt","minimum","maximum","constant","while",
               "tuple","get_tuple_element"]:
        c = text.count(f"stablehlo.{op}")
        if c: print(f"  stablehlo.{op}: {c}")
    print(f"  stablehlo.power: {text.count('stablehlo.power')} (should be 0)")
    print(f"  stablehlo.concatenate: {text.count('stablehlo.concatenate')} (should be 0)")


if __name__ == "__main__":
    main()
