# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fused q_t_update: all phase transitions inlined into a single function.

Uses lax.select/lax.pow instead of jnp.where/jnp.power and groups computations
to reduce the number of XLA kernels (~80 -> fewer fused kernels).
"""

import jax.numpy as jnp
from jax import lax

from ..core.common import constants as const
from ..core.definitions import Q


def _fast_pow(x, c):
    """Replace lax.pow with exp(c*log(x)) for better GPU kernel fusion.

    stablehlo.power blocks XLA fusion; exp+multiply+log fuse into elementwise kernels.
    """
    return jnp.exp(c * jnp.log(x))


def q_t_update_fused(t, p, rho, q, dt, qnc):
    """
    Fused q_t_update with optimizations for GPU kernel fusion.

    Uses lax.select and exp(c*log(x)) instead of lax.pow for better fusion.
    """
    # Precompute commonly used values
    tmelt = const.tmelt
    qmin = const.qmin
    tfrz_het2 = const.tfrz_het2
    tfrz_hom = const.tfrz_hom
    rv = const.rv

    # ============================================================
    # SATURATION COMPUTATIONS (inlined thermo functions)
    # ============================================================
    # qsat_rho - saturation over liquid water
    C1ES = 610.78
    C3LES = 17.269
    C4LES = 35.86
    qsat_w = (C1ES * jnp.exp(C3LES * (t - tmelt) / (t - C4LES))) / (rho * rv * t)

    # qsat_ice_rho - saturation over ice
    C3IES = 21.875
    C4IES = 7.66
    qvsi = (C1ES * jnp.exp(C3IES * (t - tmelt) / (t - C4IES))) / (rho * rv * t)

    # qsat_rho_tmelt
    qsat_tmelt = C1ES / (rho * rv * tmelt)

    # ============================================================
    # ACTIVATION MASKS
    # ============================================================
    qmax_precip = jnp.maximum(q.g, jnp.maximum(q.i, jnp.maximum(q.r, q.s)))
    qmax_all = jnp.maximum(q.c, qmax_precip)

    mask = (qmax_all > qmin) | ((t < tfrz_het2) & (q.v > qvsi))
    is_sig_present = jnp.maximum(q.g, jnp.maximum(q.i, q.s)) > qmin

    # Supersaturation
    dvsw = q.v - qsat_w
    dvsi = q.v - qvsi
    dvsw0 = lax.select(is_sig_present, q.v - qsat_tmelt, jnp.zeros_like(t))

    t_below_tmelt = t < tmelt
    t_at_least_tmelt = ~t_below_tmelt

    # ============================================================
    # SNOW PROPERTIES (inlined)
    # ============================================================
    TMIN_SN = tmelt - 40.0
    TMAX_SN = tmelt
    QSMIN_SN = 2.0e-6
    XA1 = -1.65
    XA2 = 5.45e-2
    XA3 = 3.27e-4
    XB1 = 1.42
    XB2 = 1.19e-2
    XB3 = 9.6e-5
    N0S0 = 8.0e5
    N0S1 = 13.5 * 5.65e5
    N0S2 = -0.107
    N0S3 = 13.5
    N0S4 = 0.5 * N0S1
    N0S5 = 1.0e6
    N0S6 = 1.0e2 * N0S1
    N0S7 = 1.0e9

    tc_sn = jnp.maximum(jnp.minimum(t, TMAX_SN), TMIN_SN) - tmelt
    alf_sn = _fast_pow(10.0, XA1 + tc_sn * (XA2 + tc_sn * XA3))
    bet_sn = XB1 + tc_sn * (XB2 + tc_sn * XB3)

    qs_ratio = (q.s + QSMIN_SN) * rho / const.ams
    n0s_val = N0S3 * _fast_pow(qs_ratio, 4.0 - 3.0 * bet_sn) / (alf_sn * alf_sn * alf_sn)
    y_sn = jnp.exp(N0S2 * tc_sn)
    n0smn = jnp.maximum(N0S4 * y_sn, N0S5)
    n0smx = jnp.minimum(N0S6 * y_sn, N0S7)
    n_snow = lax.select(
        q.s > qmin, jnp.minimum(n0smx, jnp.maximum(n0smn, n0s_val)), jnp.full_like(t, N0S0)
    )

    # Snow lambda
    A2_LAM = const.ams * 2.0
    LMD_0 = 1.0e10
    BX_LAM = 1.0 / (const.bms + 1.0)
    l_snow = lax.select(
        q.s > qmin, _fast_pow(A2_LAM * n_snow / (q.s * rho + 1e-30), BX_LAM), jnp.full_like(t, LMD_0)
    )

    # ============================================================
    # ICE PROPERTIES (inlined)
    # ============================================================
    A_COOP = 5.0
    B_COOP = 0.304
    NIMAX = 250.0e3
    n_ice = jnp.minimum(NIMAX, A_COOP * jnp.exp(B_COOP * (tmelt - t))) / rho

    MI_MAX = 1.0e-9
    m_ice = jnp.maximum(const.m0_ice, jnp.minimum(q.i / (n_ice + 1e-30), MI_MAX))

    # Ice sticking
    A_FREEZ = 0.09
    B_MAX_EXP = 1.0
    EFF_MIN = 0.075
    EFF_FAC = 3.5e-3
    TCRIT = tmelt - 85.0
    x_ice = jnp.maximum(
        jnp.maximum(jnp.minimum(jnp.exp(A_FREEZ * (t - tmelt)), B_MAX_EXP), EFF_MIN),
        EFF_FAC * (t - TCRIT),
    )

    # Deposition factor
    KAPPA = 2.4e-2
    B_DF = 1.94
    A_DF = const.als * const.als / (KAPPA * rv)
    CX_DF = 2.22e-5 * _fast_pow(tmelt, -B_DF) * 101325.0
    x_df = CX_DF / const.rd * _fast_pow(t, B_DF - 1.0)
    eta = lax.select(
        t_below_tmelt & is_sig_present,
        x_df / (1.0 + A_DF * x_df * qvsi / (t * t)),
        jnp.zeros_like(t),
    )

    # ============================================================
    # PHASE TRANSITIONS (grouped for fusion)
    # ============================================================
    zero = jnp.zeros_like(t)

    # --- Cloud to Rain (autoconversion + accretion) ---
    QMIN_AC = 1.0e-6
    TAU_MAX = 0.9
    TAU_MIN = 1.0e-30
    A_PHI = 600.0
    B_PHI = 0.68
    C_PHI = 5.0e-5
    AC_KERNEL = 5.25
    X3 = 2.0
    X2 = 2.6e-10
    X1 = 9.44e9
    AU_KERNEL = X1 / (20.0 * X2) * (X3 + 2.0) * (X3 + 4.0) / ((X3 + 1.0) * (X3 + 1.0))

    tau_cr = jnp.maximum(TAU_MIN, jnp.minimum(1.0 - q.c / (q.c + q.r + 1e-30), TAU_MAX))
    phi_cr = _fast_pow(tau_cr, B_PHI)
    one_minus_phi = 1.0 - phi_cr
    phi_cr = A_PHI * phi_cr * one_minus_phi * one_minus_phi * one_minus_phi
    qc_ratio = q.c * q.c / (qnc + 1e-30)
    one_minus_tau = 1.0 - tau_cr
    xau = AU_KERNEL * qc_ratio * qc_ratio * (1.0 + phi_cr / (one_minus_tau * one_minus_tau + 1e-30))
    tau_ratio = tau_cr / (tau_cr + C_PHI)
    tau_ratio_sq = tau_ratio * tau_ratio
    xac = AC_KERNEL * q.c * q.r * tau_ratio_sq * tau_ratio_sq
    sx2x_c_r = lax.select((q.c > QMIN_AC) & (t > tfrz_hom), xau + xac, zero)

    # --- Rain to Vapor (evaporation) ---
    B1_RV = 0.16667
    B2_RV = 0.55555
    C1_RV = 0.61
    C2_RV = -0.0163
    C3_RV = 1.111e-4
    A1_RV = 1.536e-3
    A2_RV = 1.0
    A3_RV = 19.0621

    tc_rv = t - tmelt
    evap_max = (C1_RV + tc_rv * (C2_RV + C3_RV * tc_rv)) * (-dvsw) / dt
    qr_rho = q.r * rho + 1e-30
    sx2x_r_v = lax.select(
        (q.r > qmin) & (dvsw + q.c <= 0.0),
        jnp.minimum(
            A1_RV * (A2_RV + A3_RV * _fast_pow(qr_rho, B1_RV)) * (-dvsw) * _fast_pow(qr_rho, B2_RV),
            evap_max,
        ),
        zero,
    )

    # --- Cloud x Ice (freezing/melting) ---
    sx2x_c_i_raw = lax.select((q.c > qmin) & (t < tfrz_hom), q.c / dt, zero)
    sx2x_c_i_raw = lax.select((q.i > qmin) & (t > tmelt), -q.i / dt, sx2x_c_i_raw)
    sx2x_i_c = -jnp.minimum(sx2x_c_i_raw, 0.0)
    sx2x_c_i = jnp.maximum(sx2x_c_i_raw, 0.0)

    # --- Cloud to Snow (riming) ---
    ECS = 0.9
    B_RIM_CS = -(const.v1s + 3.0)
    C_RIM_CS = 2.61 * ECS * const.v0s
    sx2x_c_s = lax.select(
        (jnp.minimum(q.c, q.s) > qmin) & (t > tfrz_hom),
        C_RIM_CS * n_snow * q.c * _fast_pow(l_snow, B_RIM_CS),
        zero,
    )

    # --- Cloud to Graupel (riming) ---
    A_RIM_CG = 4.43
    B_RIM_CG = 0.94878
    sx2x_c_g = lax.select(
        (jnp.minimum(q.c, q.g) > qmin) & (t > tfrz_hom),
        A_RIM_CG * q.c * _fast_pow(q.g * rho + 1e-30, B_RIM_CG),
        zero,
    )

    # --- Vapor x Ice (deposition/sublimation) ---
    AMI = 130.0
    B_EXP_VI = -0.67
    A_FACT_VI = 4.0 * AMI ** (-1.0 / 3.0)

    vi_raw = (A_FACT_VI * eta) * rho * q.i * _fast_pow(m_ice + 1e-30, B_EXP_VI) * dvsi
    vi_raw = lax.select(
        vi_raw > 0.0,
        jnp.minimum(vi_raw, dvsi / dt),
        jnp.maximum(jnp.maximum(vi_raw, dvsi / dt), -q.i / dt),
    )
    sx2x_v_i_base = lax.select((q.i > qmin) & t_below_tmelt & is_sig_present, vi_raw, zero)
    sx2x_i_v = lax.select(t_below_tmelt & is_sig_present, -jnp.minimum(sx2x_v_i_base, 0.0), zero)
    sx2x_v_i_pos = lax.select(t_below_tmelt & is_sig_present, jnp.maximum(sx2x_v_i_base, 0.0), zero)

    ice_dep = lax.select(t_below_tmelt & is_sig_present, jnp.minimum(sx2x_v_i_pos, dvsi / dt), zero)

    # --- Ice deposition nucleation ---
    sx2x_v_i_nuc = lax.select(
        (q.i <= qmin)
        & (((t < tfrz_het2) & (dvsi > 0.0)) | ((t <= const.tfrz_het1) & (q.c > qmin))),
        jnp.minimum(const.m0_ice * n_ice, jnp.maximum(0.0, dvsi)) / dt,
        zero,
    )
    sx2x_v_i = lax.select(t_below_tmelt, sx2x_v_i_pos + sx2x_v_i_nuc, zero)

    # --- Deposition auto conversion ---
    M0_S = 3.0e-9
    B_DEP = 0.666666667
    XCRIT = 1.0
    dac = lax.select(
        q.i > qmin,
        jnp.maximum(0.0, ice_dep)
        * B_DEP
        / (_fast_pow(M0_S / (m_ice + 1e-30), B_DEP) - XCRIT + 1e-30),
        zero,
    )

    # --- Ice to Snow ---
    QI0 = 0.0
    C_IAU = 1.0e-3
    C_AGG_IS = 2.61 * const.v0s
    B_AGG_IS = -(const.v1s + 3.0)
    sx2x_i_s = lax.select(
        t_below_tmelt & is_sig_present & (q.i > qmin),
        x_ice
        * (
            dac
            + C_IAU * jnp.maximum(0.0, q.i - QI0)
            + q.i * C_AGG_IS * n_snow * _fast_pow(l_snow, B_AGG_IS)
        ),
        zero,
    )

    # --- Ice to Graupel ---
    A_CT_IG = 1.72
    B_CT_IG = 0.875
    C_AGG_IG = 2.46
    B_AGG_IG = 0.94878
    ig_agg = lax.select(
        (q.i > qmin) & (q.g > qmin),
        x_ice * q.i * C_AGG_IG * _fast_pow(rho * q.g + 1e-30, B_AGG_IG),
        zero,
    )
    ig_coll = lax.select(
        (q.i > qmin) & (q.r > qmin), A_CT_IG * q.i * _fast_pow(rho * q.r + 1e-30, B_CT_IG), zero
    )
    sx2x_i_g = lax.select(t_below_tmelt & is_sig_present, ig_agg + ig_coll, zero)

    # --- Snow to Graupel ---
    A_RIM_SG = 0.5
    B_RIM_SG = 0.75
    sx2x_s_g = lax.select(
        (jnp.minimum(q.c, q.s) > qmin) & (t > tfrz_hom) & t_below_tmelt & is_sig_present,
        A_RIM_SG * q.c * _fast_pow(q.s * rho + 1e-30, B_RIM_SG),
        zero,
    )

    # --- Rain to Graupel ---
    TFRZ_RAIN = tmelt - 2.0
    A1_RG = 9.95e-5
    B1_RG = 1.75
    C2_RG = 0.66
    C3_RG = 1.0
    C4_RG = 0.1
    A2_RG = 1.24e-3
    B2_RG = 1.625
    QS_CRIT = 1.0e-7

    maskinner_rg = (dvsw + q.c <= 0.0) | (q.r > C4_RG * q.c)
    mask_rg = (q.r > qmin) & (t < TFRZ_RAIN)

    rg_imm = lax.select(
        mask_rg & (t > tfrz_hom) & maskinner_rg,
        (jnp.exp(C2_RG * (TFRZ_RAIN - t)) - C3_RG) * A1_RG * _fast_pow(q.r * rho + 1e-30, B1_RG),
        zero,
    )
    rg_hom = lax.select(mask_rg & (t <= tfrz_hom), q.r / dt, zero)
    rg_coll = lax.select(
        (jnp.minimum(q.i, q.r) > qmin) & (q.s > QS_CRIT),
        A2_RG * (q.i / (m_ice + 1e-30)) * _fast_pow(rho * q.r + 1e-30, B2_RG),
        zero,
    )
    sx2x_r_g = lax.select(
        t_below_tmelt & is_sig_present, jnp.maximum(rg_imm, rg_hom) + rg_coll, zero
    )

    # --- Vapor x Snow ---
    NU_VS = 1.75e-5
    A0_VS = 1.0
    A1_VS = 0.4182 * jnp.sqrt(const.v0s / NU_VS)
    A2_VS = -(const.v1s + 1.0) / 2.0
    EPS_VS = 1.0e-15
    QS_LIM = 1.0e-7
    CNX_VS = 4.0
    B_VS = 0.8
    C1_VS = 31282.3
    C2_VS = 0.241897
    C3_VS = 0.28003
    C4_VS = -0.146293e-6

    vs_cold = (
        (CNX_VS * n_snow * eta / rho)
        * (A0_VS + A1_VS * _fast_pow(l_snow, A2_VS))
        * dvsi
        / (l_snow * l_snow + EPS_VS)
    )
    vs_cold = lax.select(vs_cold > 0.0, jnp.minimum(vs_cold, dvsi / dt - ice_dep), vs_cold)
    vs_cold = lax.select(q.s <= QS_LIM, jnp.minimum(vs_cold, 0.0), vs_cold)

    vs_warm = (C1_VS / p + C2_VS) * jnp.minimum(0.0, dvsw0) * _fast_pow(q.s * rho + 1e-30, B_VS)
    vs_mid = (C3_VS + C4_VS * p) * dvsw * _fast_pow(q.s * rho + 1e-30, B_VS)

    sx2x_v_s_raw = lax.select(
        t < tmelt, vs_cold, lax.select(t > (tmelt - const.tx * dvsw0), vs_warm, vs_mid)
    )
    sx2x_v_s_raw = lax.select(
        (q.s > qmin) & is_sig_present, jnp.maximum(sx2x_v_s_raw, -q.s / dt), zero
    )
    sx2x_s_v = lax.select(is_sig_present, -jnp.minimum(sx2x_v_s_raw, 0.0), zero)
    sx2x_v_s = lax.select(is_sig_present, jnp.maximum(sx2x_v_s_raw, 0.0), zero)

    # --- Vapor x Graupel ---
    A1_VG = 0.398561
    A2_VG = -0.00152398
    A3_VG = 2554.99
    A4_VG = 2.6531e-7
    A5_VG = 0.153907
    A6_VG = -7.86703e-7
    A7_VG = 0.0418521
    A8_VG = -4.7524e-8
    B_VG = 0.6

    qg_rho = q.g * rho + 1e-30
    qg_pow = _fast_pow(qg_rho, B_VG)
    vg_cold = (A1_VG + A2_VG * t + A3_VG / p + A4_VG * p) * dvsi * qg_pow
    vg_warm = (A5_VG + A6_VG * p) * jnp.minimum(0.0, dvsw0) * qg_pow
    vg_mid = (A7_VG + A8_VG * p) * dvsw * qg_pow

    sx2x_v_g_raw = lax.select(
        t < tmelt, vg_cold, lax.select(t > (tmelt - const.tx * dvsw0), vg_warm, vg_mid)
    )
    sx2x_v_g_raw = lax.select(
        (q.g > qmin) & is_sig_present, jnp.maximum(sx2x_v_g_raw, -q.g / dt), zero
    )
    sx2x_g_v = lax.select(is_sig_present, -jnp.minimum(sx2x_v_g_raw, 0.0), zero)
    sx2x_v_g = lax.select(is_sig_present, jnp.maximum(sx2x_v_g_raw, 0.0), zero)

    # --- Snow to Rain (melting) ---
    C1_SR = 79.6863
    C2_SR = 0.612654e-3
    A_SR = const.tx - 389.5
    B_SR = 0.8
    sx2x_s_r = lax.select(
        (t > jnp.maximum(tmelt, tmelt - const.tx * dvsw0)) & (q.s > qmin) & is_sig_present,
        (C1_SR / p + C2_SR) * (t - tmelt + A_SR * dvsw0) * _fast_pow(q.s * rho + 1e-30, B_SR),
        zero,
    )

    # --- Graupel to Rain (melting) ---
    A_MELT = const.tx - 389.5
    B_MELT = 0.6
    C1_MELT = 12.31698
    C2_MELT = 7.39441e-5
    sx2x_g_r = lax.select(
        (t > jnp.maximum(tmelt, tmelt - const.tx * dvsw0)) & (q.g > qmin) & is_sig_present,
        (C1_MELT / p + C2_MELT) * (t - tmelt + A_MELT * dvsw0) * _fast_pow(q.g * rho + 1e-30, B_MELT),
        zero,
    )

    # ============================================================
    # ABOVE TMELT ADJUSTMENTS
    # ============================================================
    sx2x_c_r = lax.select(t_at_least_tmelt, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r)
    sx2x_c_s = lax.select(t_at_least_tmelt, zero, sx2x_c_s)
    sx2x_c_g = lax.select(t_at_least_tmelt, zero, sx2x_c_g)
    ice_dep = lax.select(t_at_least_tmelt, zero, ice_dep)

    # ============================================================
    # SINKS AND SOURCES
    # ============================================================
    sink_v = sx2x_v_s + sx2x_v_i + sx2x_v_g
    sink_c = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g
    sink_r = sx2x_r_v + sx2x_r_g
    sink_s = lax.select(is_sig_present, sx2x_s_v + sx2x_s_r + sx2x_s_g, zero)
    sink_i = lax.select(is_sig_present, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, zero)
    sink_g = lax.select(is_sig_present, sx2x_g_v + sx2x_g_r, zero)

    # ============================================================
    # SINK SATURATION CLAMPING (CRITICAL!)
    # Rescale sink rates when they exceed available mass (q/dt)
    # ============================================================

    # Vapor sink clamping
    stot_v = q.v / dt
    sink_v_saturated = (sink_v > stot_v) & (q.v > qmin)
    scale_v = stot_v / (sink_v + 1e-30)
    sx2x_v_s = lax.select(sink_v_saturated, sx2x_v_s * scale_v, sx2x_v_s)
    sx2x_v_i = lax.select(sink_v_saturated, sx2x_v_i * scale_v, sx2x_v_i)
    sx2x_v_g = lax.select(sink_v_saturated, sx2x_v_g * scale_v, sx2x_v_g)
    sink_v = lax.select(sink_v_saturated, sx2x_v_s + sx2x_v_i + sx2x_v_g, sink_v)

    # Cloud sink clamping
    stot_c = q.c / dt
    sink_c_saturated = (sink_c > stot_c) & (q.c > qmin)
    scale_c = stot_c / (sink_c + 1e-30)
    sx2x_c_r = lax.select(sink_c_saturated, sx2x_c_r * scale_c, sx2x_c_r)
    sx2x_c_s = lax.select(sink_c_saturated, sx2x_c_s * scale_c, sx2x_c_s)
    sx2x_c_i = lax.select(sink_c_saturated, sx2x_c_i * scale_c, sx2x_c_i)
    sx2x_c_g = lax.select(sink_c_saturated, sx2x_c_g * scale_c, sx2x_c_g)
    sink_c = lax.select(sink_c_saturated, sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g, sink_c)

    # Rain sink clamping
    stot_r = q.r / dt
    sink_r_saturated = (sink_r > stot_r) & (q.r > qmin)
    scale_r = stot_r / (sink_r + 1e-30)
    sx2x_r_v = lax.select(sink_r_saturated, sx2x_r_v * scale_r, sx2x_r_v)
    sx2x_r_g = lax.select(sink_r_saturated, sx2x_r_g * scale_r, sx2x_r_g)
    sink_r = lax.select(sink_r_saturated, sx2x_r_v + sx2x_r_g, sink_r)

    # Snow sink clamping
    stot_s = q.s / dt
    sink_s_saturated = (sink_s > stot_s) & (q.s > qmin)
    scale_s = stot_s / (sink_s + 1e-30)
    sx2x_s_v = lax.select(sink_s_saturated, sx2x_s_v * scale_s, sx2x_s_v)
    sx2x_s_r = lax.select(sink_s_saturated, sx2x_s_r * scale_s, sx2x_s_r)
    sx2x_s_g = lax.select(sink_s_saturated, sx2x_s_g * scale_s, sx2x_s_g)
    sink_s = lax.select(sink_s_saturated, sx2x_s_v + sx2x_s_r + sx2x_s_g, sink_s)

    # Ice sink clamping
    stot_i = q.i / dt
    sink_i_saturated = (sink_i > stot_i) & (q.i > qmin)
    scale_i = stot_i / (sink_i + 1e-30)
    sx2x_i_v = lax.select(sink_i_saturated, sx2x_i_v * scale_i, sx2x_i_v)
    sx2x_i_c = lax.select(sink_i_saturated, sx2x_i_c * scale_i, sx2x_i_c)
    sx2x_i_s = lax.select(sink_i_saturated, sx2x_i_s * scale_i, sx2x_i_s)
    sx2x_i_g = lax.select(sink_i_saturated, sx2x_i_g * scale_i, sx2x_i_g)
    sink_i = lax.select(sink_i_saturated, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, sink_i)

    # Graupel sink clamping
    stot_g = q.g / dt
    sink_g_saturated = (sink_g > stot_g) & (q.g > qmin)
    scale_g = stot_g / (sink_g + 1e-30)
    sx2x_g_v = lax.select(sink_g_saturated, sx2x_g_v * scale_g, sx2x_g_v)
    sx2x_g_r = lax.select(sink_g_saturated, sx2x_g_r * scale_g, sx2x_g_r)
    sink_g = lax.select(sink_g_saturated, sx2x_g_v + sx2x_g_r, sink_g)

    # ============================================================
    # WATER CONTENT UPDATES
    # ============================================================
    dqdt_v = sx2x_r_v + sx2x_s_v + sx2x_i_v + sx2x_g_v - sink_v
    dqdt_c = sx2x_i_c - sink_c
    dqdt_r = sx2x_c_r + sx2x_s_r + sx2x_g_r - sink_r
    dqdt_s = sx2x_v_s + sx2x_c_s + sx2x_i_s - sink_s
    dqdt_i = sx2x_v_i + sx2x_c_i - sink_i
    dqdt_g = sx2x_v_g + sx2x_c_g + sx2x_r_g + sx2x_s_g + sx2x_i_g - sink_g

    qv_new = lax.select(mask, jnp.maximum(0.0, q.v + dqdt_v * dt), q.v)
    qc_new = lax.select(mask, jnp.maximum(0.0, q.c + dqdt_c * dt), q.c)
    qr_new = lax.select(mask, jnp.maximum(0.0, q.r + dqdt_r * dt), q.r)
    qs_new = lax.select(mask, jnp.maximum(0.0, q.s + dqdt_s * dt), q.s)
    qi_new = lax.select(mask, jnp.maximum(0.0, q.i + dqdt_i * dt), q.i)
    qg_new = lax.select(mask, jnp.maximum(0.0, q.g + dqdt_g * dt), q.g)

    # ============================================================
    # UPDATE TEMPERATURE
    # ============================================================
    # Compute total water contents after update
    qice_new = qs_new + qi_new + qg_new
    qliq_new = qc_new + qr_new
    qtot_new = qv_new + qice_new + qliq_new

    # Heat capacity: cv = cvd + (cvv - cvd) * qtot + (clw - cvv) * qliq + (ci - cvv) * qice
    cv = (
        const.cvd
        + (const.cvv - const.cvd) * qtot_new
        + (const.clw - const.cvv) * qliq_new
        + (const.ci - const.cvv) * qice_new
    )

    # Temperature change from phase transitions
    # dT = dt * ((dqdt_c + dqdt_r) * (lvc - (clw - cvv) * t) + (dqdt_i + dqdt_s + dqdt_g) * (lsc - (ci - cvv) * t)) / cv
    t_new = lax.select(
        mask,
        t
        + dt
        * (
            (dqdt_c + dqdt_r) * (const.lvc - (const.clw - const.cvv) * t)
            + (dqdt_i + dqdt_s + dqdt_g) * (const.lsc - (const.ci - const.cvv) * t)
        )
        / cv,
        t,
    )

    q_out = Q(v=qv_new, c=qc_new, r=qr_new, s=qs_new, i=qi_new, g=qg_new)
    return q_out, t_new
