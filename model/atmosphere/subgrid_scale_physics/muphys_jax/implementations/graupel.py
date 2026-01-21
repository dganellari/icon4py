# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Graupel microphysics implementation.
Complete composition of phase transitions and precipitation.
"""

import jax
# jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from functools import partial

from ..core import properties as props, thermo, transitions as trans
from ..core.common import constants as const
from ..core.common.backend import jit_compile

# Import from core modules
from ..core.definitions import Q, TempState
from ..core.scans import precip_scan_batched, precip_scan_unrolled, precip_scan_tiled, temperature_scan_step

# Pallas import (optional)
try:
    from ..core.scans_pallas import precip_scan_pallas, PALLAS_AVAILABLE
except ImportError:
    PALLAS_AVAILABLE = False
    precip_scan_pallas = None


# ============================================================================
# Main Physics Functions
# ============================================================================


def q_t_update(t, p, rho, q, dt, qnc):
    """
    Update water species and temperature via phase transitions.
    Corresponds to graupel.py:158-362 in the GT4Py implementation.
    """
    # Activation mask
    mask = jnp.where(
        (jnp.maximum(q.c, jnp.maximum(q.g, jnp.maximum(q.i, jnp.maximum(q.r, q.s)))) > const.qmin)
        | ((t < const.tfrz_het2) & (q.v > thermo.qsat_ice_rho(t, rho))),
        True,
        False,
    )

    is_sig_present = jnp.maximum(q.g, jnp.maximum(q.i, q.s)) > const.qmin

    dvsw = q.v - thermo.qsat_rho(t, rho)
    qvsi = thermo.qsat_ice_rho(t, rho)
    dvsi = q.v - qvsi

    # Snow properites
    n_snow = props.snow_number(t, rho, q.s)
    l_snow = props.snow_lambda(rho, q.s, n_snow)

    # Define conversion 'matrix'
    sx2x_c_r = trans.cloud_to_rain(t, q.c, q.r, qnc)
    sx2x_r_v = trans.rain_to_vapor(t, rho, q.c, q.r, dvsw, dt)
    sx2x_c_i = trans.cloud_x_ice(t, q.c, q.i, dt)
    sx2x_i_c = -jnp.minimum(sx2x_c_i, 0.0)
    sx2x_c_i = jnp.maximum(sx2x_c_i, 0.0)

    sx2x_c_s = trans.cloud_to_snow(t, q.c, q.s, n_snow, l_snow)
    sx2x_c_g = trans.cloud_to_graupel(t, rho, q.c, q.g)

    t_below_tmelt = t < const.tmelt
    t_at_least_tmelt = ~t_below_tmelt

    n_ice = props.ice_number(t, rho)
    m_ice = props.ice_mass(q.i, n_ice)
    x_ice = props.ice_sticking(t)

    eta = jnp.where(t_below_tmelt & is_sig_present, props.deposition_factor(t, qvsi), 0.0)
    sx2x_v_i = jnp.where(
        t_below_tmelt & is_sig_present, trans.vapor_x_ice(q.i, m_ice, eta, dvsi, rho, dt), 0.0
    )
    sx2x_i_v = jnp.where(t_below_tmelt & is_sig_present, -jnp.minimum(sx2x_v_i, 0.0), 0.0)
    sx2x_v_i = jnp.where(t_below_tmelt & is_sig_present, jnp.maximum(sx2x_v_i, 0.0), sx2x_i_v)

    ice_dep = jnp.where(t_below_tmelt & is_sig_present, jnp.minimum(sx2x_v_i, dvsi / dt), 0.0)
    sx2x_i_s = jnp.where(
        t_below_tmelt & is_sig_present,
        props.deposition_auto_conversion(q.i, m_ice, ice_dep)
        + trans.ice_to_snow(q.i, n_snow, l_snow, x_ice),
        0.0,
    )
    sx2x_i_g = jnp.where(
        t_below_tmelt & is_sig_present, trans.ice_to_graupel(rho, q.r, q.g, q.i, x_ice), 0.0
    )
    sx2x_s_g = jnp.where(
        t_below_tmelt & is_sig_present, trans.snow_to_graupel(t, rho, q.c, q.s), 0.0
    )
    sx2x_r_g = jnp.where(
        t_below_tmelt & is_sig_present,
        trans.rain_to_graupel(t, rho, q.c, q.r, q.i, q.s, m_ice, dvsw, dt),
        0.0,
    )

    sx2x_v_i = jnp.where(
        t_below_tmelt, sx2x_v_i + props.ice_deposition_nucleation(t, q.c, q.i, n_ice, dvsi, dt), 0.0
    )
    sx2x_c_r = jnp.where(t_at_least_tmelt, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r)
    sx2x_c_s = jnp.where(t_at_least_tmelt, 0.0, sx2x_c_s)
    sx2x_c_g = jnp.where(t_at_least_tmelt, 0.0, sx2x_c_g)
    ice_dep = jnp.where(t_at_least_tmelt, 0.0, ice_dep)
    eta = jnp.where(t_at_least_tmelt, 0.0, eta)

    dvsw0 = jnp.where(is_sig_present, q.v - thermo.qsat_rho_tmelt(rho), 0.0)
    sx2x_v_s = jnp.where(
        is_sig_present,
        trans.vapor_x_snow(t, p, rho, q.s, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt),
        0.0,
    )
    sx2x_s_v = jnp.where(is_sig_present, -jnp.minimum(sx2x_v_s, 0.0), 0.0)
    sx2x_v_s = jnp.where(is_sig_present, jnp.maximum(sx2x_v_s, 0.0), 0.0)

    sx2x_v_g = jnp.where(
        is_sig_present, trans.vapor_x_graupel(t, p, rho, q.g, dvsw, dvsi, dvsw0, dt), 0.0
    )
    sx2x_g_v = jnp.where(is_sig_present, -jnp.minimum(sx2x_v_g, 0.0), 0.0)
    sx2x_v_g = jnp.where(is_sig_present, jnp.maximum(sx2x_v_g, 0.0), 0.0)

    sx2x_s_r = jnp.where(is_sig_present, trans.snow_to_rain(t, p, rho, dvsw0, q.s), 0.0)
    sx2x_g_r = jnp.where(is_sig_present, trans.graupel_to_rain(t, p, rho, dvsw0, q.g), 0.0)

    # Sink calculation 
    sink_v = sx2x_v_s + sx2x_v_i + sx2x_v_g
    sink_c = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g
    sink_r = sx2x_r_v + sx2x_r_g
    sink_s = jnp.where(is_sig_present, sx2x_s_v + sx2x_s_r + sx2x_s_g, 0.0)
    sink_i = jnp.where(is_sig_present, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, 0.0)
    sink_g = jnp.where(is_sig_present, sx2x_g_v + sx2x_g_r, 0.0)

    stot = q.v / dt
    sink_v_saturated = (sink_v > stot) & (q.v > const.qmin)
    sx2x_v_s = jnp.where(sink_v_saturated, sx2x_v_s * stot / sink_v, sx2x_v_s)
    sx2x_v_i = jnp.where(sink_v_saturated, sx2x_v_i * stot / sink_v, sx2x_v_i)
    sx2x_v_g = jnp.where(sink_v_saturated, sx2x_v_g * stot / sink_v, sx2x_v_g)
    sink_v = jnp.where(sink_v_saturated, sx2x_v_s + sx2x_v_i + sx2x_v_g, sink_v)

    stot = q.c / dt
    sink_c_saturated = (sink_c > stot) & (q.c > const.qmin)
    sx2x_c_r = jnp.where(sink_c_saturated, sx2x_c_r * stot / sink_c, sx2x_c_r)
    sx2x_c_s = jnp.where(sink_c_saturated, sx2x_c_s * stot / sink_c, sx2x_c_s)
    sx2x_c_i = jnp.where(sink_c_saturated, sx2x_c_i * stot / sink_c, sx2x_c_i)
    sx2x_c_g = jnp.where(sink_c_saturated, sx2x_c_g * stot / sink_c, sx2x_c_g)
    sink_c = jnp.where(sink_c_saturated, sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g, sink_c)

    stot = q.r / dt
    sink_r_saturated = (sink_r > stot) & (q.r > const.qmin)
    sx2x_r_v = jnp.where(sink_r_saturated, sx2x_r_v * stot / sink_r, sx2x_r_v)
    sx2x_r_g = jnp.where(sink_r_saturated, sx2x_r_g * stot / sink_r, sx2x_r_g)
    sink_r = jnp.where(sink_r_saturated, sx2x_r_v + sx2x_r_g, sink_r)

    stot = q.s / dt
    sink_s_saturated = (sink_s > stot) & (q.s > const.qmin)
    sx2x_s_v = jnp.where(sink_s_saturated, sx2x_s_v * stot / sink_s, sx2x_s_v)
    sx2x_s_r = jnp.where(sink_s_saturated, sx2x_s_r * stot / sink_s, sx2x_s_r)
    sx2x_s_g = jnp.where(sink_s_saturated, sx2x_s_g * stot / sink_s, sx2x_s_g)
    sink_s = jnp.where(sink_s_saturated, sx2x_s_v + sx2x_s_r + sx2x_s_g, sink_s)

    stot = q.i / dt
    sink_i_saturated = (sink_i > stot) & (q.i > const.qmin)
    sx2x_i_v = jnp.where(sink_i_saturated, sx2x_i_v * stot / sink_i, sx2x_i_v)
    sx2x_i_c = jnp.where(sink_i_saturated, sx2x_i_c * stot / sink_i, sx2x_i_c)
    sx2x_i_s = jnp.where(sink_i_saturated, sx2x_i_s * stot / sink_i, sx2x_i_s)
    sx2x_i_g = jnp.where(sink_i_saturated, sx2x_i_g * stot / sink_i, sx2x_i_g)
    sink_i = jnp.where(sink_i_saturated, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, sink_i)

    stot = q.g / dt
    sink_g_saturated = (sink_g > stot) & (q.g > const.qmin)
    sx2x_g_v = jnp.where(sink_g_saturated, sx2x_g_v * stot / sink_g, sx2x_g_v)
    sx2x_g_r = jnp.where(sink_g_saturated, sx2x_g_r * stot / sink_g, sx2x_g_r)
    sink_g = jnp.where(sink_g_saturated, sx2x_g_v + sx2x_g_r, sink_g)

    # water content updates:
    dqdt_v = sx2x_r_v + sx2x_s_v + sx2x_i_v + sx2x_g_v - sink_v
    qv = jnp.where(mask, jnp.maximum(0.0, q.v + dqdt_v * dt), q.v)

    dqdt_c = sx2x_i_c - sink_c
    qc = jnp.where(mask, jnp.maximum(0.0, q.c + dqdt_c * dt), q.c)

    dqdt_r = sx2x_c_r + sx2x_s_r + sx2x_g_r - sink_r
    qr = jnp.where(mask, jnp.maximum(0.0, q.r + dqdt_r * dt), q.r)

    dqdt_s = sx2x_v_s + sx2x_c_s + sx2x_i_s - sink_s
    qs = jnp.where(mask, jnp.maximum(0.0, q.s + dqdt_s * dt), q.s)

    dqdt_i = sx2x_v_i + sx2x_c_i - sink_i
    qi = jnp.where(mask, jnp.maximum(0.0, q.i + dqdt_i * dt), q.i)

    dqdt_g = sx2x_v_g + sx2x_c_g + sx2x_r_g + sx2x_s_g + sx2x_i_g - sink_g
    qg = jnp.where(mask, jnp.maximum(0.0, q.g + dqdt_g * dt), q.g)

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv = (
        const.cvd
        + (const.cvv - const.cvd) * qtot
        + (const.clw - const.cvv) * qliq
        + (const.ci - const.cvv) * qice
    )

    t = jnp.where(
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

    return Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg), t


def temperature_update_scan(t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask):
    """Temperature update scan with energy flux.

    Note: All inputs are (ncells, nlev). Internally uses (nlev, ncells) for the scan
          to avoid repeated transposes in the scan loop.
    """
    ncells, nlev = t.shape

    # Use same dtype as input arrays to preserve precision
    init_state = TempState(
        t=jnp.zeros(ncells, dtype=t.dtype),
        eflx=jnp.zeros(ncells, dtype=t.dtype),
        activated=jnp.zeros(ncells, dtype=bool),
    )

    # Transpose once at API boundary: (ncells, nlev) -> (nlev, ncells)
    inputs = (
        jnp.swapaxes(t, 0, 1),
        jnp.swapaxes(t_kp1, 0, 1),
        jnp.swapaxes(ei_old, 0, 1),
        jnp.swapaxes(pr, 0, 1),
        jnp.swapaxes(pflx_tot, 0, 1),
        jnp.swapaxes(qv, 0, 1),
        jnp.swapaxes(qliq, 0, 1),
        jnp.swapaxes(qice, 0, 1),
        jnp.swapaxes(rho, 0, 1),
        jnp.swapaxes(dz, 0, 1),
        jnp.full((nlev, ncells), dt, dtype=t.dtype),
        jnp.swapaxes(mask, 0, 1),
    )

    final_state, outputs = lax.scan(temperature_scan_step, init_state, inputs)

    # Transpose back: (nlev, ncells) -> (ncells, nlev)
    return TempState(
        t=jnp.swapaxes(outputs.t, 0, 1),
        eflx=jnp.swapaxes(outputs.eflx, 0, 1),
        activated=jnp.swapaxes(outputs.activated, 0, 1)
    )


def precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt, fused=False, tiled=False, tile_size=4, optimize_layout=True, unrolled=False, pallas=False):
    """
    Apply precipitation sedimentation and temperature effects.
    From graupel.py:366-424.

    Args:
        fused: If True, use fused precipitation+temperature scan (90 kernels).
               If False, use separate scans (180 kernels).
        tiled: If True, use tiled scan to process multiple levels per iteration.
        tile_size: Number of levels to process per tiled scan iteration.
        optimize_layout: If True, keep arrays in GPU-optimal layout (nlev, ncells).

    Note: All inputs are (ncells, nlev). Internally we use (nlev, ncells) for scans
          to avoid repeated transposes in the scan loop when optimize_layout=True.
    """
    if fused:
        return precipitation_effects_fused(
            last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
        )

    ncells, nlev = t.shape

    # Store initial state for energy calculation
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = thermo.internal_energy(t, q_in.v, qliq, qice, rho, dz)

    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)

    # Velocity scale factors
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)

    # Fall speed parameters (from GT4Py idx namespace)
    params_list = [
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]

    if optimize_layout:
        # === Keep in GPU-optimal layout: (nlev, ncells) throughout ===
        zeta_T = jnp.swapaxes(zeta, 0, 1)  # (nlev, ncells)
        rho_T = jnp.swapaxes(rho, 0, 1)   # (nlev, ncells)
        q_list_T = [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
                    jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)]
        vc_list_T = [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
                     jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)]
        mask_list_T = [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
                       jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)]

        # Run batched precipitation scans with vertical-major data
        if pallas:
            # PALLAS: Custom GPU kernel with carry in registers
            if not PALLAS_AVAILABLE:
                raise RuntimeError("Pallas not available. Install with: pip install jax[cuda12_pallas]")
            results = precip_scan_pallas(
                params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T
            )
        elif unrolled:
            # UNROLLED: Single fused kernel (no lax.scan overhead)
            results = precip_scan_unrolled(
                params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T
            )
        elif tiled:
            # TILED: Reduce scan iterations by tile_size factor
            results = precip_scan_tiled(
                params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T, tile_size=tile_size
            )
        else:
            results = precip_scan_batched(
                params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T
            )

        # Unpack results (in nlev, ncells format)
        (qr_T, pr_T), (qs_T, ps_T), (qi_T, pi_T), (qg_T, pg_T) = results

        # === TRANSPOSE BACK only at the end: (nlev, ncells) -> (ncells, nlev) ===
        qr = jnp.swapaxes(qr_T, 0, 1)
        qs = jnp.swapaxes(qs_T, 0, 1)
        qi = jnp.swapaxes(qi_T, 0, 1)
        qg = jnp.swapaxes(qg_T, 0, 1)
        pr = jnp.swapaxes(pr_T, 0, 1)
        ps = jnp.swapaxes(ps_T, 0, 1)
        pi = jnp.swapaxes(pi_T, 0, 1)
        pg = jnp.swapaxes(pg_T, 0, 1)

        # Update for temperature scan - keep in (ncells, nlev) layout
        qliq = q_in.c + qr
        qice = qs + qi + qg
        pflx_tot = ps + pi + pg

        # Shift temperature for next level
        t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
        t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)

        kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

        # Temperature update scan
        result_t = temperature_update_scan(
            t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
        )
        t_new = result_t.t
        eflx = result_t.eflx
    else:
        # Original layout (more transposes, suboptimal for GPU)
        # === TRANSPOSE ONCE at API boundary: (ncells, nlev) -> (nlev, ncells) ===
        zeta_T = jnp.swapaxes(zeta, 0, 1)
        rho_T = jnp.swapaxes(rho, 0, 1)

        # Run batched precipitation scans with vertical-major data
        if tiled:
            from ..core.scans import precip_scan_batched_tiled
            results = precip_scan_batched_tiled(
                params_list,
                zeta_T,  # (nlev, ncells)
                rho_T,   # (nlev, ncells)
                [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
                 jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)],  # (nlev, ncells) each
                [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
                 jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)],
                [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
                 jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)],
                tile_size=tile_size
            )
        else:
            results = precip_scan_batched(
                params_list,
                zeta_T,  # (nlev, ncells)
                rho_T,   # (nlev, ncells)
                [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
                 jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)],  # (nlev, ncells) each
                [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
                 jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)],
                [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
                 jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)],
            )

        # Unpack results (still in nlev, ncells format)
        (qr_T, pr_T), (qs_T, ps_T), (qi_T, pi_T), (qg_T, pg_T) = results

        # === TRANSPOSE BACK: (nlev, ncells) -> (ncells, nlev) ===
        qr = jnp.swapaxes(qr_T, 0, 1)
        qs = jnp.swapaxes(qs_T, 0, 1)
        qi = jnp.swapaxes(qi_T, 0, 1)
        qg = jnp.swapaxes(qg_T, 0, 1)
        pr = jnp.swapaxes(pr_T, 0, 1)
        ps = jnp.swapaxes(ps_T, 0, 1)
        pi = jnp.swapaxes(pi_T, 0, 1)
        pg = jnp.swapaxes(pg_T, 0, 1)

        # Update for temperature scan
        qliq = q_in.c + qr
        qice = qs + qi + qg
        pflx_tot = ps + pi + pg

        # Shift temperature for next level
        t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
        t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)

        kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

        # Temperature update scan
        result_t = temperature_update_scan(
            t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
        )
        t_new = result_t.t
        eflx = result_t.eflx

    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt
    """
    Apply precipitation sedimentation and temperature effects.
    From graupel.py:366-424.

    Args:
        fused: If True, use fused precipitation+temperature scan (90 kernels).
               If False, use separate scans (180 kernels).
        tiled: If True, use tiled scan to process multiple levels per iteration.
        tile_size: Number of levels to process per tiled scan iteration.

    Note: All inputs are (ncells, nlev). Internally we use (nlev, ncells) for scans
          to avoid repeated transposes in the scan loop.
    """
    if fused:
        return precipitation_effects_fused(
            last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
        )

    _, nlev = t.shape

    # Store initial state for energy calculation
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = thermo.internal_energy(t, q_in.v, qliq, qice, rho, dz)

    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)

    # Velocity scale factors
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)

    # Fall speed parameters (from GT4Py idx namespace)
    params_list = [
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]

    # === TRANSPOSE ONCE at API boundary: (ncells, nlev) -> (nlev, ncells) ===
    zeta_T = jnp.swapaxes(zeta, 0, 1)
    rho_T = jnp.swapaxes(rho, 0, 1)

    # Run batched precipitation scans with vertical-major data
    if tiled:
        from ..core.scans import precip_scan_batched_tiled
        results = precip_scan_batched_tiled(
            params_list,
            zeta_T,  # (nlev, ncells)
            rho_T,   # (nlev, ncells)
            [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
             jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)],  # (nlev, ncells) each
            [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
             jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)],
            [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
             jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)],
            tile_size=tile_size
        )
    else:
        results = precip_scan_batched(
            params_list,
            zeta_T,  # (nlev, ncells)
            rho_T,   # (nlev, ncells)
            [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
             jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)],  # (nlev, ncells) each
            [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
             jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)],
            [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
             jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)],
        )

    # Unpack results (still in nlev, ncells format)
    (qr_T, pr_T), (qs_T, ps_T), (qi_T, pi_T), (qg_T, pg_T) = results

    # === TRANSPOSE BACK: (nlev, ncells) -> (ncells, nlev) ===
    qr = jnp.swapaxes(qr_T, 0, 1)
    qs = jnp.swapaxes(qs_T, 0, 1)
    qi = jnp.swapaxes(qi_T, 0, 1)
    qg = jnp.swapaxes(qg_T, 0, 1)
    pr = jnp.swapaxes(pr_T, 0, 1)
    ps = jnp.swapaxes(ps_T, 0, 1)
    pi = jnp.swapaxes(pi_T, 0, 1)
    pg = jnp.swapaxes(pg_T, 0, 1)

    # Update for temperature scan
    qliq = q_in.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg

    # Shift temperature for next level
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)

    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    # Temperature update scan
    result_t = temperature_update_scan(
        t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
    )
    t_new = result_t.t
    eflx = result_t.eflx

    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt
    """
    Apply precipitation sedimentation and temperature effects.
    From graupel.py:366-424.

    Args:
        fused: If True, use fused scan (90 kernel launches).
               If False, use separate scans (180 kernel launches).

    Note: All inputs are (ncells, nlev). Internally we use (nlev, ncells) for scans
          to avoid repeated transposes in the scan loop.
    """
    if fused:
        return precipitation_effects_fused(
            last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
        )

    _, nlev = t.shape

    # Store initial state for energy calculation
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = thermo.internal_energy(t, q_in.v, qliq, qice, rho, dz)

    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)

    # Velocity scale factors
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)

    # Fall speed parameters (from GT4Py idx namespace)
    params_list = [
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]

    # === TRANSPOSE ONCE at API boundary: (ncells, nlev) -> (nlev, ncells) ===
    zeta_T = jnp.swapaxes(zeta, 0, 1)
    rho_T = jnp.swapaxes(rho, 0, 1)

    # Run batched precipitation scans with vertical-major data
    results = precip_scan_batched(
        params_list,
        zeta_T,  # (nlev, ncells)
        rho_T,   # (nlev, ncells)
        [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
         jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)],  # (nlev, ncells) each
        [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
         jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)],
        [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
         jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)],
    )

    # Unpack results (still in nlev, ncells format)
    (qr_T, pr_T), (qs_T, ps_T), (qi_T, pi_T), (qg_T, pg_T) = results

    # === TRANSPOSE BACK: (nlev, ncells) -> (ncells, nlev) ===
    qr = jnp.swapaxes(qr_T, 0, 1)
    qs = jnp.swapaxes(qs_T, 0, 1)
    qi = jnp.swapaxes(qi_T, 0, 1)
    qg = jnp.swapaxes(qg_T, 0, 1)
    pr = jnp.swapaxes(pr_T, 0, 1)
    ps = jnp.swapaxes(ps_T, 0, 1)
    pi = jnp.swapaxes(pi_T, 0, 1)
    pg = jnp.swapaxes(pg_T, 0, 1)

    # Update for temperature scan
    qliq = q_in.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg

    # Shift temperature for next level
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)

    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    # Temperature update scan
    result_t = temperature_update_scan(
        t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
    )
    t_new = result_t.t
    eflx = result_t.eflx

    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt


def precipitation_effects_fused(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    FUSED implementation: Single scan for precipitation + temperature.
    
    Reduces kernel launches from 180 (2×90) to 90.
    Expected speedup: ~1.3-1.4x
    """
    from ..core.scans import precip_scan_step_fast, temperature_scan_step
    
    ncells, nlev = t.shape
    
    # Setup (same as unfused version)
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = thermo.internal_energy(t, q_in.v, qliq, qice, rho, dz)
    
    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)
    
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)
    
    params_list = [
        (14.58, 0.111, 1.0e-12),
        (57.80, 0.16666666666666666, 1.0e-12),
        (1.25, 0.160, 1.0e-12),
        (12.24, 0.217, 1.0e-08),
    ]
    
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)
    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g
    
    # Fused scan step
    def fused_step(carry, inputs):
        """Process one vertical level: do precipitation then temperature."""
        # Unpack carry
        precip_carries, temp_carry = carry
        
        # Unpack inputs
        # precip_inputs_per_species: tuple of 4 species, each is a tuple of 8 arrays (ncells,)
        # temp_inputs: tuple of 12 arrays (ncells,)
        precip_inputs_per_species, temp_inputs = inputs
        
        # === STEP 1: Precipitation for all 4 species ===
        new_precip_carries = []
        q_outs = []
        flx_outs = []
        
        for i in range(4):
            carry_i = precip_carries[i]
            # Each species' inputs: (prefactor, exponent, offset, zeta, vc, q, rho, mask)
            inputs_i = precip_inputs_per_species[i]
            new_carry_i, (q_out, flx_out) = precip_scan_step_fast(carry_i, inputs_i)
            new_precip_carries.append(new_carry_i)
            q_outs.append(q_out)
            flx_outs.append(flx_out)
        
        qr, qs, qi, qg = q_outs
        pr, ps, pi, pg = flx_outs
        
        # === STEP 2: Temperature using fresh precipitation results ===
        # Unpack temp inputs (13 elements)
        t_in, t_kp1_in, ei_old_in, _, _, qv_in, qc_in, qliq_in, qice_in, rho_in, dz_in, dt_in, mask_in = temp_inputs
        
        # Update with fresh precipitation results
        qliq_updated = qc_in + qr  # qc + qr (use qc from inputs)
        qice_updated = qs + qi + qg
        pflx_tot_updated = ps + pi + pg
        
        updated_temp_inputs = (
            t_in, t_kp1_in, ei_old_in, pr, pflx_tot_updated,
            qv_in, qliq_updated, qice_updated, rho_in, dz_in, dt_in, mask_in
        )
        
        new_temp_carry, temp_output = temperature_scan_step(temp_carry, updated_temp_inputs)
        
        # Return new carry and outputs
        new_carry = (tuple(new_precip_carries), new_temp_carry)
        outputs = (qr, qs, qi, qg, pr, ps, pi, pg, temp_output.t, temp_output.eflx)
        
        return new_carry, outputs
    
    # === TRANSPOSE ONCE at API boundary: (ncells, nlev) -> (nlev, ncells) ===
    # Transpose shared arrays once, reuse for all 4 species
    zeta_T = jnp.swapaxes(zeta, 0, 1)
    rho_T = jnp.swapaxes(rho, 0, 1)

    # Prepare per-species data
    q_list = [q_in.r, q_in.s, q_in.i, q_in.g]
    vc_list = [vc_r, vc_s, vc_i, vc_g]
    mask_list = [kmin_r, kmin_s, kmin_i, kmin_g]

    # Build per-species precipitation inputs with shape (nlev, ncells)
    # Each tuple element is: (prefactor, exponent, offset, zeta, vc, q, rho, mask)
    precip_inputs_per_species = []
    for i, params in enumerate(params_list):
        prefactor, exponent, offset = params
        # Broadcast scalars to (nlev, ncells) shape so scan can iterate over them
        species_inputs = (
            jnp.full((nlev, ncells), prefactor, dtype=t.dtype),
            jnp.full((nlev, ncells), exponent, dtype=t.dtype),
            jnp.full((nlev, ncells), offset, dtype=t.dtype),
            zeta_T,  # Reuse transposed array
            jnp.swapaxes(vc_list[i], 0, 1),  # (nlev, ncells)
            jnp.swapaxes(q_list[i], 0, 1),  # (nlev, ncells)
            rho_T,  # Reuse transposed array
            jnp.swapaxes(mask_list[i], 0, 1),  # (nlev, ncells)
        )
        precip_inputs_per_species.append(species_inputs)
    
    # Temperature inputs with shape (nlev, ncells) - reuse rho_T
    dz_T = jnp.swapaxes(dz, 0, 1)
    temp_inputs_all = (
        jnp.swapaxes(t, 0, 1), jnp.swapaxes(t_kp1, 0, 1), jnp.swapaxes(ei_old, 0, 1),
        jnp.zeros((nlev, ncells), dtype=t.dtype),  # pr placeholder
        jnp.zeros((nlev, ncells), dtype=t.dtype),  # pflx_tot placeholder
        jnp.swapaxes(q_in.v, 0, 1), jnp.swapaxes(q_in.c, 0, 1),
        jnp.swapaxes(qliq, 0, 1), jnp.swapaxes(qice, 0, 1),
        rho_T,  # Reuse transposed array
        dz_T,   # Reuse transposed array
        jnp.full((nlev, ncells), dt, dtype=t.dtype),
        jnp.swapaxes(kmin_rsig, 0, 1)
    )
    
    # Combine inputs: tuple of (precip_inputs_per_species, temp_inputs_all)
    fused_inputs = (tuple(precip_inputs_per_species), temp_inputs_all)
    
    # Initial carry - OPTIMIZED: Only 4 elements (removed rho and vc from carry)
    init_precip_carries = tuple([
        (jnp.zeros(ncells, dtype=t.dtype),  # q_prev
         jnp.zeros(ncells, dtype=t.dtype),  # flx_prev
         jnp.zeros(ncells, dtype=t.dtype),  # rhox_prev
         jnp.zeros(ncells, dtype=bool))     # activated_prev
        for _ in range(4)
    ])
    
    init_temp_carry = TempState(
        t=jnp.zeros(ncells, dtype=t.dtype),
        eflx=jnp.zeros(ncells, dtype=t.dtype),
        activated=jnp.zeros(ncells, dtype=bool)
    )
    
    init_carry = (init_precip_carries, init_temp_carry)
    
    # Run fused scan!
    final_carry, outputs = lax.scan(fused_step, init_carry, fused_inputs)
    
    # Unpack outputs
    qr, qs, qi, qg, pr, ps, pi, pg, t_new, eflx = outputs
    
    # Transpose back - use swapaxes instead of .T
    qr, qs, qi, qg = jnp.swapaxes(qr, 0, 1), jnp.swapaxes(qs, 0, 1), jnp.swapaxes(qi, 0, 1), jnp.swapaxes(qg, 0, 1)
    pr, ps, pi, pg = jnp.swapaxes(pr, 0, 1), jnp.swapaxes(ps, 0, 1), jnp.swapaxes(pi, 0, 1), jnp.swapaxes(pg, 0, 1)
    t_new, eflx = jnp.swapaxes(t_new, 0, 1), jnp.swapaxes(eflx, 0, 1)
    
    pflx_tot = ps + pi + pg
    
    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt


def graupel(last_level, dz, te, p, rho, q, dt, qnc, use_fused_scans=False, use_tiled_scans=False, tile_size=4, optimize_layout=True, use_unrolled=False, use_pallas=False):
    """
    Top-level graupel microphysics function.
    From graupel.py:427-456.

    Args:
        use_fused_scans: If True, use fused precipitation+temperature scan (90 kernels).
        use_tiled_scans: If True, use tiled scans (reduces iterations by tile_size).
        tile_size: Number of levels per tiled scan iteration.
        optimize_layout: If True, minimize transposes for better GPU memory layout.
        use_unrolled: If True, use unrolled loop (single kernel, no lax.scan).
        use_pallas: If True, use Pallas GPU kernel (carry in registers).
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects(
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt,
        fused=use_fused_scans, tiled=use_tiled_scans, tile_size=tile_size,
        optimize_layout=optimize_layout, unrolled=use_unrolled, pallas=use_pallas
    )

    return (
        t_final,
        Q(v=q_updated.v, c=q_updated.c, r=qr, s=qs, i=qi, g=qg),
        pflx,
        pr,
        ps,
        pi,
        pg,
        pre,
    )
    """
    Top-level graupel microphysics function.
    From graupel.py:427-456.
    
    Args:
        use_fused_scans: If True, use fused precipitation+temperature scan (90 kernels).
                        If False, use separate scans (180 kernels, baseline).
        use_tiled_scans: If True, use tiled scans to process multiple levels per iteration.
        tile_size: Number of levels per tiled scan iteration.
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects(
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt,
        fused=use_fused_scans, tiled=use_tiled_scans, tile_size=tile_size
    )

    return (
        t_final,
        Q(v=q_updated.v, c=q_updated.c, r=qr, s=qs, i=qi, g=qg),
        pflx,
        pr,
        ps,
        pi,
        pg,
        pre,
    )
    """
    Top-level graupel microphysics function.
    From graupel.py:427-456.
    
    Args:
        use_fused_scans: If True, use fused precipitation+temperature scan (90 kernels).
                        If False, use separate scans (180 kernels, baseline).
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects(
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt,
        fused=use_fused_scans
    )

    return (
        t_final,
        Q(v=q_updated.v, c=q_updated.c, r=qr, s=qs, i=qi, g=qg),
        pflx,
        pr,
        ps,
        pi,
        pg,
        pre,
    )


# ============================================================================
# JIT-compiled entry point (backend-switchable) TODO(dganellari):
# ============================================================================


@partial(jax.jit, static_argnames=['use_fused_scans', 'use_tiled_scans', 'tile_size', 'optimize_layout', 'use_unrolled', 'use_pallas'])
def graupel_run(dz, te, p, rho, q_in, dt, qnc, last_level=None, use_fused_scans=False, use_tiled_scans=False, tile_size=4, optimize_layout=True, use_unrolled=False, use_pallas=False):
    """
    JIT-compiled graupel driver.

    Args:
        use_fused_scans: If True, use fused precipitation+temperature scan (90 kernels).
        use_tiled_scans: If True, use tiled scans (reduces iterations by tile_size).
        tile_size: Number of levels per tiled scan iteration.
        optimize_layout: If True, minimize transposes for better GPU memory layout.
        use_unrolled: If True, use static unrolled loop.
        use_pallas: If True, use Pallas GPU kernel (target DaCe perf).
    """
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel(last_level, dz, te, p, rho, q_in, dt, qnc,
                   use_fused_scans=use_fused_scans, use_tiled_scans=use_tiled_scans,
                   tile_size=tile_size, optimize_layout=optimize_layout,
                   use_unrolled=use_unrolled, use_pallas=use_pallas)
    """
    JIT-compiled graupel driver (backend-switchable via environment variable).

    Args:
        use_fused_scans: If True, use fused precipitation+temperature scan (90 kernels).
                        If False, use separate scans (180 kernels, baseline).
                        MUST be static (known at compile time) for JIT compilation.
        use_tiled_scans: If True, use tiled scans to process multiple levels per iteration.
        tile_size: Number of levels per tiled scan iteration.
                        MUST be static (known at compile time) for JIT compilation.

    Note: Buffer donation (donate_argnums) is used to reduce D2D memory copies by allowing
          XLA to reuse input buffer memory for outputs where possible.
    """
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel(last_level, dz, te, p, rho, q_in, dt, qnc, use_fused_scans=use_fused_scans, use_tiled_scans=use_tiled_scans, tile_size=tile_size)


__all__ = ["graupel", "graupel_run", "precipitation_effects", "q_t_update"]
