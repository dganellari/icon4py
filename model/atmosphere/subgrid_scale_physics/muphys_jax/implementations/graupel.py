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

import jax.numpy as jnp
from jax import lax

from ..core import properties as props, thermo, transitions as trans
from ..core.common import constants as const
from ..core.common.backend import jit_compile

# Import from core modules
from ..core.definitions import Q, TempState
from ..core.scans import precip_scan_batched, temperature_scan_step


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
    """Temperature update scan with energy flux."""
    ncells, nlev = t.shape

    init_state = TempState(
        t=jnp.zeros(ncells), eflx=jnp.zeros(ncells), activated=jnp.zeros(ncells, dtype=bool)
    )

    inputs = (
        t.T,
        t_kp1.T,
        ei_old.T,
        pr.T,
        pflx_tot.T,
        qv.T,
        qliq.T,
        qice.T,
        rho.T,
        dz.T,
        jnp.full((nlev, ncells), dt),
        mask.T,
    )

    final_state, outputs = lax.scan(temperature_scan_step, init_state, inputs)

    return TempState(t=outputs.t.T, eflx=outputs.eflx.T, activated=outputs.activated.T)


def precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    Apply precipitation sedimentation and temperature effects.
    From graupel.py:366-424.

    Optimized to batch all 4 precipitation scans via vmap for better GPU utilization.
    """
    # Store initial state for endergy calculation
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
    # Order: rain, snow, ice, graupel
    params_list = [
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]

    # Run batched precipitation scans (all 4 species in parallel via vmap)
    results = precip_scan_batched(
        params_list,
        zeta,
        rho,
        [q_in.r, q_in.s, q_in.i, q_in.g],
        [vc_r, vc_s, vc_i, vc_g],
        [kmin_r, kmin_s, kmin_i, kmin_g],
    )

    # Unpack results: rain, snow, ice, graupel
    (qr, pr), (qs, ps), (qi, pi), (qg, pg) = results

    # Update for temperature scan
    qliq = q_in.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg

    # Shift temperature for next level
    ncells, nlev = t.shape
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


def graupel(last_level, dz, te, p, rho, q, dt, qnc):
    """
    Top-level graupel microphysics function.
    From graupel.py:427-456.
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
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt
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


@jit_compile
def graupel_run(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """
    JIT-compiled graupel driver (backend-switchable via environment variable).
    """
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel(last_level, dz, te, p, rho, q_in, dt, qnc)


__all__ = ["graupel", "graupel_run", "precipitation_effects", "q_t_update"]
