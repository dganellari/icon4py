# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Complete JAX implementation of the graupel microphysics scheme.
This is a direct translation of GT4Py graupel.py to JAX.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

import constants as const
import transitions as trans


# ============================================================================
# Data Structures (matching GT4Py Q definition)
# ============================================================================

class Q(NamedTuple):
    """6 water species"""
    v: jnp.ndarray  # vapor
    c: jnp.ndarray  # cloud
    r: jnp.ndarray  # rain
    s: jnp.ndarray  # snow
    i: jnp.ndarray  # ice
    g: jnp.ndarray  # graupel


# ============================================================================
# Property Functions (from GT4Py properties.py)
# ============================================================================

def snow_number(t, rho, qs):
    """Compute snow number concentration."""
    TMIN = const.tmelt - 40.0
    TMAX = const.tmelt
    QSMIN = 2.0e-6
    XA1 = -1.65e0
    XA2 = 5.45e-2
    XA3 = 3.27e-4
    XB1 = 1.42e0
    XB2 = 1.19e-2
    XB3 = 9.60e-5
    N0S0 = 8.00e5
    N0S1 = 13.5 * 5.65e05
    N0S2 = -0.107
    N0S3 = 13.5
    N0S4 = 0.5 * N0S1
    N0S5 = 1.0e6
    N0S6 = 1.0e2 * N0S1
    N0S7 = 1.0e9

    tc = jnp.maximum(jnp.minimum(t, TMAX), TMIN) - const.tmelt
    alf = jnp.power(10.0, (XA1 + tc * (XA2 + tc * XA3)))
    bet = XB1 + tc * (XB2 + tc * XB3)
    n0s = N0S3 * jnp.power(((qs + QSMIN) * rho / const.ams), (4.0 - 3.0 * bet)) / (alf * alf * alf)
    y = jnp.exp(N0S2 * tc)
    n0smn = jnp.maximum(N0S4 * y, N0S5)
    n0smx = jnp.minimum(N0S6 * y, N0S7)

    return jnp.where(qs > const.qmin, jnp.minimum(n0smx, jnp.maximum(n0smn, n0s)), N0S0)


def snow_lambda(rho, qs, ns):
    """Compute snow slope parameter (lambda)."""
    A2 = const.ams * 2.0
    LMD_0 = 1.0e10
    BX = 1.0 / (const.bms + 1.0)
    QSMIN = 0.0e-6

    return jnp.where(qs > const.qmin, jnp.power((A2 * ns / ((qs + QSMIN) * rho)), BX), LMD_0)


def ice_number(t, rho):
    """Calculate ice number concentration."""
    A_COOP = 5.000
    B_COOP = 0.304
    NIMAX = 250.0e3

    return jnp.minimum(NIMAX, A_COOP * jnp.exp(B_COOP * (const.tmelt - t))) / rho


def ice_mass(qi, ni):
    """Compute ice crystal mass."""
    MI_MAX = 1.0e-9
    return jnp.maximum(const.m0_ice, jnp.minimum(qi / ni, MI_MAX))


def ice_sticking(t):
    """Compute ice sticking efficiency."""
    A_FREEZ = 0.09
    B_MAX_EXP = 1.00
    EFF_MIN = 0.075
    EFF_FAC = 3.5e-3
    TCRIT = const.tmelt - 85.0

    return jnp.maximum(
        jnp.maximum(jnp.minimum(jnp.exp(A_FREEZ * (t - const.tmelt)), B_MAX_EXP), EFF_MIN),
        EFF_FAC * (t - TCRIT)
    )


def deposition_factor(t, qvsi):
    """Compute deposition factor for ice/snow growth."""
    KAPPA = 2.40e-2
    B = 1.94
    A = const.als * const.als / (KAPPA * const.rv)
    CX = 2.22e-5 * jnp.power(const.tmelt, (-B)) * 101325.0

    x = CX / const.rd * jnp.power(t, B - 1.0)
    return (CX / const.rd * jnp.power(t, B - 1.0)) / (1.0 + A * x * qvsi / (t * t))


def deposition_auto_conversion(qi, m_ice, ice_dep):
    """Automatic conversion of deposited ice to snow."""
    M0_S = 3.0e-9
    B_DEP = 0.666666666666666667
    XCRIT = 1.0

    return jnp.where(
        qi > const.qmin,
        jnp.maximum(0.0, ice_dep) * B_DEP / (jnp.power((M0_S / m_ice), B_DEP) - XCRIT),
        0.0
    )


def ice_deposition_nucleation(t, qc, qi, ni, dvsi, dt):
    """Vapor deposition for new ice nucleation."""
    return jnp.where(
        (qi <= const.qmin) & (((t < const.tfrz_het2) & (dvsi > 0.0)) | ((t <= const.tfrz_het1) & (qc > const.qmin))),
        jnp.minimum(const.m0_ice * ni, jnp.maximum(0.0, dvsi)) / dt,
        0.0
    )


def vel_scale_factor_default(xrho):
    """Default velocity scale factor."""
    return xrho


def vel_scale_factor_ice(xrho):
    """Velocity scale factor for ice."""
    B_I = 0.66666666666666667
    return jnp.power(xrho, B_I)


def vel_scale_factor_snow(xrho, rho, t, qs):
    """Velocity scale factor for snow."""
    B_S = -0.16666666666666667
    return xrho * jnp.power(snow_number(t, rho, qs), B_S)


# ============================================================================
# Thermodynamic Functions (from GT4Py thermo.py)
# ============================================================================

def qsat_rho(t, rho):
    """Saturation specific humidity over liquid water."""
    C1ES = 610.78
    C3LES = 17.269
    C4LES = 35.86

    return (C1ES * jnp.exp(C3LES * (t - const.tmelt) / (t - C4LES))) / (rho * const.rv * t)


def qsat_ice_rho(t, rho):
    """Saturation specific humidity over ice."""
    C1ES = 610.78
    C3IES = 21.875
    C4IES = 7.66

    return (C1ES * jnp.exp(C3IES * (t - const.tmelt) / (t - C4IES))) / (rho * const.rv * t)


def qsat_rho_tmelt(rho):
    """Saturation specific humidity at melting temperature."""
    C1ES = 610.78
    return C1ES / (rho * const.rv * const.tmelt)


def internal_energy(t, qv, qliq, qice, rho, dz):
    """Compute internal energy from temperature."""
    qtot = qliq + qice + qv
    cv = const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice

    return rho * dz * (cv * t - qliq * const.lvc - qice * const.lsc)


# ============================================================================
# Scan Operators (from GT4Py graupel.py)
# ============================================================================

class PrecipState(NamedTuple):
    """State for precipitation scan."""
    q_update: jnp.ndarray
    flx: jnp.ndarray
    rho: jnp.ndarray
    vc: jnp.ndarray
    activated: jnp.ndarray


def precip_scan_step(previous_level, inputs):
    """Single vertical level of precipitation scan."""
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    current_level_activated = previous_level.activated | mask
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * previous_level.flx
    flx_partial = jnp.minimum(
        rho_x * vc * prefactor * jnp.power(rho_x + offset, exponent),
        flx_eff
    )

    rhox_prev = (previous_level.q_update + q) * 0.5 * previous_level.rho
    vt = jnp.where(
        previous_level.activated,
        previous_level.vc * prefactor * jnp.power(rhox_prev + offset, exponent),
        0.0
    )

    next_q_update = jnp.where(
        current_level_activated,
        (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho),
        q
    )

    next_flx = jnp.where(
        current_level_activated,
        (next_q_update * rho * vt + flx_partial) * 0.5,
        0.0
    )

    current_level = PrecipState(
        q_update=next_q_update,
        flx=next_flx,
        rho=rho,
        vc=vc,
        activated=current_level_activated
    )

    return current_level, current_level


def precip_scan(prefactor, exponent, offset, zeta, vc, q, rho, mask):
    """Precipitation scan (top to bottom)."""
    ncells, nlev = q.shape

    init_state = PrecipState(
        q_update=jnp.zeros(ncells),
        flx=jnp.zeros(ncells),
        rho=jnp.zeros(ncells),
        vc=jnp.zeros(ncells),
        activated=jnp.zeros(ncells, dtype=bool)
    )

    inputs = (prefactor.T, exponent.T, offset.T, zeta.T, vc.T, q.T, rho.T, mask.T)
    final_state, outputs = lax.scan(precip_scan_step, init_state, inputs)

    return PrecipState(
        q_update=outputs.q_update.T,
        flx=outputs.flx.T,
        rho=outputs.rho.T,
        vc=outputs.vc.T,
        activated=outputs.activated.T
    )


class TempState(NamedTuple):
    """State for temperature update scan."""
    t: jnp.ndarray
    eflx: jnp.ndarray
    activated: jnp.ndarray


def temperature_update_step(previous_level, inputs):
    """Single vertical level of temperature update scan."""
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask = inputs

    current_level_activated = previous_level.activated | mask

    eflx = jnp.where(
        current_level_activated,
        dt * (
            pr * (const.clw * t - const.cvd * t_kp1 - const.lvc)
            + pflx_tot * (const.ci * t - const.cvd * t_kp1 - const.lsc)
        ),
        previous_level.eflx
    )

    e_int = ei_old + previous_level.eflx - eflx

    # Temperature from internal energy (inlined)
    qtot = qliq + qice + qv
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho * dz
    t_new = jnp.where(
        current_level_activated,
        (e_int + rho * dz * (qliq * const.lvc + qice * const.lsc)) / cv,
        t
    )

    current_level = TempState(t=t_new, eflx=eflx, activated=current_level_activated)
    return current_level, current_level


def temperature_update_scan(t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask):
    """Temperature update scan with energy flux."""
    ncells, nlev = t.shape

    init_state = TempState(
        t=jnp.zeros(ncells),
        eflx=jnp.zeros(ncells),
        activated=jnp.zeros(ncells, dtype=bool)
    )

    inputs = (t.T, t_kp1.T, ei_old.T, pr.T, pflx_tot.T, qv.T, qliq.T, qice.T, rho.T, dz.T,
              jnp.full((nlev, ncells), dt), mask.T)

    final_state, outputs = lax.scan(temperature_update_step, init_state, inputs)

    return TempState(t=outputs.t.T, eflx=outputs.eflx.T, activated=outputs.activated.T)


# ============================================================================
# Field Operators (main physics)
# ============================================================================

def q_t_update(t, p, rho, q, dt, qnc):
    """
    Update water species and temperature via phase transitions.
    This is the massive 200-line function from graupel.py:158-362.
    """
    # Activation mask
    mask = jnp.where(
        (jnp.maximum(q.c, jnp.maximum(q.g, jnp.maximum(q.i, jnp.maximum(q.r, q.s)))) > const.qmin)
        | ((t < const.tfrz_het2) & (q.v > qsat_ice_rho(t, rho))),
        True,
        False
    )

    is_sig_present = jnp.maximum(q.g, jnp.maximum(q.i, q.s)) > const.qmin

    # Saturation deficits
    dvsw = q.v - qsat_rho(t, rho)
    qvsi = qsat_ice_rho(t, rho)
    dvsi = q.v - qvsi

    # Snow properties
    n_snow = snow_number(t, rho, q.s)
    l_snow = snow_lambda(rho, q.s, n_snow)

    # === Phase transitions (14 total) ===
    sx2x_c_r = trans.cloud_to_rain(t, q.c, q.r, qnc)
    sx2x_r_v = trans.rain_to_vapor(t, rho, q.c, q.r, dvsw, dt)
    sx2x_c_i = trans.cloud_x_ice(t, q.c, q.i, dt)
    sx2x_i_c = -jnp.minimum(sx2x_c_i, 0.0)
    sx2x_c_i = jnp.maximum(sx2x_c_i, 0.0)

    sx2x_c_s = trans.cloud_to_snow(t, q.c, q.s, n_snow, l_snow)
    sx2x_c_g = trans.cloud_to_graupel(t, rho, q.c, q.g)

    t_below_tmelt = t < const.tmelt
    t_at_least_tmelt = ~t_below_tmelt

    n_ice = ice_number(t, rho)
    m_ice = ice_mass(q.i, n_ice)
    x_ice = ice_sticking(t)

    eta = jnp.where(t_below_tmelt & is_sig_present, deposition_factor(t, qvsi), 0.0)
    sx2x_v_i = jnp.where(
        t_below_tmelt & is_sig_present,
        trans.vapor_x_ice(q.i, m_ice, eta, dvsi, rho, dt),
        0.0
    )
    sx2x_i_v = jnp.where(t_below_tmelt & is_sig_present, -jnp.minimum(sx2x_v_i, 0.0), 0.0)
    sx2x_v_i = jnp.where(t_below_tmelt & is_sig_present, jnp.maximum(sx2x_v_i, 0.0), sx2x_i_v)

    ice_dep = jnp.where(t_below_tmelt & is_sig_present, jnp.minimum(sx2x_v_i, dvsi / dt), 0.0)
    sx2x_i_s = jnp.where(
        t_below_tmelt & is_sig_present,
        deposition_auto_conversion(q.i, m_ice, ice_dep) + trans.ice_to_snow(q.i, n_snow, l_snow, x_ice),
        0.0
    )
    sx2x_i_g = jnp.where(
        t_below_tmelt & is_sig_present,
        trans.ice_to_graupel(rho, q.r, q.g, q.i, x_ice),
        0.0
    )
    sx2x_s_g = jnp.where(t_below_tmelt & is_sig_present, trans.snow_to_graupel(t, rho, q.c, q.s), 0.0)
    sx2x_r_g = jnp.where(
        t_below_tmelt & is_sig_present,
        trans.rain_to_graupel(t, rho, q.c, q.r, q.i, q.s, m_ice, dvsw, dt),
        0.0
    )

    sx2x_v_i = jnp.where(
        t_below_tmelt,
        sx2x_v_i + ice_deposition_nucleation(t, q.c, q.i, n_ice, dvsi, dt),
        0.0
    )
    sx2x_c_r = jnp.where(t_at_least_tmelt, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r)
    sx2x_c_s = jnp.where(t_at_least_tmelt, 0.0, sx2x_c_s)
    sx2x_c_g = jnp.where(t_at_least_tmelt, 0.0, sx2x_c_g)
    ice_dep = jnp.where(t_at_least_tmelt, 0.0, ice_dep)
    eta = jnp.where(t_at_least_tmelt, 0.0, eta)

    dvsw0 = jnp.where(is_sig_present, q.v - qsat_rho_tmelt(rho), 0.0)
    sx2x_v_s = jnp.where(
        is_sig_present,
        trans.vapor_x_snow(t, p, rho, q.s, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt),
        0.0
    )
    sx2x_s_v = jnp.where(is_sig_present, -jnp.minimum(sx2x_v_s, 0.0), 0.0)
    sx2x_v_s = jnp.where(is_sig_present, jnp.maximum(sx2x_v_s, 0.0), 0.0)

    sx2x_v_g = jnp.where(is_sig_present, trans.vapor_x_graupel(t, p, rho, q.g, dvsw, dvsi, dvsw0, dt), 0.0)
    sx2x_g_v = jnp.where(is_sig_present, -jnp.minimum(sx2x_v_g, 0.0), 0.0)
    sx2x_v_g = jnp.where(is_sig_present, jnp.maximum(sx2x_v_g, 0.0), 0.0)

    sx2x_s_r = jnp.where(is_sig_present, trans.snow_to_rain(t, p, rho, dvsw0, q.s), 0.0)
    sx2x_g_r = jnp.where(is_sig_present, trans.graupel_to_rain(t, p, rho, dvsw0, q.g), 0.0)

    # === Sink calculation and saturation limiting ===
    sink_v = sx2x_v_s + sx2x_v_i + sx2x_v_g
    sink_c = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g
    sink_r = sx2x_r_v + sx2x_r_g
    sink_s = jnp.where(is_sig_present, sx2x_s_v + sx2x_s_r + sx2x_s_g, 0.0)
    sink_i = jnp.where(is_sig_present, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, 0.0)
    sink_g = jnp.where(is_sig_present, sx2x_g_v + sx2x_g_r, 0.0)

    # Saturation limiters for each species
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

    # === Update water species ===
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

    # === Update temperature via latent heat ===
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
        t + dt * (
            (dqdt_c + dqdt_r) * (const.lvc - (const.clw - const.cvv) * t)
            + (dqdt_i + dqdt_s + dqdt_g) * (const.lsc - (const.ci - const.cvv) * t)
        ) / cv,
        t
    )

    return Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg), t


def precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    Apply precipitation sedimentation and temperature effects.
    From graupel.py:366-424.
    """
    # Store initial state for energy calculation
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = internal_energy(t, q_in.v, qliq, qice, rho, dz)

    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)

    # Velocity scale factors
    vc_r = vel_scale_factor_default(xrho)
    vc_s = vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = vel_scale_factor_ice(xrho)
    vc_g = vel_scale_factor_default(xrho)

    # Fall speed parameters (from idx namespace in GT4Py)
    # These are hardcoded constants for each species
    PREFACTOR_R, EXPONENT_R, OFFSET_R = 130.0, 0.5, 0.0
    PREFACTOR_S, EXPONENT_S, OFFSET_S = const.v0s, const.v1s, 0.0
    PREFACTOR_I, EXPONENT_I, OFFSET_I = 27.7, 0.216, 0.0
    PREFACTOR_G, EXPONENT_G, OFFSET_G = 442.0, 0.89, 0.0

    # Run 4 precipitation scans
    result_r = precip_scan(
        jnp.full_like(zeta, PREFACTOR_R), jnp.full_like(zeta, EXPONENT_R),
        jnp.full_like(zeta, OFFSET_R), zeta, vc_r, q_in.r, rho, kmin_r
    )
    qr, pr = result_r.q_update, result_r.flx

    result_s = precip_scan(
        jnp.full_like(zeta, PREFACTOR_S), jnp.full_like(zeta, EXPONENT_S),
        jnp.full_like(zeta, OFFSET_S), zeta, vc_s, q_in.s, rho, kmin_s
    )
    qs, ps = result_s.q_update, result_s.flx

    result_i = precip_scan(
        jnp.full_like(zeta, PREFACTOR_I), jnp.full_like(zeta, EXPONENT_I),
        jnp.full_like(zeta, OFFSET_I), zeta, vc_i, q_in.i, rho, kmin_i
    )
    qi, pi = result_i.q_update, result_i.flx

    result_g = precip_scan(
        jnp.full_like(zeta, PREFACTOR_G), jnp.full_like(zeta, EXPONENT_G),
        jnp.full_like(zeta, OFFSET_G), zeta, vc_g, q_in.g, rho, kmin_g
    )
    qg, pg = result_g.q_update, result_g.flx

    # Update for temperature scan
    qliq = q_in.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg

    # Shift temperature for next level (concat_where equivalent)
    ncells, nlev = t.shape
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)  # Use last level for boundary
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)  # Mask beyond last_lev

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

    return t_final, Q(v=q_updated.v, c=q_updated.c, r=qr, s=qs, i=qi, g=qg), pflx, pr, ps, pi, pg, pre


# ============================================================================
# JIT-compiled entry point (backend-switchable)
# ============================================================================

from backend import jit_compile

@jit_compile
def graupel_run(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """
    JIT-compiled graupel driver (backend-switchable via environment variable).

    Args:
        dz: Layer thickness [m] - shape (ncells, nlev)
        te: Temperature [K] - shape (ncells, nlev)
        p: Pressure [Pa] - shape (ncells, nlev)
        rho: Density [kg/m3] - shape (ncells, nlev)
        q_in: Q NamedTuple with 6 water species - each shape (ncells, nlev)
        dt: Time step [s] - scalar
        qnc: Cloud number concentration [1/m3] - scalar
        last_level: Last vertical level index (defaults to nlev-1)

    Returns:
        Tuple of (t_out, q_out, pflx, pr, ps, pi, pg, pre)
    """
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel(last_level, dz, te, p, rho, q_in, dt, qnc)
