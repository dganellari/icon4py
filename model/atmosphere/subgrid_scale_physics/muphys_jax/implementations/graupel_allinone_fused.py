# ICON4Py - JAX all-in-one fused scan implementation
#
# This implementation fuses all precipitation species and temperature updates
# into a single lax.scan step, matching the GT4Py pattern for optimal performance.
import jax

import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import NamedTuple

from ..core import properties as props, thermo, transitions as trans
from ..core.common import constants as const
from ..core.common.backend import jit_compile
from ..core.definitions import Q, TempState

class PrecipStateQx(NamedTuple):
    x: jnp.ndarray
    p: jnp.ndarray
    vc: jnp.ndarray
    activated: jnp.ndarray

class IntegrationState(NamedTuple):
    r: PrecipStateQx
    s: PrecipStateQx
    i: PrecipStateQx
    g: PrecipStateQx
    t_state: TempState
    rho: jnp.ndarray
    pflx_tot: jnp.ndarray


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


def precipitation_effects_allinone_fused(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    All-in-one fused scan: single JAX scan for precipitation + temperature.

    Matches GT4Py's "all-in-one" scan pattern for optimal performance.
    """
    ncells, nlev = t.shape

    # Setup (same as other implementations)
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
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]

    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)
    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    # Prepare arrays in (nlev, ncells) order for scan
    t_T = t.T
    t_kp1_T = t_kp1.T
    rho_T = rho.T
    dz_T = dz.T
    q_tuple = tuple(v.T for v in q_in)
    masks_tuple = (kmin_r.T, kmin_s.T, kmin_i.T, kmin_g.T)
    dt_arr = jnp.full((nlev, ncells), dt)

    # Initial carry
    init_carry = IntegrationState(
        r=PrecipStateQx(jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,), dtype=bool)),
        s=PrecipStateQx(jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,), dtype=bool)),
        i=PrecipStateQx(jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,), dtype=bool)),
        g=PrecipStateQx(jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,), dtype=bool)),
        t_state=TempState(jnp.zeros((ncells,)), jnp.zeros((ncells,)), jnp.zeros((ncells,), dtype=bool)),
        rho=jnp.zeros((ncells,)),
        pflx_tot=jnp.zeros((ncells,)),
    )

    # Prepare params and const dicts
    params = {
        'r': params_list[0],
        's': params_list[1],
        'i': params_list[2],
        'g': params_list[3],
    }

    const_dict = {
        'clw': const.clw, 'cvd': const.cvd, 'lvc': const.lvc, 'ci': const.ci,
        'cvv': const.cvv, 'lsc': const.lsc, 'rho_00': const.rho_00,
        'internal_energy': thermo.internal_energy,
        'vel_scale_factor_default': props.vel_scale_factor_default,
        'vel_scale_factor_snow': props.vel_scale_factor_snow,
        'vel_scale_factor_ice': props.vel_scale_factor_ice,
    }

    # Compute fall speed corrections (same as other implementations)
    vc_r_T = props.vel_scale_factor_default(xrho).T
    vc_s_T = props.vel_scale_factor_snow(xrho, rho, t, q_in.s).T
    vc_i_T = props.vel_scale_factor_ice(xrho).T
    vc_g_T = props.vel_scale_factor_default(xrho).T

    # Prepare scan inputs
    scan_inputs = (t_T, t_kp1_T, rho_T, q_tuple, masks_tuple, dt_arr, dz_T, vc_r_T, vc_s_T, vc_i_T, vc_g_T)

    # Run the fused scan
    final_carry, outputs = lax.scan(
        lambda carry, x: fused_scan_step_allinone(carry, x, params, const_dict),
        init_carry,
        scan_inputs
    )

    # Unpack outputs
    r_x, s_x, i_x, g_x, r_p, s_p, i_p, g_p, t_out, eflx = outputs

    # Transpose back to (ncells, nlev)
    def tb(x): return x.T
    return (
        tb(r_x), tb(s_x), tb(i_x), tb(g_x), tb(t_out),
        tb(r_p + s_p + i_p + g_p), tb(r_p), tb(s_p), tb(i_p), tb(g_p), tb(eflx)
    )


def fused_scan_step_allinone(prev: IntegrationState, inputs, params, const):
    """All-in-one fused scan step using real physics."""
    t, t_kp1, rho, q_tuple, masks_tuple, dt, dz, vc_r, vc_s, vc_i, vc_g = inputs
    q_v, q_c, q_r, q_s, q_i, q_g = q_tuple
    mask_r, mask_s, mask_i, mask_g = masks_tuple
    zeta = dt / (2.0 * dz)

    # Precip species update
    r = precip_qx_level_update_allinone(prev.r, prev.rho, *params['r'], zeta, vc_r, q_r, rho, mask_r)
    s = precip_qx_level_update_allinone(prev.s, prev.rho, *params['s'], zeta, vc_s, q_s, rho, mask_s)
    i = precip_qx_level_update_allinone(prev.i, prev.rho, *params['i'], zeta, vc_i, q_i, rho, mask_i)
    g = precip_qx_level_update_allinone(prev.g, prev.rho, *params['g'], zeta, vc_g, q_g, rho, mask_g)

    q_v, q_c, q_r, q_s, q_i, q_g = q_tuple

    qliq = q_c + r.x
    qice = s.x + i.x + g.x
    kmin_rsig = mask_r | mask_s | mask_i | mask_g

    # Temperature update
    t_state = temperature_update_allinone(
        prev.t_state, t, t_kp1, r.p, s.p + i.p + g.p, q_v, qliq, qice, rho, dz, dt, kmin_rsig, const
    )

    pflx_tot = s.p + i.p + g.p + r.p
    return IntegrationState(r=r, s=s, i=i, g=g, t_state=t_state, rho=rho, pflx_tot=pflx_tot), (
        r.x, s.x, i.x, g.x, r.p, s.p, i.p, g.p, t_state.t, t_state.eflx
    )


def precip_qx_level_update_allinone(prev: PrecipStateQx, prev_rho, prefactor, exponent, offset, zeta, vc, q, rho, mask):
    """Precipitation update using real physics."""
    current_activated = prev.activated | mask
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * prev.p
    flx_partial = jnp.minimum(rho_x * vc * prefactor * jnp.power(rho_x + offset, exponent), flx_eff)
    rhox_prev = (prev.x + q) * 0.5 * prev_rho
    vt = jnp.where(prev.activated, prev.vc * prefactor * jnp.power(rhox_prev + offset, exponent), 0.0)
    x = jnp.where(current_activated, (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho), q)
    p = jnp.where(current_activated, (x * rho * vt + flx_partial) * 0.5, 0.0)
    return PrecipStateQx(x=x, p=p, vc=vc, activated=current_activated)


def temperature_update_allinone(prev: TempState, t, t_kp1, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask, const):
    """Temperature update using real physics."""
    current_activated = prev.activated | mask
    eflx = jnp.where(current_activated,
        pr * (const['clw'] * t - const['cvd'] * t_kp1 - const['lvc']) + pflx_tot * (const['ci'] * t - const['cvd'] * t_kp1 - const['lsc']),
        prev.eflx)
    e_int = (
        const['internal_energy'](t, qv, qliq, qice, rho, dz)
        + dt * prev.eflx
        - dt * eflx
    )
    qtot = qliq + qice + qv
    cv = (const['cvd'] * (1.0 - qtot) + const['cvv'] * qv + const['clw'] * qliq + const['ci'] * qice) * rho * dz
    t_new = jnp.where(current_activated, (e_int + rho * dz * (qliq * const['lvc'] + qice * const['lsc'])) / cv, t)
    return TempState(t=t_new, eflx=eflx, activated=current_activated)


def graupel_allinone_fused(last_level, dz, te, p, rho, q, dt, qnc):
    """
    All-in-one fused graupel microphysics function.
    Single JAX scan combining precipitation and temperature updates.
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects (all-in-one fused scan)
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects_allinone_fused(
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


@partial(jax.jit, static_argnames=['last_level'])
def graupel_allinone_fused_run(dz, te, p, rho, q_in, dt, qnc, last_level=None, **kwargs):
    """
    JIT-compiled all-in-one fused graupel driver.
    Ignores extra kwargs for compatibility with driver.
    """
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel_allinone_fused(last_level, dz, te, p, rho, q_in, dt, qnc)


__all__ = ["graupel_allinone_fused", "graupel_allinone_fused_run"]