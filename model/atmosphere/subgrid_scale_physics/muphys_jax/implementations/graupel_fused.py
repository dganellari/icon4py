# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fused scan implementation - combines precipitation + temperature scans.

This reduces kernel launches from 180 (90×2) to 90 (single fused scan).
"""

import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass

from ..core.common import constants as const
from ..core.definitions import TempState


@dataclass
class FusedCarry:
    """Combined carry state for fused precipitation + temperature scan."""
    # Precipitation state (4 species)
    q_precip: tuple  # (q_r, q_s, q_i, q_g)
    flx_precip: tuple  # (flx_r, flx_s, flx_i, flx_g)
    rho_prev: tuple
    vc_prev: tuple
    activated_precip: tuple
    
    # Temperature state
    t_prev: jnp.ndarray
    eflx_prev: jnp.ndarray
    activated_temp: jnp.ndarray


def fused_scan_step(carry, inputs):
    """
    Fused scan step: precipitation + temperature in one iteration.
    
    This eliminates the data dependency between scans by computing
    precipitation first, then immediately using results for temperature.
    """
    # Unpack inputs
    (precip_inputs, temp_inputs) = inputs
    
    # === PRECIPITATION STEP (4 species in parallel) ===
    new_q_precip = []
    new_flx_precip = []
    new_rho_prev = []
    new_vc_prev = []
    new_activated_precip = []
    
    for i in range(4):  # rain, snow, ice, graupel
        prefactor, exponent, offset, zeta, vc, q, rho, mask = precip_inputs[i]
        
        q_prev, flx_prev, rho_prev, vc_prev, activated_prev = (
            carry.q_precip[i],
            carry.flx_precip[i],
            carry.rho_prev[i],
            carry.vc_prev[i],
            carry.activated_precip[i],
        )
        
        # Update activation
        activated = activated_prev | mask
        
        # Physics computation (inlined from precip_scan_step_fast)
        rho_x = q * rho
        flx_eff = (rho_x / zeta) + 2.0 * flx_prev
        fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
        flx_partial = lax.min(rho_x * vc * fall_speed, flx_eff)
        
        rhox_prev = (q_prev + q) * 0.5 * rho_prev
        vt_active = vc_prev * prefactor * lax.pow(rhox_prev + offset, exponent)
        vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q))
        
        q_activated = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)
        flx_activated = (q_activated * rho * vt + flx_partial) * 0.5
        
        q_out = lax.select(activated, q_activated, q)
        flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q))
        
        new_q_precip.append(q_out)
        new_flx_precip.append(flx_out)
        new_rho_prev.append(rho)
        new_vc_prev.append(vc)
        new_activated_precip.append(activated)
    
    # === TEMPERATURE STEP (uses fresh precipitation results) ===
    t, t_kp1, ei_old, qv, qc, qliq_in, qice_in, rho, dz, dt, mask_temp = temp_inputs
    
    # Use updated precipitation to compute qliq and qice
    qr, qs, qi, qg = new_q_precip
    pr, ps, pi, pg = new_flx_precip
    
    qliq = qc + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg
    
    activated_temp = carry.activated_temp | mask_temp
    
    # Energy flux from precipitation
    cvd_t_kp1 = const.cvd * t_kp1
    eflx_new = dt * (
        pr * (const.clw * t - cvd_t_kp1 - const.lvc)
        + pflx_tot * (const.ci * t - cvd_t_kp1 - const.lsc)
    )
    
    e_int = ei_old + carry.eflx_prev - eflx_new
    
    # Temperature update
    qtot = qliq + qice + qv
    rho_dz = rho * dz
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho_dz
    t_new = (e_int + rho_dz * (qliq * const.lvc + qice * const.lsc)) / cv
    
    eflx_out = lax.select(activated_temp, eflx_new, carry.eflx_prev)
    t_out = lax.select(activated_temp, t_new, t)
    
    # === BUILD NEW CARRY ===
    new_carry = FusedCarry(
        q_precip=tuple(new_q_precip),
        flx_precip=tuple(new_flx_precip),
        rho_prev=tuple(new_rho_prev),
        vc_prev=tuple(new_vc_prev),
        activated_precip=tuple(new_activated_precip),
        t_prev=t_out,
        eflx_prev=eflx_out,
        activated_temp=activated_temp,
    )
    
    # === OUTPUTS ===
    outputs = (*new_q_precip, *new_flx_precip, t_out, eflx_out)
    
    return new_carry, outputs


def precipitation_effects_fused(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    Fused precipitation + temperature scan.
    
    Replaces the 2 separate scans (precip_scan_batched + temperature_update_scan)
    with a single fused scan that processes both in one pass.
    
    Reduces kernel launches: 180 → 90 (~1.4x speedup expected)
    """
    from ..core import properties as props, thermo
    
    ncells, nlev = t.shape
    
    # === SETUP (same as before) ===
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
    
    # Shift temperature for next level
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)
    
    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g
    
    # === PREPARE INPUTS FOR FUSED SCAN ===
    # Transpose to (nlev, ncells) for scan
    q_list = [q_in.r, q_in.s, q_in.i, q_in.g]
    vc_list = [vc_r, vc_s, vc_i, vc_g]
    mask_list = [kmin_r, kmin_s, kmin_i, kmin_g]
    
    # Broadcast parameters for each level
    precip_inputs_per_level = []
    for level in range(nlev):
        level_precip_inputs = []
        for i, params in enumerate(params_list):
            prefactor, exponent, offset = params
            level_precip_inputs.append((
                prefactor,
                exponent,
                offset,
                zeta[:, level],
                vc_list[i][:, level],
                q_list[i][:, level],
                rho[:, level],
                mask_list[i][:, level],
            ))
        precip_inputs_per_level.append(tuple(level_precip_inputs))
    
    temp_inputs_per_level = (
        t.T,
        t_kp1.T,
        ei_old.T,
        q_in.v.T,
        q_in.c.T,
        qliq.T,
        qice.T,
        rho.T,
        dz.T,
        jnp.full((nlev, ncells), dt, dtype=t.dtype),
        kmin_rsig.T,
    )
    
    # Combine inputs
    fused_inputs = (precip_inputs_per_level, temp_inputs_per_level)
    
    # === INITIAL CARRY ===
    init_carry = FusedCarry(
        q_precip=tuple([jnp.zeros(ncells, dtype=t.dtype) for _ in range(4)]),
        flx_precip=tuple([jnp.zeros(ncells, dtype=t.dtype) for _ in range(4)]),
        rho_prev=tuple([jnp.zeros(ncells, dtype=t.dtype) for _ in range(4)]),
        vc_prev=tuple([jnp.zeros(ncells, dtype=t.dtype) for _ in range(4)]),
        activated_precip=tuple([jnp.zeros(ncells, dtype=bool) for _ in range(4)]),
        t_prev=jnp.zeros(ncells, dtype=t.dtype),
        eflx_prev=jnp.zeros(ncells, dtype=t.dtype),
        activated_temp=jnp.zeros(ncells, dtype=bool),
    )
    
    # === RUN FUSED SCAN (single pass!) ===
    final_carry, outputs = lax.scan(fused_scan_step, init_carry, fused_inputs)
    
    # === UNPACK OUTPUTS ===
    qr, qs, qi, qg, pr, ps, pi, pg, t_new, eflx = outputs
    
    # Transpose back to (ncells, nlev)
    qr = qr.T
    qs = qs.T
    qi = qi.T
    qg = qg.T
    pr = pr.T
    ps = ps.T
    pi = pi.T
    pg = pg.T
    t_new = t_new.T
    eflx = eflx.T
    
    pflx_tot = ps + pi + pg
    
    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt
