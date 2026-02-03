# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
IREE-optimized scan operators for vertical physics in muphys microphysics.

Key differences from baseline scans:
1. Uses lax.fori_loop instead of lax.scan where possible
2. Avoids vmap for species batching (explicit Python loop)
3. Uses simple tuple carries instead of NamedTuples/dicts
4. Avoids dynamic_update_slice inside loops (causes IREE CUDA issues)
"""

import jax
import jax.numpy as jnp
from jax import lax

from .common import constants as const
from .definitions import TempState


def precip_scan_step_iree(carry, inputs):
    """
    Precipitation scan step with simple tuple carry.
    
    Same as baseline but uses tuple instead of named fields.
    """
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = carry
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    activated = activated_prev | mask

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

    new_carry = (q_out, flx_out, rho, vc, activated)
    outputs = (q_out, flx_out)

    return new_carry, outputs


def _single_species_scan_iree(params, zeta, rho, q, vc, mask):
    """
    Single species precipitation scan using lax.scan with tuple carry.
    
    This is called sequentially for each species (no vmap).
    """
    prefactor, exponent, offset = params
    ncells, nlev = q.shape

    prefactor_arr = jnp.broadcast_to(prefactor, (nlev, ncells))
    exponent_arr = jnp.broadcast_to(exponent, (nlev, ncells))
    offset_arr = jnp.broadcast_to(offset, (nlev, ncells))

    # Simple tuple carry
    init_carry = (
        jnp.zeros(ncells, dtype=q.dtype),  # q_prev
        jnp.zeros(ncells, dtype=q.dtype),  # flx_prev
        jnp.zeros(ncells, dtype=q.dtype),  # rho_prev
        jnp.zeros(ncells, dtype=q.dtype),  # vc_prev
        jnp.zeros(ncells, dtype=bool),     # activated
    )

    inputs = (prefactor_arr, exponent_arr, offset_arr, zeta.T, vc.T, q.T, rho.T, mask.T)

    final_carry, outputs = lax.scan(precip_scan_step_iree, init_carry, inputs)
    q_out, flx_out = outputs

    return q_out.T, flx_out.T


def precip_scan_sequential(params_list, zeta, rho, q_list, vc_list, mask_list):
    """
    Run 4 precipitation scans sequentially (no vmap).
    
    IREE-optimized: explicit Python loop instead of vmap.
    This allows IREE to compile each scan as a separate operation.
    """
    results = []
    for i in range(4):
        params = params_list[i]
        q_out, flx_out = _single_species_scan_iree(params, zeta, rho, q_list[i], vc_list[i], mask_list[i])
        results.append((q_out, flx_out))
    return results


def temperature_scan_step_iree(carry, inputs):
    """
    Temperature scan step with simple tuple carry.
    """
    eflx_prev, activated_prev = carry
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask = inputs

    activated = activated_prev | mask

    cvd_t_kp1 = const.cvd * t_kp1
    eflx_new = dt * (
        pr * (const.clw * t - cvd_t_kp1 - const.lvc)
        + pflx_tot * (const.ci * t - cvd_t_kp1 - const.lsc)
    )

    e_int = ei_old + eflx_prev - eflx_new

    qtot = qliq + qice + qv
    rho_dz = rho * dz
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho_dz
    t_new = (e_int + rho_dz * (qliq * const.lvc + qice * const.lsc)) / cv

    eflx = lax.select(activated, eflx_new, eflx_prev)
    t_out = lax.select(activated, t_new, t)

    new_carry = (eflx, activated)
    outputs = (t_out, eflx)

    return new_carry, outputs


def temperature_scan_iree(t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask):
    """
    Temperature correction scan using lax.scan with simple tuple carry.
    
    Returns TempState for compatibility with caller.
    """
    ncells, nlev = t.shape
    
    # Simple tuple carry (eflx_prev, activated_prev)
    init_carry = (
        jnp.zeros(ncells, dtype=t.dtype),  # eflx_prev
        jnp.zeros(ncells, dtype=bool),      # activated_prev
    )
    
    inputs = (
        t.T, t_kp1.T, ei_old.T, pr.T, pflx_tot.T,
        qv.T, qliq.T, qice.T, rho.T, dz.T,
        jnp.broadcast_to(dt, (nlev, ncells)), mask.T
    )
    
    final_carry, outputs = lax.scan(temperature_scan_step_iree, init_carry, inputs)
    t_out, eflx = outputs
    
    return TempState(
        t=t_out.T,
        eflx=eflx.T,
        activated=final_carry[1]
    )


# ============================================================================
# Alternative: fori_loop with scan-based output (hybrid approach)
# ============================================================================

def precip_fori_single_level(k, state_and_arrays):
    """
    Process single level in fori_loop, accumulating outputs via returned tuple.
    
    This avoids dynamic_update_slice by using tuple accumulation.
    """
    state, q_all, vc_all, rho_all, zeta_all, mask_all, params, q_outs, flx_outs = state_and_arrays
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = state
    prefactor, exponent, offset = params
    
    # Extract level k
    q_k = q_all[:, k]
    vc_k = vc_all[:, k]
    rho_k = rho_all[:, k]
    zeta_k = zeta_all[:, k]
    mask_k = mask_all[:, k]
    
    activated = activated_prev | mask_k
    
    rho_x = q_k * rho_k
    flx_eff = (rho_x / zeta_k) + 2.0 * flx_prev
    
    fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
    flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)
    
    rhox_prev = (q_prev + q_k) * 0.5 * rho_prev
    vt_active = vc_prev * prefactor * lax.pow(rhox_prev + offset, exponent)
    vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q_k))
    
    q_activated = (zeta_k * (flx_eff - flx_partial)) / ((1.0 + zeta_k * vt) * rho_k)
    flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5
    
    q_result = lax.select(activated, q_activated, q_k)
    flx_result = lax.select(activated, flx_activated, jnp.zeros_like(q_k))
    
    # Update output arrays using .at[].set() - IREE should handle this
    q_outs_new = q_outs.at[:, k].set(q_result)
    flx_outs_new = flx_outs.at[:, k].set(flx_result)
    
    new_state = (q_result, flx_result, rho_k, vc_k, activated)
    return (new_state, q_all, vc_all, rho_all, zeta_all, mask_all, params, q_outs_new, flx_outs_new)


def precip_scan_fori(params, zeta, rho, q, vc, mask):
    """
    Single species precipitation scan using fori_loop with .at[].set().
    """
    prefactor, exponent, offset = params[0], params[1], params[2]
    ncells, nlev = q.shape
    
    init_state = (
        jnp.zeros(ncells, dtype=q.dtype),  # q_prev
        jnp.zeros(ncells, dtype=q.dtype),  # flx_prev
        jnp.zeros(ncells, dtype=q.dtype),  # rho_prev
        jnp.zeros(ncells, dtype=q.dtype),  # vc_prev
        jnp.zeros(ncells, dtype=bool),     # activated
    )
    
    q_outs = jnp.zeros_like(q)
    flx_outs = jnp.zeros_like(q)
    
    init_all = (init_state, q, vc, rho, zeta, mask, (prefactor, exponent, offset), q_outs, flx_outs)
    
    final_all = lax.fori_loop(0, nlev, precip_fori_single_level, init_all)
    
    return final_all[7], final_all[8]  # q_outs, flx_outs


def precip_scan_fori_sequential(params_list, zeta, rho, q_list, vc_list, mask_list):
    """
    Run 4 precipitation scans via fori_loop sequentially.
    """
    results = []
    for i in range(4):
        params = jnp.array(params_list[i], dtype=zeta.dtype)
        q_out, flx_out = precip_scan_fori(params, zeta, rho, q_list[i], vc_list[i], mask_list[i])
        results.append((q_out, flx_out))
    return results


# ============================================================================
# Alternative: Fully unrolled (Python loop at trace time)
# ============================================================================

def precip_scan_unrolled(params, zeta, rho, q, vc, mask, nlev=90):
    """
    Precipitation scan with Python-level unrolling.
    
    For small nlev (like 90), this may be faster because
    IREE can see all operations and optimize globally.
    """
    prefactor, exponent, offset = params[0], params[1], params[2]
    ncells = q.shape[0]
    
    q_prev = jnp.zeros(ncells, dtype=q.dtype)
    flx_prev = jnp.zeros(ncells, dtype=q.dtype)
    rho_prev = jnp.zeros(ncells, dtype=q.dtype)
    vc_prev = jnp.zeros(ncells, dtype=q.dtype)
    activated = jnp.zeros(ncells, dtype=bool)
    
    q_outs = []
    flx_outs = []
    
    for k in range(nlev):
        q_k = q[:, k]
        vc_k = vc[:, k]
        rho_k = rho[:, k]
        zeta_k = zeta[:, k]
        mask_k = mask[:, k]
        
        activated = activated | mask_k
        
        rho_x = q_k * rho_k
        flx_eff = (rho_x / zeta_k) + 2.0 * flx_prev
        
        fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
        flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)
        
        rhox_prev = (q_prev + q_k) * 0.5 * rho_prev
        vt_active = vc_prev * prefactor * lax.pow(rhox_prev + offset, exponent)
        vt = lax.select(activated, vt_active, jnp.zeros_like(q_k))
        
        q_activated = (zeta_k * (flx_eff - flx_partial)) / ((1.0 + zeta_k * vt) * rho_k)
        flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5
        
        q_result = lax.select(activated, q_activated, q_k)
        flx_result = lax.select(activated, flx_activated, jnp.zeros_like(q_k))
        
        q_outs.append(q_result)
        flx_outs.append(flx_result)
        
        q_prev = q_result
        flx_prev = flx_result
        rho_prev = rho_k
        vc_prev = vc_k
    
    return jnp.stack(q_outs, axis=1), jnp.stack(flx_outs, axis=1)
