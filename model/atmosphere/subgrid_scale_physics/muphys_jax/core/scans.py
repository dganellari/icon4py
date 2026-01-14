# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Scan operators for vertical physics in muphys microphysics.

Optimized for GPU execution with:
- lax.pow for better kernel fusion
- lax.select for branchless conditionals
- Batched precipitation scans via vmap for parallel execution
"""

import jax
import jax.numpy as jnp
from jax import lax

from .definitions import TempState
from .common import constants as const


def precip_scan_step_fast(carry, inputs):
    """
    Optimized precipitation scan step using tuple carry.

    Carry: (q_update, flx, rho_prev, vc_prev, activated) - 5 arrays
    Inputs: (prefactor, exponent, offset, zeta, vc, q, rho, mask)
    """
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = carry
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    # Update activation mask
    activated = activated_prev | mask

    # Calculate precipitation flux with terminal velocity
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * flx_prev

    # Inlined fall speed - use lax.pow for better fusion
    fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
    flx_partial = lax.min(rho_x * vc * fall_speed, flx_eff)

    # Density-weighted specific mass from previous level
    rhox_prev = (q_prev + q) * 0.5 * rho_prev

    # Terminal velocity - branchless with lax.select
    vt_active = vc_prev * prefactor * lax.pow(rhox_prev + offset, exponent)
    vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q))

    # Compute activated values
    q_activated = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)
    flx_activated = (q_activated * rho * vt + flx_partial) * 0.5

    # Branchless selection
    q_out = lax.select(activated, q_activated, q)
    flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q))

    new_carry = (q_out, flx_out, rho, vc, activated)
    outputs = (q_out, flx_out)

    return new_carry, outputs


def _single_species_scan(params, zeta, rho, q, vc, mask):
    """
    Single species precipitation scan for use with vmap.

    Args:
        params: (prefactor, exponent, offset) - scalars
        zeta: dt/(2*dz) - shape (ncells, nlev)
        rho: density - shape (ncells, nlev)
        q: specific mass - shape (ncells, nlev)
        vc: velocity correction - shape (ncells, nlev)
        mask: activation mask - shape (ncells, nlev)

    Returns:
        (q_update, flx) - each shape (ncells, nlev)
    """
    prefactor, exponent, offset = params
    ncells, nlev = q.shape

    # Broadcast parameters
    prefactor_arr = jnp.broadcast_to(prefactor, (nlev, ncells))
    exponent_arr = jnp.broadcast_to(exponent, (nlev, ncells))
    offset_arr = jnp.broadcast_to(offset, (nlev, ncells))

    init_carry = (
        jnp.zeros(ncells),
        jnp.zeros(ncells),
        jnp.zeros(ncells),
        jnp.zeros(ncells),
        jnp.zeros(ncells, dtype=bool),
    )

    inputs = (prefactor_arr, exponent_arr, offset_arr, zeta.T, vc.T, q.T, rho.T, mask.T)

    final_carry, outputs = lax.scan(precip_scan_step_fast, init_carry, inputs)
    q_out, flx_out = outputs

    return q_out.T, flx_out.T


def precip_scan_batched(params_list, zeta, rho, q_list, vc_list, mask_list):
    """
    Batch 4 precipitation scans together for better GPU utilization.

    Uses jax.vmap to run all 4 species (rain, snow, ice, graupel) in parallel.

    Args:
        params_list: List of 4 (prefactor, exponent, offset) tuples
        zeta: Common zeta array - shape (ncells, nlev)
        rho: Common rho array - shape (ncells, nlev)
        q_list: List of 4 q arrays [qr, qs, qi, qg]
        vc_list: List of 4 vc arrays
        mask_list: List of 4 mask arrays

    Returns:
        List of 4 (q_update, flx) tuples
    """
    # Stack inputs for vectorization
    params_stacked = jnp.array(params_list)  # (4, 3)
    q_stacked = jnp.stack(q_list, axis=0)  # (4, ncells, nlev)
    vc_stacked = jnp.stack(vc_list, axis=0)  # (4, ncells, nlev)
    mask_stacked = jnp.stack(mask_list, axis=0)  # (4, ncells, nlev)

    # vmap over the 4 species (axis 0)
    batched_scan = jax.vmap(
        lambda p, q, vc, m: _single_species_scan(p, zeta, rho, q, vc, m),
        in_axes=(0, 0, 0, 0)
    )

    q_updates, flxs = batched_scan(params_stacked, q_stacked, vc_stacked, mask_stacked)

    # Unstack results
    return [(q_updates[i], flxs[i]) for i in range(4)]


def temperature_scan_step(previous_level, inputs):
    """
    Single step of temperature update scan operator.

    Args:
        previous_level: TempState from the level above (or initial state)
        inputs: Tuple of (t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask)

    Returns:
        current_level: TempState for this level
        current_level: TempState (output to accumulate)
    """
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask = inputs

    # Activation mask (cumulative OR from top to bottom)
    current_level_activated = previous_level.activated | mask

    # Energy flux from precipitation
    cvd_t_kp1 = const.cvd * t_kp1
    eflx_new = dt * (
        pr * (const.clw * t - cvd_t_kp1 - const.lvc) +
        pflx_tot * (const.ci * t - cvd_t_kp1 - const.lsc)
    )

    # Internal energy update
    e_int = ei_old + previous_level.eflx - eflx_new

    # Calculate temperature from internal energy
    qtot = qliq + qice + qv
    rho_dz = rho * dz
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho_dz
    t_new = (e_int + rho_dz * (qliq * const.lvc + qice * const.lsc)) / cv

    # Branchless selection
    eflx = lax.select(current_level_activated, eflx_new, previous_level.eflx)
    t_out = lax.select(current_level_activated, t_new, t)

    result = TempState(t=t_out, eflx=eflx, activated=current_level_activated)
    return result, result
