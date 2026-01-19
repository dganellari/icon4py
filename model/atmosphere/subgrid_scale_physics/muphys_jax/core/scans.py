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
- lax.select for branchless conditionals which maps directly to XLA Select primitive (low level)
- Batched precipitation scans via vmap for parallel execution (increase data parallelism)
"""

import jax
import jax.numpy as jnp
from jax import lax

from .common import constants as const
from .definitions import TempState


def precip_scan_step_fast(carry, inputs):
    """Precipitation scan step using plain tuple carry for performance."""
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = carry
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    # Update activation mask
    activated = activated_prev | mask

    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * flx_prev

    # Inlined fall speed - use lax.pow for better fusion: this can fuse into one kernel with lax pow
    fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
    flx_partial = lax.min(rho_x * vc * fall_speed, flx_eff)

    rhox_prev = (q_prev + q) * 0.5 * rho_prev

    # Terminal velocity - branchless with lax.select instead of jnp.where to avoid possible temp buffers
    vt_active = vc_prev * prefactor * lax.pow(rhox_prev + offset, exponent)
    vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q))

    q_activated = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)
    flx_activated = (q_activated * rho * vt + flx_partial) * 0.5

    # Branchless selection
    q_out = lax.select(activated, q_activated, q)
    flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q))

    new_carry = (q_out, flx_out, rho, vc, activated)
    outputs = (q_out, flx_out)

    return new_carry, outputs


def _single_species_scan(params, zeta, rho, q, vc, mask):
    """Single species precipitation scan for use with vmap."""
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
    """Batch 4 precipitation scans via vmap for parallel execution."""
    # Stack inputs for vectorization
    params_stacked = jnp.array(params_list)  # (4, 3)
    q_stacked = jnp.stack(q_list, axis=0)  # (4, ncells, nlev)
    vc_stacked = jnp.stack(vc_list, axis=0)  # (4, ncells, nlev)
    mask_stacked = jnp.stack(mask_list, axis=0)  # (4, ncells, nlev)

    # vmap over the 4 species (axis 0)
    batched_scan = jax.vmap(
        lambda p, q, vc, m: _single_species_scan(p, zeta, rho, q, vc, m), in_axes=(0, 0, 0, 0)
    )

    q_updates, flxs = batched_scan(params_stacked, q_stacked, vc_stacked, mask_stacked)

    # Unstack results
    return [(q_updates[i], flxs[i]) for i in range(4)]


def temperature_scan_step(previous_level, inputs):
    """
    JAX equivalent of GT4Py _temperature_update scan_operator.

    Computes both branches and uses lax.select (branchless) instead of if/else.
    Returns (carry, output) tuple for lax.scan.
    """
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask = inputs

    current_level_activated = previous_level.activated | mask

    # Energy flux from precipitation
    cvd_t_kp1 = const.cvd * t_kp1
    eflx_new = dt * (
        pr * (const.clw * t - cvd_t_kp1 - const.lvc)
        + pflx_tot * (const.ci * t - cvd_t_kp1 - const.lsc)
    )

    e_int = ei_old + previous_level.eflx - eflx_new

    # Inlined T_from_internal_energy_scalar
    qtot = qliq + qice + qv
    rho_dz = rho * dz
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho_dz
    t_new = (e_int + rho_dz * (qliq * const.lvc + qice * const.lsc)) / cv

    # Branchless selection (lax.select instead of if/else)
    eflx = lax.select(current_level_activated, eflx_new, previous_level.eflx)
    t_out = lax.select(current_level_activated, t_new, t)

    result = TempState(t=t_out, eflx=eflx, activated=current_level_activated)
    return result, result
