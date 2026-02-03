# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Scan operators for TRANSPOSED (nlev, ncells) data layout.

These scans work directly with (nlev, ncells) data without any transposes,
for maximum GPU performance with coalesced memory access.
"""

import jax
import jax.numpy as jnp
from jax import lax

from .common import constants as const
from .definitions import TempState


def precip_scan_step_transposed(carry, inputs):
    """Precipitation scan step for transposed layout.

    Identical logic to baseline but expects (ncells,) shaped inputs
    at each level from the scan over the nlev dimension.
    """
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = carry
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    # Update activation mask
    activated = activated_prev | mask

    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * flx_prev

    # Inlined fall speed
    fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
    flx_partial = lax.min(rho_x * vc * fall_speed, flx_eff)

    rhox_prev = (q_prev + q) * 0.5 * rho_prev

    # Terminal velocity
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


def _single_species_scan_transposed(params, zeta, rho, q, vc, mask):
    """Single species precipitation scan for TRANSPOSED (nlev, ncells) layout.

    No transposes - data comes in as (nlev, ncells) and goes out the same way.
    """
    prefactor, exponent, offset = params
    nlev, ncells = q.shape

    # Broadcast parameters to (nlev, ncells)
    prefactor_arr = jnp.broadcast_to(prefactor, (nlev, ncells))
    exponent_arr = jnp.broadcast_to(exponent, (nlev, ncells))
    offset_arr = jnp.broadcast_to(offset, (nlev, ncells))

    # Initial carry - (ncells,) shaped
    init_carry = (
        jnp.zeros(ncells),
        jnp.zeros(ncells),
        jnp.zeros(ncells),
        jnp.zeros(ncells),
        jnp.zeros(ncells, dtype=bool),
    )

    # Inputs are already in (nlev, ncells) - scan over nlev dimension
    # Each element fed to scan step is (ncells,) shaped
    inputs = (prefactor_arr, exponent_arr, offset_arr, zeta, vc, q, rho, mask)

    final_carry, outputs = lax.scan(precip_scan_step_transposed, init_carry, inputs)
    q_out, flx_out = outputs

    # Output is already (nlev, ncells) - no transpose needed
    return q_out, flx_out


def precip_scan_batched_transposed(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Batch 4 precipitation scans via vmap for TRANSPOSED layout.

    All inputs/outputs in (nlev, ncells) layout.
    """
    # Stack inputs for vectorization
    params_stacked = jnp.array(params_list)  # (4, 3)
    q_stacked = jnp.stack(q_list, axis=0)  # (4, nlev, ncells)
    vc_stacked = jnp.stack(vc_list, axis=0)  # (4, nlev, ncells)
    mask_stacked = jnp.stack(mask_list, axis=0)  # (4, nlev, ncells)

    # vmap over the 4 species (axis 0)
    batched_scan = jax.vmap(
        lambda p, q, vc, m: _single_species_scan_transposed(p, zeta, rho, q, vc, m),
        in_axes=(0, 0, 0, 0)
    )

    q_updates, flxs = batched_scan(params_stacked, q_stacked, vc_stacked, mask_stacked)

    # Unstack results - still (nlev, ncells) each
    return [(q_updates[i], flxs[i]) for i in range(4)]


def temperature_scan_step_transposed(previous_level, inputs):
    """Temperature scan step for TRANSPOSED layout.

    Same logic as baseline, operates on (ncells,) data at each level.
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

    # Branchless selection
    eflx = lax.select(current_level_activated, eflx_new, previous_level.eflx)
    t_out = lax.select(current_level_activated, t_new, t)

    result = TempState(t=t_out, eflx=eflx, activated=current_level_activated)
    return result, result


def temperature_update_scan_transposed(t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask):
    """Temperature update scan for TRANSPOSED (nlev, ncells) layout.

    No transposes - all data in (nlev, ncells) layout.
    """
    nlev, ncells = t.shape

    init_state = TempState(
        t=jnp.zeros(ncells),
        eflx=jnp.zeros(ncells),
        activated=jnp.zeros(ncells, dtype=bool)
    )

    # Inputs already in (nlev, ncells) - scan over nlev
    inputs = (
        t,
        t_kp1,
        ei_old,
        pr,
        pflx_tot,
        qv,
        qliq,
        qice,
        rho,
        dz,
        jnp.full((nlev, ncells), dt),
        mask,
    )

    final_state, outputs = lax.scan(temperature_scan_step_transposed, init_state, inputs)

    # Output already (nlev, ncells)
    return TempState(t=outputs.t, eflx=outputs.eflx, activated=outputs.activated)


__all__ = [
    "precip_scan_batched_transposed",
    "temperature_update_scan_transposed",
]
