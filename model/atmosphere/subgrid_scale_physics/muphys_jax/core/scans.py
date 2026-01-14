# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Scan operators for vertical physics in muphys microphysics.
"""

import jax
import jax.numpy as jnp
from jax import lax

from .definitions import PrecipState, TempState
from .common import constants as const


def precip_scan_step(previous_level, inputs):
    """
    Single step of precipitation scan operator.

    This is the scan function that processes one vertical level at a time.

    Args:
        previous_level: PrecipState from the level above (or initial state)
        inputs: Tuple of (prefactor, exponent, offset, zeta, vc, q, rho, mask)

    Returns:
        current_level: PrecipState for this level
        current_level: PrecipState (output to accumulate)
    """
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    # Update activation mask (activated if any previous level or current level is masked)
    current_level_activated = previous_level.activated | mask

    # Calculate precipitation flux with terminal velocity
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * previous_level.flx

    # Inlined fall speed calculation
    flx_partial = jnp.minimum(
        rho_x * vc * prefactor * jnp.power(rho_x + offset, exponent),
        flx_eff
    )

    # Density-weighted specific mass from previous level
    rhox_prev = (previous_level.q_update + q) * 0.5 * previous_level.rho

    # Terminal velocity (depends on previous level activation)
    vt = jnp.where(
        previous_level.activated,
        previous_level.vc * prefactor * jnp.power(rhox_prev + offset, exponent),
        0.0
    )

    # Update specific mass and flux (depends on current level activation)
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
        activated=current_level_activated,
    )

    return current_level, current_level


def precip_scan(prefactor, exponent, offset, zeta, vc, q, rho, mask):
    """
    Precipitation scan operator for vertical sedimentation.

    Scans vertically (forward, top to bottom) to compute precipitation sedimentation.

    Args:
        prefactor: Fall speed prefactor - shape (ncells, nlev)
        exponent: Fall speed exponent - shape (ncells, nlev)
        offset: Fall speed offset - shape (ncells, nlev)
        zeta: dt/(2*dz) time/space ratio - shape (ncells, nlev)
        vc: Fall speed correction - shape (ncells, nlev)
        q: Specific mass of hydrometeor - shape (ncells, nlev)
        rho: Air density - shape (ncells, nlev)
        mask: Activation mask - shape (ncells, nlev)

    Returns:
        PrecipState with q_update, flx, rho, vc, activated - each shape (ncells, nlev)
    """
    ncells, nlev = q.shape

    # Initial state (all zeros/False)
    init_state = PrecipState(
        q_update=jnp.zeros(ncells),
        flx=jnp.zeros(ncells),
        rho=jnp.zeros(ncells),
        vc=jnp.zeros(ncells),
        activated=jnp.zeros(ncells, dtype=bool),
    )

    # Transpose to (nlev, ncells) for scanning over axis 0
    inputs = (
        prefactor.T,  # (nlev, ncells)
        exponent.T,
        offset.T,
        zeta.T,
        vc.T,
        q.T,
        rho.T,
        mask.T,
    )

    # Scan over vertical levels (axis 0)
    final_state, outputs = lax.scan(precip_scan_step, init_state, inputs)

    # Transpose outputs back to (ncells, nlev)
    result = PrecipState(
        q_update=outputs.q_update.T,
        flx=outputs.flx.T,
        rho=outputs.rho.T,
        vc=outputs.vc.T,
        activated=outputs.activated.T,
    )

    return result


def temperature_scan_step(previous_level, inputs):
    """
    Single step of temperature update scan operator.

    Computes temperature changes from latent heat and precipitation flux.

    Args:
        previous_level: TempState from the level above (or initial state)
        inputs: Tuple of (t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask)

    Returns:
        current_level: TempState for this level
        current_level: TempState (output to accumlate)
    """
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask = inputs

    # Activation mask (cumulative OR from top to bottom)
    current_level_activated = previous_level.activated | mask

    # Energy flux from precipitation (always compute)
    eflx_new = dt * (
        pr * (const.clw * t - const.cvd * t_kp1 - const.lvc) +
        pflx_tot * (const.ci * t - const.cvd * t_kp1 - const.lsc)
    )

    # Internal energy update
    e_int = ei_old + previous_level.eflx - eflx_new

    # Calculate temperature from internal energy
    qtot = qliq + qice + qv  # total water specific mass
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho * dz
    t_new = (e_int + rho * dz * (qliq * const.lvc + qice * const.lsc)) / cv

    # Conditional selection based on activation mask
    eflx = jnp.where(current_level_activated, eflx_new, previous_level.eflx)
    t_out = jnp.where(current_level_activated, t_new, t)

    current_level = TempState(t=t_out, eflx=eflx, activated=current_level_activated)

    return current_level, current_level


def temperature_scan(zeta, lheat, q_update, rho):
    """
    Temperature update scan operator.

    Scans vertically (forward, top to bottom) to compute temperature changes
    from latent heat release due to phase transitions.

    Args:
        zeta: dt/(2*dz) time/space ratio - shape (ncells, nlev)
        lheat: Latent heat coefficient - shape (ncells, nlev)
        q_update: Specific mass change from precipitation - shape (ncells, nlev)
        rho: Air density - shape (ncells, nlev)

    Returns:
        TempState with te_update and flx - each shape (ncells, nlev)
    """
    ncells, nlev = q_update.shape

    # Initial state (all zeros)
    init_state = TempState(
        te_update=jnp.zeros(ncells),
        flx=jnp.zeros(ncells),
    )

    # Transpose to (nlev, ncells) for scanning over axis 0
    inputs = (
        zeta.T,       # (nlev, ncells)
        lheat.T,
        q_update.T,
        rho.T,
    )

    # Scan over vertical levels (axis 0)
    final_state, outputs = lax.scan(temperature_scan_step, init_state, inputs)

    # Transpose outputs back to (ncells, nlev)
    result = TempState(
        te_update=outputs.te_update.T,
        flx=outputs.flx.T,
    )

    return result
