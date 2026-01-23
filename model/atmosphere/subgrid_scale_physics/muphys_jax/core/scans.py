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
    """Precipitation scan step using plain tuple carry for performance.

    Used by the fused scan. Processes a single species at one vertical level.

    OPTIMIZED: Carry only contains (q_prev, flx_prev, rhox_prev, activated_prev)
    instead of (q_prev, flx_prev, rho_prev, vc_prev, activated_prev).
    This reduces D2D memory copies by 40% (2 fewer arrays per iteration).
    """
    q_prev, flx_prev, rhox_prev, activated_prev = carry
    prefactor, exponent, offset, zeta, vc, q, rho, mask = inputs

    # Update activation mask
    activated = activated_prev | mask

    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * flx_prev

    # Inlined fall speed - use lax.pow for better fusion
    fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
    flx_partial = lax.min(rho_x * vc * fall_speed, flx_eff)

    # Terminal velocity - use rhox_prev from carry (already computed in previous iteration)
    vt_active = vc * prefactor * lax.pow(rhox_prev + offset, exponent)
    vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q))

    q_activated = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)
    flx_activated = (q_activated * rho * vt + flx_partial) * 0.5

    # Branchless selection
    q_out = lax.select(activated, q_activated, q)
    flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q))

    # Compute rhox for next iteration (to be used as rhox_prev)
    rhox_next = q_out * rho

    new_carry = (q_out, flx_out, rhox_next, activated)
    outputs = (q_out, flx_out)

    return new_carry, outputs


def _single_species_scan(params, zeta, rho, q, vc, mask):
    """Single species precipitation scan.

    IMPORTANT: All inputs should be in vertical-major format (nlev, ncells) to avoid transposes.
    """
    prefactor, exponent, offset = params
    nlev, ncells = q.shape

    # Broadcast parameters
    prefactor_arr = jnp.broadcast_to(prefactor, (nlev, ncells))
    exponent_arr = jnp.broadcast_to(exponent, (nlev, ncells))
    offset_arr = jnp.broadcast_to(offset, (nlev, ncells))

    # Use same dtype as input arrays to preserve precision
    # OPTIMIZED: Only 4 carry elements instead of 5 (removed rho and vc)
    init_carry = (
        jnp.zeros(ncells, dtype=q.dtype),  # q_prev
        jnp.zeros(ncells, dtype=q.dtype),  # flx_prev
        jnp.zeros(ncells, dtype=q.dtype),  # rhox_prev
        jnp.zeros(ncells, dtype=bool),      # activated_prev
    )

    # No transposes needed - data is already (nlev, ncells)
    inputs = (prefactor_arr, exponent_arr, offset_arr, zeta, vc, q, rho, mask)

    final_carry, outputs = lax.scan(precip_scan_step_fast, init_carry, inputs)
    q_out, flx_out = outputs

    # Return in same format (nlev, ncells) - no transpose
    return q_out, flx_out


def precip_scan_batched(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Process 4 precipitation scans sequentially (no vmap).

    OPTIMIZED: Sequential processing eliminates vmap transpose overhead.
    Trade-off: Loses some parallelism but reduces memory traffic.

    IMPORTANT: All inputs should be in vertical-major format (nlev, ncells) to avoid transposes.
    """
    # Process each species sequentially - no vmap, no stacking, no transpose!
    results = []
    for i in range(4):
        q_out, flx_out = _single_species_scan(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i]
        )
        results.append((q_out, flx_out))

    return results


# ============================================================================
# Tiled scan with buffer donation for reduced D2D copies
# ============================================================================

def _single_species_scan_tiled(params, zeta, rho, q, vc, mask, tile_size=4):
    """Single species scan processing multiple levels per iteration.

    Reduces scan iterations from nlev to nlev/tile_size, cutting D2D copies.
    """
    prefactor, exponent, offset = params
    nlev, ncells = q.shape
    dtype = q.dtype

    # Pad to multiple of tile_size
    nlev_padded = ((nlev + tile_size - 1) // tile_size) * tile_size
    pad_size = nlev_padded - nlev

    if pad_size > 0:
        pad_fn = lambda x: jnp.concatenate([x, jnp.zeros((pad_size, ncells), dtype=x.dtype)], axis=0)
        zeta = pad_fn(zeta)
        vc = pad_fn(vc)
        q = pad_fn(q)
        rho = pad_fn(rho)
        mask = pad_fn(mask.astype(dtype)).astype(bool)

    # Reshape to (n_tiles, tile_size, ncells)
    n_tiles = nlev_padded // tile_size
    zeta_t = zeta.reshape(n_tiles, tile_size, ncells)
    vc_t = vc.reshape(n_tiles, tile_size, ncells)
    q_t = q.reshape(n_tiles, tile_size, ncells)
    rho_t = rho.reshape(n_tiles, tile_size, ncells)
    mask_t = mask.reshape(n_tiles, tile_size, ncells)

    # Broadcast params
    prefactor_t = jnp.full((n_tiles, tile_size, ncells), prefactor, dtype=dtype)
    exponent_t = jnp.full((n_tiles, tile_size, ncells), exponent, dtype=dtype)
    offset_t = jnp.full((n_tiles, tile_size, ncells), offset, dtype=dtype)

    def tile_step(carry, tile_inputs):
        """Process one tile of levels."""
        q_prev, flx_prev, rhox_prev, activated_prev = carry
        pref_tile, exp_tile, off_tile, zeta_tile, vc_tile, q_tile, rho_tile, mask_tile = tile_inputs

        q_out_tile = []
        flx_out_tile = []

        # Process tile_size levels within this step
        for t in range(tile_size):
            activated = activated_prev | mask_tile[t]
            rho_x = q_tile[t] * rho_tile[t]
            flx_eff = (rho_x / zeta_tile[t]) + 2.0 * flx_prev

            fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
            flx_partial = lax.min(rho_x * vc_tile[t] * fall_speed, flx_eff)

            vt_active = vc_tile[t] * prefactor * lax.pow(rhox_prev + offset, exponent)
            vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q_tile[t]))

            q_activated = (zeta_tile[t] * (flx_eff - flx_partial)) / ((1.0 + zeta_tile[t] * vt) * rho_tile[t])
            flx_activated = (q_activated * rho_tile[t] * vt + flx_partial) * 0.5

            q_out_t = lax.select(activated, q_activated, q_tile[t])
            flx_out_t = lax.select(activated, flx_activated, jnp.zeros_like(q_tile[t]))

            q_out_tile.append(q_out_t)
            flx_out_tile.append(flx_out_t)

            # Update carry
            q_prev = q_out_t
            flx_prev = flx_out_t
            rhox_prev = q_out_t * rho_tile[t]
            activated_prev = activated

        new_carry = (q_prev, flx_prev, rhox_prev, activated_prev)
        outputs = (jnp.stack(q_out_tile), jnp.stack(flx_out_tile))
        return new_carry, outputs

    init_carry = (
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=bool),
    )

    inputs = (prefactor_t, exponent_t, offset_t, zeta_t, vc_t, q_t, rho_t, mask_t)
    _, (q_out_tiles, flx_out_tiles) = lax.scan(tile_step, init_carry, inputs)

    # Reshape back and remove padding
    q_out = q_out_tiles.reshape(nlev_padded, ncells)[:nlev]
    flx_out = flx_out_tiles.reshape(nlev_padded, ncells)[:nlev]

    return q_out, flx_out


def precip_scan_tiled(params_list, zeta, rho, q_list, vc_list, mask_list, tile_size=4):
    """Process 4 precipitation scans with tiling.

    Reduces scan iterations by factor of tile_size (e.g., 90 -> 23 with tile_size=4).
    This cuts D2D memory copies proportionally.
    """
    results = []
    for i in range(4):
        q_out, flx_out = _single_species_scan_tiled(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i], tile_size=tile_size
        )
        results.append((q_out, flx_out))
    return results


# ============================================================================
# JIT-compiled version with buffer donation
# ============================================================================

def precip_scan_tiled_donated(params_list, zeta, rho, q_list, vc_list, mask_list, tile_size=4):
    """Tiled precipitation scan with optimized buffer handling.

    This version processes multiple levels per scan iteration (tile_size),
    reducing D2D memory copies by factor of tile_size.

    Buffer donation happens at the caller (graupel_run) level where
    donate_argnums can be specified on the JIT decorator.
    """
    return precip_scan_tiled(params_list, zeta, rho, q_list, vc_list, mask_list, tile_size=tile_size)


def _precip_column_static_unroll(params, zeta, rho, q, vc, mask, nlev=90):
    """Single species precipitation - STATIC UNROLL (true single kernel).

    Python loop unrolled at JAX trace time -> single fused XLA kernel.
    nlev must be known at trace time (default 90).
    """
    prefactor, exponent, offset = params
    _, ncells = q.shape
    dtype = q.dtype

    # Build computation graph by unrolling Python loop
    q_levels = []
    flx_levels = []

    q_prev = jnp.zeros(ncells, dtype=dtype)
    flx_prev = jnp.zeros(ncells, dtype=dtype)
    rhox_prev = jnp.zeros(ncells, dtype=dtype)
    activated_prev = jnp.zeros(ncells, dtype=bool)

    for k in range(nlev):
        zeta_k = zeta[k]
        vc_k = vc[k]
        q_k = q[k]
        rho_k = rho[k]
        mask_k = mask[k]

        activated = activated_prev | mask_k
        rho_x = q_k * rho_k
        flx_eff = (rho_x / zeta_k) + 2.0 * flx_prev

        fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
        flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)

        vt_active = vc_k * prefactor * lax.pow(rhox_prev + offset, exponent)
        vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q_k))

        q_activated = (zeta_k * (flx_eff - flx_partial)) / ((1.0 + zeta_k * vt) * rho_k)
        flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5

        q_k_out = lax.select(activated, q_activated, q_k)
        flx_k_out = lax.select(activated, flx_activated, jnp.zeros_like(q_k))

        q_levels.append(q_k_out)
        flx_levels.append(flx_k_out)

        # Update for next iteration
        q_prev = q_k_out
        flx_prev = flx_k_out
        rhox_prev = q_k_out * rho_k
        activated_prev = activated

    # Stack all levels into output arrays
    q_out = jnp.stack(q_levels, axis=0)
    flx_out = jnp.stack(flx_levels, axis=0)

    return q_out, flx_out


def precip_scan_unrolled(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Process 4 precipitation scans with STATIC UNROLL - true single kernel.

    Python loop unrolled at trace time -> single fused XLA computation.
    Target: Match DaCe performance (~14ms).
    """
    nlev = q_list[0].shape[0]
    results = []
    for i in range(4):
        q_out, flx_out = _precip_column_static_unroll(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i], nlev=nlev
        )
        results.append((q_out, flx_out))
    return results


def precip_scan_step_unified(carry, inputs):
    """Single unified precipitation scan step for all 4 species.

    OPTIMIZED: Process all 4 species (rain, snow, ice, graupel) in a single scan.
    This eliminates vmap overhead while maintaining parallelism within each level.

    Carry structure: (q_prev_0, flx_prev_0, rhox_prev_0, activated_prev_0,
                      q_prev_1, flx_prev_1, rhox_prev_1, activated_prev_1,
                      q_prev_2, flx_prev_2, rhox_prev_2, activated_prev_2,
                      q_prev_3, flx_prev_3, rhox_prev_3, activated_prev_3)
    """
    # Unpack carry for 4 species (16 elements total)
    q_prev_0, flx_prev_0, rhox_prev_0, activated_prev_0 = carry[0:4]
    q_prev_1, flx_prev_1, rhox_prev_1, activated_prev_1 = carry[4:8]
    q_prev_2, flx_prev_2, rhox_prev_2, activated_prev_2 = carry[8:12]
    q_prev_3, flx_prev_3, rhox_prev_3, activated_prev_3 = carry[12:16]

    # Unpack inputs: params for 4 species + shared variables
    (prefactor_0, exponent_0, offset_0, prefactor_1, exponent_1, offset_1,
     prefactor_2, exponent_2, offset_2, prefactor_3, exponent_3, offset_3,
     zeta, vc_0, vc_1, vc_2, vc_3,
     q_0, q_1, q_2, q_3, rho,
     mask_0, mask_1, mask_2, mask_3) = inputs

    # Process species 0
    activated_0 = activated_prev_0 | mask_0
    rho_x_0 = q_0 * rho
    flx_eff_0 = (rho_x_0 / zeta) + 2.0 * flx_prev_0
    fall_speed_0 = prefactor_0 * lax.pow(rho_x_0 + offset_0, exponent_0)
    flx_partial_0 = lax.min(rho_x_0 * vc_0 * fall_speed_0, flx_eff_0)
    vt_active_0 = vc_0 * prefactor_0 * lax.pow(rhox_prev_0 + offset_0, exponent_0)
    vt_0 = lax.select(activated_prev_0, vt_active_0, jnp.zeros_like(q_0))
    q_activated_0 = (zeta * (flx_eff_0 - flx_partial_0)) / ((1.0 + zeta * vt_0) * rho)
    flx_activated_0 = (q_activated_0 * rho * vt_0 + flx_partial_0) * 0.5
    q_out_0 = lax.select(activated_0, q_activated_0, q_0)
    flx_out_0 = lax.select(activated_0, flx_activated_0, jnp.zeros_like(q_0))
    rhox_next_0 = q_out_0 * rho

    # Process species 1
    activated_1 = activated_prev_1 | mask_1
    rho_x_1 = q_1 * rho
    flx_eff_1 = (rho_x_1 / zeta) + 2.0 * flx_prev_1
    fall_speed_1 = prefactor_1 * lax.pow(rho_x_1 + offset_1, exponent_1)
    flx_partial_1 = lax.min(rho_x_1 * vc_1 * fall_speed_1, flx_eff_1)
    vt_active_1 = vc_1 * prefactor_1 * lax.pow(rhox_prev_1 + offset_1, exponent_1)
    vt_1 = lax.select(activated_prev_1, vt_active_1, jnp.zeros_like(q_1))
    q_activated_1 = (zeta * (flx_eff_1 - flx_partial_1)) / ((1.0 + zeta * vt_1) * rho)
    flx_activated_1 = (q_activated_1 * rho * vt_1 + flx_partial_1) * 0.5
    q_out_1 = lax.select(activated_1, q_activated_1, q_1)
    flx_out_1 = lax.select(activated_1, flx_activated_1, jnp.zeros_like(q_1))
    rhox_next_1 = q_out_1 * rho

    # Process species 2
    activated_2 = activated_prev_2 | mask_2
    rho_x_2 = q_2 * rho
    flx_eff_2 = (rho_x_2 / zeta) + 2.0 * flx_prev_2
    fall_speed_2 = prefactor_2 * lax.pow(rho_x_2 + offset_2, exponent_2)
    flx_partial_2 = lax.min(rho_x_2 * vc_2 * fall_speed_2, flx_eff_2)
    vt_active_2 = vc_2 * prefactor_2 * lax.pow(rhox_prev_2 + offset_2, exponent_2)
    vt_2 = lax.select(activated_prev_2, vt_active_2, jnp.zeros_like(q_2))
    q_activated_2 = (zeta * (flx_eff_2 - flx_partial_2)) / ((1.0 + zeta * vt_2) * rho)
    flx_activated_2 = (q_activated_2 * rho * vt_2 + flx_partial_2) * 0.5
    q_out_2 = lax.select(activated_2, q_activated_2, q_2)
    flx_out_2 = lax.select(activated_2, flx_activated_2, jnp.zeros_like(q_2))
    rhox_next_2 = q_out_2 * rho

    # Process species 3
    activated_3 = activated_prev_3 | mask_3
    rho_x_3 = q_3 * rho
    flx_eff_3 = (rho_x_3 / zeta) + 2.0 * flx_prev_3
    fall_speed_3 = prefactor_3 * lax.pow(rho_x_3 + offset_3, exponent_3)
    flx_partial_3 = lax.min(rho_x_3 * vc_3 * fall_speed_3, flx_eff_3)
    vt_active_3 = vc_3 * prefactor_3 * lax.pow(rhox_prev_3 + offset_3, exponent_3)
    vt_3 = lax.select(activated_prev_3, vt_active_3, jnp.zeros_like(q_3))
    q_activated_3 = (zeta * (flx_eff_3 - flx_partial_3)) / ((1.0 + zeta * vt_3) * rho)
    flx_activated_3 = (q_activated_3 * rho * vt_3 + flx_partial_3) * 0.5
    q_out_3 = lax.select(activated_3, q_activated_3, q_3)
    flx_out_3 = lax.select(activated_3, flx_activated_3, jnp.zeros_like(q_3))
    rhox_next_3 = q_out_3 * rho

    # Pack carry and outputs
    new_carry = (
        q_out_0, flx_out_0, rhox_next_0, activated_0,
        q_out_1, flx_out_1, rhox_next_1, activated_1,
        q_out_2, flx_out_2, rhox_next_2, activated_2,
        q_out_3, flx_out_3, rhox_next_3, activated_3,
    )
    outputs = ((q_out_0, flx_out_0), (q_out_1, flx_out_1),
               (q_out_2, flx_out_2), (q_out_3, flx_out_3))

    return new_carry, outputs


def precip_scan_unified(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Single unified precipitation scan for all 4 species.

    OPTIMIZED: Process all 4 species in a single scan (one loop over 90 levels).
    Eliminates vmap overhead and transposes while maintaining level-wise parallelism.

    IMPORTANT: All inputs should be in vertical-major format (nlev, ncells).
    """
    nlev, ncells = q_list[0].shape

    # Extract parameters for all 4 species
    prefactor_0, exponent_0, offset_0 = params_list[0]
    prefactor_1, exponent_1, offset_1 = params_list[1]
    prefactor_2, exponent_2, offset_2 = params_list[2]
    prefactor_3, exponent_3, offset_3 = params_list[3]

    # Broadcast parameters to (nlev, ncells)
    prefactor_0_arr = jnp.broadcast_to(prefactor_0, (nlev, ncells))
    exponent_0_arr = jnp.broadcast_to(exponent_0, (nlev, ncells))
    offset_0_arr = jnp.broadcast_to(offset_0, (nlev, ncells))
    prefactor_1_arr = jnp.broadcast_to(prefactor_1, (nlev, ncells))
    exponent_1_arr = jnp.broadcast_to(exponent_1, (nlev, ncells))
    offset_1_arr = jnp.broadcast_to(offset_1, (nlev, ncells))
    prefactor_2_arr = jnp.broadcast_to(prefactor_2, (nlev, ncells))
    exponent_2_arr = jnp.broadcast_to(exponent_2, (nlev, ncells))
    offset_2_arr = jnp.broadcast_to(offset_2, (nlev, ncells))
    prefactor_3_arr = jnp.broadcast_to(prefactor_3, (nlev, ncells))
    exponent_3_arr = jnp.broadcast_to(exponent_3, (nlev, ncells))
    offset_3_arr = jnp.broadcast_to(offset_3, (nlev, ncells))

    # Initialize carry for all 4 species (16 elements)
    dtype = q_list[0].dtype
    init_carry = (
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=bool),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=bool),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=bool),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype), jnp.zeros(ncells, dtype=bool),
    )

    # Pack inputs: params + shared variables + species data
    inputs = (
        prefactor_0_arr, exponent_0_arr, offset_0_arr,
        prefactor_1_arr, exponent_1_arr, offset_1_arr,
        prefactor_2_arr, exponent_2_arr, offset_2_arr,
        prefactor_3_arr, exponent_3_arr, offset_3_arr,
        zeta, vc_list[0], vc_list[1], vc_list[2], vc_list[3],
        q_list[0], q_list[1], q_list[2], q_list[3], rho,
        mask_list[0], mask_list[1], mask_list[2], mask_list[3]
    )

    # Single scan over 90 vertical levels
    final_carry, outputs = lax.scan(precip_scan_step_unified, init_carry, inputs)

    # Unpack outputs for 4 species
    results = [
        (outputs[0][0], outputs[0][1]),
        (outputs[1][0], outputs[1][1]),
        (outputs[2][0], outputs[2][1]),
        (outputs[3][0], outputs[3][1]),
    ]

    return results


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
