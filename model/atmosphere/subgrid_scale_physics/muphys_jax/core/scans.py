# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Scan operators for vertical physics in muphys microphysics.

All scan variants consolidated here. One shared precip_scan_step with
layout-specific wrappers (row-major vs column-major) and batching strategies
(vmap, sequential, tiled, unrolled).

Layout handling:
- Row-major (ncells, nlev): baseline functions — transpose internally before scanning
- Column-major (nlev, ncells): transposed functions — scan directly, no transposes
"""

import jax
import jax.numpy as jnp
from jax import lax

from .common import constants as const
from .definitions import TempState


# ============================================================================
# Core scan steps (shared by all layout/batching variants)
# ============================================================================


def precip_scan_step(carry, inputs):
    """Precipitation scan step — original Fortran formula with 5-element carry.

    Carry: (q_prev, flx_prev, rho_prev, vc_prev, activated_prev)
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


def temperature_scan_step(previous_level, inputs):
    """Temperature update scan step using TempState carry.

    JAX equivalent of GT4Py _temperature_update scan_operator.
    Computes both branches and uses lax.select (branchless).
    """
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask = inputs

    current_level_activated = previous_level.activated | mask

    cvd_t_kp1 = const.cvd * t_kp1
    eflx_new = dt * (
        pr * (const.clw * t - cvd_t_kp1 - const.lvc)
        + pflx_tot * (const.ci * t - cvd_t_kp1 - const.lsc)
    )

    e_int = ei_old + previous_level.eflx - eflx_new

    qtot = qliq + qice + qv
    rho_dz = rho * dz
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho_dz
    t_new = (e_int + rho_dz * (qliq * const.lvc + qice * const.lsc)) / cv

    eflx = lax.select(current_level_activated, eflx_new, previous_level.eflx)
    t_out = lax.select(current_level_activated, t_new, t)

    result = TempState(t=t_out, eflx=eflx, activated=current_level_activated)
    return result, result


# ============================================================================
# Helpers
# ============================================================================


def _init_carry_5(ncells, dtype):
    """Initial carry for 5-element scan step."""
    return (
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=bool),
    )


def _init_carry_4(ncells, dtype):
    """Initial carry for 4-element scan step (used by tiled/unrolled variants)."""
    return (
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=bool),
    )


def _broadcast_params(prefactor, exponent, offset, nlev, ncells):
    """Broadcast scalar params to (nlev, ncells)."""
    return (
        jnp.broadcast_to(prefactor, (nlev, ncells)),
        jnp.broadcast_to(exponent, (nlev, ncells)),
        jnp.broadcast_to(offset, (nlev, ncells)),
    )


# ============================================================================
# Single-species scan runners (layout-specific entry/exit)
# ============================================================================


def _single_species_scan(params, zeta, rho, q, vc, mask):
    """Single species precipitation scan — ROW-MAJOR (ncells, nlev).

    Transposes to (nlev, ncells) for scan, transposes back on output.
    """
    prefactor, exponent, offset = params
    ncells, nlev = q.shape

    prefactor_arr, exponent_arr, offset_arr = _broadcast_params(
        prefactor, exponent, offset, nlev, ncells
    )
    init_carry = _init_carry_5(ncells, q.dtype)

    inputs = (prefactor_arr, exponent_arr, offset_arr, zeta.T, vc.T, q.T, rho.T, mask.T)
    final_carry, outputs = lax.scan(precip_scan_step, init_carry, inputs)
    q_out, flx_out = outputs

    return q_out.T, flx_out.T


def _single_species_scan_transposed(params, zeta, rho, q, vc, mask):
    """Single species precipitation scan — COLUMN-MAJOR (nlev, ncells).

    No transposes needed.
    """
    prefactor, exponent, offset = params
    nlev, ncells = q.shape

    prefactor_arr, exponent_arr, offset_arr = _broadcast_params(
        prefactor, exponent, offset, nlev, ncells
    )
    init_carry = _init_carry_5(ncells, q.dtype)

    inputs = (prefactor_arr, exponent_arr, offset_arr, zeta, vc, q, rho, mask)
    final_carry, outputs = lax.scan(precip_scan_step, init_carry, inputs)
    q_out, flx_out = outputs

    return q_out, flx_out


# ============================================================================
# Batched precipitation scans (4 species)
# ============================================================================


def precip_scan_batched(params_list, zeta, rho, q_list, vc_list, mask_list):
    """4 precipitation scans via vmap — ROW-MAJOR (ncells, nlev) layout."""
    params_stacked = jnp.array(params_list)
    q_stacked = jnp.stack(q_list, axis=0)
    vc_stacked = jnp.stack(vc_list, axis=0)
    mask_stacked = jnp.stack(mask_list, axis=0)

    batched_scan = jax.vmap(
        lambda p, q, vc, m: _single_species_scan(p, zeta, rho, q, vc, m), in_axes=(0, 0, 0, 0)
    )

    q_updates, flxs = batched_scan(params_stacked, q_stacked, vc_stacked, mask_stacked)
    return [(q_updates[i], flxs[i]) for i in range(4)]


def precip_scan_batched_transposed(params_list, zeta, rho, q_list, vc_list, mask_list):
    """4 precipitation scans via vmap — COLUMN-MAJOR (nlev, ncells) layout."""
    params_stacked = jnp.array(params_list)
    q_stacked = jnp.stack(q_list, axis=0)
    vc_stacked = jnp.stack(vc_list, axis=0)
    mask_stacked = jnp.stack(mask_list, axis=0)

    batched_scan = jax.vmap(
        lambda p, q, vc, m: _single_species_scan_transposed(p, zeta, rho, q, vc, m),
        in_axes=(0, 0, 0, 0),
    )

    q_updates, flxs = batched_scan(params_stacked, q_stacked, vc_stacked, mask_stacked)
    return [(q_updates[i], flxs[i]) for i in range(4)]


def precip_scan_sequential(params_list, zeta, rho, q_list, vc_list, mask_list):
    """4 precipitation scans sequentially (no vmap) — ROW-MAJOR (ncells, nlev).

    IREE-optimized: explicit Python loop instead of vmap.
    """
    results = []
    for i in range(4):
        q_out, flx_out = _single_species_scan(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i]
        )
        results.append((q_out, flx_out))
    return results


# ============================================================================
# Temperature update scans
# ============================================================================


def temperature_update_scan_transposed(
    t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask
):
    """Temperature update scan — COLUMN-MAJOR (nlev, ncells) layout."""
    nlev, ncells = t.shape

    init_state = TempState(
        t=jnp.zeros(ncells), eflx=jnp.zeros(ncells), activated=jnp.zeros(ncells, dtype=bool)
    )

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

    final_state, outputs = lax.scan(temperature_scan_step, init_state, inputs)
    return TempState(t=outputs.t, eflx=outputs.eflx, activated=outputs.activated)


# ============================================================================
# IREE-specific: temperature scan with tuple carry (avoids NamedTuple issues)
# ============================================================================


def _temperature_scan_step_iree(carry, inputs):
    """Temperature scan step with simple tuple carry for IREE compatibility."""
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
    """Temperature scan with simple tuple carry — ROW-MAJOR (ncells, nlev).

    Returns TempState for compatibility with caller.
    """
    ncells, nlev = t.shape

    init_carry = (
        jnp.zeros(ncells, dtype=t.dtype),
        jnp.zeros(ncells, dtype=bool),
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
        jnp.broadcast_to(dt, (nlev, ncells)),
        mask.T,
    )

    final_carry, outputs = lax.scan(_temperature_scan_step_iree, init_carry, inputs)
    t_out, eflx = outputs

    return TempState(t=t_out.T, eflx=eflx.T, activated=final_carry[1])


# ============================================================================
# IREE-specific: fori_loop variant
# ============================================================================


def _precip_fori_single_level(k, state_and_arrays):
    """Process single level in fori_loop, accumulating outputs via .at[].set()."""
    state, q_all, vc_all, rho_all, zeta_all, mask_all, params, q_outs, flx_outs = state_and_arrays
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = state
    prefactor, exponent, offset = params

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

    q_outs_new = q_outs.at[:, k].set(q_result)
    flx_outs_new = flx_outs.at[:, k].set(flx_result)

    new_state = (q_result, flx_result, rho_k, vc_k, activated)
    return (new_state, q_all, vc_all, rho_all, zeta_all, mask_all, params, q_outs_new, flx_outs_new)


def precip_scan_fori(params, zeta, rho, q, vc, mask):
    """Single species precipitation scan using fori_loop with .at[].set()."""
    prefactor, exponent, offset = params[0], params[1], params[2]
    ncells, nlev = q.shape

    init_state = _init_carry_5(ncells, q.dtype)

    q_outs = jnp.zeros_like(q)
    flx_outs = jnp.zeros_like(q)

    init_all = (init_state, q, vc, rho, zeta, mask, (prefactor, exponent, offset), q_outs, flx_outs)
    final_all = lax.fori_loop(0, nlev, _precip_fori_single_level, init_all)

    return final_all[7], final_all[8]


# ============================================================================
# Tiled scan (reduces D2D copies) — COLUMN-MAJOR (nlev, ncells)
# ============================================================================


def _single_species_scan_tiled(params, zeta, rho, q, vc, mask, tile_size=4):
    """Single species scan processing multiple levels per iteration.

    Reduces scan iterations from nlev to nlev/tile_size, cutting D2D copies.
    Uses optimized 4-element carry (q, flx, rhox, activated).
    """
    prefactor, exponent, offset = params
    nlev, ncells = q.shape
    dtype = q.dtype

    nlev_padded = ((nlev + tile_size - 1) // tile_size) * tile_size
    pad_size = nlev_padded - nlev

    if pad_size > 0:
        pad_fn = lambda x: jnp.concatenate(
            [x, jnp.zeros((pad_size, ncells), dtype=x.dtype)], axis=0
        )
        zeta = pad_fn(zeta)
        vc = pad_fn(vc)
        q = pad_fn(q)
        rho = pad_fn(rho)
        mask = pad_fn(mask.astype(dtype)).astype(bool)

    n_tiles = nlev_padded // tile_size
    zeta_t = zeta.reshape(n_tiles, tile_size, ncells)
    vc_t = vc.reshape(n_tiles, tile_size, ncells)
    q_t = q.reshape(n_tiles, tile_size, ncells)
    rho_t = rho.reshape(n_tiles, tile_size, ncells)
    mask_t = mask.reshape(n_tiles, tile_size, ncells)

    prefactor_t = jnp.full((n_tiles, tile_size, ncells), prefactor, dtype=dtype)
    exponent_t = jnp.full((n_tiles, tile_size, ncells), exponent, dtype=dtype)
    offset_t = jnp.full((n_tiles, tile_size, ncells), offset, dtype=dtype)

    def tile_step(carry, tile_inputs):
        q_prev, flx_prev, rhox_prev, activated_prev = carry
        pref_tile, exp_tile, off_tile, zeta_tile, vc_tile, q_tile, rho_tile, mask_tile = tile_inputs

        q_out_tile = []
        flx_out_tile = []

        for t in range(tile_size):
            activated = activated_prev | mask_tile[t]
            rho_x = q_tile[t] * rho_tile[t]
            flx_eff = (rho_x / zeta_tile[t]) + 2.0 * flx_prev

            fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
            flx_partial = lax.min(rho_x * vc_tile[t] * fall_speed, flx_eff)

            vt_active = vc_tile[t] * prefactor * lax.pow(rhox_prev + offset, exponent)
            vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q_tile[t]))

            q_activated = (zeta_tile[t] * (flx_eff - flx_partial)) / (
                (1.0 + zeta_tile[t] * vt) * rho_tile[t]
            )
            flx_activated = (q_activated * rho_tile[t] * vt + flx_partial) * 0.5

            q_out_t = lax.select(activated, q_activated, q_tile[t])
            flx_out_t = lax.select(activated, flx_activated, jnp.zeros_like(q_tile[t]))

            q_out_tile.append(q_out_t)
            flx_out_tile.append(flx_out_t)

            q_prev = q_out_t
            flx_prev = flx_out_t
            rhox_prev = q_out_t * rho_tile[t]
            activated_prev = activated

        new_carry = (q_prev, flx_prev, rhox_prev, activated_prev)
        outputs = (jnp.stack(q_out_tile), jnp.stack(flx_out_tile))
        return new_carry, outputs

    init_carry = _init_carry_4(ncells, dtype)

    inputs = (prefactor_t, exponent_t, offset_t, zeta_t, vc_t, q_t, rho_t, mask_t)
    _, (q_out_tiles, flx_out_tiles) = lax.scan(tile_step, init_carry, inputs)

    q_out = q_out_tiles.reshape(nlev_padded, ncells)[:nlev]
    flx_out = flx_out_tiles.reshape(nlev_padded, ncells)[:nlev]

    return q_out, flx_out


def precip_scan_tiled(params_list, zeta, rho, q_list, vc_list, mask_list, tile_size=4):
    """4 precipitation scans with tiling — COLUMN-MAJOR (nlev, ncells)."""
    results = []
    for i in range(4):
        q_out, flx_out = _single_species_scan_tiled(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i], tile_size=tile_size
        )
        results.append((q_out, flx_out))
    return results


# ============================================================================
# Static unrolled scan — COLUMN-MAJOR (nlev, ncells)
# ============================================================================


def _precip_column_static_unroll(params, zeta, rho, q, vc, mask, nlev=90):
    """Single species precipitation with static unroll.

    Python loop unrolled at JAX trace time -> single fused XLA kernel.
    Uses optimized 4-element carry (q, flx, rhox, activated).
    """
    prefactor, exponent, offset = params
    _, ncells = q.shape
    dtype = q.dtype

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

        q_prev = q_k_out
        flx_prev = flx_k_out
        rhox_prev = q_k_out * rho_k
        activated_prev = activated

    q_out = jnp.stack(q_levels, axis=0)
    flx_out = jnp.stack(flx_levels, axis=0)

    return q_out, flx_out


def precip_scan_unrolled(params_list, zeta, rho, q_list, vc_list, mask_list):
    """4 precipitation scans with static unroll — COLUMN-MAJOR (nlev, ncells)."""
    nlev = q_list[0].shape[0]
    results = []
    for i in range(4):
        q_out, flx_out = _precip_column_static_unroll(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i], nlev=nlev
        )
        results.append((q_out, flx_out))
    return results
