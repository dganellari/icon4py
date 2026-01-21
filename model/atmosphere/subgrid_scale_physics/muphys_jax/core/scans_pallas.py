# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pallas GPU kernels for vertical scans in graupel microphysics.

NOTE: Pallas API is experimental and version-dependent.
This module provides a fallback using vmap + lax.fori_loop which
achieves similar performance characteristics by keeping carry in registers.
"""

import jax
import jax.numpy as jnp
from jax import lax, vmap
from functools import partial

# Check Pallas availability
try:
    from jax.experimental import pallas as pl
    PALLAS_AVAILABLE = True
except ImportError:
    PALLAS_AVAILABLE = False


def _single_cell_scan(params, zeta_col, rho_col, q_col, vc_col, mask_col):
    """Process single cell's vertical column. Carry stays in registers.

    This function processes one horizontal cell through all vertical levels.
    When vmapped across cells, each cell's carry state stays in GPU registers.
    """
    prefactor, exponent, offset = params
    nlev = q_col.shape[0]
    dtype = q_col.dtype

    def body_fn(k, state):
        q_out, flx_out, q_prev, flx_prev, rhox_prev, activated_prev = state

        zeta_k = zeta_col[k]
        vc_k = vc_col[k]
        q_k = q_col[k]
        rho_k = rho_col[k]
        mask_k = mask_col[k]

        activated = activated_prev | mask_k
        rho_x = q_k * rho_k
        flx_eff = (rho_x / zeta_k) + 2.0 * flx_prev

        fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
        flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)

        vt_active = vc_k * prefactor * lax.pow(rhox_prev + offset, exponent)
        vt = lax.select(activated_prev, vt_active, 0.0)

        q_activated = (zeta_k * (flx_eff - flx_partial)) / ((1.0 + zeta_k * vt) * rho_k)
        flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5

        q_k_out = lax.select(activated, q_activated, q_k)
        flx_k_out = lax.select(activated, flx_activated, 0.0)

        q_out = q_out.at[k].set(q_k_out)
        flx_out = flx_out.at[k].set(flx_k_out)

        return (q_out, flx_out, q_k_out, flx_k_out, q_k_out * rho_k, activated)

    init_state = (
        jnp.zeros(nlev, dtype=dtype),  # q_out
        jnp.zeros(nlev, dtype=dtype),  # flx_out
        jnp.zeros((), dtype=dtype),     # q_prev (scalar)
        jnp.zeros((), dtype=dtype),     # flx_prev (scalar)
        jnp.zeros((), dtype=dtype),     # rhox_prev (scalar)
        jnp.zeros((), dtype=bool),      # activated_prev (scalar)
    )

    final_state = lax.fori_loop(0, nlev, body_fn, init_state)
    return final_state[0], final_state[1]


def _single_species_scan_vmap(params, zeta, rho, q, vc, mask):
    """Single species scan using vmap over cells.

    Each cell processes its vertical column independently.
    Carry state (4 scalars per cell) stays in registers during the loop.
    This is structurally similar to a Pallas kernel with one thread per cell.

    Input shape: (nlev, ncells)
    """
    prefactor, exponent, offset = params
    nlev, ncells = q.shape

    # Transpose to (ncells, nlev) for vmap over cells
    zeta_T = jnp.swapaxes(zeta, 0, 1)
    rho_T = jnp.swapaxes(rho, 0, 1)
    q_T = jnp.swapaxes(q, 0, 1)
    vc_T = jnp.swapaxes(vc, 0, 1)
    mask_T = jnp.swapaxes(mask, 0, 1)

    # vmap over cells - each cell's scan runs independently
    scan_cell = partial(_single_cell_scan, params)
    q_out_T, flx_out_T = vmap(scan_cell)(zeta_T, rho_T, q_T, vc_T, mask_T)

    # Transpose back to (nlev, ncells)
    q_out = jnp.swapaxes(q_out_T, 0, 1)
    flx_out = jnp.swapaxes(flx_out_T, 0, 1)

    return q_out, flx_out


def precip_scan_pallas(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Process 4 precipitation scans using vmap-based parallel cell processing.

    This achieves similar benefits to Pallas:
    - Each cell processes independently (parallelism across cells)
    - Carry state is scalar per cell (fits in registers)
    - No inter-cell communication during vertical scan

    The key difference from precip_scan_batched is that here we vmap
    over CELLS (horizontal), not over LEVELS (vertical). This means
    the sequential dependency is within each cell, not across cells.
    """
    results = []
    for i in range(4):
        q_out, flx_out = _single_species_scan_vmap(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i]
        )
        results.append((q_out, flx_out))
    return results
