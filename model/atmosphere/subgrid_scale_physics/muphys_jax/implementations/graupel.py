# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Graupel microphysics implementation.
Complete composition of phase transitions and precipitation.
Original (ncells, nlev) layout with tiled/unrolled scan options.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from ..core import properties as props, thermo
from ..core.common import constants as const
from ..core.common.backend import jit_compile

from ..core.definitions import Q
from ..core.scans import precip_scan_batched, precip_scan_unrolled, precip_scan_tiled
from .graupel_baseline import q_t_update, temperature_update_scan


def precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt, tiled=False, tile_size=4, unrolled=False):
    """
    Apply precipitation sedimentation and temperature effects.

    All inputs are (ncells, nlev). Internally uses (nlev, ncells) for scans.
    """
    ncells, nlev = t.shape

    # Store initial state for energy calculation
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = thermo.internal_energy(t, q_in.v, qliq, qice, rho, dz)

    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)

    # Velocity scale factors
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)

    # Fall speed parameters
    params_list = [
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]

    # Transpose to GPU-optimal layout: (ncells, nlev) -> (nlev, ncells)
    zeta_T = jnp.swapaxes(zeta, 0, 1)
    rho_T = jnp.swapaxes(rho, 0, 1)
    q_list_T = [jnp.swapaxes(q_in.r, 0, 1), jnp.swapaxes(q_in.s, 0, 1),
                jnp.swapaxes(q_in.i, 0, 1), jnp.swapaxes(q_in.g, 0, 1)]
    vc_list_T = [jnp.swapaxes(vc_r, 0, 1), jnp.swapaxes(vc_s, 0, 1),
                 jnp.swapaxes(vc_i, 0, 1), jnp.swapaxes(vc_g, 0, 1)]
    mask_list_T = [jnp.swapaxes(kmin_r, 0, 1), jnp.swapaxes(kmin_s, 0, 1),
                   jnp.swapaxes(kmin_i, 0, 1), jnp.swapaxes(kmin_g, 0, 1)]

    # Run batched precipitation scans
    if unrolled:
        results = precip_scan_unrolled(
            params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T
        )
    elif tiled:
        results = precip_scan_tiled(
            params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T, tile_size=tile_size
        )
    else:
        results = precip_scan_batched(
            params_list, zeta_T, rho_T, q_list_T, vc_list_T, mask_list_T
        )

    # Unpack and transpose back: (nlev, ncells) -> (ncells, nlev)
    (qr_T, pr_T), (qs_T, ps_T), (qi_T, pi_T), (qg_T, pg_T) = results
    qr = jnp.swapaxes(qr_T, 0, 1)
    qs = jnp.swapaxes(qs_T, 0, 1)
    qi = jnp.swapaxes(qi_T, 0, 1)
    qg = jnp.swapaxes(qg_T, 0, 1)
    pr = jnp.swapaxes(pr_T, 0, 1)
    ps = jnp.swapaxes(ps_T, 0, 1)
    pi = jnp.swapaxes(pi_T, 0, 1)
    pg = jnp.swapaxes(pg_T, 0, 1)

    # Temperature update scan
    qliq = q_in.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)
    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    result_t = temperature_update_scan(
        t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
    )
    t_new = result_t.t
    eflx = result_t.eflx

    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt


def graupel(last_level, dz, te, p, rho, q, dt, qnc, use_tiled_scans=False, tile_size=4, use_unrolled=False):
    """
    Top-level graupel microphysics function.
    Original (ncells, nlev) layout implementation.
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects(
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt,
        tiled=use_tiled_scans, tile_size=tile_size, unrolled=use_unrolled
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


# ============================================================================
# JIT-compiled entry points
# ============================================================================


@partial(jax.jit, static_argnames=['use_tiled_scans', 'tile_size', 'use_unrolled'])
def graupel_run(dz, te, p, rho, q_in, dt, qnc, last_level=None, use_tiled_scans=False, tile_size=4, use_unrolled=False):
    """JIT-compiled graupel driver (original ncells x nlev layout)."""
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel(last_level, dz, te, p, rho, q_in, dt, qnc,
                   use_tiled_scans=use_tiled_scans, tile_size=tile_size,
                   use_unrolled=use_unrolled)


__all__ = ["graupel", "graupel_run", "precipitation_effects", "q_t_update"]
