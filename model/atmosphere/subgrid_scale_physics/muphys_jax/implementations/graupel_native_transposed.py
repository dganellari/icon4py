# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Graupel microphysics in (nlev, ncells) layout.

All inputs/outputs are (nlev, ncells). No transposes during computation.
Data must be pre-transposed before calling.
"""

import jax.numpy as jnp

from ..core import properties as props, thermo
from ..core.common import constants as const
from ..core.common.backend import jit_compile
from ..core.definitions import Q
from ..core.scans import (
    precip_scan_batched_transposed,
    temperature_update_scan_transposed,
)

from .q_t_update_fused import q_t_update_fused
from .graupel_baseline import q_t_update as q_t_update_native


from ..core.optimized_precip import (
    optimized_precip_transposed_p,
    is_optimized_enabled,
)
from ..core.optimized_graupel import (
    optimized_graupel_p,
    is_graupel_optimized_enabled,
)


def precipitation_effects_native_transposed(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """Precipitation sedimentation and temperature effects. (nlev, ncells) layout."""
    if is_optimized_enabled():
        return optimized_precip_transposed_p.bind(
            kmin_r, kmin_i, kmin_s, kmin_g,
            q_in.v, q_in.c, q_in.r, q_in.s, q_in.i, q_in.g,
            t, rho, dz,
            last_lev=last_lev,
            dt=dt
        )

    return _precipitation_effects_native_transposed_jax(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
    )


def _precipitation_effects_native_transposed_jax(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """JAX fallback for precipitation_effects in (nlev, ncells) layout."""
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

    # Batched precipitation scans in (nlev, ncells) layout
    results = precip_scan_batched_transposed(
        params_list,
        zeta,
        rho,
        [q_in.r, q_in.s, q_in.i, q_in.g],
        [vc_r, vc_s, vc_i, vc_g],
        [kmin_r, kmin_s, kmin_i, kmin_g],
    )

    # Unpack results: rain, snow, ice, graupel (still in nlev, ncells layout)
    (qr, pr), (qs, ps), (qi, pi), (qg, pg) = results

    # Update for temperature scan
    qliq = q_in.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg

    # Shift temperature for next level
    nlev, ncells = t.shape
    t_kp1 = jnp.concatenate([t[1:, :], t[-1:, :]], axis=0)
    t_kp1 = jnp.where(jnp.arange(nlev)[:, None] < last_lev, t_kp1, t)

    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    # Temperature update scan
    result_t = temperature_update_scan_transposed(
        t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
    )
    t_new = result_t.t
    eflx = result_t.eflx

    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt


def graupel_native_transposed(last_level, dz, te, p, rho, q, dt, qnc):
    """
    Top-level graupel in (nlev, ncells) layout.

    Uses injected HLO when configured, otherwise falls back to JAX scans.
    See optimized_graupel.py and optimized_precip.py for HLO configuration.
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Full-graupel HLO path
    if is_graupel_optimized_enabled():
        results = optimized_graupel_p.bind(
            kmin_r, kmin_i, kmin_s, kmin_g,
            te, p, rho, dz,
            q.v, q.c, q.r, q.s, q.i, q.g,
            dt=dt
        )
        # results: t_final, qv, qc, qr, qs, qi, qg, pflx, pr, ps, pi, pg, eflx
        t_final, qv, qc, qr, qs, qi, qg, pflx, pr, ps, pi, pg, eflx = results
        return (
            t_final,
            Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg),
            pflx,
            pr,
            ps,
            pi,
            pg,
            eflx,
        )

    # Phase transitions (fused version for fewer kernel launches)
    q_updated, t_updated = q_t_update_fused(te, p, rho, q, dt, qnc)

    # Precipitation sedimentation
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects_native_transposed(
        last_level, kmin_r, kmin_i, kmin_s, kmin_g, q_updated, t_updated, rho, dz, dt
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


@jit_compile
def graupel_run_native_transposed(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """JIT-compiled entry point. All arrays must be (nlev, ncells)."""
    if last_level is None:
        last_level = te.shape[0] - 1  # nlev is first dimension

    return graupel_native_transposed(last_level, dz, te, p, rho, q_in, dt, qnc)


__all__ = [
    "graupel_native_transposed",
    "graupel_run_native_transposed",
    "precipitation_effects_native_transposed",
    "q_t_update_native",
]
