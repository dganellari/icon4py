# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Graupel for IREE ROCm (MI250X/MI300).

(nlev, ncells) layout, Python-unrolled loops, vmap over 4 species.
"""

import jax
import jax.numpy as jnp
from jax import lax

from ..core import properties as props, thermo
from ..core.common import constants as const
from ..core.definitions import Q
from .q_t_update_fused import q_t_update_fused


@jax.jit
def graupel_run_iree_rocm(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """Graupel in (nlev, ncells) layout, Python-unrolled, vmap over species."""
    nlev, ncells = te.shape
    if last_level is None:
        last_level = nlev - 1

    # Phase transitions
    q_updated, t_updated = q_t_update_fused(te, p, rho, q_in, dt, qnc)

    kmin_r = q_updated.r > const.qmin
    kmin_i = q_updated.i > const.qmin
    kmin_s = q_updated.s > const.qmin
    kmin_g = q_updated.g > const.qmin

    # Velocity scale factors
    zeta = dt / (2.0 * dz)
    xrho = jnp.sqrt(const.rho_00 / rho)
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t_updated, q_updated.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)

    # Pre-precipitation internal energy
    qliq_pre = q_updated.c + q_updated.r
    qice_pre = q_updated.s + q_updated.i + q_updated.g
    ei_old = thermo.internal_energy(t_updated, q_updated.v, qliq_pre, qice_pre, rho, dz)

    # Precipitation scan (Python-unrolled, vmap over 4 species)
    q_stacked = jnp.stack(
        [q_updated.r, q_updated.s, q_updated.i, q_updated.g], axis=0
    )  # (4, nlev, ncells)
    vc_stacked = jnp.stack([vc_r, vc_s, vc_i, vc_g], axis=0)
    mask_stacked = jnp.stack([kmin_r, kmin_s, kmin_i, kmin_g], axis=0)
    params_arr = jnp.array(
        [
            [14.58, 0.111, 1.0e-12],
            [57.80, 0.16666666666666666, 1.0e-12],
            [1.25, 0.160, 1.0e-12],
            [12.24, 0.217, 1.0e-08],
        ]
    )  # (4, 3)

    def _precip_single_species(params, q, vc, mask):
        prefactor, exponent, offset = params[0], params[1], params[2]

        q_prev = jnp.zeros(ncells, dtype=te.dtype)
        flx_prev = jnp.zeros(ncells, dtype=te.dtype)
        rho_prev = jnp.zeros(ncells, dtype=te.dtype)
        vc_prev = jnp.zeros(ncells, dtype=te.dtype)
        activated = jnp.zeros(ncells, dtype=bool)

        q_outs = []
        flx_outs = []

        for k in range(nlev):
            q_k = q[k]        # (ncells,) — contiguous
            vc_k = vc[k]
            rho_k = rho[k]    # captured from outer scope
            zeta_k = zeta[k]
            mask_k = mask[k]

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

        return jnp.stack(q_outs, axis=0), jnp.stack(flx_outs, axis=0)  # (nlev, ncells)

    batched_precip = jax.vmap(_precip_single_species, in_axes=(0, 0, 0, 0))
    q_out_all, flx_out_all = batched_precip(params_arr, q_stacked, vc_stacked, mask_stacked)
    # q_out_all, flx_out_all: (4, nlev, ncells)

    qr, pr = q_out_all[0], flx_out_all[0]
    qs, ps = q_out_all[1], flx_out_all[1]
    qi, pi = q_out_all[2], flx_out_all[2]
    qg, pg = q_out_all[3], flx_out_all[3]

    # Temperature update (Python-unrolled)
    qliq = q_updated.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg

    t_kp1 = jnp.concatenate([t_updated[1:, :], t_updated[-1:, :]], axis=0)
    t_kp1 = jnp.where(jnp.arange(nlev)[:, None] < last_level, t_kp1, t_updated)

    kmin_any = kmin_r | kmin_s | kmin_i | kmin_g

    eflx_prev = jnp.zeros(ncells, dtype=te.dtype)
    activated_t = jnp.zeros(ncells, dtype=bool)
    t_outs = []
    eflx_outs = []

    for k in range(nlev):
        t_k = t_updated[k]
        t_kp1_k = t_kp1[k]
        ei_old_k = ei_old[k]
        pr_k = pr[k]
        pflx_k = pflx_tot[k]
        qv_k = q_updated.v[k]
        qliq_k = qliq[k]
        qice_k = qice[k]
        rho_k = rho[k]
        dz_k = dz[k]
        mask_k = kmin_any[k]

        activated_t = activated_t | mask_k

        cvd_t_kp1 = const.cvd * t_kp1_k
        eflx_new = dt * (
            pr_k * (const.clw * t_k - cvd_t_kp1 - const.lvc)
            + pflx_k * (const.ci * t_k - cvd_t_kp1 - const.lsc)
        )

        e_int = ei_old_k + eflx_prev - eflx_new
        qtot = qliq_k + qice_k + qv_k
        rho_dz = rho_k * dz_k
        cv = (
            const.cvd * (1.0 - qtot)
            + const.cvv * qv_k
            + const.clw * qliq_k
            + const.ci * qice_k
        ) * rho_dz
        t_new = (e_int + rho_dz * (qliq_k * const.lvc + qice_k * const.lsc)) / cv

        eflx_result = lax.select(activated_t, eflx_new, eflx_prev)
        t_result = lax.select(activated_t, t_new, t_k)

        t_outs.append(t_result)
        eflx_outs.append(eflx_result)

        eflx_prev = eflx_result

    t_out = jnp.stack(t_outs, axis=0)    # (nlev, ncells)
    eflx = jnp.stack(eflx_outs, axis=0)  # (nlev, ncells)

    q_out = Q(v=q_updated.v, c=q_updated.c, r=qr, s=qs, i=qi, g=qg)
    return t_out, q_out, pflx_tot + pr, pr, ps, pi, pg, eflx / dt


# Alias for compatibility
graupel_iree_rocm = graupel_run_iree_rocm
