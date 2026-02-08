# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Graupel for the IREE backend.

Uses split JIT boundaries (2 stages) to stay within IREE's memory allocation limits,
Python-level unrolled scans, and vmap for species batching.
"""

import jax
import jax.numpy as jnp
from jax import lax

from ..core.common import constants as const
from ..core.definitions import Q, TempState
from ..core import properties as props
from ..core import thermo
from ..core.scans import precip_scan_batched  # Keep vmap version - sequential was slower
from .graupel_baseline import q_t_update


# ============================================================================
# Python-unrolled scans - IREE sees all 90 levels at compile time
# ============================================================================

def _precip_unrolled_single(prefactor, exponent, offset, zeta, rho, q, vc, mask):
    """Single species precipitation with Python-level unrolling (90 iterations)."""
    ncells, nlev = q.shape
    
    q_prev = jnp.zeros(ncells, dtype=q.dtype)
    flx_prev = jnp.zeros(ncells, dtype=q.dtype)
    rho_prev = jnp.zeros(ncells, dtype=q.dtype)
    vc_prev = jnp.zeros(ncells, dtype=q.dtype)
    activated = jnp.zeros(ncells, dtype=bool)
    
    q_outs = []
    flx_outs = []
    
    for k in range(nlev):
        q_k = q[:, k]
        vc_k = vc[:, k]
        rho_k = rho[:, k]
        zeta_k = zeta[:, k]
        mask_k = mask[:, k]
        
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
    
    return jnp.stack(q_outs, axis=1), jnp.stack(flx_outs, axis=1)


def _precip_unrolled_batched(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Run 4 unrolled precipitation scans via vmap."""
    params_arr = jnp.array(params_list)  # (4, 3)
    q_stacked = jnp.stack(q_list, axis=0)  # (4, ncells, nlev)
    vc_stacked = jnp.stack(vc_list, axis=0)
    mask_stacked = jnp.stack(mask_list, axis=0)
    
    def single_scan(params, q, vc, mask):
        return _precip_unrolled_single(params[0], params[1], params[2], zeta, rho, q, vc, mask)
    
    batched = jax.vmap(single_scan, in_axes=(0, 0, 0, 0))
    q_out, flx_out = batched(params_arr, q_stacked, vc_stacked, mask_stacked)
    
    return [(q_out[i], flx_out[i]) for i in range(4)]


def _temp_unrolled(t, t_kp1, ei_old, pr, pflx_tot, qv, qliq, qice, rho, dz, dt, mask):
    """Temperature correction with Python-level unrolling (90 iterations)."""
    ncells, nlev = t.shape
    
    eflx_prev = jnp.zeros(ncells, dtype=t.dtype)
    activated = jnp.zeros(ncells, dtype=bool)
    
    t_outs = []
    eflx_outs = []
    
    for k in range(nlev):
        t_k = t[:, k]
        t_kp1_k = t_kp1[:, k]
        ei_old_k = ei_old[:, k]
        pr_k = pr[:, k]
        pflx_k = pflx_tot[:, k]
        qv_k = qv[:, k]
        qliq_k = qliq[:, k]
        qice_k = qice[:, k]
        rho_k = rho[:, k]
        dz_k = dz[:, k]
        mask_k = mask[:, k]
        
        activated = activated | mask_k
        
        cvd_t_kp1 = const.cvd * t_kp1_k
        eflx_new = dt * (
            pr_k * (const.clw * t_k - cvd_t_kp1 - const.lvc)
            + pflx_k * (const.ci * t_k - cvd_t_kp1 - const.lsc)
        )
        
        e_int = ei_old_k + eflx_prev - eflx_new
        
        qtot = qliq_k + qice_k + qv_k
        rho_dz = rho_k * dz_k
        cv = (const.cvd * (1.0 - qtot) + const.cvv * qv_k + const.clw * qliq_k + const.ci * qice_k) * rho_dz
        t_new = (e_int + rho_dz * (qliq_k * const.lvc + qice_k * const.lsc)) / cv
        
        eflx_result = lax.select(activated, eflx_new, eflx_prev)
        t_result = lax.select(activated, t_new, t_k)
        
        t_outs.append(t_result)
        eflx_outs.append(eflx_result)
        
        eflx_prev = eflx_result
    
    return jnp.stack(t_outs, axis=1), jnp.stack(eflx_outs, axis=1)


# ============================================================================
# Main JIT-compiled stages
# ============================================================================

@jax.jit
def _iree_step1_phase_and_precip(t, p, rho, dz, q, dt, qnc):
    """
    Combined step 1+2: Phase transitions + Precipitation.
    
    Uses Python-unrolled precipitation scans.
    """
    # Phase transitions
    q_mid, t_mid = q_t_update(t, p, rho, q, dt, qnc)
    
    # Precipitation
    xrho = jnp.sqrt(const.rho_00 / rho)
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t_mid, q_mid.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)
    
    kmin_r = q_mid.r > const.qmin
    kmin_s = q_mid.s > const.qmin
    kmin_i = q_mid.i > const.qmin
    kmin_g = q_mid.g > const.qmin
    
    zeta = dt / (2.0 * dz)
    
    params_list = [
        (14.58, 0.111, 1.0e-12),
        (57.80, 0.16666666666666666, 1.0e-12),
        (1.25, 0.160, 1.0e-12),
        (12.24, 0.217, 1.0e-08),
    ]
    
    # Python-unrolled precipitation with vmap batching
    results = _precip_unrolled_batched(
        params_list, zeta, rho,
        [q_mid.r, q_mid.s, q_mid.i, q_mid.g],
        [vc_r, vc_s, vc_i, vc_g],
        [kmin_r, kmin_s, kmin_i, kmin_g]
    )
    
    (qr, pr), (qs, ps), (qi, pi), (qg, pg) = results
    kmin_any = kmin_r | kmin_s | kmin_i | kmin_g
    
    return t_mid, q_mid, qr, qs, qi, qg, pr, ps, pi, pg, kmin_any


@jax.jit
def _iree_step2_temperature(t_mid, q_mid, qr, qs, qi, qg, pr, ps, pi, pg, kmin_any, rho, dz, dt):
    """
    Step 3: Temperature correction using Python-unrolled scan.
    """
    qliq = q_mid.c + qr
    qice = qs + qi + qg
    pflx_tot = ps + pi + pg
    
    ei_old = thermo.internal_energy(t_mid, q_mid.v, qliq, qice, rho, dz)
    t_kp1 = jnp.concatenate([t_mid[:, 1:], t_mid[:, -1:]], axis=1)
    
    # Python-unrolled temperature scan
    t_out, eflx = _temp_unrolled(
        t_mid, t_kp1, ei_old, pr, pflx_tot,
        q_mid.v, qliq, qice, rho, dz, dt, kmin_any
    )
    
    return t_out, eflx / dt, pr, pflx_tot + pr


def graupel_run_iree(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """
    IREE-optimized graupel driver with 2-stage split JIT.
    
    Uses Python-unrolled scans (90 iterations visible at compile time).
    """
    # Stage 1: Phase transitions + Precipitation
    t_mid, q_mid, qr, qs, qi, qg, pr, ps, pi, pg, kmin_any = _iree_step1_phase_and_precip(
        te, p, rho, dz, q_in, dt, qnc
    )
    
    # Stage 2: Temperature correction
    t_out, eflx, prr_tot, pflx_tot = _iree_step2_temperature(
        t_mid, q_mid, qr, qs, qi, qg, pr, ps, pi, pg, kmin_any, rho, dz, dt
    )
    
    q_out = Q(v=q_mid.v, c=q_mid.c, r=qr, s=qs, i=qi, g=qg)
    return t_out, q_out, pflx_tot, pr, ps, pi, pg, eflx


# Alias
graupel_iree = graupel_run_iree