# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generated/unrolled precipitation_effects implementation.

This module provides an alternative implementation of precipitation_effects
that uses Python for-loops which get unrolled at trace time, instead of
lax.scan which creates while loops in the HLO.

The unrolled version may have different performance characteristics:
- Larger compiled kernel (more instructions)
- No while loop overhead
- Better instruction-level parallelism within a level
- All operations are statically known at compile time

Usage:
    from muphys_jax.implementations.generated_precip import precipitation_effects_unrolled

    # Use as drop-in replacement for precipitation_effects
    qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx = precipitation_effects_unrolled(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt
    )
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Tuple

from ..core.common import constants as const
from ..core import properties as props
from ..core import thermo
from ..core.definitions import Q, TempState


# Number of vertical levels - must be known at trace time for unrolling
NLEV = 90


# ============================================================================
# Scan body functions (single step)
# ============================================================================

def precip_scan_step_single(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    prefactor: float,
    exponent: float,
    offset: float,
    zeta: jnp.ndarray,
    vc: jnp.ndarray,
    q: jnp.ndarray,
    rho: jnp.ndarray,
    mask: jnp.ndarray
) -> Tuple[Tuple, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Single precipitation scan step - extracted for unrolling.

    Args:
        carry: (q_prev, flx_prev, rho_prev, vc_prev, activated_prev)
        prefactor, exponent, offset: Fall speed parameters
        zeta: dt / (2 * dz) for this level
        vc: Velocity scale factor for this level
        q: Hydrometeor mixing ratio for this level
        rho: Air density for this level
        mask: Activation mask (kmin) for this level

    Returns:
        new_carry, (q_out, flx_out)
    """
    q_prev, flx_prev, rho_prev, vc_prev, activated_prev = carry

    # Update activation mask
    activated = activated_prev | mask

    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * flx_prev

    # Fall speed calculation
    fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
    flx_partial = lax.min(rho_x * vc * fall_speed, flx_eff)

    rhox_prev = (q_prev + q) * 0.5 * rho_prev

    # Terminal velocity - branchless
    vt_active = vc_prev * prefactor * lax.pow(rhox_prev + offset, exponent)
    vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q))

    q_activated = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)
    flx_activated = (q_activated * rho * vt + flx_partial) * 0.5

    # Branchless selection
    q_out = lax.select(activated, q_activated, q)
    flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q))

    new_carry = (q_out, flx_out, rho, vc, activated)
    return new_carry, (q_out, flx_out)


def temperature_scan_step_single(
    carry: Tuple[jnp.ndarray, jnp.ndarray],
    t: jnp.ndarray,
    t_kp1: jnp.ndarray,
    ei_old: jnp.ndarray,
    pr: jnp.ndarray,
    pflx_tot: jnp.ndarray,
    qv: jnp.ndarray,
    qliq: jnp.ndarray,
    qice: jnp.ndarray,
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    dt: float,
    mask: jnp.ndarray
) -> Tuple[Tuple, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Single temperature scan step - extracted for unrolling.

    Args:
        carry: (eflx_prev, activated_prev)
        Per-level inputs for temperature calculation

    Returns:
        new_carry, (t_out, eflx_out)
    """
    eflx_prev, activated_prev = carry

    current_level_activated = activated_prev | mask

    # Energy flux from precipitation
    cvd_t_kp1 = const.cvd * t_kp1
    eflx_new = dt * (
        pr * (const.clw * t - cvd_t_kp1 - const.lvc)
        + pflx_tot * (const.ci * t - cvd_t_kp1 - const.lsc)
    )

    e_int = ei_old + eflx_prev - eflx_new

    # Temperature from internal energy
    qtot = qliq + qice + qv
    rho_dz = rho * dz
    cv = (const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice) * rho_dz
    t_new = (e_int + rho_dz * (qliq * const.lvc + qice * const.lsc)) / cv

    # Branchless selection
    eflx = lax.select(current_level_activated, eflx_new, eflx_prev)
    t_out = lax.select(current_level_activated, t_new, t)

    new_carry = (eflx, current_level_activated)
    return new_carry, (t_out, eflx)


# ============================================================================
# Unrolled precipitation scan for single species
# ============================================================================

def precip_scan_unrolled_single_species(
    prefactor: float,
    exponent: float,
    offset: float,
    zeta: jnp.ndarray,  # (ncells, nlev)
    rho: jnp.ndarray,   # (ncells, nlev)
    q: jnp.ndarray,     # (ncells, nlev)
    vc: jnp.ndarray,    # (ncells, nlev)
    mask: jnp.ndarray,  # (ncells, nlev) - kmin mask
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unrolled precipitation scan for a single hydrometeor species.

    Uses Python for-loop which unrolls at trace time instead of lax.scan.
    """
    ncells = q.shape[0]
    nlev = q.shape[1]

    # Initialize carry
    q_prev = jnp.zeros(ncells)
    flx_prev = jnp.zeros(ncells)
    rho_prev = jnp.zeros(ncells)
    vc_prev = jnp.zeros(ncells)
    activated_prev = jnp.zeros(ncells, dtype=bool)

    # Output lists
    q_out_list = []
    flx_out_list = []

    # Unrolled loop over levels
    for k in range(nlev):
        carry = (q_prev, flx_prev, rho_prev, vc_prev, activated_prev)

        # Get per-level values
        zeta_k = zeta[:, k]
        vc_k = vc[:, k]
        q_k = q[:, k]
        rho_k = rho[:, k]
        mask_k = mask[:, k]

        # Execute step
        new_carry, (q_out_k, flx_out_k) = precip_scan_step_single(
            carry, prefactor, exponent, offset, zeta_k, vc_k, q_k, rho_k, mask_k
        )

        # Update carry
        q_prev, flx_prev, rho_prev, vc_prev, activated_prev = new_carry

        # Collect outputs
        q_out_list.append(q_out_k)
        flx_out_list.append(flx_out_k)

    # Stack outputs: (nlev, ncells) -> transpose to (ncells, nlev)
    q_out = jnp.stack(q_out_list, axis=1)
    flx_out = jnp.stack(flx_out_list, axis=1)

    return q_out, flx_out


def temperature_scan_unrolled(
    t: jnp.ndarray,         # (ncells, nlev)
    t_kp1: jnp.ndarray,     # (ncells, nlev)
    ei_old: jnp.ndarray,    # (ncells, nlev)
    pr: jnp.ndarray,        # (ncells, nlev)
    pflx_tot: jnp.ndarray,  # (ncells, nlev)
    qv: jnp.ndarray,        # (ncells, nlev)
    qliq: jnp.ndarray,      # (ncells, nlev)
    qice: jnp.ndarray,      # (ncells, nlev)
    rho: jnp.ndarray,       # (ncells, nlev)
    dz: jnp.ndarray,        # (ncells, nlev)
    dt: float,
    mask: jnp.ndarray,      # (ncells, nlev)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unrolled temperature scan.

    Uses Python for-loop which unrolls at trace time instead of lax.scan.
    """
    ncells = t.shape[0]
    nlev = t.shape[1]

    # Initialize carry
    eflx_prev = jnp.zeros(ncells)
    activated_prev = jnp.zeros(ncells, dtype=bool)

    # Output lists
    t_out_list = []
    eflx_out_list = []

    # Unrolled loop over levels
    for k in range(nlev):
        carry = (eflx_prev, activated_prev)

        # Get per-level values
        t_k = t[:, k]
        t_kp1_k = t_kp1[:, k]
        ei_old_k = ei_old[:, k]
        pr_k = pr[:, k]
        pflx_tot_k = pflx_tot[:, k]
        qv_k = qv[:, k]
        qliq_k = qliq[:, k]
        qice_k = qice[:, k]
        rho_k = rho[:, k]
        dz_k = dz[:, k]
        mask_k = mask[:, k]

        # Execute step
        new_carry, (t_out_k, eflx_out_k) = temperature_scan_step_single(
            carry, t_k, t_kp1_k, ei_old_k, pr_k, pflx_tot_k,
            qv_k, qliq_k, qice_k, rho_k, dz_k, dt, mask_k
        )

        # Update carry
        eflx_prev, activated_prev = new_carry

        # Collect outputs
        t_out_list.append(t_out_k)
        eflx_out_list.append(eflx_out_k)

    # Stack outputs: (nlev, ncells) -> already (ncells,) per level -> (ncells, nlev)
    t_out = jnp.stack(t_out_list, axis=1)
    eflx_out = jnp.stack(eflx_out_list, axis=1)

    return t_out, eflx_out


# ============================================================================
# Main unrolled precipitation_effects function
# ============================================================================

def precipitation_effects_unrolled(
    last_lev: int,
    kmin_r: jnp.ndarray,
    kmin_i: jnp.ndarray,
    kmin_s: jnp.ndarray,
    kmin_g: jnp.ndarray,
    q_in: Q,
    t: jnp.ndarray,
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, ...]:
    """
    Unrolled version of precipitation_effects.

    Drop-in replacement for precipitation_effects that uses Python for-loops
    which get unrolled at trace time, eliminating while loops in the HLO.

    Args:
        last_lev: Last vertical level index
        kmin_r, kmin_i, kmin_s, kmin_g: Activation masks per species
        q_in: Q dataclass with hydrometeor mixing ratios
        t: Temperature (ncells, nlev)
        rho: Air density (ncells, nlev)
        dz: Layer thickness (ncells, nlev)
        dt: Time step

    Returns:
        qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx
    """
    ncells, nlev = t.shape

    # Store initial state for energy calculation
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = thermo.internal_energy(t, q_in.v, qliq, qice, rho, dz)

    # Compute zeta = dt / (2 * dz)
    zeta = dt / (2.0 * dz)

    # Velocity scale factors
    xrho = jnp.sqrt(const.rho_00 / rho)
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)

    # Fall speed parameters: (prefactor, exponent, offset)
    params_r = (14.58, 0.111, 1.0e-12)
    params_s = (57.80, 0.16666666666666666, 1.0e-12)
    params_i = (1.25, 0.160, 1.0e-12)
    params_g = (12.24, 0.217, 1.0e-08)

    # Run unrolled precipitation scans for each species
    qr, pr = precip_scan_unrolled_single_species(
        params_r[0], params_r[1], params_r[2], zeta, rho, q_in.r, vc_r, kmin_r
    )
    qs, ps = precip_scan_unrolled_single_species(
        params_s[0], params_s[1], params_s[2], zeta, rho, q_in.s, vc_s, kmin_s
    )
    qi, pi = precip_scan_unrolled_single_species(
        params_i[0], params_i[1], params_i[2], zeta, rho, q_in.i, vc_i, kmin_i
    )
    qg, pg = precip_scan_unrolled_single_species(
        params_g[0], params_g[1], params_g[2], zeta, rho, q_in.g, vc_g, kmin_g
    )

    # Update for temperature scan
    qliq_new = q_in.c + qr
    qice_new = qs + qi + qg
    pflx_tot = ps + pi + pg

    # Shift temperature for next level
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    t_kp1 = jnp.where(jnp.arange(nlev) < last_lev, t_kp1, t)

    # Combined activation mask
    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g

    # Run unrolled temperature scan
    t_new, eflx = temperature_scan_unrolled(
        t, t_kp1, ei_old, pr, pflx_tot, q_in.v, qliq_new, qice_new, rho, dz, dt, kmin_rsig
    )

    return qr, qs, qi, qg, t_new, pflx_tot + pr, pr, ps, pi, pg, eflx / dt


# ============================================================================
# JIT-compiled entry points
# ============================================================================

@partial(jax.jit, static_argnames=['last_lev'])
def precipitation_effects_unrolled_jit(
    last_lev: int,
    kmin_r: jnp.ndarray,
    kmin_i: jnp.ndarray,
    kmin_s: jnp.ndarray,
    kmin_g: jnp.ndarray,
    qv: jnp.ndarray,
    qc: jnp.ndarray,
    qr: jnp.ndarray,
    qs: jnp.ndarray,
    qi: jnp.ndarray,
    qg: jnp.ndarray,
    t: jnp.ndarray,
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, ...]:
    """JIT-compiled entry point with flattened inputs (no Q dataclass)."""
    q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
    return precipitation_effects_unrolled(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
    )


# ============================================================================
# Graupel with unrolled precipitation_effects
# ============================================================================

def graupel_with_unrolled_precip(last_level, dz, te, p, rho, q, dt, qnc):
    """
    Graupel microphysics using unrolled precipitation_effects.

    Drop-in replacement for graupel() that uses the unrolled version
    of precipitation_effects.
    """
    from .graupel_baseline import q_t_update

    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions (same as original)
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects - UNROLLED VERSION
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects_unrolled(
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


@partial(jax.jit, static_argnames=['last_level'])
def graupel_unrolled_run(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """
    JIT-compiled graupel driver using unrolled precipitation_effects.
    """
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel_with_unrolled_precip(last_level, dz, te, p, rho, q_in, dt, qnc)


# ============================================================================
# Export the HLO
# ============================================================================

def export_unrolled_hlo(output_path: str = "precip_unrolled.hlo"):
    """
    Export the unrolled precipitation_effects to HLO for inspection.

    Usage:
        from muphys_jax.implementations.generated_precip import export_unrolled_hlo
        export_unrolled_hlo("my_unrolled.hlo")
    """
    ncells, nlev = 20480, 90
    last_lev = nlev - 1

    # Create dummy inputs with correct shapes
    kmin_r = jnp.ones((ncells, nlev), dtype=bool)
    kmin_i = jnp.ones((ncells, nlev), dtype=bool)
    kmin_s = jnp.ones((ncells, nlev), dtype=bool)
    kmin_g = jnp.ones((ncells, nlev), dtype=bool)
    qv = jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.01
    qc = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-5
    qr = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-6
    qs = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-6
    qi = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-7
    qg = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-7
    t = jnp.ones((ncells, nlev), dtype=jnp.float64) * 280.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float64) * 100.0
    dt = 30.0

    # Lower to HLO
    lowered = jax.jit(
        lambda *args: precipitation_effects_unrolled_jit(last_lev, *args),
    ).lower(
        kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz, dt
    )

    hlo_text = lowered.as_text()

    with open(output_path, 'w') as f:
        f.write(hlo_text)

    print(f"Exported unrolled HLO to: {output_path}")
    return hlo_text


__all__ = [
    "precipitation_effects_unrolled",
    "precipitation_effects_unrolled_jit",
    "graupel_with_unrolled_precip",
    "graupel_unrolled_run",
    "export_unrolled_hlo",
    "NLEV",
]
