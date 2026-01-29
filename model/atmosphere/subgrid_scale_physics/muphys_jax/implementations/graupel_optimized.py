# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Graupel microphysics with optimized HLO injection support.

This is a drop-in replacement for graupel_baseline that can use
optimized HLO for precipitation_effects when configured.

Usage:
    # Default (no optimization):
    from muphys_jax.implementations.graupel_optimized import graupel_run

    # With optimized HLO injection:
    export MUPHYS_OPTIMIZED_HLO=/path/to/precip_effect.serialized
    export MUPHYS_USE_OPTIMIZED=1
    from muphys_jax.implementations.graupel_optimized import graupel_run

    # Or configure programmatically:
    from muphys_jax.core.optimized_precip import configure_optimized_precip
    configure_optimized_precip(hlo_path="...", use_optimized=True)
"""

from ..core.common import constants as const
from ..core.common.backend import jit_compile
from ..core.definitions import Q

# Import everything from graupel_baseline - no duplication
from .graupel_baseline import (
    q_t_update,
    precipitation_effects as _precipitation_effects_original,
)

# Import the optimized precipitation_effects (auto-configures from env vars)
from ..core.optimized_precip import (
    precipitation_effects_optimized,
    configure_optimized_precip,
    is_optimized_enabled,
)


# ============================================================================
# Precipitation Effects - Uses Optimized Version When Configured
# ============================================================================

def precipitation_effects(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    Apply precipitation sedimentation and temperature effects.

    This function dispatches to either:
    - The optimized HLO version (if MUPHYS_USE_OPTIMIZED=1)
    - The original JAX implementation (default)
    """
    if is_optimized_enabled():
        # Use the custom primitive with optimized HLO
        return precipitation_effects_optimized(
            last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
        )

    # Original JAX implementation from graupel_baseline
    return _precipitation_effects_original(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
    )


# ============================================================================
# Top-Level Graupel Function
# ============================================================================

def graupel(last_level, dz, te, p, rho, q, dt, qnc):
    """Top-level graupel microphysics function."""
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions (from graupel_baseline)
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects (dispatches to optimized if configured)
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects(
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


# ============================================================================
# JIT-compiled entry point
# ============================================================================

@jit_compile
def graupel_run(dz, te, p, rho, q_in, dt, qnc, last_level=None):
    """JIT-compiled graupel driver with optional HLO optimization."""
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel(last_level, dz, te, p, rho, q_in, dt, qnc)


__all__ = [
    "graupel",
    "graupel_run",
    "precipitation_effects",
    "configure_optimized_precip",
    "is_optimized_enabled",
]
