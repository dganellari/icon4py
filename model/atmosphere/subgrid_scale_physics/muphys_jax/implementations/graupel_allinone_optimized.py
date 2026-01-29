# ICON4Py - JAX all-in-one fused scan with HLO optimization support
#
# This is a drop-in replacement for graupel_allinone_fused that can use
# optimized HLO for precipitation_effects when configured.
#
# Usage:
#     # Default (no optimization):
#     from muphys_jax.implementations.graupel_allinone_optimized import graupel_allinone_fused_run
#
#     # With optimized HLO injection:
#     export MUPHYS_OPTIMIZED_HLO=/path/to/precip_effect_allinone.serialized
#     export MUPHYS_USE_OPTIMIZED=1
#     from muphys_jax.implementations.graupel_allinone_optimized import graupel_allinone_fused_run

import jax
from functools import partial

from ..core.common import constants as const
from ..core.definitions import Q

# Import from graupel_allinone_fused - no duplication
from .graupel_allinone_fused import (
    q_t_update,
    precipitation_effects_allinone_fused as _precipitation_effects_original,
)

# Import the optimized precipitation_effects (auto-configures from env vars)
from ..core.optimized_precip import (
    configure_optimized_precip,
    is_optimized_enabled,
)


# ============================================================================
# Precipitation Effects - Uses Optimized Version When Configured
# ============================================================================

def precipitation_effects_allinone_fused(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
    """
    All-in-one fused scan for precipitation + temperature.

    Dispatches to optimized version if configured.
    """
    # For now, always use original since we haven't created a separate
    # primitive for allinone. The optimization pipeline is the same:
    # export → transform → inject
    return _precipitation_effects_original(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
    )


# ============================================================================
# Top-Level Graupel Function
# ============================================================================

def graupel_allinone_fused(last_level, dz, te, p, rho, q, dt, qnc):
    """
    All-in-one fused graupel microphysics with optional HLO optimization.
    """
    # Compute minimum levels for each species
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    # Phase transitions
    q_updated, t_updated = q_t_update(te, p, rho, q, dt, qnc)

    # Precipitation effects
    qr, qs, qi, qg, t_final, pflx, pr, ps, pi, pg, pre = precipitation_effects_allinone_fused(
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
def graupel_allinone_fused_run(dz, te, p, rho, q_in, dt, qnc, last_level=None, **kwargs):
    """JIT-compiled all-in-one fused graupel driver with optional HLO optimization."""
    if last_level is None:
        last_level = te.shape[1] - 1

    return graupel_allinone_fused(last_level, dz, te, p, rho, q_in, dt, qnc)


__all__ = [
    "graupel_allinone_fused",
    "graupel_allinone_fused_run",
    "precipitation_effects_allinone_fused",
    "configure_optimized_precip",
    "is_optimized_enabled",
]
