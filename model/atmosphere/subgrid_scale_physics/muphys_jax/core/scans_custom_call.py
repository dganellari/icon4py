# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom JAX primitive with MLIR lowering for precipitation scans.

This approach defines a custom primitive that can have optimized lowering
to StableHLO/MLIR, potentially generating better loop structures than
lax.scan.

NOTE: This is experimental. For production use, Triton is recommended.
"""

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from functools import partial
import numpy as np

# Define custom primitive
precip_scan_p = core.Primitive("precip_scan")
precip_scan_p.multiple_results = True


def precip_scan_prim(zeta, rho, q, vc, mask, prefactor, exponent, offset):
    """Call the custom precipitation scan primitive."""
    return precip_scan_p.bind(
        zeta, rho, q, vc, mask,
        prefactor=prefactor, exponent=exponent, offset=offset
    )


def _precip_scan_abstract(zeta, rho, q, vc, mask, *, prefactor, exponent, offset):
    """Abstract evaluation - returns output shapes/dtypes."""
    # Output shapes match input q shape
    return (
        core.ShapedArray(q.shape, q.dtype),  # q_out
        core.ShapedArray(q.shape, q.dtype),  # flx_out
    )


precip_scan_p.def_abstract_eval(_precip_scan_abstract)


def _precip_scan_impl(zeta, rho, q, vc, mask, *, prefactor, exponent, offset):
    """Default implementation using lax.scan (fallback)."""
    from jax import lax

    nlev, ncells = q.shape
    dtype = q.dtype

    def scan_step(carry, inputs):
        q_prev, flx_prev, rhox_prev, activated_prev = carry
        zeta_k, vc_k, q_k, rho_k, mask_k = inputs

        activated = activated_prev | mask_k
        rho_x = q_k * rho_k
        flx_eff = (rho_x / zeta_k) + 2.0 * flx_prev

        fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
        flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)

        vt_active = vc_k * prefactor * lax.pow(rhox_prev + offset, exponent)
        vt = lax.select(activated_prev, vt_active, jnp.zeros_like(q_k))

        q_activated = (zeta_k * (flx_eff - flx_partial)) / ((1.0 + zeta_k * vt) * rho_k)
        flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5

        q_out = lax.select(activated, q_activated, q_k)
        flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q_k))

        rhox_next = q_out * rho_k
        new_carry = (q_out, flx_out, rhox_next, activated)
        return new_carry, (q_out, flx_out)

    init_carry = (
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=dtype),
        jnp.zeros(ncells, dtype=bool),
    )

    inputs = (zeta, vc, q, rho, mask)
    _, (q_out, flx_out) = lax.scan(scan_step, init_carry, inputs)

    return q_out, flx_out


precip_scan_p.def_impl(_precip_scan_impl)


def _precip_scan_lowering(ctx, zeta, rho, q, vc, mask, *, prefactor, exponent, offset):
    """MLIR lowering for precipitation scan.

    This generates StableHLO that could be optimized by XLA.
    For true optimization, we'd generate a custom while loop structure.

    NOTE: This is a placeholder - real optimization would require
    generating MLIR that XLA can better optimize, or using custom_call
    to invoke a pre-compiled CUDA kernel.
    """
    # For now, fall back to default lowering via impl
    # A real implementation would generate optimized MLIR here
    return mlir.lower_fun(_precip_scan_impl, multiple_results=True)(
        ctx, zeta, rho, q, vc, mask,
        prefactor=prefactor, exponent=exponent, offset=offset
    )


# Register lowering rules
mlir.register_lowering(precip_scan_p, _precip_scan_lowering)


def _single_species_scan_custom(params, zeta, rho, q, vc, mask):
    """Single species scan using custom primitive."""
    prefactor, exponent, offset = params
    return precip_scan_prim(zeta, rho, q, vc, mask, prefactor, exponent, offset)


def precip_scan_custom(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Process 4 precipitation scans using custom primitive.

    This provides a foundation for MLIR-level optimization.
    """
    results = []
    for i in range(4):
        q_out, flx_out = _single_species_scan_custom(
            params_list[i], zeta, rho, q_list[i], vc_list[i], mask_list[i]
        )
        results.append((q_out, flx_out))
    return results
