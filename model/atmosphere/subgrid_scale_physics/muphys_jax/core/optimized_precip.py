# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Optimized precipitation_effects using custom JAX primitive with HLO injection.

This module provides a drop-in replacement for precipitation_effects that can:
1. Use the original JAX implementation (default)
2. Use an optimized HLO module loaded from disk
3. Seamlessly integrate into the graupel pipeline within JIT

The injection works by:
1. Defining a custom JAX primitive (optimized_precip_p)
2. During MLIR lowering, if optimized HLO is configured, emit a custom_call
   that references the pre-compiled HLO module
3. XLA will execute the optimized HLO instead of re-tracing the original

Usage:
    # Option 1: Environment variables
    export MUPHYS_OPTIMIZED_HLO=/path/to/optimized.serialized
    export MUPHYS_USE_OPTIMIZED=1
    python your_script.py

    # Option 2: Programmatic configuration
    from muphys_jax.core.optimized_precip import configure_optimized_precip
    configure_optimized_precip(
        hlo_path="/path/to/optimized.serialized",
        use_optimized=True
    )

    # Then use graupel_optimized which calls precipitation_effects_optimized
    from muphys_jax.implementations.graupel_optimized import graupel_run
"""

import os
from functools import partial
from typing import Tuple, Optional, Callable
import pathlib

import jax
import jax.numpy as jnp
from jax import core as jax_core  # ShapedArray is always here
from jax import lax
from jax.interpreters import mlir
from jax.interpreters import batching
from jax._src.interpreters import ad
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

# Primitive may be in jax.extend.core (JAX 0.5+) or jax.core (older)
try:
    from jax.extend.core import Primitive
except (ImportError, AttributeError):
    from jax.core import Primitive

from .definitions import Q


# ============================================================================
# Global configuration
# ============================================================================

_OPTIMIZED_HLO_PATH: Optional[str] = None
_USE_OPTIMIZED: bool = False
_CACHED_EXECUTABLE = None
_CACHED_HLO_MODULE: Optional[bytes] = None  # Raw serialized HLO bytes


def configure_optimized_precip(hlo_path: Optional[str] = None, use_optimized: bool = False):
    """
    Configure the optimized precipitation_effects.

    Args:
        hlo_path: Path to optimized HLO file (serialized executable or .hlo text)
        use_optimized: Whether to use optimized version
    """
    global _OPTIMIZED_HLO_PATH, _USE_OPTIMIZED, _CACHED_EXECUTABLE, _CACHED_HLO_MODULE
    _OPTIMIZED_HLO_PATH = hlo_path
    _USE_OPTIMIZED = use_optimized
    _CACHED_EXECUTABLE = None  # Clear cache when config changes
    _CACHED_HLO_MODULE = None


def is_optimized_enabled() -> bool:
    """Check if optimized version is enabled."""
    return _USE_OPTIMIZED and _OPTIMIZED_HLO_PATH is not None


# ============================================================================
# Custom Primitive Definition
# ============================================================================

# Define the primitive
optimized_precip_p = Primitive("optimized_precipitation_effects")
optimized_precip_p.multiple_results = True


def _precip_effect_abstract_eval(
    kmin_r, kmin_i, kmin_s, kmin_g,
    q_v, q_c, q_r, q_s, q_i, q_g,
    t, rho, dz,
    *, last_lev, dt
):
    """Abstract evaluation - defines output shapes/types."""
    # All outputs have shape (ncells, nlev) and dtype float64
    # except for masks which are bool
    ncells, nlev = q_v.shape

    f64_shape = jax_core.ShapedArray((ncells, nlev), jnp.float64)

    # precipitation_effects returns 11 outputs:
    # qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx
    return (
        f64_shape,  # qr
        f64_shape,  # qs
        f64_shape,  # qi
        f64_shape,  # qg
        f64_shape,  # t_new
        f64_shape,  # pflx_tot
        f64_shape,  # pr
        f64_shape,  # ps
        f64_shape,  # pi
        f64_shape,  # pg
        f64_shape,  # eflx
    )


def _precip_effect_impl(
    kmin_r, kmin_i, kmin_s, kmin_g,
    q_v, q_c, q_r, q_s, q_i, q_g,
    t, rho, dz,
    *, last_lev, dt
):
    """
    Concrete implementation - called in eager mode or as fallback.

    This imports and calls the original JAX implementation.
    """
    from ..implementations.graupel_baseline import precipitation_effects as original_precip

    q_in = Q(v=q_v, c=q_c, r=q_r, s=q_s, i=q_i, g=q_g)
    return original_precip(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt)


# Register abstract eval and implementation
optimized_precip_p.def_abstract_eval(_precip_effect_abstract_eval)
optimized_precip_p.def_impl(_precip_effect_impl)


# ============================================================================
# MLIR Lowering - This is where HLO injection happens
# ============================================================================

def _load_hlo_module():
    """Load the HLO module from disk (cached)."""
    global _CACHED_HLO_MODULE

    if _CACHED_HLO_MODULE is not None:
        return _CACHED_HLO_MODULE

    if not _OPTIMIZED_HLO_PATH:
        return None

    try:
        path = pathlib.Path(_OPTIMIZED_HLO_PATH)

        if path.suffix == '.serialized':
            # Already serialized executable - read as bytes
            with open(path, 'rb') as f:
                _CACHED_HLO_MODULE = f.read()
        elif path.suffix in ('.hlo', '.stablehlo'):
            # Text format - need to read as text
            with open(path, 'r') as f:
                _CACHED_HLO_MODULE = f.read()
        else:
            # Try as binary first
            with open(path, 'rb') as f:
                _CACHED_HLO_MODULE = f.read()

        print(f"Loaded HLO module from: {_OPTIMIZED_HLO_PATH}")
        return _CACHED_HLO_MODULE

    except Exception as e:
        print(f"Failed to load HLO module: {e}")
        return None


def _precip_effect_lowering(ctx, *args, last_lev, dt):
    """
    MLIR lowering rule for the primitive.

    If optimized HLO is available, parse and inline it.
    Otherwise, fall back to the default JAX lowering.
    """
    if is_optimized_enabled():
        hlo_module = _load_hlo_module()
        if hlo_module is not None:
            return _stablehlo_injection_lowering(ctx, *args, last_lev=last_lev, dt=dt)
        else:
            print("HLO module not loaded, falling back to JAX")
            return _fallback_lowering(ctx, *args, last_lev=last_lev, dt=dt)
    else:
        return _fallback_lowering(ctx, *args, last_lev=last_lev, dt=dt)


def _fallback_lowering(ctx, *args, last_lev, dt):
    """
    Fallback lowering: inline the original JAX implementation.

    This re-traces the original function and inlines its HLO.
    """
    # Get input avals
    avals_in = [v.type for v in args]

    # Reconstruct Q namedtuple from individual arrays
    # args order: kmin_r, kmin_i, kmin_s, kmin_g, q_v, q_c, q_r, q_s, q_i, q_g, t, rho, dz
    kmin_r, kmin_i, kmin_s, kmin_g = args[0:4]
    q_v, q_c, q_r, q_s, q_i, q_g = args[4:10]
    t, rho, dz = args[10:13]

    # Import original implementation
    from ..implementations.graupel_baseline import precipitation_effects as original_precip

    # Create the Q namedtuple - but we're in MLIR land, so we need to call
    # the original function's lowering directly

    # For fallback, we use mlir.lower_fun to get the lowering of the original
    def original_wrapper(kmin_r, kmin_i, kmin_s, kmin_g, q_v, q_c, q_r, q_s, q_i, q_g, t, rho, dz):
        q_in = Q(v=q_v, c=q_c, r=q_r, s=q_s, i=q_i, g=q_g)
        return original_precip(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt)

    # Use mlir.lower_fun to get the standard lowering
    lowering_fn = mlir.lower_fun(original_wrapper, multiple_results=True)
    return lowering_fn(ctx, *args)


def _custom_call_lowering(ctx, *args, last_lev, dt):  # noqa: ARG001
    """DEPRECATED: Use _stablehlo_injection_lowering instead."""
    return _stablehlo_injection_lowering(ctx, *args, last_lev=last_lev, dt=dt)


# Counter for unique function names
_MERGE_COUNTER = 0


def _stablehlo_injection_lowering(ctx, *args, last_lev, dt):  # noqa: ARG001
    """
    Lowering that inlines optimized StableHLO using merge_mlir_modules.

    This approach:
    1. Parses the StableHLO text into an MLIR module
    2. Merges it into the current module using mlir.merge_mlir_modules
    3. Emits a func.call to the merged function

    Note: last_lev and dt are baked into the optimized HLO module at export time.
    """
    global _CACHED_HLO_MODULE, _MERGE_COUNTER

    # Get the HLO module content
    hlo_text = _CACHED_HLO_MODULE
    if isinstance(hlo_text, bytes):
        hlo_text = hlo_text.decode('utf-8')

    # Get output types from abstract eval
    avals_out = ctx.avals_out
    result_types = [mlir.aval_to_ir_type(aval) for aval in avals_out]

    # Get the current MLIR module
    dst_module = ctx.module_context.module

    try:
        # Parse the optimized StableHLO into a new module
        # Must be done within the same MLIR context
        src_module = ir.Module.parse(hlo_text)

        # Generate unique name for the merged function
        _MERGE_COUNTER += 1
        desired_name = f"_injected_precip_effect_{_MERGE_COUNTER}"

        # Merge src_module into dst_module
        # This renames the "main" function and makes it private
        actual_name = mlir.merge_mlir_modules(
            dst_module,
            desired_name,
            src_module,
        )

        # Build the call to the merged function
        # Get the function type from result_types
        from jax._src.lib.mlir.dialects import func as func_dialect

        # Create the call operation
        call_op = func_dialect.CallOp(
            result_types,
            ir.FlatSymbolRefAttr.get(actual_name),
            list(args),
        )

        # Return the results
        return list(call_op.results)

    except Exception as e:
        print(f"WARNING: Failed to inject StableHLO: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to JAX tracing")
        return _fallback_lowering(ctx, *args, last_lev=last_lev, dt=dt)


# Register the lowering rule
mlir.register_lowering(optimized_precip_p, _precip_effect_lowering)


# ============================================================================
# Batching rule (for vmap support)
# ============================================================================

def _precip_effect_batch(batched_args, batch_dims, *, last_lev, dt):
    """Batching rule for vmap."""
    # For simplicity, vmap over the first dimension
    # This maintains compatibility with the original function

    # Check if any input is batched
    if all(bd is None for bd in batch_dims):
        # No batching, just call directly
        return optimized_precip_p.bind(*batched_args, last_lev=last_lev, dt=dt), (None,) * 11

    # Find the batch dimension
    batch_size = None
    for arg, bd in zip(batched_args, batch_dims):
        if bd is not None:
            batch_size = arg.shape[bd]
            break

    # Use vmap to handle batching
    def unbatched_call(*args):
        return optimized_precip_p.bind(*args, last_lev=last_lev, dt=dt)

    in_axes = batch_dims
    out_axes = (0,) * 11  # All outputs batched on axis 0

    batched_fn = jax.vmap(unbatched_call, in_axes=in_axes, out_axes=out_axes)
    results = batched_fn(*batched_args)

    return results, (0,) * 11


batching.primitive_batchers[optimized_precip_p] = _precip_effect_batch


# ============================================================================
# Public API
# ============================================================================

def precipitation_effects_optimized(
    last_lev: int,
    kmin_r: jnp.ndarray,
    kmin_i: jnp.ndarray,
    kmin_s: jnp.ndarray,
    kmin_g: jnp.ndarray,
    q_in: Q,
    t: jnp.ndarray,
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    dt: float
) -> Tuple[jnp.ndarray, ...]:
    """
    Drop-in replacement for precipitation_effects.

    Uses optimized HLO if configured, otherwise falls back to JAX implementation.

    Args:
        last_lev: Last vertical level index
        kmin_r, kmin_i, kmin_s, kmin_g: Species activation masks
        q_in: Q namedtuple with species mixing ratios
        t: Temperature
        rho: Air density
        dz: Layer thickness
        dt: Time step

    Returns:
        Tuple of (qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx)
    """
    return optimized_precip_p.bind(
        kmin_r, kmin_i, kmin_s, kmin_g,
        q_in.v, q_in.c, q_in.r, q_in.s, q_in.i, q_in.g,
        t, rho, dz,
        last_lev=last_lev,
        dt=dt
    )


# ============================================================================
# Convenience: Environment variable configuration
# ============================================================================

def auto_configure():
    """
    Auto-configure from environment variables.

    Set these environment variables:
        MUPHYS_OPTIMIZED_HLO=/path/to/optimized.serialized
        MUPHYS_USE_OPTIMIZED=1
    """
    hlo_path = os.environ.get("MUPHYS_OPTIMIZED_HLO")
    use_optimized = os.environ.get("MUPHYS_USE_OPTIMIZED", "0") == "1"

    if hlo_path and use_optimized:
        configure_optimized_precip(hlo_path=hlo_path, use_optimized=True)
        print(f"Auto-configured optimized precipitation_effects from: {hlo_path}")


# Auto-configure on import
auto_configure()
