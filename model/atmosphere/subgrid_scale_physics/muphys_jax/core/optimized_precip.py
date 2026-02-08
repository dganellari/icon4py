# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom JAX primitive that injects pre-compiled StableHLO for precipitation_effects.

When configured (via configure_optimized_precip or MUPHYS_OPTIMIZED_HLO env var),
the primitive replaces the JAX-traced precipitation scan with a pre-compiled HLO
module during MLIR lowering. Otherwise falls back to the standard JAX implementation.
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
_TRANSPOSED_LAYOUT: bool = False  # If True, HLO expects (nlev, ncells) layout


def configure_optimized_precip(hlo_path: Optional[str] = None, use_optimized: bool = False,
                                transposed: bool = False):
    """
    Configure the optimized precipitation_effects.

    Args:
        hlo_path: Path to optimized HLO file (serialized executable or .hlo text)
        use_optimized: Whether to use optimized version
        transposed: If True, the HLO expects transposed (nlev, ncells) layout.
                   Inputs will be transposed before calling and outputs transposed back.
    """
    global _OPTIMIZED_HLO_PATH, _USE_OPTIMIZED, _CACHED_EXECUTABLE, _CACHED_HLO_MODULE, _TRANSPOSED_LAYOUT
    _OPTIMIZED_HLO_PATH = hlo_path
    _USE_OPTIMIZED = use_optimized
    _TRANSPOSED_LAYOUT = transposed
    _CACHED_EXECUTABLE = None  # Clear cache when config changes
    _CACHED_HLO_MODULE = None


def is_optimized_enabled() -> bool:
    """Check if optimized version is enabled."""
    return _USE_OPTIMIZED and _OPTIMIZED_HLO_PATH is not None


# ============================================================================
# Custom Primitive Definition
# ============================================================================

# Define the primitive for original (ncells, nlev) layout
optimized_precip_p = Primitive("optimized_precipitation_effects")
optimized_precip_p.multiple_results = True

# Define a separate primitive for transposed (nlev, ncells) layout
# This allows proper shape tracking through JAX's tracing
optimized_precip_transposed_p = Primitive("optimized_precipitation_effects_transposed")
optimized_precip_transposed_p.multiple_results = True


def _precip_effect_abstract_eval(
    kmin_r, kmin_i, kmin_s, kmin_g,
    q_v, q_c, q_r, q_s, q_i, q_g,
    t, rho, dz,
    *, last_lev, dt
):
    """Abstract evaluation for original (ncells, nlev) layout."""
    # All outputs have shape (ncells, nlev) and dtype float64
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


def _precip_effect_transposed_abstract_eval(
    kmin_r, kmin_i, kmin_s, kmin_g,
    q_v, q_c, q_r, q_s, q_i, q_g,
    t, rho, dz,
    *, last_lev, dt
):
    """Abstract evaluation for transposed (nlev, ncells) layout."""
    # Inputs are (nlev, ncells), outputs are also (nlev, ncells)
    nlev, ncells = q_v.shape

    f64_shape = jax_core.ShapedArray((nlev, ncells), jnp.float64)

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


def _precip_effect_transposed_impl(
    kmin_r, kmin_i, kmin_s, kmin_g,
    q_v, q_c, q_r, q_s, q_i, q_g,
    t, rho, dz,
    *, last_lev, dt
):
    """
    Concrete implementation for transposed layout - called in eager mode or as fallback.

    Transposes inputs, calls original, transposes outputs back.
    """
    from ..implementations.graupel_baseline import precipitation_effects as original_precip

    # Transpose inputs from (nlev, ncells) to (ncells, nlev)
    kmin_r_t = jnp.transpose(kmin_r)
    kmin_i_t = jnp.transpose(kmin_i)
    kmin_s_t = jnp.transpose(kmin_s)
    kmin_g_t = jnp.transpose(kmin_g)
    q_v_t = jnp.transpose(q_v)
    q_c_t = jnp.transpose(q_c)
    q_r_t = jnp.transpose(q_r)
    q_s_t = jnp.transpose(q_s)
    q_i_t = jnp.transpose(q_i)
    q_g_t = jnp.transpose(q_g)
    t_t = jnp.transpose(t)
    rho_t = jnp.transpose(rho)
    dz_t = jnp.transpose(dz)

    q_in = Q(v=q_v_t, c=q_c_t, r=q_r_t, s=q_s_t, i=q_i_t, g=q_g_t)
    results = original_precip(last_lev, kmin_r_t, kmin_i_t, kmin_s_t, kmin_g_t, q_in, t_t, rho_t, dz_t, dt)

    # Transpose outputs from (ncells, nlev) back to (nlev, ncells)
    return tuple(jnp.transpose(r) for r in results)


# Register abstract eval and implementation for original layout
optimized_precip_p.def_abstract_eval(_precip_effect_abstract_eval)
optimized_precip_p.def_impl(_precip_effect_impl)

# Register abstract eval and implementation for transposed layout
optimized_precip_transposed_p.def_abstract_eval(_precip_effect_transposed_abstract_eval)
optimized_precip_transposed_p.def_impl(_precip_effect_transposed_impl)


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

    # Get output types from abstract eval (in original ncells×nlev layout)
    avals_out = ctx.avals_out
    avals_in = ctx.avals_in

    # Debug: print shapes
    print(f"[HLO INJECTION DEBUG] Input avals: {[str(a) for a in avals_in]}")
    print(f"[HLO INJECTION DEBUG] Output avals: {[str(a) for a in avals_out]}")
    print(f"[HLO INJECTION DEBUG] Arg MLIR types: {[str(arg.type) for arg in args]}")

    # Get the current MLIR module
    dst_module = ctx.module_context.module

    try:
        # Parse the optimized StableHLO into a new module
        # Must be done within the same MLIR context
        print(f"[HLO INJECTION DEBUG] Parsing StableHLO ({len(hlo_text)} chars)...")
        src_module = ir.Module.parse(hlo_text)
        print(f"[HLO INJECTION DEBUG] Parsed successfully")

        # Generate unique name for the merged function
        _MERGE_COUNTER += 1
        desired_name = f"_injected_precip_effect_{_MERGE_COUNTER}"

        # Merge src_module into dst_module
        # This renames the "main" function and makes it private
        print(f"[HLO INJECTION DEBUG] Merging as '{desired_name}'...")
        actual_name = mlir.merge_mlir_modules(
            dst_module,
            desired_name,
            src_module,
        )
        print(f"[HLO INJECTION DEBUG] Merged successfully as '{actual_name}'")

        # Build the call to the merged function
        from jax._src.lib.mlir.dialects import func as func_dialect

        # Generate result types from abstract eval
        result_types = [mlir.aval_to_ir_type(aval) for aval in avals_out]
        print(f"[HLO INJECTION DEBUG] Result types: {[str(t) for t in result_types]}")

        # Create the call
        call_op = func_dialect.CallOp(
            result_types,
            ir.FlatSymbolRefAttr.get(actual_name),
            list(args),
        )
        print(f"[HLO INJECTION DEBUG] Created call successfully!")
        return list(call_op.results)

    except Exception as e:
        print(f"[HLO INJECTION ERROR] Failed to inject StableHLO: {e}")
        import traceback
        traceback.print_exc()
        print("[HLO INJECTION ERROR] Falling back to JAX tracing")
        return _fallback_lowering(ctx, *args, last_lev=last_lev, dt=dt)


def _fallback_lowering_transposed(ctx, *args, last_lev, dt):
    """
    Fallback lowering for transposed primitive: inline the JAX implementation with transposes.
    """
    from ..implementations.graupel_baseline import precipitation_effects as original_precip

    def transposed_wrapper(kmin_r, kmin_i, kmin_s, kmin_g, q_v, q_c, q_r, q_s, q_i, q_g, t, rho, dz):
        # Transpose inputs from (nlev, ncells) to (ncells, nlev)
        kmin_r_t = jnp.transpose(kmin_r)
        kmin_i_t = jnp.transpose(kmin_i)
        kmin_s_t = jnp.transpose(kmin_s)
        kmin_g_t = jnp.transpose(kmin_g)
        q_v_t = jnp.transpose(q_v)
        q_c_t = jnp.transpose(q_c)
        q_r_t = jnp.transpose(q_r)
        q_s_t = jnp.transpose(q_s)
        q_i_t = jnp.transpose(q_i)
        q_g_t = jnp.transpose(q_g)
        t_t = jnp.transpose(t)
        rho_t = jnp.transpose(rho)
        dz_t = jnp.transpose(dz)

        q_in = Q(v=q_v_t, c=q_c_t, r=q_r_t, s=q_s_t, i=q_i_t, g=q_g_t)
        results = original_precip(last_lev, kmin_r_t, kmin_i_t, kmin_s_t, kmin_g_t, q_in, t_t, rho_t, dz_t, dt)

        # Transpose outputs from (ncells, nlev) back to (nlev, ncells)
        return tuple(jnp.transpose(r) for r in results)

    lowering_fn = mlir.lower_fun(transposed_wrapper, multiple_results=True)
    return lowering_fn(ctx, *args)


def _precip_effect_transposed_lowering(ctx, *args, last_lev, dt):
    """
    MLIR lowering rule for the transposed primitive.

    If optimized HLO is available (transposed layout), inject it.
    Otherwise, fall back to the default JAX lowering with transposes.
    """
    if is_optimized_enabled():
        hlo_module = _load_hlo_module()
        if hlo_module is not None:
            return _stablehlo_injection_lowering(ctx, *args, last_lev=last_lev, dt=dt)
        else:
            print("HLO module not loaded, falling back to JAX (transposed)")
            return _fallback_lowering_transposed(ctx, *args, last_lev=last_lev, dt=dt)
    else:
        return _fallback_lowering_transposed(ctx, *args, last_lev=last_lev, dt=dt)


# Register the lowering rules
mlir.register_lowering(optimized_precip_p, _precip_effect_lowering)
mlir.register_lowering(optimized_precip_transposed_p, _precip_effect_transposed_lowering)


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


def _precip_effect_transposed_batch(batched_args, batch_dims, *, last_lev, dt):
    """Batching rule for vmap (transposed primitive)."""
    if all(bd is None for bd in batch_dims):
        return optimized_precip_transposed_p.bind(*batched_args, last_lev=last_lev, dt=dt), (None,) * 11

    def unbatched_call(*args):
        return optimized_precip_transposed_p.bind(*args, last_lev=last_lev, dt=dt)

    batched_fn = jax.vmap(unbatched_call, in_axes=batch_dims, out_axes=(0,) * 11)
    results = batched_fn(*batched_args)
    return results, (0,) * 11


batching.primitive_batchers[optimized_precip_transposed_p] = _precip_effect_transposed_batch


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
    precipitation_effects with optional HLO injection.

    If _TRANSPOSED_LAYOUT is set, transposes to (nlev, ncells) before calling
    the primitive and transposes outputs back to (ncells, nlev).
    """
    if _TRANSPOSED_LAYOUT:
        # Transpose all inputs from (ncells, nlev) to (nlev, ncells)
        kmin_r_t = jnp.transpose(kmin_r)
        kmin_i_t = jnp.transpose(kmin_i)
        kmin_s_t = jnp.transpose(kmin_s)
        kmin_g_t = jnp.transpose(kmin_g)
        q_v_t = jnp.transpose(q_in.v)
        q_c_t = jnp.transpose(q_in.c)
        q_r_t = jnp.transpose(q_in.r)
        q_s_t = jnp.transpose(q_in.s)
        q_i_t = jnp.transpose(q_in.i)
        q_g_t = jnp.transpose(q_in.g)
        t_t = jnp.transpose(t)
        rho_t = jnp.transpose(rho)
        dz_t = jnp.transpose(dz)

        # Call transposed primitive (expects nlev×ncells, returns nlev×ncells)
        results = optimized_precip_transposed_p.bind(
            kmin_r_t, kmin_i_t, kmin_s_t, kmin_g_t,
            q_v_t, q_c_t, q_r_t, q_s_t, q_i_t, q_g_t,
            t_t, rho_t, dz_t,
            last_lev=last_lev,
            dt=dt
        )

        # Transpose outputs back from (nlev, ncells) to (ncells, nlev)
        return tuple(jnp.transpose(r) for r in results)
    else:
        # Original layout - no transposes needed
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
        MUPHYS_TRANSPOSED=1  (if HLO expects nlev×ncells layout)
    """
    hlo_path = os.environ.get("MUPHYS_OPTIMIZED_HLO")
    use_optimized = os.environ.get("MUPHYS_USE_OPTIMIZED", "0") == "1"
    transposed = os.environ.get("MUPHYS_TRANSPOSED", "0") == "1"

    if hlo_path and use_optimized:
        configure_optimized_precip(hlo_path=hlo_path, use_optimized=True, transposed=transposed)
        layout = "transposed (nlev×ncells)" if transposed else "original (ncells×nlev)"
        print(f"Auto-configured optimized precipitation_effects from: {hlo_path} [{layout}]")


# Auto-configure on import
auto_configure()
