# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Optimized q_t_update using custom JAX primitive with HLO injection.

This module provides a drop-in replacement for q_t_update (phase transitions)
that can inject optimized StableHLO for better kernel fusion on GPU.

The q_t_update function computes all phase transitions between water species:
- 14 transitions (cloud_to_rain, ice_to_snow, vapor_x_ice, etc.)
- ~27 power operations, ~7 exponentials
- Purely element-wise (no while loops or scans)

The issue: XLA generates ~80 separate kernel launches, each taking ~0.3ms,
resulting in ~24ms total for q_t_update (75% of graupel time).

Solution: Inject optimized StableHLO with better kernel fusion.

Usage:
    from muphys_jax.core.optimized_q_t_update import (
        configure_optimized_q_t_update,
        q_t_update_optimized
    )

    # Configure HLO injection
    configure_optimized_q_t_update(
        hlo_path="/path/to/q_t_update_transposed.stablehlo",
        use_optimized=True
    )

    # Use in graupel code
    q_out, t_out = q_t_update_optimized(t, p, rho, q_in, dt, qnc)
"""

import os
from functools import partial
from typing import Tuple, Optional
import pathlib

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax import lax
from jax.interpreters import mlir
from jax.interpreters import batching
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
_CACHED_HLO_MODULE: Optional[str] = None
_DT_VALUE: float = 30.0  # Default dt, should match the exported HLO


def configure_optimized_q_t_update(
    hlo_path: Optional[str] = None,
    use_optimized: bool = False,
    dt: float = 30.0
):
    """
    Configure the optimized q_t_update.

    Args:
        hlo_path: Path to optimized HLO file (.stablehlo text format)
        use_optimized: Whether to use optimized version
        dt: The dt value baked into the HLO (must match export)
    """
    global _OPTIMIZED_HLO_PATH, _USE_OPTIMIZED, _CACHED_HLO_MODULE, _DT_VALUE
    _OPTIMIZED_HLO_PATH = hlo_path
    _USE_OPTIMIZED = use_optimized
    _DT_VALUE = dt
    _CACHED_HLO_MODULE = None  # Clear cache when config changes


def is_optimized_enabled() -> bool:
    """Check if optimized version is enabled."""
    return _USE_OPTIMIZED and _OPTIMIZED_HLO_PATH is not None


def get_dt_value() -> float:
    """Get the dt value that was baked into the HLO."""
    return _DT_VALUE


# ============================================================================
# Custom Primitive Definition
# ============================================================================

# Primitive for transposed (nlev, ncells) layout - native transposed mode
optimized_q_t_update_p = Primitive("optimized_q_t_update")
optimized_q_t_update_p.multiple_results = True


def _q_t_update_abstract_eval(t, p, rho, qv, qc, qr, qs, qi, qg, qnc):
    """Abstract evaluation for q_t_update primitive.

    Note: The exported StableHLO has qnc as a scalar (tensor<f64>),
    so this primitive also expects qnc to be a scalar.
    """
    # Inputs: t, p, rho, qv, qc, qr, qs, qi, qg are (nlev, ncells)
    # qnc is a scalar (tensor<f64>)
    # Outputs: qv, qc, qr, qs, qi, qg, t_new are (nlev, ncells)
    nlev, ncells = t.shape
    f64_shape = jax_core.ShapedArray((nlev, ncells), jnp.float64)

    return (
        f64_shape,  # qv_out
        f64_shape,  # qc_out
        f64_shape,  # qr_out
        f64_shape,  # qs_out
        f64_shape,  # qi_out
        f64_shape,  # qg_out
        f64_shape,  # t_out
    )


def _q_t_update_impl(t, p, rho, qv, qc, qr, qs, qi, qg, qnc):
    """
    Concrete implementation - called in eager mode or as fallback.

    This imports and calls the original JAX implementation.
    """
    from ..implementations.graupel_native_transposed import q_t_update_native

    q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
    q_out, t_out = q_t_update_native(t, p, rho, q_in, _DT_VALUE, qnc)
    return q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g, t_out


# Register abstract eval and implementation
optimized_q_t_update_p.def_abstract_eval(_q_t_update_abstract_eval)
optimized_q_t_update_p.def_impl(_q_t_update_impl)


# ============================================================================
# MLIR Lowering - HLO Injection
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

        with open(path, 'r') as f:
            _CACHED_HLO_MODULE = f.read()

        print(f"Loaded q_t_update HLO module from: {_OPTIMIZED_HLO_PATH}")
        return _CACHED_HLO_MODULE

    except Exception as e:
        print(f"Failed to load q_t_update HLO module: {e}")
        return None


# Counter for unique function names
_MERGE_COUNTER = 0


def _q_t_update_lowering(ctx, *args):
    """
    MLIR lowering rule for the primitive.

    If optimized HLO is available, parse and inline it.
    Otherwise, fall back to the default JAX lowering.
    """
    if is_optimized_enabled():
        hlo_module = _load_hlo_module()
        if hlo_module is not None:
            return _stablehlo_injection_lowering(ctx, *args)
        else:
            print("q_t_update HLO module not loaded, falling back to JAX")
            return _fallback_lowering(ctx, *args)
    else:
        return _fallback_lowering(ctx, *args)


def _fallback_lowering(ctx, *args):
    """
    Fallback lowering: inline the original JAX implementation.
    """
    from ..implementations.graupel_native_transposed import q_t_update_native

    def original_wrapper(t, p, rho, qv, qc, qr, qs, qi, qg, qnc):
        q_in = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)
        q_out, t_out = q_t_update_native(t, p, rho, q_in, _DT_VALUE, qnc)
        return q_out.v, q_out.c, q_out.r, q_out.s, q_out.i, q_out.g, t_out

    lowering_fn = mlir.lower_fun(original_wrapper, multiple_results=True)
    return lowering_fn(ctx, *args)


def _stablehlo_injection_lowering(ctx, *args):
    """
    Lowering that inlines optimized StableHLO using merge_mlir_modules.

    This approach:
    1. Parses the StableHLO text into an MLIR module
    2. Merges it into the current module using mlir.merge_mlir_modules
    3. Emits a func.call to the merged function

    Note: dt is baked into the optimized HLO module at export time.
    """
    global _CACHED_HLO_MODULE, _MERGE_COUNTER

    hlo_text = _CACHED_HLO_MODULE
    if isinstance(hlo_text, bytes):
        hlo_text = hlo_text.decode('utf-8')

    # Get output types from abstract eval
    avals_out = ctx.avals_out
    avals_in = ctx.avals_in

    # Debug info
    print(f"[Q_T_UPDATE HLO DEBUG] Input avals: {[str(a) for a in avals_in]}")
    print(f"[Q_T_UPDATE HLO DEBUG] Output avals: {[str(a) for a in avals_out]}")

    # Get the current MLIR module
    dst_module = ctx.module_context.module

    try:
        # Parse the optimized StableHLO into a new module
        print(f"[Q_T_UPDATE HLO DEBUG] Parsing StableHLO ({len(hlo_text)} chars)...")
        src_module = ir.Module.parse(hlo_text)
        print(f"[Q_T_UPDATE HLO DEBUG] Parsed successfully")

        # Generate unique name for the merged function
        _MERGE_COUNTER += 1
        desired_name = f"_injected_q_t_update_{_MERGE_COUNTER}"

        # Merge src_module into dst_module
        print(f"[Q_T_UPDATE HLO DEBUG] Merging as '{desired_name}'...")
        actual_name = mlir.merge_mlir_modules(
            dst_module,
            desired_name,
            src_module,
        )
        print(f"[Q_T_UPDATE HLO DEBUG] Merged successfully as '{actual_name}'")

        # Build the call to the merged function
        from jax._src.lib.mlir.dialects import func as func_dialect

        # Generate result types from abstract eval
        result_types = [mlir.aval_to_ir_type(aval) for aval in avals_out]

        # Create the call
        call_op = func_dialect.CallOp(
            result_types,
            ir.FlatSymbolRefAttr.get(actual_name),
            list(args),
        )
        print(f"[Q_T_UPDATE HLO DEBUG] Created call successfully!")
        return list(call_op.results)

    except Exception as e:
        print(f"[Q_T_UPDATE HLO ERROR] Failed to inject StableHLO: {e}")
        import traceback
        traceback.print_exc()
        print("[Q_T_UPDATE HLO ERROR] Falling back to JAX tracing")
        return _fallback_lowering(ctx, *args)


# Register the lowering rule
mlir.register_lowering(optimized_q_t_update_p, _q_t_update_lowering)


# ============================================================================
# Batching rule (for vmap support)
# ============================================================================

def _q_t_update_batch(batched_args, batch_dims):
    """Batching rule for vmap."""
    if all(bd is None for bd in batch_dims):
        return optimized_q_t_update_p.bind(*batched_args), (None,) * 7

    def unbatched_call(*args):
        return optimized_q_t_update_p.bind(*args)

    batched_fn = jax.vmap(unbatched_call, in_axes=batch_dims, out_axes=(0,) * 7)
    results = batched_fn(*batched_args)
    return results, (0,) * 7


batching.primitive_batchers[optimized_q_t_update_p] = _q_t_update_batch


# ============================================================================
# Public API
# ============================================================================

def q_t_update_optimized(
    t: jnp.ndarray,
    p: jnp.ndarray,
    rho: jnp.ndarray,
    q_in: Q,
    dt: float,  # Note: dt is ignored when using optimized HLO (baked in)
    qnc: float  # Note: qnc is a SCALAR in the exported HLO
) -> Tuple[Q, jnp.ndarray]:
    """
    Drop-in replacement for q_t_update_native.

    Uses optimized HLO if configured, otherwise falls back to JAX implementation.

    IMPORTANT: When using optimized HLO:
    - The dt parameter is ignored (baked into the HLO at export time)
    - The qnc parameter must be a scalar (baked into the HLO)
    Make sure these values match what was used during export.

    Args:
        t: Temperature (nlev, ncells)
        p: Pressure (nlev, ncells)
        rho: Air density (nlev, ncells)
        q_in: Q namedtuple with species mixing ratios (nlev, ncells each)
        dt: Time step (ignored when using optimized HLO)
        qnc: Cloud number concentration - SCALAR value

    Returns:
        Tuple of (Q namedtuple with updated species, updated temperature)
    """
    # Check if dt matches the baked-in value
    if is_optimized_enabled() and abs(dt - _DT_VALUE) > 1e-10:
        print(f"WARNING: dt={dt} does not match baked-in dt={_DT_VALUE}")

    # Ensure qnc is a scalar JAX array
    qnc_scalar = jnp.asarray(qnc, dtype=jnp.float64)

    # Call the primitive
    qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, t_out = optimized_q_t_update_p.bind(
        t, p, rho, q_in.v, q_in.c, q_in.r, q_in.s, q_in.i, q_in.g, qnc_scalar
    )

    q_out = Q(v=qv_out, c=qc_out, r=qr_out, s=qs_out, i=qi_out, g=qg_out)
    return q_out, t_out


# ============================================================================
# Environment variable configuration
# ============================================================================

def auto_configure():
    """
    Auto-configure from environment variables.

    Set these environment variables:
        MUPHYS_Q_T_UPDATE_HLO=/path/to/q_t_update.stablehlo
        MUPHYS_USE_OPTIMIZED_Q_T_UPDATE=1
        MUPHYS_Q_T_UPDATE_DT=30.0
    """
    hlo_path = os.environ.get("MUPHYS_Q_T_UPDATE_HLO")
    use_optimized = os.environ.get("MUPHYS_USE_OPTIMIZED_Q_T_UPDATE", "0") == "1"
    dt_str = os.environ.get("MUPHYS_Q_T_UPDATE_DT", "30.0")

    try:
        dt = float(dt_str)
    except ValueError:
        dt = 30.0

    if hlo_path and use_optimized:
        configure_optimized_q_t_update(hlo_path=hlo_path, use_optimized=True, dt=dt)
        print(f"Auto-configured optimized q_t_update from: {hlo_path} [dt={dt}]")


# Auto-configure on import
auto_configure()
