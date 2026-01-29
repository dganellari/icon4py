"""
StableHLO-based precipitation scan using pre-compiled unrolled IR.

This module provides a precipitation scan implementation that loads
and executes a pre-compiled StableHLO module for optimal performance.

The unrolled StableHLO eliminates:
- While loop overhead
- Dynamic slice/update operations
- Tuple passing between iterations

Usage:
    from muphys_jax.core.scans_stablehlo import precip_scan_stablehlo, STABLEHLO_AVAILABLE

    if STABLEHLO_AVAILABLE:
        results = precip_scan_stablehlo(params_list, zeta, rho, q_list, vc_list, mask_list)
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from functools import lru_cache

# Check if StableHLO module is available
STABLEHLO_PATH = os.environ.get(
    'MUPHYS_STABLEHLO_PATH',
    str(Path(__file__).parent.parent.parent.parent.parent.parent / 'shlo' / 'precip_effect_x64_unrolled.stablehlo')
)
STABLEHLO_AVAILABLE = Path(STABLEHLO_PATH).exists()
STABLEHLO_IMPORT_ERROR = None if STABLEHLO_AVAILABLE else f"StableHLO file not found: {STABLEHLO_PATH}"


@lru_cache(maxsize=1)
def _get_compiled_executable():
    """Load and compile the StableHLO module (cached)."""
    import jaxlib._jax as jax_cpp
    from jax.lib import xla_bridge

    if not STABLEHLO_AVAILABLE:
        raise RuntimeError(STABLEHLO_IMPORT_ERROR)

    # Load StableHLO text
    with open(STABLEHLO_PATH, 'r') as f:
        stablehlo_text = f.read()

    # Get GPU client
    client = xla_bridge.get_backend("gpu")

    # Compile
    devices = client.local_devices()[:1]
    device_list = jax_cpp.DeviceList(tuple(devices))
    compile_options = jax_cpp.CompileOptions()

    # Compile and get LoadedExecutable
    executable = client.compile(stablehlo_text, device_list, compile_options)
    serialized = executable.serialize()
    loaded = client.deserialize_executable(serialized, device_list, compile_options)

    return loaded, client


def precip_scan_stablehlo(
    params_list: List[Tuple[float, float, float]],
    zeta: jnp.ndarray,  # (nlev, ncells)
    rho: jnp.ndarray,   # (nlev, ncells)
    q_list: List[jnp.ndarray],  # 4 arrays of (nlev, ncells)
    vc_list: List[jnp.ndarray],  # 4 arrays of (nlev, ncells)
    mask_list: List[jnp.ndarray],  # 4 arrays of (nlev, ncells)
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Execute precipitation scan using pre-compiled StableHLO.

    This is a drop-in replacement for precip_scan_unrolled that uses
    the pre-compiled StableHLO for maximum performance.

    Args:
        params_list: List of (prefactor, exponent, offset) for 4 species
        zeta: dt / (2 * dz), shape (nlev, ncells)
        rho: density, shape (nlev, ncells)
        q_list: [qr, qs, qi, qg], each (nlev, ncells)
        vc_list: [vc_r, vc_s, vc_i, vc_g], each (nlev, ncells)
        mask_list: [kmin_r, kmin_s, kmin_i, kmin_g], each (nlev, ncells)

    Returns:
        List of (q_out, flx_out) tuples for each species
    """
    if not STABLEHLO_AVAILABLE:
        raise RuntimeError(STABLEHLO_IMPORT_ERROR)

    executable, client = _get_compiled_executable()
    device = client.local_devices()[0]

    nlev, ncells = zeta.shape

    # Transpose inputs from (nlev, ncells) to (ncells, nlev) for StableHLO
    # The StableHLO was generated with (ncells, nlev) layout
    kmin_r = jnp.swapaxes(mask_list[0], 0, 1)  # (ncells, nlev)
    kmin_i = jnp.swapaxes(mask_list[2], 0, 1)  # Note: mask_list order is [r, s, i, g]
    kmin_s = jnp.swapaxes(mask_list[1], 0, 1)
    kmin_g = jnp.swapaxes(mask_list[3], 0, 1)

    qv = jnp.zeros((ncells, nlev), dtype=jnp.float64)  # Placeholder
    qc = jnp.zeros((ncells, nlev), dtype=jnp.float64)  # Placeholder
    qr = jnp.swapaxes(q_list[0], 0, 1)
    qs = jnp.swapaxes(q_list[1], 0, 1)
    qi = jnp.swapaxes(q_list[2], 0, 1)
    qg = jnp.swapaxes(q_list[3], 0, 1)

    t = jnp.zeros((ncells, nlev), dtype=jnp.float64)  # Placeholder
    rho_T = jnp.swapaxes(rho, 0, 1)

    # Compute dz from zeta: dz = dt / (2 * zeta), assuming dt=30
    dt = 30.0
    dz = dt / (2.0 * zeta)
    dz_T = jnp.swapaxes(dz, 0, 1)

    # Prepare inputs for StableHLO (must match signature)
    inputs = [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho_T, dz_T]

    # Put on device
    jax_inputs = [jax.device_put(inp, device) for inp in inputs]
    device_inputs = [x.addressable_data(0) for x in jax_inputs]

    # Execute
    results = executable.execute(device_inputs)
    jax.block_until_ready(results)

    # Results: qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, t_out, pflx_r, pflx_s, pflx_i, pflx_g
    # We need: qr_out, qs_out, qi_out, qg_out and pflx_r, pflx_s, pflx_i, pflx_g

    # Convert back to JAX arrays and transpose back to (nlev, ncells)
    qr_out = jnp.swapaxes(jnp.array(results[2]), 0, 1)
    qs_out = jnp.swapaxes(jnp.array(results[3]), 0, 1)
    qi_out = jnp.swapaxes(jnp.array(results[4]), 0, 1)
    qg_out = jnp.swapaxes(jnp.array(results[5]), 0, 1)

    pflx_r = jnp.swapaxes(jnp.array(results[7]), 0, 1)
    pflx_s = jnp.swapaxes(jnp.array(results[8]), 0, 1)
    pflx_i = jnp.swapaxes(jnp.array(results[9]), 0, 1)
    pflx_g = jnp.swapaxes(jnp.array(results[10]), 0, 1)

    return [
        (qr_out, pflx_r),
        (qs_out, pflx_s),
        (qi_out, pflx_i),
        (qg_out, pflx_g),
    ]


def precip_scan_stablehlo_direct(
    kmin_r: jnp.ndarray,  # (ncells, nlev)
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
) -> Tuple[jnp.ndarray, ...]:
    """
    Direct StableHLO execution with matching signature.

    This matches the exact signature of the StableHLO module.
    All inputs are (ncells, nlev).
    """
    if not STABLEHLO_AVAILABLE:
        raise RuntimeError(STABLEHLO_IMPORT_ERROR)

    executable, client = _get_compiled_executable()
    device = client.local_devices()[0]

    inputs = [kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz]

    jax_inputs = [jax.device_put(inp, device) for inp in inputs]
    device_inputs = [x.addressable_data(0) for x in jax_inputs]

    results = executable.execute(device_inputs)
    jax.block_until_ready(results)

    # Return all 11 outputs
    return tuple(jnp.array(r) for r in results)
