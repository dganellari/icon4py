# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
MLIR-based graupel microphysics implementation.

This module provides the graupel microphysics routines using MLIR-generated
GPU kernels for optimal performance. The interface matches muphys_jax for
easy comparison and drop-in replacement.
"""

import numpy as np
from typing import Tuple, List

from ..core.precip_scans_mlir import (
    precip_scan_mlir,
    generate_precip_scan_mlir,
    MLIR_AVAILABLE,
    MLIR_IMPORT_ERROR,
)

__all__ = [
    "graupel_run",
    "precipitation_effects",
    "MLIR_AVAILABLE",
    "MLIR_IMPORT_ERROR",
]

# Physical constants (same as muphys_jax)
TMELT = 273.15  # Melting temperature of ice [K]
RVAP = 461.51   # Gas constant for water vapor [J/(kg*K)]
LVAP = 2.5008e6 # Latent heat of vaporization [J/kg]
LSUB = 2.8345e6 # Latent heat of sublimation [J/kg]
LFUS = LSUB - LVAP  # Latent heat of fusion [J/kg]
CPD = 1004.64   # Specific heat of dry air [J/(kg*K)]
RHO_00 = 1.225  # Reference air density [kg/m^3]
QMIN = 1.0e-15  # Minimum mixing ratio threshold

# Fall speed parameters for precipitation species (from muphys_jax)
# (prefactor, exponent, offset)
RAIN_PARAMS = (14.58, 0.111, 1.0e-12)
SNOW_PARAMS = (57.80, 0.16666666666666666, 1.0e-12)
ICE_PARAMS = (1.25, 0.160, 1.0e-12)
GRAUPEL_PARAMS = (12.24, 0.217, 1.0e-08)


def vel_scale_factor_default(xrho: np.ndarray) -> np.ndarray:
    """Default velocity scale factor."""
    return xrho


def vel_scale_factor_snow(xrho: np.ndarray, rho: np.ndarray, t: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Velocity scale factor for snow (simplified)."""
    return xrho * 0.8


def vel_scale_factor_ice(xrho: np.ndarray) -> np.ndarray:
    """Velocity scale factor for ice."""
    return xrho * 0.5


def precipitation_effects(
    dz: np.ndarray,
    t: np.ndarray,
    rho: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qi: np.ndarray,
    qg: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precipitation effects using MLIR-generated GPU kernel.

    This is the main precipitation routine that:
    1. Computes velocity scale factors
    2. Computes activation masks
    3. Calls MLIR precipitation scan kernel
    4. Returns updated mixing ratios and fluxes

    Args:
        dz: Layer thickness [m] (nlev x ncells)
        t: Temperature [K] (nlev x ncells)
        rho: Air density [kg/m^3] (nlev x ncells)
        qr, qs, qi, qg: Mixing ratios [kg/kg] (nlev x ncells)
        dt: Time step [s]

    Returns:
        Tuple of:
        - qr_out, qs_out, qi_out, qg_out: Updated mixing ratios
        - t_out: Updated temperature
        - pflx: Total precipitation flux
        - pr, ps, pi, pg: Individual species precipitation rates
    """
    if not MLIR_AVAILABLE:
        raise RuntimeError(
            f"MLIR not available: {MLIR_IMPORT_ERROR}\n"
            "Install with: pip install mlir-python-bindings"
        )

    nlev, ncells = qr.shape

    # Compute zeta = dt / (2 * dz)
    zeta = dt / (2.0 * dz)

    # Compute xrho for velocity scaling
    xrho = np.sqrt(RHO_00 / rho)

    # Compute velocity scale factors for each species
    vc_r = vel_scale_factor_default(xrho)
    vc_s = vel_scale_factor_snow(xrho, rho, t, qs)
    vc_i = vel_scale_factor_ice(xrho)
    vc_g = vel_scale_factor_default(xrho)

    # Compute activation masks (species present above threshold)
    mask_r = qr > QMIN
    mask_s = qs > QMIN
    mask_i = qi > QMIN
    mask_g = qg > QMIN

    # Parameters for each species
    params_list = [RAIN_PARAMS, SNOW_PARAMS, ICE_PARAMS, GRAUPEL_PARAMS]

    # Call MLIR kernel
    results = precip_scan_mlir(
        params_list=params_list,
        zeta=zeta,
        rho=rho,
        q_list=[qr, qs, qi, qg],
        vc_list=[vc_r, vc_s, vc_i, vc_g],
        mask_list=[mask_r, mask_s, mask_i, mask_g]
    )

    # Unpack results
    (qr_out, flx_rain), (qs_out, flx_snow), (qi_out, flx_ice), (qg_out, flx_graupel) = results

    # Compute precipitation rates at surface (bottom level flux)
    pr = flx_rain[-1, :]   # Rain precipitation rate
    ps = flx_snow[-1, :]   # Snow precipitation rate
    pi = flx_ice[-1, :]    # Ice precipitation rate
    pg = flx_graupel[-1, :] # Graupel precipitation rate

    # Total precipitation flux
    pflx = flx_rain + flx_snow + flx_ice + flx_graupel

    # Update temperature due to phase changes (simplified)
    # Latent heat release from precipitation
    dq_rain = qr - qr_out
    dq_snow = qs - qs_out
    dq_ice = qi - qi_out
    dq_graupel = qg - qg_out

    t_out = t.copy()
    t_out += (LVAP * dq_rain + LSUB * (dq_snow + dq_ice + dq_graupel)) / CPD

    return qr_out, qs_out, qi_out, qg_out, t_out, pflx, pr, ps, pi, pg


def graupel_run(
    nlev: int,
    dz: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    rho: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qi: np.ndarray,
    qg: np.ndarray,
    dt: float,
    qnc: float = 100.0
) -> Tuple[np.ndarray, ...]:
    """
    Main graupel microphysics driver function.

    This is the top-level entry point matching the muphys_jax interface.

    Args:
        nlev: Number of vertical levels
        dz: Layer thickness [m]
        t: Temperature [K]
        p: Pressure [Pa]
        rho: Air density [kg/m^3]
        qv: Water vapor mixing ratio [kg/kg]
        qc: Cloud water mixing ratio [kg/kg]
        qr: Rain mixing ratio [kg/kg]
        qs: Snow mixing ratio [kg/kg]
        qi: Ice mixing ratio [kg/kg]
        qg: Graupel mixing ratio [kg/kg]
        dt: Time step [s]
        qnc: Cloud droplet number concentration [m^-3]

    Returns:
        Tuple of:
        - t_out: Updated temperature
        - q_out: Stacked array [qv, qc, qr, qs, qi, qg]
        - pflx: Precipitation flux
        - pr, ps, pi, pg: Surface precipitation rates
        - pre: Total surface precipitation
    """
    if not MLIR_AVAILABLE:
        raise RuntimeError(
            f"MLIR not available: {MLIR_IMPORT_ERROR}\n"
            "Install with: pip install mlir-python-bindings"
        )

    # Note: Full microphysics would include phase transitions (q_t_update)
    # For now, we focus on precipitation effects (the scan-heavy part)
    # Full physics remains in muphys_jax

    # Run precipitation effects (MLIR-accelerated)
    qr_out, qs_out, qi_out, qg_out, t_out, pflx, pr, ps, pi, pg = precipitation_effects(
        dz=dz,
        t=t,
        rho=rho,
        qr=qr,
        qs=qs,
        qi=qi,
        qg=qg,
        dt=dt
    )

    # Total surface precipitation
    pre = pr + ps + pi + pg

    # Pack output mixing ratios
    q_out = np.stack([qv, qc, qr_out, qs_out, qi_out, qg_out], axis=0)

    return t_out, q_out, pflx, pr, ps, pi, pg, pre
