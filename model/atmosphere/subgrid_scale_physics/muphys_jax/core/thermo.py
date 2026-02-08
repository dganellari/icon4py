# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Thermodynamic functions for muphys microphysics.
Saturation vapor pressure and internal energy calculations.
"""

import jax.numpy as jnp

from .common import constants as const


def qsat_rho(t, rho):
    """Saturation specific humidity over liquid water."""
    C1ES = 610.78
    C3LES = 17.269
    C4LES = 35.86

    return (C1ES * jnp.exp(C3LES * (t - const.tmelt) / (t - C4LES))) / (rho * const.rv * t)


def qsat_ice_rho(t, rho):
    """Saturation specific humidity over ice."""
    C1ES = 610.78
    C3IES = 21.875
    C4IES = 7.66

    return (C1ES * jnp.exp(C3IES * (t - const.tmelt) / (t - C4IES))) / (rho * const.rv * t)


def qsat_rho_tmelt(rho):
    """Saturation specific humidity at melting temperature."""
    C1ES = 610.78
    return C1ES / (rho * const.rv * const.tmelt)


def internal_energy(t, qv, qliq, qice, rho, dz):
    """Compute internal energy from temperature."""
    qtot = qliq + qice + qv
    cv = const.cvd * (1.0 - qtot) + const.cvv * qv + const.clw * qliq + const.ci * qice

    return rho * dz * (cv * t - qliq * const.lvc - qice * const.lsc)


__all__ = [
    "internal_energy",
    "qsat_ice_rho",
    "qsat_rho",
    "qsat_rho_tmelt",
]
