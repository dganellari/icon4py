# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Property functions for muphys microphysics.
"""

import jax.numpy as jnp
from .common import constants as const


def snow_number(t, rho, qs):
    """Compute snow number concentration."""
    TMIN = const.tmelt - 40.0
    TMAX = const.tmelt
    QSMIN = 2.0e-6
    XA1 = -1.65e0
    XA2 = 5.45e-2
    XA3 = 3.27e-4
    XB1 = 1.42e0
    XB2 = 1.19e-2
    XB3 = 9.60e-5
    N0S0 = 8.00e5
    N0S1 = 13.5 * 5.65e05
    N0S2 = -0.107
    N0S3 = 13.5
    N0S4 = 0.5 * N0S1
    N0S5 = 1.0e6
    N0S6 = 1.0e2 * N0S1
    N0S7 = 1.0e9

    tc = jnp.maximum(jnp.minimum(t, TMAX), TMIN) - const.tmelt
    alf = jnp.power(10.0, (XA1 + tc * (XA2 + tc * XA3)))
    bet = XB1 + tc * (XB2 + tc * XB3)
    n0s = N0S3 * jnp.power(((qs + QSMIN) * rho / const.ams), (4.0 - 3.0 * bet)) / (alf * alf * alf)
    y = jnp.exp(N0S2 * tc)
    n0smn = jnp.maximum(N0S4 * y, N0S5)
    n0smx = jnp.minimum(N0S6 * y, N0S7)

    return jnp.where(qs > const.qmin, jnp.minimum(n0smx, jnp.maximum(n0smn, n0s)), N0S0)


def snow_lambda(rho, qs, ns):
    """Compute snow slope parameter (lambda)."""
    A2 = const.ams * 2.0
    LMD_0 = 1.0e10
    BX = 1.0 / (const.bms + 1.0)
    QSMIN = 0.0e-6

    return jnp.where(qs > const.qmin, jnp.power((A2 * ns / ((qs + QSMIN) * rho)), BX), LMD_0)


def ice_number(t, rho):
    """Calculate ice number concentration."""
    A_COOP = 5.000
    B_COOP = 0.304
    NIMAX = 250.0e3

    return jnp.minimum(NIMAX, A_COOP * jnp.exp(B_COOP * (const.tmelt - t))) / rho


def ice_mass(qi, ni):
    """Compute ice crystal mass."""
    MI_MAX = 1.0e-9
    return jnp.maximum(const.m0_ice, jnp.minimum(qi / ni, MI_MAX))


def ice_sticking(t):
    """Compute ice sticking efficiency."""
    A_FREEZ = 0.09
    B_MAX_EXP = 1.00
    EFF_MIN = 0.075
    EFF_FAC = 3.5e-3
    TCRIT = const.tmelt - 85.0

    return jnp.maximum(
        jnp.maximum(jnp.minimum(jnp.exp(A_FREEZ * (t - const.tmelt)), B_MAX_EXP), EFF_MIN),
        EFF_FAC * (t - TCRIT)
    )


def deposition_factor(t, qvsi):
    """Compute deposition factor for ice/snow growth."""
    KAPPA = 2.40e-2
    B = 1.94
    A = const.als * const.als / (KAPPA * const.rv)
    CX = 2.22e-5 * jnp.power(const.tmelt, (-B)) * 101325.0

    x = CX / const.rd * jnp.power(t, B - 1.0)
    return (CX / const.rd * jnp.power(t, B - 1.0)) / (1.0 + A * x * qvsi / (t * t))


def deposition_auto_conversion(qi, m_ice, ice_dep):
    """Automatic conversion of deposited ice to snow."""
    M0_S = 3.0e-9
    B_DEP = 0.666666666666666667
    XCRIT = 1.0

    return jnp.where(
        qi > const.qmin,
        jnp.maximum(0.0, ice_dep) * B_DEP / (jnp.power((M0_S / m_ice), B_DEP) - XCRIT),
        0.0
    )


def ice_deposition_nucleation(t, qc, qi, ni, dvsi, dt):
    """Vapor deposition for new ice nucleation."""
    return jnp.where(
        (qi <= const.qmin) & (((t < const.tfrz_het2) & (dvsi > 0.0)) | ((t <= const.tfrz_het1) & (qc > const.qmin))),
        jnp.minimum(const.m0_ice * ni, jnp.maximum(0.0, dvsi)) / dt,
        0.0
    )


def vel_scale_factor_default(xrho):
    """Default velocity scale factor."""
    return xrho


def vel_scale_factor_ice(xrho):
    """Velocity scale factor for ice."""
    B_I = 0.66666666666666667
    return jnp.power(xrho, B_I)


def vel_scale_factor_snow(xrho, rho, t, qs):
    """Velocity scale factor for snow."""
    B_S = -0.16666666666666667
    return xrho * jnp.power(snow_number(t, rho, qs), B_S)


__all__ = [
    'snow_number',
    'snow_lambda',
    'ice_number',
    'ice_mass',
    'ice_sticking',
    'deposition_factor',
    'deposition_auto_conversion',
    'ice_deposition_nucleation',
    'vel_scale_factor_default',
    'vel_scale_factor_ice',
    'vel_scale_factor_snow',
]
