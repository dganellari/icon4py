# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Phase transition functions for muphys microphysics.
Implements all 14 phase transitions between water species.
"""

import jax.numpy as jnp

from .common import constants as const


def cloud_to_graupel(t, rho, qc, qg):
    """Conversion rate from cloud to graupel (riming)."""
    A_RIM = 4.43
    B_RIM = 0.94878
    mask = (jnp.minimum(qc, qg) > const.qmin) & (t > const.tfrz_hom)
    return jnp.where(mask, A_RIM * qc * jnp.power(qg * rho, B_RIM), 0.0)


def cloud_to_rain(t, qc, qr, nc):
    """Conversion rate from cloud to rain (autoconversion + accretion)."""
    QMIN_AC = 1.0e-6
    TAU_MAX = 0.90e0
    TAU_MIN = 1.0e-30
    A_PHI = 6.0e2
    B_PHI = 0.68e0
    C_PHI = 5.0e-5
    AC_KERNEL = 5.25e0
    X3 = 2.0e0
    X2 = 2.6e-10
    X1 = 9.44e9
    AU_KERNEL = X1 / (20.0 * X2) * (X3 + 2.0) * (X3 + 4.0) / ((X3 + 1.0) * (X3 + 1.0))

    tau = jnp.maximum(TAU_MIN, jnp.minimum(1.0 - qc / (qc + qr), TAU_MAX))
    phi = jnp.power(tau, B_PHI)
    phi = A_PHI * phi * jnp.power(1.0 - phi, 3.0)
    xau = AU_KERNEL * jnp.power(qc * qc / nc, 2.0) * (1.0 + phi / jnp.power(1.0 - tau, 2.0))
    xac = AC_KERNEL * qc * qr * jnp.power(tau / (tau + C_PHI), 4.0)

    mask = (qc > QMIN_AC) & (t > const.tfrz_hom)
    return jnp.where(mask, xau + xac, 0.0)


def cloud_to_snow(t, qc, qs, ns, lam):
    """Conversion rate from cloud to snow (riming)."""
    ECS = 0.9
    B_RIM = -(const.v1s + 3.0)
    C_RIM = 2.61 * ECS * const.v0s
    mask = (jnp.minimum(qc, qs) > const.qmin) & (t > const.tfrz_hom)
    return jnp.where(mask, C_RIM * ns * qc * jnp.power(lam, B_RIM), 0.0)


def cloud_x_ice(t, qc, qi, dt):
    """
    Cloud-ice exchange (freezing or melting).
    Positive = cloud freezes to ice.
    Negative = ice melts to cloud.
    """
    # Homogeneous freezing
    result = jnp.where((qc > const.qmin) & (t < const.tfrz_hom), qc / dt, 0.0)
    # Melting
    tmelt = const.tmelt
    result = jnp.where((qi > const.qmin) & (t > tmelt), -qi / dt, result)
    return result


def graupel_to_rain(t, p, rho, dvsw0, qg):
    """Conversion rate from graupel to rain (melting)."""
    A_MELT = const.tx - 389.5
    B_MELT = 0.6
    C1_MELT = 12.31698
    C2_MELT = 7.39441e-05
    tmelt = const.tmelt
    mask = (t > jnp.maximum(tmelt, tmelt - const.tx * dvsw0)) & (qg > const.qmin)
    return jnp.where(
        mask,
        (C1_MELT / p + C2_MELT) * (t - tmelt + A_MELT * dvsw0) * jnp.power(qg * rho, B_MELT),
        0.0,
    )


def ice_to_graupel(rho, qr, qg, qi, sticking_eff):
    """Conversion rate from ice to graupel (aggregation + collection)."""
    A_CT = 1.72
    B_CT = 0.875
    C_AGG_CT = 2.46
    B_AGG_CT = 0.94878

    # Aggregation with graupel
    result = jnp.where(
        (qi > const.qmin) & (qg > const.qmin),
        sticking_eff * qi * C_AGG_CT * jnp.power(rho * qg, B_AGG_CT),
        0.0,
    )

    # Collection by rain
    result = jnp.where(
        (qi > const.qmin) & (qr > const.qmin),
        result + A_CT * qi * jnp.power(rho * qr, B_CT),
        result,
    )
    return result


def ice_to_snow(qi, ns, lam, sticking_eff):
    """Conversion rate from ice to snow (autoconversion + aggregation)."""
    QI0 = 0.0
    C_IAU = 1.0e-3
    C_AGG = 2.61 * const.v0s
    B_AGG = -(const.v1s + 3.0)

    mask = qi > const.qmin
    return jnp.where(
        mask,
        sticking_eff
        * (C_IAU * jnp.maximum(0.0, qi - QI0) + qi * C_AGG * ns * jnp.power(lam, B_AGG)),
        0.0,
    )


def rain_to_graupel(t, rho, qc, qr, qi, qs, mi, dvsw, dt):
    """Conversion rate from rain to graupel (freezing + collection)."""
    tmelt = const.tmelt
    TFRZ_RAIN = tmelt - 2.0
    A1 = 9.95e-5
    B1 = 1.75
    C2 = 0.66
    C3 = 1.0
    C4 = 0.1
    A2 = 1.24e-3
    B2 = 1.625
    QS_CRIT = 1.0e-7

    maskinner = (dvsw + qc <= 0.0) | (qr > C4 * qc)
    mask = (qr > const.qmin) & (t < TFRZ_RAIN)

    # Immersion freezing
    result = jnp.where(
        mask & (t > const.tfrz_hom) & maskinner,
        (jnp.exp(C2 * (TFRZ_RAIN - t)) - C3) * (A1 * jnp.power(qr * rho, B1)),
        0.0,
    )

    # Homogeneous freezing
    result = jnp.where(mask & (t <= const.tfrz_hom), qr / dt, result)

    # Collection by ice
    result = jnp.where(
        (jnp.minimum(qi, qr) > const.qmin) & (qs > QS_CRIT),
        result + A2 * (qi / mi) * jnp.power(rho * qr, B2),
        result,
    )
    return result


def rain_to_vapor(t, rho, qc, qr, dvsw, dt):
    """Conversion rate from rain to vapor (evaporation)."""
    tmelt = const.tmelt
    B1_RV = 0.16667
    B2_RV = 0.55555
    C1_RV = 0.61
    C2_RV = -0.0163
    C3_RV = 1.111e-4
    A1_RV = 1.536e-3
    A2_RV = 1.0e0
    A3_RV = 19.0621e0

    tc = t - tmelt
    evap_max = (C1_RV + tc * (C2_RV + C3_RV * tc)) * (-dvsw) / dt

    mask = (qr > const.qmin) & (dvsw + qc <= 0.0)
    return jnp.where(
        mask,
        jnp.minimum(
            A1_RV
            * (A2_RV + A3_RV * jnp.power(qr * rho, B1_RV))
            * (-dvsw)
            * jnp.power(qr * rho, B2_RV),
            evap_max,
        ),
        0.0,
    )


def snow_to_graupel(t, rho, qc, qs):
    """Conversion rate from snow to graupel (riming)."""
    A_RIM_CT = 0.5
    B_RIM_CT = 0.75
    mask = (jnp.minimum(qc, qs) > const.qmin) & (t > const.tfrz_hom)
    return jnp.where(mask, A_RIM_CT * qc * jnp.power(qs * rho, B_RIM_CT), 0.0)


def snow_to_rain(t, p, rho, dvsw0, qs):
    """Conversion rate from snow to rain (melting)."""
    tmelt = const.tmelt
    C1_SR = 79.6863
    C2_SR = 0.612654e-3
    A_SR = const.tx - 389.5
    B_SR = 0.8

    mask = (t > jnp.maximum(tmelt, tmelt - const.tx * dvsw0)) & (qs > const.qmin)
    return jnp.where(
        mask, (C1_SR / p + C2_SR) * (t - tmelt + A_SR * dvsw0) * jnp.power(qs * rho, B_SR), 0.0
    )


def vapor_x_graupel(t, p, rho, qg, dvsw, dvsi, dvsw0, dt):
    """
    Vapor-graupel exchange (deposition/sublimation).
    Positive = vapor deposits on graupel.
    Negative = graupel sublimes to vapor.
    """
    tmelt = const.tmelt
    A1_VG = 0.398561
    A2_VG = -0.00152398
    A3 = 2554.99
    A4 = 2.6531e-7
    A5 = 0.153907
    A6 = -7.86703e-07
    A7 = 0.0418521
    A8 = -4.7524e-8
    B_VG = 0.6

    # Below freezing: use dvsi
    result_cold = (A1_VG + A2_VG * t + A3 / p + A4 * p) * dvsi * jnp.power(qg * rho, B_VG)

    # Above tmelt - tx*dvsw0: use minimum(0, dvsw0)
    result_warm = (A5 + A6 * p) * jnp.minimum(0.0, dvsw0) * jnp.power(qg * rho, B_VG)

    # Between: use dvsw
    result_mid = (A7 + A8 * p) * dvsw * jnp.power(qg * rho, B_VG)

    result = jnp.where(
        t < tmelt, result_cold, jnp.where(t > (tmelt - const.tx * dvsw0), result_warm, result_mid)
    )

    return jnp.where(qg > const.qmin, jnp.maximum(result, -qg / dt), 0.0)


def vapor_x_ice(qi, mi, eta, dvsi, rho, dt):
    """
    Vapor-ice exchange (deposition/sublimation).
    Positive = vapor deposits on ice.
    Negative = ice sublimes to vapor.
    """
    AMI = 130.0
    B_EXP = -0.67
    A_FACT = 4.0 * AMI ** (-1.0 / 3.0)

    result = (A_FACT * eta) * rho * qi * jnp.power(mi, B_EXP) * dvsi

    # Limit deposition/sublimation rates
    result = jnp.where(
        result > 0.0,
        jnp.minimum(result, dvsi / dt),
        jnp.maximum(jnp.maximum(result, dvsi / dt), -qi / dt),
    )

    return jnp.where(qi > const.qmin, result, 0.0)


def vapor_x_snow(t, p, rho, qs, ns, lam, eta, ice_dep, dvsw, dvsi, dvsw0, dt):
    """
    Vapor-snow exchange (deposition/sublimation).
    Positive = vapor deposits on snow.
    Negative = snow sublimes to vapor.
    """
    tmelt = const.tmelt
    NU = 1.75e-5
    A0_VS = 1.0
    A1_VS = 0.4182 * jnp.sqrt(const.v0s / NU)
    A2_VS = -(const.v1s + 1.0) / 2.0
    EPS = 1.0e-15
    QS_LIM = 1.0e-7
    CNX = 4.0
    B_VS = 0.8
    C1_VS = 31282.3
    C2_VS = 0.241897
    C3_VS = 0.28003
    C4_VS = -0.146293e-6

    # Below freezing: deposition/sublimation
    result_cold = (
        (CNX * ns * eta / rho) * (A0_VS + A1_VS * jnp.power(lam, A2_VS)) * dvsi / (lam * lam + EPS)
    )
    result_cold = jnp.where(
        (result_cold > 0.0), jnp.minimum(result_cold, dvsi / dt - ice_dep), result_cold
    )
    result_cold = jnp.where((qs <= QS_LIM), jnp.minimum(result_cold, 0.0), result_cold)

    # Above tmelt - tx*dvsw0: melting with dvsw0
    result_warm = (C1_VS / p + C2_VS) * jnp.minimum(0.0, dvsw0) * jnp.power(qs * rho, B_VS)

    # Between: use dvsw
    result_mid = (C3_VS + C4_VS * p) * dvsw * jnp.power(qs * rho, B_VS)

    result = jnp.where(
        t < tmelt, result_cold, jnp.where(t > (tmelt - const.tx * dvsw0), result_warm, result_mid)
    )

    return jnp.where(qs > const.qmin, jnp.maximum(result, -qs / dt), 0.0)
