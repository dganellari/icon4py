# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical constants for muphys_jax microphysics (self-contained, no GT4Py dependency)."""

# =============================================================================
# Thermodynamic constants (from t_d)
# =============================================================================

# Dry air
rd = 287.04          # [J/K/kg] gas constant
cpd = 1004.64        # [J/K/kg] specific heat at constant pressure
cvd = 717.60         # [J/K/kg] specific heat at constant volume => cpd - rd

# Water vapor
rv = 461.51          # [J/K/kg] gas constant for water vapor
cpv = 1869.46        # [J/K/kg] specific heat at constant pressure
cvv = 1407.95        # [J/K/kg] specific heat at constant volume => cpv - rv

# Liquid water
clw = 4192.6641119999995  # [J/K/kg] specific heat capacity of liquid water

# Phase changes
tmelt = 273.15       # [K] melting temperature of ice/snow
alv = 2.5008e6       # [J/kg] latent heat for vaporisation
als = 2.8345e6       # [J/kg] latent heat for sublimation
alf = 333700.0       # [J/kg] latent heat for fusion => als - alv

# =============================================================================
# Microphysics constants (from g_ct)
# =============================================================================

# Reference values
rho_00 = 1.225       # [kg/m3] reference air density

# Thresholds
qmin = 1.0e-15       # threshold for computation

# Ice properties
ci = 2108.0          # [J/K/kg] specific heat of ice
m0_ice = 1.0e-12     # [kg] initial crystal mass for cloud ice nucleation

# Snow parameters (mass-size and fall speed relations)
ams = 0.069          # Formfactor in the mass-size relation of snow particles
bms = 2.0            # Exponent in the mass-size relation of snow particles
v0s = 25.0           # prefactor in snow fall speed
v1s = 0.5            # Exponent in the terminal velocity for snow

# Temperature thresholds
tx = 3339.5
tfrz_het1 = 267.15   # [K] temperature for het. freezing of cloud water with supersat => TMELT - 6.0
tfrz_het2 = 248.15   # [K] temperature for het. freezing of cloud water => TMELT - 25.0
tfrz_hom = 236.15    # [K] temperature for hom. freezing of cloud water => TMELT - 37.0

# Invariant parts of enthalpy
lvc = 3135383.2031928  # invariant part of vaporization enthalpy => alv - (cpv - clw) * tmelt
lsc = 2899657.201      # invariant part of sublimation enthalpy => als - (cpv - ci) * tmelt