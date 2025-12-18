# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Physical constants for muphys microphysics.
"""

# Microphysics constants (g_ct)
rho_00 = 1.225  # reference air density
q1 = 8.0e-6
qmin = 1.0e-15  # threshold for computation
ams = 0.069  # Formfactor in the mass-size relation of snow particles
bms = 2.0  # Exponent in the mass-size relation of snow particles
v0s = 25.0  # prefactor in snow fall speed
v1s = 0.5  # Exponent in the terminal velocity for snow
m0_ice = 1.0e-12  # initial crystal mass for cloud ice nucleation
ci = 2108.0  # specific heat of ice
tx = 3339.5
tfrz_het1 = 267.15  # temperature for het. freezing of cloud water with supersat
tfrz_het2 = 248.15  # temperature for het. freezing of cloud water
tfrz_hom = 236.15  # temperature for hom. freezing of cloud water
lvc = 3135383.2031928  # invariant part of vaporization enthalpy
lsc = 2899657.201  # invariant part of sublimation enthalpy

# Thermodynamic constants (t_d)
rd = 287.04  # [J/K/kg] gas constant for dry air
cpd = 1004.64  # [J/K/kg] specific heat at constant pressure
cvd = 717.60  # [J/K/kg] specific heat at constant volume
con_m = 1.50e-5  # [m^2/s] kinematic viscosity of dry air
con_h = 2.20e-5  # [m^2/s] scalar conductivity of dry air
con0_h = 2.40e-2  # [J/m/s/K] thermal conductivity of dry air
eta0d = 1.717e-5  # [N*s/m2] dyn viscosity of dry air at tmelt
rv = 461.51  # [J/K/kg] gas constant for water vapor
cpv = 1869.46  # [J/K/kg] specific heat at constant pressure
cvv = 1407.95  # [J/K/kg] specific heat at constant volume
dv0 = 2.22e-5  # [m^2/s] diff coeff of H2O vapor in dry air at tmelt
rhoh2o = 1000.0  # [kg/m3] density of liquid water
rhoice = 916.7  # [kg/m3] density of pure ice
cv_i = 2000.0

# Phase changes
alv = 2.5008e6  # [J/kg] latent heat for vaporisation
als = 2.8345e6  # [J/kg] latent heat for sublimation
alf = 333700.0  # [J/kg] latent heat for fusion
tmelt = 273.15  # [K] melting temperature of ice/snow
clw = 4192.6641119999995  # specific heat capacity of liquid water
