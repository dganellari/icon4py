# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Physical constants for muphys microphysics."""

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.frozen import g_ct, t_d

# Thermodynamic constants (from t_d)
rd = t_d.rd
rv = t_d.rv
cvd = t_d.cvd
cvv = t_d.cvv
clw = t_d.clw
tmelt = t_d.tmelt
als = t_d.als

# Microphysics constants (from g_ct)
ci = g_ct.ci
qmin = g_ct.qmin
rho_00 = g_ct.rho_00
ams = g_ct.ams
bms = g_ct.bms
v0s = g_ct.v0s
v1s = g_ct.v1s
m0_ice = g_ct.m0_ice
tx = g_ct.tx
tfrz_het1 = g_ct.tfrz_het1
tfrz_het2 = g_ct.tfrz_het2
tfrz_hom = g_ct.tfrz_hom
lvc = g_ct.lvc
lsc = g_ct.lsc