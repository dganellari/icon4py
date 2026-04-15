# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Load graupel input/reference data from NetCDF files."""

from __future__ import annotations

import pathlib
from typing import Any

import jax.numpy as jnp
import netCDF4
import numpy as np

from ..core.definitions import Q


def calc_dz(z: np.ndarray) -> np.ndarray:
    """
    Calculate layer thickness from geometric height.

    Args:
        z: Geometric height array of shape (nlev, ncells), must be float64

    Returns:
        Layer thickness array of shape (nlev, ncells)
    """
    ksize = z.shape[0]
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


def load_graupel_inputs(
    input_file: str | pathlib.Path,
    timestep: int = 0,
    as_jax: bool = True,
) -> tuple[Any, Any, Any, Any, Q, float, float, int, int]:
    """
    Load graupel inputs from NetCDF file.

    Args:
        input_file: Path to input NetCDF file
        timestep: Time index to load (default: 0)
        as_jax: If True, return JAX arrays; if False, return numpy arrays

    Returns:
        Tuple of (dz, t, p, rho, q, dt, qnc, ncells, nlev)
        where q is a Q namedtuple with species mixing ratios
    """
    ds = netCDF4.Dataset(input_file, "r")

    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])

    # Calculate dz from geometric height
    # Must be float64 before dz calculation or we lose precision
    zg = np.asarray(ds.variables["zg"]).astype(np.float64)
    dz_calc = calc_dz(zg)
    dz_transposed = np.transpose(dz_calc)  # (height, ncells) -> (ncells, height)

    def load_var(varname: str) -> np.ndarray:
        """Load and transpose a variable to (ncells, nlev) layout."""
        var = ds.variables[varname]
        if var.dimensions[0] == "time":
            var = var[timestep, :, :]
        return np.transpose(var).astype(np.float64)

    # Load all variables as numpy float64
    t = load_var("ta")
    p = load_var("pfull")
    rho = load_var("rho")
    qv = load_var("hus")
    qc = load_var("clw")
    qr = load_var("qr")
    qs = load_var("qs")
    qi = load_var("cli")
    qg = load_var("qg")

    ds.close()

    dt = 30.0
    qnc = 100.0

    if as_jax:
        dz = jnp.array(dz_transposed, dtype=jnp.float64)
        t = jnp.array(t, dtype=jnp.float64)
        p = jnp.array(p, dtype=jnp.float64)
        rho = jnp.array(rho, dtype=jnp.float64)
        q = Q(
            v=jnp.array(qv, dtype=jnp.float64),
            c=jnp.array(qc, dtype=jnp.float64),
            r=jnp.array(qr, dtype=jnp.float64),
            s=jnp.array(qs, dtype=jnp.float64),
            i=jnp.array(qi, dtype=jnp.float64),
            g=jnp.array(qg, dtype=jnp.float64),
        )
    else:
        dz = dz_transposed
        q = Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg)

    return dz, t, p, rho, q, dt, qnc, ncells, nlev


def load_graupel_reference(reference_file: str | pathlib.Path) -> dict[str, np.ndarray]:
    """
    Load reference output from NetCDF file.

    Args:
        reference_file: Path to reference NetCDF file

    Returns:
        Dictionary with reference fields as numpy arrays of shape (ncells, nlev):
        - t: Temperature
        - qv: Water vapor mixing ratio
        - qc: Cloud water mixing ratio
        - qi: Ice mixing ratio
        - qr: Rain mixing ratio
        - qs: Snow mixing ratio
        - qg: Graupel mixing ratio
    """
    with netCDF4.Dataset(reference_file, mode="r") as nc:
        return {
            "t": np.array(nc.variables["ta"][:], dtype=np.float64).T,
            "qv": np.array(nc.variables["hus"][:], dtype=np.float64).T,
            "qc": np.array(nc.variables["clw"][:], dtype=np.float64).T,
            "qi": np.array(nc.variables["cli"][:], dtype=np.float64).T,
            "qr": np.array(nc.variables["qr"][:], dtype=np.float64).T,
            "qs": np.array(nc.variables["qs"][:], dtype=np.float64).T,
            "qg": np.array(nc.variables["qg"][:], dtype=np.float64).T,
        }


def load_precip_inputs(
    input_file: str | pathlib.Path,
    timestep: int = 0,
) -> tuple[
    int,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Q,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    float,
    int,
    int,
]:
    """
    Load inputs specifically for precipitation_effects function.

    Args:
        input_file: Path to input NetCDF file
        timestep: Time index to load (default: 0)

    Returns:
        Tuple of (last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev)
    """
    from ..core.common import constants as const

    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(input_file, timestep, as_jax=True)

    last_lev = nlev - 1

    # Compute kmin masks (species present above threshold)
    kmin_r = q.r > const.qmin
    kmin_i = q.i > const.qmin
    kmin_s = q.s > const.qmin
    kmin_g = q.g > const.qmin

    return last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev


__all__ = [
    "calc_dz",
    "load_graupel_inputs",
    "load_graupel_reference",
    "load_precip_inputs",
]
