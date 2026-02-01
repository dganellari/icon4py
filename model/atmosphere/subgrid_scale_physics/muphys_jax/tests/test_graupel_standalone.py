# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone test for JAX graupel implementation.
This test is independent of the GT4Py muphys infrastructure.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Final

# Configure JAX before any JAX imports
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from muphys_jax.core.definitions import Q as Q_jax
from muphys_jax.implementations.graupel_baseline import graupel_run


def _calc_dz(z: np.ndarray) -> np.ndarray:
    """
    Calculate layer thickness from geometric height.
    
    Args:
        z: Geometric height array of shape (nlev, ncells)
    
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


@dataclasses.dataclass(frozen=True)
class MuphysGraupelExperiment:
    """Configuration for a graupel microphysics test experiment."""
    name: str
    uri: str = ""
    dt: float = 30.0
    qnc: float = 100.0

    @property
    def input_file(self) -> pathlib.Path:
        """Path to input NetCDF file."""
        # Match GT4Py test path structure: testdata/muphys/graupel_only/{name}/
        # From muphys_jax/tests/ go up to icon4py root: ../../../../../..
        base = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent
        return base / "testdata" / "muphys" / "graupel_only" / self.name / "input.nc"

    @property
    def reference_file(self) -> pathlib.Path:
        """Path to reference output NetCDF file."""
        # Match GT4Py test path structure: testdata/muphys/graupel_only/{name}/
        # From muphys_jax/tests/ go up to icon4py root: ../../../../../..
        base = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent
        return base / "testdata" / "muphys_graupel_data" / self.name / "reference.nc"

    def __str__(self):
        return self.name


def load_input(filename: pathlib.Path) -> dict:
    """
    Load input data from NetCDF file.
    
    Args:
        filename: Path to input NetCDF file
    
    Returns:
        Dictionary with input fields as numpy arrays of shape (ncells, nlev)
    """
    with netCDF4.Dataset(filename, mode="r") as nc:
        try:
            ncells = len(nc.dimensions["cell"])
        except KeyError:
            ncells = len(nc.dimensions["ncells"])
        nlev = len(nc.dimensions["height"])

        # Calculate dz from geometric height (ensure float64 like GT4Py)
        zg = np.asarray(nc.variables["zg"]).astype(np.float64)
        dz = _calc_dz(zg)
        dz = np.transpose(dz)  # (height, ncells) -> (ncells, height)

        def load_var(varname: str) -> np.ndarray:
            """Load and transpose a variable."""
            var = nc.variables[varname]
            if var.dimensions[0] == "time":
                var = var[0, :, :]
            return np.transpose(var).astype(np.float64)

        return {
            "ncells": ncells,
            "nlev": nlev,
            "dz": dz,
            "t": load_var("ta"),
            "p": load_var("pfull"),
            "rho": load_var("rho"),
            "qv": load_var("hus"),
            "qc": load_var("clw"),
            "qr": load_var("qr"),
            "qs": load_var("qs"),
            "qi": load_var("cli"),
            "qg": load_var("qg"),
        }


def load_reference(filename: pathlib.Path) -> dict:
    """
    Load reference output from NetCDF file.
    
    Args:
        filename: Path to reference NetCDF file
    
    Returns:
        Dictionary with reference fields as numpy arrays of shape (ncells, nlev)
    """
    with netCDF4.Dataset(filename, mode="r") as nc:
        return {
            "t": np.array(nc.variables["ta"][:], dtype=np.float64).T,
            "qv": np.array(nc.variables["hus"][:], dtype=np.float64).T,
            "qc": np.array(nc.variables["clw"][:], dtype=np.float64).T,
            "qi": np.array(nc.variables["cli"][:], dtype=np.float64).T,
            "qr": np.array(nc.variables["qr"][:], dtype=np.float64).T,
            "qs": np.array(nc.variables["qs"][:], dtype=np.float64).T,
            "qg": np.array(nc.variables["qg"][:], dtype=np.float64).T,
        }


GRAUPEL_EXPERIMENTS: Final = [
    MuphysGraupelExperiment("mini", "https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=mini.tar.gz"),
    # MuphysGraupelExperiment("tiny", "https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=tiny.tar.gz"),  # incomplete data
    MuphysGraupelExperiment("R2B05", "https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=R2B05.tar.gz"),
]


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", GRAUPEL_EXPERIMENTS, ids=str)
def test_graupel_run_jax(experiment: MuphysGraupelExperiment):
    """
    Test the JAX implementation of the Graupel microphysics scheme.
    
    This test loads NetCDF input/reference data directly and compares
    the JAX implementation output against reference values.
    """

    # Load input data using NetCDF directly
    inp = load_input(experiment.input_file)
    
    # Convert to JAX arrays
    q_jax = Q_jax(
        v=jnp.array(inp["qv"]),
        c=jnp.array(inp["qc"]),
        i=jnp.array(inp["qi"]),
        r=jnp.array(inp["qr"]),
        s=jnp.array(inp["qs"]),
        g=jnp.array(inp["qg"]),
    )

    # Run the JAX graupel implementation
    t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
        dz=jnp.array(inp["dz"]),
        te=jnp.array(inp["t"]),
        p=jnp.array(inp["p"]),
        rho=jnp.array(inp["rho"]),
        q_in=q_jax,
        dt=jnp.array(experiment.dt),
        qnc=jnp.array(experiment.qnc),
    )

    # Load reference data
    ref = load_reference(experiment.reference_file)

    # Convert output back to numpy for comparison
    t_out_np = np.array(t_out)
    qv_out_np = np.array(q_out.v)
    qc_out_np = np.array(q_out.c)
    qi_out_np = np.array(q_out.i)
    qr_out_np = np.array(q_out.r)
    qs_out_np = np.array(q_out.s)
    qg_out_np = np.array(q_out.g)

    # Tolerance for comparison (matching GT4Py test)
    rtol = 1e-14
    atol = 1e-16

    # Verify results
    np.testing.assert_allclose(t_out_np, ref["t"], rtol=rtol, atol=atol, err_msg="Temperature mismatch")
    np.testing.assert_allclose(qv_out_np, ref["qv"], rtol=rtol, atol=atol, err_msg="Water vapor mismatch")
    np.testing.assert_allclose(qc_out_np, ref["qc"], rtol=rtol, atol=atol, err_msg="Cloud water mismatch")
    np.testing.assert_allclose(qi_out_np, ref["qi"], rtol=rtol, atol=atol, err_msg="Ice mismatch")
    np.testing.assert_allclose(qr_out_np, ref["qr"], rtol=rtol, atol=atol, err_msg="Rain mismatch")
    np.testing.assert_allclose(qs_out_np, ref["qs"], rtol=rtol, atol=atol, err_msg="Snow mismatch")
    np.testing.assert_allclose(qg_out_np, ref["qg"], rtol=rtol, atol=atol, err_msg="Graupel mismatch")
