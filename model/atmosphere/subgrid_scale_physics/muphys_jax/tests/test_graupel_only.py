# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Final

import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from muphys_jax.core.definitions import Q as Q_jax
from muphys_jax.implementations.graupel_baseline import graupel_run


def _calc_dz(z: np.ndarray) -> np.ndarray:
    """Calculate layer thickness from geometric height."""
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
    name: str
    uri: str
    dt: float = 30.0
    qnc: float = 100.0

    @property
    def input_file(self) -> pathlib.Path:
        # Assumes test data is in testdata/muphys_graupel_data/{name}/
        return pathlib.Path(__file__).parent.parent.parent.parent.parent / "testdata" / "muphys_graupel_data" / self.name / "input.nc"

    @property
    def reference_file(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent.parent.parent.parent.parent / "testdata" / "muphys_graupel_data" / self.name / "reference.nc"

    def __str__(self):
        return self.name


def load_input(filename: pathlib.Path) -> dict:
    """Load input from NetCDF file."""
    with netCDF4.Dataset(filename, mode="r") as nc:
        try:
            ncells = len(nc.dimensions["cell"])
        except KeyError:
            ncells = len(nc.dimensions["ncells"])
        nlev = len(nc.dimensions["height"])

        # Calculate dz
        dz = _calc_dz(nc.variables["zg"])
        dz = np.transpose(dz)  # (height, ncells) -> (ncells, height)

        def load_var(varname: str) -> np.ndarray:
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
    """Load reference output from NetCDF file."""
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


class Experiments:
    # TODO currently on havogt's polybox
    MINI: Final = utils.MuphysExperiment(
        name="mini",
        type=utils.ExperimentType.GRAUPEL_ONLY,
        uri="https://polybox.ethz.ch/index.php/s/55oHBDxS2SiqAGN/download/mini.tar.gz",
    )
    TINY: Final = utils.MuphysExperiment(
        name="tiny",
        type=utils.ExperimentType.GRAUPEL_ONLY,
        uri="https://polybox.ethz.ch/index.php/s/5Ceop3iaWkbc7gf/download/tiny.tar.gz",
    )
    R2B05: Final = utils.MuphysExperiment(
        name="R2B05",
        type=utils.ExperimentType.GRAUPEL_ONLY,
        uri="https://polybox.ethz.ch/index.php/s/RBib8rFSEd7Eomo/download/R2B05.tar.gz",
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        Experiments.MINI,
        Experiments.TINY,
        Experiments.R2B05,
    ],
    ids=lambda exp: exp.name,
)
def test_graupel_only(experiment: utils.MuphysExperiment) -> None:
    """Test JAX graupel_baseline using GT4Py test infrastructure."""
    assert experiment.type == utils.ExperimentType.GRAUPEL_ONLY
    
    # Load input using GT4Py infrastructure (no backend needed for JAX)
    inp = common.GraupelInput.load(
        filename=experiment.input_file, allocator=None
    )

    # Convert GT4Py arrays to JAX arrays
    q_jax = Q_jax(
        v=jnp.array(np.array(inp.qv)),
        c=jnp.array(np.array(inp.qc)),
        r=jnp.array(np.array(inp.qr)),
        s=jnp.array(np.array(inp.qs)),
        i=jnp.array(np.array(inp.qi)),
        g=jnp.array(np.array(inp.qg)),
    )
    
    # Run JAX graupel implementation
    t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
        dz=jnp.array(np.array(inp.dz)),
        te=jnp.array(np.array(inp.t)),
        p=jnp.array(np.array(inp.p)),
        rho=jnp.array(np.array(inp.rho)),
        q_in=q_jax,
        dt=experiment.dt,
        qnc=experiment.qnc,
    )

    # Load reference using GT4Py infrastructure
    ref = common.GraupelOutput.load(
        filename=experiment.reference_file, allocator=None
    )

    rtol = 1e-14
    atol = 1e-16

    # Convert JAX outputs to numpy for comparison
    np.testing.assert_allclose(np.array(ref.qv), np.array(q_out.v), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.array(ref.qc), np.array(q_out.c), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.array(ref.qi), np.array(q_out.i), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.array(ref.qr), np.array(q_out.r), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.array(ref.qs), np.array(q_out.s), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.array(ref.qg), np.array(q_out.g), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.array(ref.t), np.array(t_out), atol=atol, rtol=rtol)
