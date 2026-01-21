# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration test for JAX graupel implementation against GT4Py reference data.
Uses the same test data as the GT4Py test_graupel_only.py.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Final

import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from icon4py.model.testing import data_handling, definitions as testing_defs

from muphys_jax.core.definitions import Q
from muphys_jax.driver.run_graupel_jax import GraupelInput
from muphys_jax.implementations.graupel import graupel_run


def _path_to_experiment_testdata(experiment: MuphysGraupelExperiment) -> pathlib.Path:
    return testing_defs.get_test_data_root_path() / "muphys_graupel_data" / experiment.name


@dataclasses.dataclass(frozen=True)
class MuphysGraupelExperiment:
    name: str
    uri: str
    dtype: np.dtype
    dt: float = 30.0
    qnc: float = 100.0

    @property
    def input_file(self) -> pathlib.Path:
        return _path_to_experiment_testdata(self) / "input.nc"

    @property
    def reference_file(self) -> pathlib.Path:
        return _path_to_experiment_testdata(self) / "reference.nc"

    def __str__(self):
        return self.name


class Experiments:
    MINI: Final = MuphysGraupelExperiment(
        name="mini",
        uri="https://polybox.ethz.ch/index.php/s/55oHBDxS2SiqAGN/download/mini.tar.gz",
        dtype=np.float32,
    )
    TINY: Final = MuphysGraupelExperiment(
        name="tiny",
        uri="https://polybox.ethz.ch/index.php/s/5Ceop3iaWkbc7gf/download/tiny.tar.gz",
        dtype=np.float64,
    )
    R2B05: Final = MuphysGraupelExperiment(
        name="R2B05",
        uri="https://polybox.ethz.ch/index.php/s/RBib8rFSEd7Eomo/download/R2B05.tar.gz",
        dtype=np.float32,
    )


def load_reference(filename: pathlib.Path) -> dict:
    """Load reference output from NetCDF file."""
    with netCDF4.Dataset(filename, mode="r") as nc:
        # Transpose from (height, ncells) to (ncells, height)
        return {
            "t": np.array(nc.variables["ta"][:]).T,
            "qv": np.array(nc.variables["hus"][:]).T,
            "qc": np.array(nc.variables["clw"][:]).T,
            "qi": np.array(nc.variables["cli"][:]).T,
            "qr": np.array(nc.variables["qr"][:]).T,
            "qs": np.array(nc.variables["qs"][:]).T,
            "qg": np.array(nc.variables["qg"][:]).T,
        }


@pytest.fixture(autouse=True)
def download_test_data(experiment: MuphysGraupelExperiment) -> None:
    """Downloads test data for an experiment (implicit fixture)."""
    data_handling.download_test_data(_path_to_experiment_testdata(experiment), uri=experiment.uri)


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
def test_graupel_jax(experiment: MuphysGraupelExperiment) -> None:
    """Test JAX graupel against reference data."""
    # Load input
    inp = GraupelInput.load(experiment.input_file)

    # Run JAX graupel
    t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
        inp.dz, inp.t, inp.p, inp.rho, inp.q, experiment.dt, experiment.qnc
    )

    # Load reference
    ref = load_reference(experiment.reference_file)

    # Set tolerances (same as GT4Py test_graupel_only.py)
    rtol = 1e-14 if experiment.dtype == np.float64 else 1e-7
    atol = 1e-16 if experiment.dtype == np.float64 else 1e-8

    # Compare outputs
    np.testing.assert_allclose(ref["qv"], np.array(q_out.v), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qc"], np.array(q_out.c), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qi"], np.array(q_out.i), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qr"], np.array(q_out.r), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qs"], np.array(q_out.s), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qg"], np.array(q_out.g), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["t"], np.array(t_out), atol=atol, rtol=rtol)