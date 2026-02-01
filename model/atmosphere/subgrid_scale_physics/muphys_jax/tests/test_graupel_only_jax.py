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

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from icon4py.model.testing import data_handling, definitions as testing_defs

from muphys_jax.core.definitions import Q
from muphys_jax.driver.run_graupel_jax import GraupelInput
from muphys_jax.implementations.graupel_baseline import graupel_run


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
        # Convert to float64 to match computation dtype
        return {
            "t": np.array(nc.variables["ta"][:], dtype=np.float64).T,
            "qv": np.array(nc.variables["hus"][:], dtype=np.float64).T,
            "qc": np.array(nc.variables["clw"][:], dtype=np.float64).T,
            "qi": np.array(nc.variables["cli"][:], dtype=np.float64).T,
            "qr": np.array(nc.variables["qr"][:], dtype=np.float64).T,
            "qs": np.array(nc.variables["qs"][:], dtype=np.float64).T,
            "qg": np.array(nc.variables["qg"][:], dtype=np.float64).T,
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
    rtol = 1e-14
    atol = 1e-16

    # Compare outputs
    np.testing.assert_allclose(ref["qv"], np.array(q_out.v), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qc"], np.array(q_out.c), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qi"], np.array(q_out.i), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qr"], np.array(q_out.r), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qs"], np.array(q_out.s), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["qg"], np.array(q_out.g), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref["t"], np.array(t_out), atol=atol, rtol=rtol)


# @pytest.mark.datatest
# @pytest.mark.parametrize(
#     "experiment",
#     [
#         Experiments.MINI,
#         Experiments.TINY,
#         Experiments.R2B05,
#     ],
#     ids=lambda exp: exp.name,
# )
# def test_graupel_jax_unrolled(experiment: MuphysGraupelExperiment) -> None:
#     """Test JAX graupel with use_unrolled=True produces same results as baseline.

#     This test verifies that the unrolled optimization produces identical results
#     to the lax.scan-based implementation (not against external reference data).
#     """
#     # Load input
#     inp = GraupelInput.load(experiment.input_file)

#     # Run baseline (lax.scan)
#     t_baseline, q_baseline, pflx_baseline, pr_b, ps_b, pi_b, pg_b, pre_b = graupel_run(
#         inp.dz, inp.t, inp.p, inp.rho, inp.q, experiment.dt, experiment.qnc,
#         use_unrolled=False,
#     )

#     # Run unrolled (Python loop unrolled at trace time)
#     t_unrolled, q_unrolled, pflx_unrolled, pr_u, ps_u, pi_u, pg_u, pre_u = graupel_run(
#         inp.dz, inp.t, inp.p, inp.rho, inp.q, experiment.dt, experiment.qnc,
#         use_unrolled=True,
#     )

#     # Both should produce identical results (same algorithm, just different trace)
#     # Use tight tolerances since this is the same JAX code
#     rtol = 1e-14 if experiment.dtype == np.float64 else 1e-6
#     atol = 1e-14 if experiment.dtype == np.float64 else 1e-7

#     # Compare outputs
#     np.testing.assert_allclose(np.array(q_baseline.v), np.array(q_unrolled.v), atol=atol, rtol=rtol)
#     np.testing.assert_allclose(np.array(q_baseline.c), np.array(q_unrolled.c), atol=atol, rtol=rtol)
#     np.testing.assert_allclose(np.array(q_baseline.i), np.array(q_unrolled.i), atol=atol, rtol=rtol)
#     np.testing.assert_allclose(np.array(q_baseline.r), np.array(q_unrolled.r), atol=atol, rtol=rtol)
#     np.testing.assert_allclose(np.array(q_baseline.s), np.array(q_unrolled.s), atol=atol, rtol=rtol)
#     np.testing.assert_allclose(np.array(q_baseline.g), np.array(q_unrolled.g), atol=atol, rtol=rtol)
#     np.testing.assert_allclose(np.array(t_baseline), np.array(t_unrolled), atol=atol, rtol=rtol)