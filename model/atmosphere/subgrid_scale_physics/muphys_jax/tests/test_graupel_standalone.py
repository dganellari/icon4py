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
import numpy as np
import pytest

from muphys_jax.implementations.graupel_baseline import graupel_run
from muphys_jax.utils.data_loading import load_graupel_inputs, load_graupel_reference


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
        base = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent
        return base / "testdata" / "muphys" / "graupel_only" / self.name / "input.nc"

    @property
    def reference_file(self) -> pathlib.Path:
        """Path to reference output NetCDF file."""
        base = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent
        return base / "testdata" / "muphys_graupel_data" / self.name / "reference.nc"

    def __str__(self):
        return self.name


GRAUPEL_EXPERIMENTS: Final = [
    MuphysGraupelExperiment("mini", "https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=mini.tar.gz"),
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
    # Load input data using shared utility
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(experiment.input_file)

    # Run the JAX graupel implementation
    t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
        dz=dz,
        te=t,
        p=p,
        rho=rho,
        q_in=q,
        dt=jnp.array(experiment.dt),
        qnc=jnp.array(experiment.qnc),
    )

    # Load reference data using shared utility
    ref = load_graupel_reference(experiment.reference_file)

    # Tolerance for comparison (matching GT4Py test)
    rtol = 1e-14
    atol = 1e-16

    # Verify results
    np.testing.assert_allclose(np.array(t_out), ref["t"], rtol=rtol, atol=atol, err_msg="Temperature mismatch")
    np.testing.assert_allclose(np.array(q_out.v), ref["qv"], rtol=rtol, atol=atol, err_msg="Water vapor mismatch")
    np.testing.assert_allclose(np.array(q_out.c), ref["qc"], rtol=rtol, atol=atol, err_msg="Cloud water mismatch")
    np.testing.assert_allclose(np.array(q_out.i), ref["qi"], rtol=rtol, atol=atol, err_msg="Ice mismatch")
    np.testing.assert_allclose(np.array(q_out.r), ref["qr"], rtol=rtol, atol=atol, err_msg="Rain mismatch")
    np.testing.assert_allclose(np.array(q_out.s), ref["qs"], rtol=rtol, atol=atol, err_msg="Snow mismatch")
    np.testing.assert_allclose(np.array(q_out.g), ref["qg"], rtol=rtol, atol=atol, err_msg="Graupel mismatch")
