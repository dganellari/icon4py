# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test for JAX graupel with HLO injection.

Same as: python run_graupel_optimized.py --input data.nc --optimized-hlo shlo/precip.hlo --mode baseline
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import tempfile
from typing import Final


# Configure JAX before any JAX imports
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


# Add tools directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tools"))
from export_precip_transposed import export_precip_transposed_hlo
from muphys_jax.utils.data_loading import load_graupel_reference
from run_graupel_optimized import run_graupel


@dataclasses.dataclass(frozen=True)
class MuphysGraupelExperiment:
    """Configuration for a graupel microphysics test experiment."""

    name: str
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
    MuphysGraupelExperiment("mini"),
    MuphysGraupelExperiment("R2B05"),
]


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", GRAUPEL_EXPERIMENTS, ids=str)
def test_graupel_hlo_injection(experiment: MuphysGraupelExperiment):
    """
    Test full graupel with HLO injection.

    Same as: python run_graupel_optimized.py --optimized-hlo <hlo> --mode baseline
    """
    if not experiment.input_file.exists():
        pytest.skip(f"Input file not found: {experiment.input_file}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Generate StableHLO
        print(f"\n[1/2] Generating StableHLO for {experiment.name}...")
        stablehlo_file, hlo_file = export_precip_transposed_hlo(
            input_file=str(experiment.input_file),
            skip_compile=True,
            output_dir=tmpdir,
        )

        # Step 2: Run full graupel with HLO injection (same as --optimized-hlo)
        print(f"\n[2/2] Running full graupel with HLO injection for {experiment.name}...")
        mean_time, std_time, result = run_graupel(
            mode="baseline",
            optimized_hlo=stablehlo_file,
            input_file=str(experiment.input_file),
            num_warmup=1,
            num_runs=1,
        )

        # Unpack result
        t_out, q_out, pflx, pr, ps, pi, pg, pre = result

        # Load reference using shared utility
        ref = load_graupel_reference(experiment.reference_file)

        # Compare (same tolerance as standalone test)
        rtol = 1e-14
        atol = 1e-16

        np.testing.assert_allclose(
            np.array(t_out), ref["t"], rtol=rtol, atol=atol, err_msg="Temperature mismatch"
        )
        np.testing.assert_allclose(
            np.array(q_out.v), ref["qv"], rtol=rtol, atol=atol, err_msg="Water vapor mismatch"
        )
        np.testing.assert_allclose(
            np.array(q_out.c), ref["qc"], rtol=rtol, atol=atol, err_msg="Cloud water mismatch"
        )
        np.testing.assert_allclose(
            np.array(q_out.i), ref["qi"], rtol=rtol, atol=atol, err_msg="Ice mismatch"
        )
        np.testing.assert_allclose(
            np.array(q_out.r), ref["qr"], rtol=rtol, atol=atol, err_msg="Rain mismatch"
        )
        np.testing.assert_allclose(
            np.array(q_out.s), ref["qs"], rtol=rtol, atol=atol, err_msg="Snow mismatch"
        )
        np.testing.assert_allclose(
            np.array(q_out.g), ref["qg"], rtol=rtol, atol=atol, err_msg="Graupel mismatch"
        )

        print(f"\n✓ Full graupel with HLO injection matches reference for {experiment.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
