# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
MLIR-based implementation of ICON microphysics.

This module provides a low-level MLIR implementation of graupel microphysics,
targeting GPU execution via MLIR's GPU dialect for optimal performance.

Key optimizations:
- Carry state in SSA values (registers) - no D2D memory copies
- All 4 precipitation species processed in single kernel launch
- Branchless execution with arith.select (no warp divergence)
- Static memref shapes for better LLVM optimization
- Memory coalescing: threads access contiguous cells

Usage:
    # As standalone driver
    python -m muphys_mlir.driver.run_graupel_mlir -o output.nc input.nc 10 30.0

    # Or import kernel for use in muphys_jax
    from muphys_mlir import precip_scan_mlir, MLIR_AVAILABLE
"""

__version__ = "0.1.0"

from .core.precip_scans_mlir import (
    MLIR_AVAILABLE,
    MLIR_IMPORT_ERROR,
    generate_precip_scan_mlir,
    precip_scan_mlir,
)
from .implementations.graupel import graupel_run, precipitation_effects

__all__ = [
    "MLIR_AVAILABLE",
    "MLIR_IMPORT_ERROR",
    "generate_precip_scan_mlir",
    "precip_scan_mlir",
    "graupel_run",
    "precipitation_effects",
]
