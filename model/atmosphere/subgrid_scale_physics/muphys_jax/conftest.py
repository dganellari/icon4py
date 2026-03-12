# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Pytest configuration for muphys_jax tests.

This file is loaded before test collection, ensuring JAX x64 mode
is enabled before any test modules import JAX.
"""

import jax


# Enable x64 precision for all tests
# This must be set before any JAX operations or test imports
jax.config.update("jax_enable_x64", True)
