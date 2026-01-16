# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Backend configuration for JAX muphys implementation.
"""

import jax

# Use JAX JIT compilation
jit_compile = jax.jit

__all__ = ['jit_compile']
