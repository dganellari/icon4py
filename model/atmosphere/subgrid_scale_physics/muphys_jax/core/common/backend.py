# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Backend configuration for JAX muphys implementation.
Switch between XLA and IREE backends via environment variable.
"""

import os
import jax

# Backend selection via environment variable
BACKEND = os.getenv('JAX_BACKEND', 'xla').lower()

if BACKEND == 'iree':
    try:
        from jax.experimental.jax2iree import jax2iree_jit as jit_compile
        print(f"[muphys_jax] Using IREE backend")
    except ImportError:
        print(f"[muphys_jax] IREE not available, falling back to XLA")
        print(f"[muphys_jax] Install with: pip install iree-compiler iree-runtime")
        jit_compile = jax.jit
        BACKEND = 'xla'
elif BACKEND == 'xla':
    jit_compile = jax.jit
    print(f"[muphys_jax] Using XLA backend")
else:
    raise ValueError(f"Unknown JAX_BACKEND: {BACKEND}. Use 'xla' or 'iree'")

# Export for use in other modules
__all__ = ['jit_compile', 'BACKEND']
