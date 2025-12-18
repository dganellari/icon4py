# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Core data structure definitions for muphys.
"""

from typing import NamedTuple
import jax.numpy as jnp


class Q(NamedTuple):
    """
    Water species state (6 species).

    Matches GT4Py Q definition from muphys/core/definitions.py
    """
    v: jnp.ndarray  # vapor
    c: jnp.ndarray  # cloud
    r: jnp.ndarray  # rain
    s: jnp.ndarray  # snow
    i: jnp.ndarray  # ice
    g: jnp.ndarray  # graupel


class PrecipState(NamedTuple):
    """State for precipitation scan operator."""
    q_update: jnp.ndarray
    flx: jnp.ndarray
    rho: jnp.ndarray
    vc: jnp.ndarray
    activated: jnp.ndarray


class TempState(NamedTuple):
    """State for temperature update scan operator."""
    t: jnp.ndarray
    eflx: jnp.ndarray
    activated: jnp.ndarray


__all__ = ['Q', 'PrecipState', 'TempState']
