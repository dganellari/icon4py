#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Compare NetCDF outputs from JAX and GT4Py graupel implementations.

Usage:
    python compare_graupel_outputs.py junk_jax.nc junk.nc
    python compare_graupel_outputs.py junk_jax.nc junk.nc --rtol 1e-4 --atol 1e-6
"""

import argparse
import netCDF4
import numpy as np


def load_netcdf(filename):
    """Load all variables from NetCDF file."""
    with netCDF4.Dataset(filename, 'r') as nc:
        data = {}
        for varname in nc.variables:
            var = nc.variables[varname]
            data[varname] = np.array(var[:])
        dims = {dim: len(nc.dimensions[dim]) for dim in nc.dimensions}
    return data, dims


def compare_fields(name, field1, field2, rtol=1e-5, atol=1e-8):
    """Compare two fields and print statistics."""
    if field1.shape != field2.shape:
        print(f"  {name}: SHAPE MISMATCH {field1.shape} vs {field2.shape}")
        return False
    
    diff = field1 - field2
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff / (np.abs(field1) + atol))
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    # Check if fields are close
    close = np.allclose(field1, field2, rtol=rtol, atol=atol)
    
    status = "✓" if close else "✗"
    print(f"  {name:12s} {status}  max_abs={max_abs_diff:.2e}  max_rel={max_rel_diff:.2e}  "
          f"mean_abs={mean_abs_diff:.2e}  rms={rms_diff:.2e}")
    
    if not close:
        print(f"              Range1: [{np.min(field1):.2e}, {np.max(field1):.2e}]")
        print(f"              Range2: [{np.min(field2):.2e}, {np.max(field2):.2e}]")
    
    return close


def main():
    parser = argparse.ArgumentParser(description="Compare NetCDF graupel outputs")
    parser.add_argument("file1", help="First NetCDF file (e.g., JAX output)")
    parser.add_argument("file2", help="Second NetCDF file (e.g., GT4Py output)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance (default: 1e-8)")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Comparing NetCDF outputs")
    print(f"{'='*70}")
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
    
    # Load both files
    data1, dims1 = load_netcdf(args.file1)
    data2, dims2 = load_netcdf(args.file2)
    
    print(f"\nDimensions:")
    print(f"  File 1: {dims1}")
    print(f"  File 2: {dims2}")
    
    # Find common variables
    vars1 = set(data1.keys())
    vars2 = set(data2.keys())
    common_vars = sorted(vars1 & vars2)
    only_in_1 = vars1 - vars2
    only_in_2 = vars2 - vars1
    
    if only_in_1:
        print(f"\nVariables only in file 1: {only_in_1}")
    if only_in_2:
        print(f"Variables only in file 2: {only_in_2}")
    
    # Compare common variables
    print(f"\nComparing {len(common_vars)} common variables:")
    print(f"{'='*70}")
    
    all_close = True
    for varname in common_vars:
        is_close = compare_fields(varname, data1[varname], data2[varname], 
                                  rtol=args.rtol, atol=args.atol)
        all_close = all_close and is_close
    
    print(f"{'='*70}")
    if all_close:
        print("✓ All fields match within tolerance!")
    else:
        print("✗ Some fields differ beyond tolerance")
    print(f"{'='*70}\n")
    
    return 0 if all_close else 1


if __name__ == "__main__":
    exit(main())
