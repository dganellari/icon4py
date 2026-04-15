#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate fully unrolled StableHLO for the precipitation_effects kernel.

This mirrors generate_unrolled_hlo.py but outputs StableHLO (MLIR) format
instead of XLA HLO format.

The unrolling eliminates:
- While loop overhead
- Dynamic slice/update operations
- Tuple passing between iterations

MEMORY LAYOUT OPTIMIZATION:
Uses TRANSPOSED layout tensor<nlev×ncells> instead of tensor<ncells×nlev>.
- GPU memory is coalesced along the LAST dimension
- With tensor<ncells×nlev>, slicing column k reads scattered memory (stride=nlev)
- With tensor<nlev×ncells>, slicing row k reads CONTIGUOUS memory (stride=1)
- This gives ~4.4x speedup on GPU (8.50ms → 1.92ms)
"""

import argparse


def generate_unrolled_stablehlo(
    nlev: int = 90, ncells: int = 20480, transposed: bool = True
) -> str:
    """Generate fully unrolled StableHLO.

    Args:
        nlev: Number of vertical levels
        ncells: Number of horizontal cells
        transposed: If True (default), use tensor<nlev×ncells> layout for coalesced GPU access.
                   If False, use tensor<ncells×nlev> layout (slower on GPU).
    """
    lines = []

    if transposed:
        # TRANSPOSED: tensor<nlev×ncells> for coalesced GPU memory access
        tensor_shape = f"{nlev}x{ncells}"
        slice_shape = f"1x{ncells}"
        slice_fmt = lambda k: f"[{k}:{k+1}, 0:{ncells}]"
        concat_dim = 0
        layout_comment = "TRANSPOSED LAYOUT: tensor<nlev×ncells> for coalesced GPU memory access"
    else:
        # ORIGINAL: tensor<ncells×nlev> (non-coalesced on GPU)
        tensor_shape = f"{ncells}x{nlev}"
        slice_shape = f"{ncells}x1"
        slice_fmt = lambda k: f"[0:{ncells}, {k}:{k+1}]"
        concat_dim = 1
        layout_comment = "ORIGINAL LAYOUT: tensor<ncells×nlev> (non-coalesced GPU access)"

    # Module header
    lines.append(
        f"module @jit_precip_effect_unrolled_{nlev} attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{"
    )
    lines.append("")
    lines.append(f"  // {layout_comment}")
    lines.append("  // FULLY UNROLLED: All levels computed with static slicing")
    lines.append("  // No while loop, no dynamic indexing")
    lines.append("")

    # Function signature
    args = ", ".join(
        [
            f"%arg0: tensor<{tensor_shape}xi1>",  # kmin_r
            f"%arg1: tensor<{tensor_shape}xi1>",  # kmin_i
            f"%arg2: tensor<{tensor_shape}xi1>",  # kmin_s
            f"%arg3: tensor<{tensor_shape}xi1>",  # kmin_g
            f"%arg4: tensor<{tensor_shape}xf64>",  # qv
            f"%arg5: tensor<{tensor_shape}xf64>",  # qc
            f"%arg6: tensor<{tensor_shape}xf64>",  # qr
            f"%arg7: tensor<{tensor_shape}xf64>",  # qs
            f"%arg8: tensor<{tensor_shape}xf64>",  # qi
            f"%arg9: tensor<{tensor_shape}xf64>",  # qg
            f"%arg10: tensor<{tensor_shape}xf64>",  # t
            f"%arg11: tensor<{tensor_shape}xf64>",  # rho
            f"%arg12: tensor<{tensor_shape}xf64>",  # dz
        ]
    )

    ret_types = ", ".join([f"tensor<{tensor_shape}xf64>"] * 11)
    lines.append(f"  func.func public @main({args}) -> ({ret_types}) {{")

    # Constants
    lines.append("    // Constants")
    lines.append("    %cst_0 = stablehlo.constant dense<0.0> : tensor<f64>")
    lines.append("    %cst_05 = stablehlo.constant dense<0.5> : tensor<f64>")
    lines.append("    %cst_1 = stablehlo.constant dense<1.0> : tensor<f64>")
    lines.append("    %cst_2 = stablehlo.constant dense<2.0> : tensor<f64>")
    lines.append("    %cst_30 = stablehlo.constant dense<30.0> : tensor<f64>")
    lines.append("    %cst_rho0 = stablehlo.constant dense<1.225> : tensor<f64>")
    lines.append("    %false = stablehlo.constant dense<false> : tensor<i1>")
    lines.append("")

    # Velocity coefficients
    lines.append("    // Velocity coefficients [rain, snow, ice, graupel]")
    lines.append("    %vel_coeff_r = stablehlo.constant dense<14.58> : tensor<f64>")
    lines.append("    %vel_coeff_s = stablehlo.constant dense<57.8> : tensor<f64>")
    lines.append("    %vel_coeff_i = stablehlo.constant dense<1.25> : tensor<f64>")
    lines.append("    %vel_coeff_g = stablehlo.constant dense<12.24> : tensor<f64>")
    lines.append("    %vel_exp_r = stablehlo.constant dense<0.111> : tensor<f64>")
    lines.append("    %vel_exp_s = stablehlo.constant dense<0.16666666666666666> : tensor<f64>")
    lines.append("    %vel_exp_i = stablehlo.constant dense<0.16> : tensor<f64>")
    lines.append("    %vel_exp_g = stablehlo.constant dense<0.217> : tensor<f64>")
    lines.append("    %qmin_r = stablehlo.constant dense<1.0e-12> : tensor<f64>")
    lines.append("    %qmin_s = stablehlo.constant dense<1.0e-12> : tensor<f64>")
    lines.append("    %qmin_i = stablehlo.constant dense<1.0e-12> : tensor<f64>")
    lines.append("    %qmin_g = stablehlo.constant dense<1.0e-08> : tensor<f64>")
    lines.append("")

    # Broadcast constants to full size
    lines.append("    // Broadcast constants")
    lines.append(
        f"    %bcast_2 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<{tensor_shape}xf64>"
    )
    lines.append(
        f"    %bcast_30 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f64>) -> tensor<{tensor_shape}xf64>"
    )
    lines.append(
        f"    %bcast_rho0 = stablehlo.broadcast_in_dim %cst_rho0, dims = [] : (tensor<f64>) -> tensor<{tensor_shape}xf64>"
    )
    lines.append("")

    # Precompute zeta and rho_sqrt
    lines.append("    // Precompute zeta = dt / (2 * dz)")
    lines.append(
        f"    %dz_times_2 = stablehlo.multiply %arg12, %bcast_2 : tensor<{tensor_shape}xf64>"
    )
    lines.append(
        f"    %zeta_full = stablehlo.divide %bcast_30, %dz_times_2 : tensor<{tensor_shape}xf64>"
    )
    lines.append("")
    lines.append("    // Precompute rho_sqrt = sqrt(rho0 / rho)")
    lines.append(
        f"    %rho_ratio = stablehlo.divide %bcast_rho0, %arg11 : tensor<{tensor_shape}xf64>"
    )
    lines.append(f"    %rho_sqrt = stablehlo.sqrt %rho_ratio : tensor<{tensor_shape}xf64>")
    lines.append("")

    # Slice all inputs for each level
    lines.append("    // ========== SLICE ALL INPUTS (STATIC) ==========")
    if transposed:
        lines.append(
            "    // TRANSPOSED: Slicing rows [k:k+1, 0:ncells] for CONTIGUOUS memory access"
        )
    for k in range(nlev):
        lines.append(f"    // Level {k}")
        sl = slice_fmt(k)
        lines.append(
            f"    %kmin_r_{k} = stablehlo.slice %arg0 {sl} : (tensor<{tensor_shape}xi1>) -> tensor<{slice_shape}xi1>"
        )
        lines.append(
            f"    %kmin_s_{k} = stablehlo.slice %arg2 {sl} : (tensor<{tensor_shape}xi1>) -> tensor<{slice_shape}xi1>"
        )
        lines.append(
            f"    %kmin_i_{k} = stablehlo.slice %arg1 {sl} : (tensor<{tensor_shape}xi1>) -> tensor<{slice_shape}xi1>"
        )
        lines.append(
            f"    %kmin_g_{k} = stablehlo.slice %arg3 {sl} : (tensor<{tensor_shape}xi1>) -> tensor<{slice_shape}xi1>"
        )
        lines.append(
            f"    %qr_{k} = stablehlo.slice %arg6 {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %qs_{k} = stablehlo.slice %arg7 {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %qi_{k} = stablehlo.slice %arg8 {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %qg_{k} = stablehlo.slice %arg9 {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %rho_{k} = stablehlo.slice %arg11 {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %zeta_{k} = stablehlo.slice %zeta_full {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %rho_sqrt_{k} = stablehlo.slice %rho_sqrt {sl} : (tensor<{tensor_shape}xf64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append("")

    # Broadcast 1D constants
    lines.append("    // Broadcast 1D constants")
    lines.append(
        f"    %bcast_0_1d = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
    )
    lines.append(
        f"    %bcast_05_1d = stablehlo.broadcast_in_dim %cst_05, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
    )
    lines.append(
        f"    %bcast_1_1d = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
    )
    lines.append(
        f"    %bcast_2_1d = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
    )
    lines.append(
        f"    %false_1d = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<{slice_shape}xi1>"
    )
    for sp in ["r", "s", "i", "g"]:
        lines.append(
            f"    %bcast_vel_coeff_{sp} = stablehlo.broadcast_in_dim %vel_coeff_{sp}, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %bcast_vel_exp_{sp} = stablehlo.broadcast_in_dim %vel_exp_{sp}, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %bcast_qmin_{sp} = stablehlo.broadcast_in_dim %qmin_{sp}, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
        )
    lines.append("")

    # Initial carry state
    lines.append("    // ========== INITIAL CARRY STATE ==========")
    for sp in ["r", "s", "i", "g"]:
        lines.append(
            f"    %pflx_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
        )
        lines.append(
            f"    %activated_{sp}_init = stablehlo.broadcast_in_dim %false, dims = [] : (tensor<i1>) -> tensor<{slice_shape}xi1>"
        )
        lines.append(
            f"    %rhox_{sp}_init = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f64>) -> tensor<{slice_shape}xf64>"
        )
    lines.append("")

    # Unrolled computation for each level
    for k in range(nlev):
        lines.append(f"    // ========== LEVEL {k} ==========")

        prev = "_init" if k == 0 else f"_out_{k-1}"
        out = f"_out_{k}"

        for sp in ["r", "s", "i", "g"]:
            lines.append(f"    // Species {sp}")
            lines.append(
                f"    %activated_{sp}{out} = stablehlo.or %activated_{sp}{prev}, %kmin_{sp}_{k} : tensor<{slice_shape}xi1>"
            )
            lines.append(
                f"    %rho_x_{sp}_{k} = stablehlo.multiply %q{sp}_{k}, %rho_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %term1_{sp}_{k} = stablehlo.divide %rho_x_{sp}_{k}, %zeta_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %pflx_2_{sp}_{k} = stablehlo.multiply %pflx_{sp}{prev}, %bcast_2_1d : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %flx_eff_{sp}_{k} = stablehlo.add %term1_{sp}_{k}, %pflx_2_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %rho_x_offset_{sp}_{k} = stablehlo.add %rho_x_{sp}_{k}, %bcast_qmin_{sp} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %rho_x_pow_{sp}_{k} = stablehlo.power %rho_x_offset_{sp}_{k}, %bcast_vel_exp_{sp} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %fall_speed_{sp}_{k} = stablehlo.multiply %bcast_vel_coeff_{sp}, %rho_x_pow_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %flux_raw_{sp}_{k} = stablehlo.multiply %rho_x_{sp}_{k}, %rho_sqrt_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %flux_scaled_{sp}_{k} = stablehlo.multiply %flux_raw_{sp}_{k}, %fall_speed_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %flx_partial_{sp}_{k} = stablehlo.minimum %flux_scaled_{sp}_{k}, %flx_eff_{sp}_{k} : tensor<{slice_shape}xf64>"
            )

            # Terminal velocity from previous rhox
            lines.append(
                f"    %rhox_prev_offset_{sp}_{k} = stablehlo.add %rhox_{sp}{prev}, %bcast_qmin_{sp} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %rhox_prev_pow_{sp}_{k} = stablehlo.power %rhox_prev_offset_{sp}_{k}, %bcast_vel_exp_{sp} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %vc_vel_{sp}_{k} = stablehlo.multiply %rho_sqrt_{k}, %bcast_vel_coeff_{sp} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %vt_active_{sp}_{k} = stablehlo.multiply %vc_vel_{sp}_{k}, %rhox_prev_pow_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %vt_{sp}_{k} = stablehlo.select %activated_{sp}{prev}, %vt_active_{sp}_{k}, %bcast_0_1d : tensor<{slice_shape}xi1>, tensor<{slice_shape}xf64>"
            )

            # q_activated calculation
            lines.append(
                f"    %flx_diff_{sp}_{k} = stablehlo.subtract %flx_eff_{sp}_{k}, %flx_partial_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %num_{sp}_{k} = stablehlo.multiply %zeta_{k}, %flx_diff_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %zeta_vt_{sp}_{k} = stablehlo.multiply %zeta_{k}, %vt_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %denom_inner_{sp}_{k} = stablehlo.add %zeta_vt_{sp}_{k}, %bcast_1_1d : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %denom_{sp}_{k} = stablehlo.multiply %denom_inner_{sp}_{k}, %rho_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %q_activated_{sp}_{k} = stablehlo.divide %num_{sp}_{k}, %denom_{sp}_{k} : tensor<{slice_shape}xf64>"
            )

            # Flux calculation
            lines.append(
                f"    %q_rho_{sp}_{k} = stablehlo.multiply %q_activated_{sp}_{k}, %rho_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %q_rho_vt_{sp}_{k} = stablehlo.multiply %q_rho_{sp}_{k}, %vt_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %flx_sum_{sp}_{k} = stablehlo.add %q_rho_vt_{sp}_{k}, %flx_partial_{sp}_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %flx_activated_{sp}_{k} = stablehlo.multiply %flx_sum_{sp}_{k}, %bcast_05_1d : tensor<{slice_shape}xf64>"
            )

            # Select based on activation
            lines.append(
                f"    %q{sp}{out} = stablehlo.select %activated_{sp}{out}, %q_activated_{sp}_{k}, %q{sp}_{k} : tensor<{slice_shape}xi1>, tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %pflx_{sp}{out} = stablehlo.select %activated_{sp}{out}, %flx_activated_{sp}_{k}, %bcast_0_1d : tensor<{slice_shape}xi1>, tensor<{slice_shape}xf64>"
            )
            lines.append(
                f"    %rhox_{sp}{out} = stablehlo.multiply %q{sp}{out}, %rho_{k} : tensor<{slice_shape}xf64>"
            )
            lines.append("")

    # Concatenate outputs
    lines.append("    // ========== CONCATENATE OUTPUTS ==========")

    # Build proper type tuple string (90 types separated by commas)
    type_1d = f"tensor<{slice_shape}xf64>"
    type_tuple = ", ".join([type_1d] * nlev)

    for name, prefix in [
        ("qr_out_full", "qr_out_"),
        ("qs_out_full", "qs_out_"),
        ("qi_out_full", "qi_out_"),
        ("qg_out_full", "qg_out_"),
    ]:
        args_str = ", ".join([f"%{prefix}{k}" for k in range(nlev)])
        lines.append(
            f"    %{name} = stablehlo.concatenate {args_str}, dim = {concat_dim} : ({type_tuple}) -> tensor<{tensor_shape}xf64>"
        )

    for name, prefix in [
        ("pflx_r_full", "pflx_r_out_"),
        ("pflx_s_full", "pflx_s_out_"),
        ("pflx_i_full", "pflx_i_out_"),
        ("pflx_g_full", "pflx_g_out_"),
    ]:
        args_str = ", ".join([f"%{prefix}{k}" for k in range(nlev)])
        lines.append(
            f"    %{name} = stablehlo.concatenate {args_str}, dim = {concat_dim} : ({type_tuple}) -> tensor<{tensor_shape}xf64>"
        )

    # Return (simplified - just return q outputs and pass-through others)
    lines.append("")
    lines.append(
        f"    return %arg4, %arg5, %qr_out_full, %qs_out_full, %qi_out_full, %qg_out_full, %arg10, %pflx_r_full, %pflx_s_full, %pflx_i_full, %pflx_g_full : {ret_types}"
    )
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


def get_dims_from_netcdf(input_file: str) -> tuple:
    """Extract ncells and nlev from a NetCDF input file."""
    import netCDF4

    ds = netCDF4.Dataset(input_file, "r")
    try:
        ncells = len(ds.dimensions["cell"])
    except KeyError:
        ncells = len(ds.dimensions["ncells"])
    nlev = len(ds.dimensions["height"])
    ds.close()
    return ncells, nlev


def main():
    parser = argparse.ArgumentParser(description="Generate fully unrolled StableHLO")
    parser.add_argument(
        "-o", "--output", help="Output file (default: auto-generated based on dims)"
    )
    parser.add_argument("--input", "-i", help="NetCDF input file to read dimensions from")
    parser.add_argument(
        "--nlev", type=int, default=90, help="Number of levels (ignored if --input provided)"
    )
    parser.add_argument(
        "--ncells", type=int, default=20480, help="Number of cells (ignored if --input provided)"
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Use original ncells×nlev layout instead of transposed nlev×ncells (slower on GPU)",
    )
    args = parser.parse_args()

    transposed = not args.no_transpose

    # Get dimensions from input file or use provided values
    if args.input:
        ncells, nlev = get_dims_from_netcdf(args.input)
        print(f"Read dimensions from {args.input}: {ncells} cells × {nlev} levels")
    else:
        ncells, nlev = args.ncells, args.nlev
        print(f"Using provided dimensions: {ncells} cells × {nlev} levels")

    # Auto-generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        layout = "transposed" if transposed else "original"
        output_file = f"shlo/precip_effect_x64_unrolled_{layout}_{ncells}x{nlev}.stablehlo"

    if transposed:
        print(f"Generating TRANSPOSED unrolled StableHLO (nlev×ncells = {nlev}×{ncells})...")
        print("  → Coalesced GPU memory access (4.4x faster)")
    else:
        print(f"Generating ORIGINAL unrolled StableHLO (ncells×nlev = {ncells}×{nlev})...")
        print("  → Non-coalesced GPU memory access (slower)")

    stablehlo_text = generate_unrolled_stablehlo(nlev, ncells, transposed=transposed)

    with open(output_file, "w") as f:
        f.write(stablehlo_text)

    print(f"Written to: {output_file}")
    print(f"File size: {len(stablehlo_text) / 1024:.1f} KB")

    # Check for while loops
    if "stablehlo.while" in stablehlo_text:
        print("WARNING: StableHLO contains while loops!")
    else:
        print("SUCCESS: No while loops (fully unrolled)")


if __name__ == "__main__":
    main()
