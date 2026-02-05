#!/usr/bin/env python3
"""
Combine q_t_update and precipitation_effects StableHLO into a single module.

This script programmatically combines two separately generated StableHLO modules:
1. q_t_update StableHLO (from generate_qt_update_stablehlo.py)
2. precipitation_effects StableHLO (from generate_unrolled_transposed.py)

The combined module fuses both phases into a single compilation unit, allowing
XLA to optimize across the boundary (eliminating intermediate materialization
and reducing kernel launch overhead).

The combined function signature:
    @main(kmin_r, kmin_i, kmin_s, kmin_g,     # bool masks (nlev x ncells)
          t, p, rho, dz,                       # physics inputs (nlev x ncells)
          qv, qc, qr, qs, qi, qg)             # water species (nlev x ncells)
    -> (t_final,                               # updated temperature
        qv_out, qc_out, qr_out, qs_out, qi_out, qg_out,  # updated species
        pflx, pr, ps, pi, pg, eflx)           # precipitation outputs

Strategy:
- Parse the q_t_update StableHLO to extract its @main as a private function
- Parse the precip StableHLO to extract its @main as a private function
- Generate a new @main that:
  1. Calls q_t_update (t, p, rho, qv..qg) -> (qv_new..qg_new, t_new)
  2. Calls precip (kmin_r..kmin_g, qv_new..qg_new, t_new, rho, dz) -> 11 outputs
  3. Returns combined outputs

Usage:
    # First generate both StableHLO modules:
    python generate_qt_update_stablehlo.py -o stablehlo/qt_update.stablehlo
    python generate_unrolled_transposed.py -o stablehlo/precip_transposed.stablehlo

    # Then combine them:
    python generate_combined_graupel.py \\
        --qt-update stablehlo/qt_update.stablehlo \\
        --precip stablehlo/precip_transposed.stablehlo \\
        -o stablehlo/graupel_combined.stablehlo
"""

import argparse
import re
import sys
import pathlib


def rename_main_function(stablehlo_text: str, new_name: str) -> str:
    """
    Rename the @main function in a StableHLO module to a private function.

    Changes:
    - 'func.func public @main(' -> 'func.func private @new_name('
    - Any 'func.call @main(' -> 'func.call @new_name('  (internal calls)
    """
    # Replace function declaration
    text = re.sub(
        r'func\.func\s+public\s+@main\s*\(',
        f'func.func private @{new_name}(',
        stablehlo_text
    )
    # Replace any internal calls to @main
    text = text.replace('func.call @main(', f'func.call @{new_name}(')
    text = text.replace('@main', f'@{new_name}')
    return text


def extract_function_body(stablehlo_text: str) -> str:
    """Extract everything between the module braces."""
    # Find the first { after module and the last }
    lines = stablehlo_text.split('\n')
    inside = []
    depth = 0
    in_module = False

    for line in lines:
        if 'module' in line and '{' in line:
            in_module = True
            depth = 1
            continue
        if in_module:
            depth += line.count('{') - line.count('}')
            if depth <= 0:
                break
            inside.append(line)

    return '\n'.join(inside)


def parse_function_signature(stablehlo_text: str):
    """Parse the @main function to get input types and return types."""
    # Find the func.func line(s) for @main
    # It may span multiple lines
    main_pattern = re.compile(
        r'func\.func\s+(?:public\s+)?@main\s*\(([^)]*)\)\s*->\s*\(([^)]*)\)',
        re.DOTALL
    )
    match = main_pattern.search(stablehlo_text)
    if not match:
        # Try single return type (no parens)
        main_pattern2 = re.compile(
            r'func\.func\s+(?:public\s+)?@main\s*\(([^)]*)\)\s*->\s*(\S+)',
            re.DOTALL
        )
        match = main_pattern2.search(stablehlo_text)

    if not match:
        print("ERROR: Could not parse @main function signature")
        sys.exit(1)

    args_str = match.group(1)
    rets_str = match.group(2)

    # Parse argument types: %arg0: tensor<90x327680xf64>, ...
    arg_types = []
    for arg_match in re.finditer(r'%\w+:\s*(tensor<[^>]+>)', args_str):
        arg_types.append(arg_match.group(1))

    # Parse return types
    ret_types = re.findall(r'tensor<[^>]+>', rets_str)

    return arg_types, ret_types


def combine_stablehlo(qt_update_file: str, precip_file: str, nlev: int, ncells: int) -> str:
    """
    Combine q_t_update and precipitation StableHLO into a single module.

    Combined function signature:
        @main(kmin_r: bool, kmin_i: bool, kmin_s: bool, kmin_g: bool,
              t: f64, p: f64, rho: f64, dz: f64,
              qv: f64, qc: f64, qr: f64, qs: f64, qi: f64, qg: f64)
        -> (t_final, qv_out, qc_out, qr_out, qs_out, qi_out, qg_out,
            pflx, pr, ps, pi, pg, eflx)
    """
    print(f"Reading q_t_update StableHLO: {qt_update_file}")
    with open(qt_update_file, 'r') as f:
        qt_text = f.read()
    print(f"  Size: {len(qt_text) / 1024:.1f} KB")

    print(f"Reading precipitation StableHLO: {precip_file}")
    with open(precip_file, 'r') as f:
        precip_text = f.read()
    print(f"  Size: {len(precip_text) / 1024:.1f} KB")

    # Parse signatures
    qt_arg_types, qt_ret_types = parse_function_signature(qt_text)
    precip_arg_types, precip_ret_types = parse_function_signature(precip_text)

    print(f"\nq_t_update: {len(qt_arg_types)} inputs -> {len(qt_ret_types)} outputs")
    print(f"  Inputs: {qt_arg_types}")
    print(f"  Outputs: {qt_ret_types}")
    print(f"\nprecip: {len(precip_arg_types)} inputs -> {len(precip_ret_types)} outputs")
    print(f"  Inputs: {precip_arg_types}")
    print(f"  Outputs: {precip_ret_types}")

    # Rename functions
    qt_body = extract_function_body(rename_main_function(qt_text, '_qt_update'))
    precip_body = extract_function_body(rename_main_function(precip_text, '_precip_effects'))

    # Type shortcuts
    tf = f'tensor<{nlev}x{ncells}xf64>'
    tb = f'tensor<{nlev}x{ncells}xi1>'

    # Build combined module
    lines = []
    lines.append(f'module @jit_graupel_combined_{nlev} attributes {{mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32}} {{')
    lines.append('')

    # Insert q_t_update function body (private)
    lines.append('  // ============ q_t_update (private) ============')
    lines.append(qt_body)
    lines.append('')

    # Insert precip function body (private)
    lines.append('  // ============ precipitation_effects (private) ============')
    lines.append(precip_body)
    lines.append('')

    # Build the combined @main
    # Combined inputs:
    #   %arg0..3: kmin_r, kmin_i, kmin_s, kmin_g (bool masks)
    #   %arg4: t, %arg5: p, %arg6: rho, %arg7: dz
    #   %arg8: qv, %arg9: qc, %arg10: qr, %arg11: qs, %arg12: qi, %arg13: qg
    #
    # q_t_update signature: (t, p, rho, qv, qc, qr, qs, qi, qg) -> (qv_new, qc_new, qr_new, qs_new, qi_new, qg_new, t_new)
    #
    # precip signature: (kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz)
    #                -> (qr, qs, qi, qg, t_new, pflx_tot, pr, ps, pi, pg, eflx)

    combined_args = ', '.join([
        f'%arg0: {tb}',   # kmin_r
        f'%arg1: {tb}',   # kmin_i
        f'%arg2: {tb}',   # kmin_s
        f'%arg3: {tb}',   # kmin_g
        f'%arg4: {tf}',   # t
        f'%arg5: {tf}',   # p
        f'%arg6: {tf}',   # rho
        f'%arg7: {tf}',   # dz
        f'%arg8: {tf}',   # qv
        f'%arg9: {tf}',   # qc
        f'%arg10: {tf}',  # qr
        f'%arg11: {tf}',  # qs
        f'%arg12: {tf}',  # qi
        f'%arg13: {tf}',  # qg
    ])

    # Return types: t_final, qv, qc, qr, qs, qi, qg, pflx, pr, ps, pi, pg, eflx
    ret_types_str = ', '.join([tf] * 13)

    lines.append('  // ============ Combined graupel (public) ============')
    lines.append(f'  func.func public @main({combined_args}) -> ({ret_types_str}) {{')
    lines.append('')
    lines.append('    // Phase 1: q_t_update(t, p, rho, qv, qc, qr, qs, qi, qg)')
    lines.append('    //   -> (qv_new, qc_new, qr_new, qs_new, qi_new, qg_new, t_new)')

    # Build the q_t_update call
    qt_call_args = ', '.join([
        '%arg4',   # t
        '%arg5',   # p
        '%arg6',   # rho
        '%arg8',   # qv
        '%arg9',   # qc
        '%arg10',  # qr
        '%arg11',  # qs
        '%arg12',  # qi
        '%arg13',  # qg
    ])
    qt_ret = ', '.join([tf] * 7)
    lines.append(f'    %qt:7 = func.call @_qt_update({qt_call_args}) : ({", ".join([tf]*9)}) -> ({qt_ret})')
    lines.append('    // qt#0=qv_new, qt#1=qc_new, qt#2=qr_new, qt#3=qs_new, qt#4=qi_new, qt#5=qg_new, qt#6=t_new')
    lines.append('')

    lines.append('    // Phase 2: precip_effects(kmin_r, kmin_i, kmin_s, kmin_g,')
    lines.append('    //                         qv_new, qc_new, qr_new, qs_new, qi_new, qg_new,')
    lines.append('    //                         t_new, rho, dz)')
    lines.append('    //   -> (qr_out, qs_out, qi_out, qg_out, t_final, pflx, pr, ps, pi, pg, eflx)')

    # Build the precip call - using q_t_update outputs
    precip_call_args = ', '.join([
        '%arg0',    # kmin_r
        '%arg1',    # kmin_i
        '%arg2',    # kmin_s
        '%arg3',    # kmin_g
        '%qt#0',    # qv_new
        '%qt#1',    # qc_new
        '%qt#2',    # qr_new
        '%qt#3',    # qs_new
        '%qt#4',    # qi_new
        '%qt#5',    # qg_new
        '%qt#6',    # t_new
        '%arg6',    # rho (unchanged)
        '%arg7',    # dz (unchanged)
    ])
    precip_ret = ', '.join([tf] * 11)
    precip_in_types = ', '.join([tb]*4 + [tf]*9)
    lines.append(f'    %pr:11 = func.call @_precip_effects({precip_call_args}) : ({precip_in_types}) -> ({precip_ret})')
    lines.append('    // pr#0=qr_out, pr#1=qs_out, pr#2=qi_out, pr#3=qg_out, pr#4=t_final,')
    lines.append('    // pr#5=pflx, pr#6=pr, pr#7=ps, pr#8=pi, pr#9=pg, pr#10=eflx')
    lines.append('')

    # Return: t_final, qv_out(=qt#0), qc_out(=qt#1), qr_out, qs_out, qi_out, qg_out,
    #         pflx, pr, ps, pi, pg, eflx
    # Note: qv and qc are NOT modified by precip_effects, only qr/qs/qi/qg are
    ret_vals = ', '.join([
        '%pr#4',   # t_final
        '%qt#0',   # qv (from q_t_update, unchanged by precip)
        '%qt#1',   # qc (from q_t_update, unchanged by precip)
        '%pr#0',   # qr_out (from precip)
        '%pr#1',   # qs_out (from precip)
        '%pr#2',   # qi_out (from precip)
        '%pr#3',   # qg_out (from precip)
        '%pr#5',   # pflx
        '%pr#6',   # pr
        '%pr#7',   # ps
        '%pr#8',   # pi
        '%pr#9',   # pg
        '%pr#10',  # eflx
    ])
    lines.append(f'    return {ret_vals} : {ret_types_str}')
    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Combine q_t_update + precip StableHLO")
    parser.add_argument("--qt-update", default="stablehlo/qt_update.stablehlo",
                       help="q_t_update StableHLO file")
    parser.add_argument("--precip", default="stablehlo/precip_transposed.stablehlo",
                       help="precipitation_effects StableHLO file")
    parser.add_argument("-o", "--output", default="stablehlo/graupel_combined.stablehlo",
                       help="Output combined StableHLO file")
    parser.add_argument("--nlev", type=int, default=90)
    parser.add_argument("--ncells", type=int, default=327680)
    args = parser.parse_args()

    print("=" * 80)
    print("COMBINING q_t_update + precipitation_effects StableHLO")
    print("=" * 80)
    print()

    combined = combine_stablehlo(args.qt_update, args.precip, args.nlev, args.ncells)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(combined)

    print(f"\nCombined StableHLO written to: {args.output}")
    print(f"  Size: {len(combined) / 1024:.1f} KB")
    print(f"  Lines: {combined.count(chr(10))}")
    print()
    print("To use this combined module:")
    print("  1. Create optimized_graupel.py with a custom primitive for the full graupel")
    print("  2. Or inject via test_graupel_native_transposed.py --optimized-hlo")


if __name__ == "__main__":
    main()
