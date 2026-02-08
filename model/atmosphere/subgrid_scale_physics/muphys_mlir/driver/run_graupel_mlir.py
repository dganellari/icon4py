#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
MLIR graupel driver compatible with muphys_jax run_graupel_jax.py interface.
Uses the same NetCDF input files and produces comparable output.

Usage:
    python -m muphys_mlir.driver.run_graupel_mlir -o output.nc input.nc 10 30.0 100.0

    # Print generated MLIR IR
    python -m muphys_mlir.driver.run_graupel_mlir --print-mlir input.nc

    # Generate only (no execution)
    python -m muphys_mlir.driver.run_graupel_mlir --generate-only input.nc
"""

import argparse
import pathlib
import time
from typing import NamedTuple

import netCDF4
import numpy as np

from ..core.precip_scans_mlir import generate_precip_scan_mlir
from ..implementations.graupel import MLIR_AVAILABLE, MLIR_IMPORT_ERROR, graupel_run


def _calc_dz(z: np.ndarray) -> np.ndarray:
    """Calculate layer thickness from geometric height (same as GT4Py version)."""
    ksize = z.shape[0]
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


class Q(NamedTuple):
    """Mixing ratios for hydrometeor species."""

    v: np.ndarray  # water vapor
    c: np.ndarray  # cloud water
    r: np.ndarray  # rain
    s: np.ndarray  # snow
    i: np.ndarray  # ice
    g: np.ndarray  # graupel


class GraupelInput(NamedTuple):
    """Input data for graupel scheme."""

    ncells: int
    nlev: int
    dz: np.ndarray
    p: np.ndarray
    rho: np.ndarray
    t: np.ndarray
    q: Q

    @classmethod
    def load(cls, filename: pathlib.Path | str) -> "GraupelInput":
        """Load input from NetCDF file."""
        with netCDF4.Dataset(filename, mode="r") as ncfile:
            # Get dimensions
            try:
                ncells = len(ncfile.dimensions["cell"])
            except KeyError:
                ncells = len(ncfile.dimensions["ncells"])
            nlev = len(ncfile.dimensions["height"])

            # Calculate layer thickness
            dz = _calc_dz(ncfile.variables["zg"])
            dz = np.transpose(dz)  # (height, ncells) -> (ncells, height)

            # Load variables (transpose from (height, ncells) to (ncells, height))
            def load_var(varname: str) -> np.ndarray:
                var = ncfile.variables[varname]
                if var.dimensions[0] == "time":
                    var = var[0, :, :]
                return np.transpose(var).astype(np.float64)

            # Create Q structure
            q = Q(
                v=load_var("hus"),  # specific humidity (vapor)
                c=load_var("clw"),  # cloud liquid water
                r=load_var("qr"),  # rain
                s=load_var("qs"),  # snow
                i=load_var("cli"),  # cloud ice
                g=load_var("qg"),  # graupel
            )

            return cls(
                ncells=ncells,
                nlev=nlev,
                dz=dz,
                t=load_var("ta"),
                p=load_var("pfull"),
                rho=load_var("rho"),
                q=q,
            )


class GraupelOutput(NamedTuple):
    """Output data from graupel scheme."""

    t: np.ndarray
    q: Q
    pflx: np.ndarray
    pr: np.ndarray
    ps: np.ndarray
    pi: np.ndarray
    pg: np.ndarray
    pre: np.ndarray

    def write(self, filename: pathlib.Path | str):
        """Write output to NetCDF file."""
        ncells = self.t.shape[0]
        nlev = self.t.shape[1]

        with netCDF4.Dataset(filename, mode="w") as ncfile:
            ncfile.createDimension("ncells", ncells)
            ncfile.createDimension("height", nlev)

            def write_field(varname: str, data: np.ndarray):
                var = ncfile.createVariable(varname, np.float64, ("height", "ncells"))
                var[...] = data.transpose()

            write_field("ta", self.t)
            write_field("hus", self.q.v)
            write_field("clw", self.q.c)
            write_field("cli", self.q.i)
            write_field("qr", self.q.r)
            write_field("qs", self.q.s)
            write_field("qg", self.q.g)
            write_field("pflx", self.pflx)
            write_field("prr_gsp", self.pr)
            write_field("prs_gsp", self.ps)
            write_field("pri_gsp", self.pi)
            write_field("prg_gsp", self.pg)
            write_field("pre_gsp", self.pre)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run MLIR graupel scheme (compatible with muphys_jax run_graupel_jax.py)"
    )
    parser.add_argument(
        "-o", metavar="output_file", dest="output_file", help="output filename", default="output.nc"
    )
    parser.add_argument("input_file", help="input NetCDF data file")
    parser.add_argument("itime", help="number of iterations", nargs="?", type=int, default=0)
    parser.add_argument("dt", help="timestep (seconds)", nargs="?", type=float, default=30.0)
    parser.add_argument(
        "qnc",
        help="cloud droplet number concentration (m^-3)",
        nargs="?",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--generate-only",
        help="only generate MLIR code, do not execute",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--print-mlir",
        help="print generated MLIR IR to stdout",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Check MLIR availability
    if not MLIR_AVAILABLE:
        print(f"MLIR not available: {MLIR_IMPORT_ERROR}")
        print("\nTo install MLIR Python bindings:")
        print("  pip install mlir-python-bindings")
        return 1

    print("MLIR graupel driver")
    print("=" * 60)

    # Load input data
    print(f"Loading input from: {args.input_file}")
    inp = GraupelInput.load(pathlib.Path(args.input_file))
    print(f"Grid size: {inp.ncells} cells x {inp.nlev} levels")
    print(f"Temperature range: {inp.t.min():.1f} - {inp.t.max():.1f} K")
    print(f"Timestep: {args.dt} s")
    print(f"Cloud droplet concentration: {args.qnc} m^-3")

    # Print MLIR if requested
    if args.print_mlir:
        print("\n" + "=" * 60)
        print("Generated MLIR IR:")
        print("=" * 60)
        mlir_code = generate_precip_scan_mlir(inp.nlev, inp.ncells)
        print(mlir_code)
        print("=" * 60)

    if args.generate_only:
        print("\n--generate-only specified, skipping execution")
        return 0

    # Transpose arrays to (nlev, ncells) for MLIR kernel
    # MLIR expects column-major for coalesced memory access
    dz = inp.dz.T
    t = inp.t.T
    p = inp.p.T
    rho = inp.rho.T
    qv = inp.q.v.T
    qc = inp.q.c.T
    qr = inp.q.r.T
    qs = inp.q.s.T
    qi = inp.q.i.T
    qg = inp.q.g.T

    # Warmup
    print("\nWarming up (MLIR compilation)...")
    t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
        nlev=inp.nlev,
        dz=dz,
        t=t,
        p=p,
        rho=rho,
        qv=qv,
        qc=qc,
        qr=qr,
        qs=qs,
        qi=qi,
        qg=qg,
        dt=args.dt,
        qnc=args.qnc,
    )
    print("Compilation complete!")

    # Run iterations with timing
    start_time = None
    num_iters = int(args.itime)

    for iteration in range(num_iters + 1):
        if iteration == 1:  # Start timing after first warmup iteration
            start_time = time.time()

        t_out, q_out, pflx, pr, ps, pi, pg, pre = graupel_run(
            nlev=inp.nlev,
            dz=dz,
            t=t,
            p=p,
            rho=rho,
            qv=qv,
            qc=qc,
            qr=qr,
            qs=qs,
            qi=qi,
            qg=qg,
            dt=args.dt,
            qnc=args.qnc,
        )

    end_time = time.time()

    if start_time is not None:
        elapsed_time = end_time - start_time
        print(f"\nFor {num_iters} iterations it took {elapsed_time:.4f} seconds!")
        print(f"Time per iteration: {elapsed_time / num_iters:.4f} seconds")

    # Transpose outputs back to (ncells, nlev)
    t_final = t_out.T
    # q_out is stacked as [qv, qc, qr, qs, qi, qg]
    qv_out = q_out[0].T
    qc_out = q_out[1].T
    qr_out = q_out[2].T
    qs_out = q_out[3].T
    qi_out = q_out[4].T
    qg_out = q_out[5].T
    pflx_out = pflx.T

    # Create output structure
    out = GraupelOutput(
        t=t_final,
        q=Q(
            v=qv_out,
            c=qc_out,
            r=qr_out,
            s=qs_out,
            i=qi_out,
            g=qg_out,
        ),
        pflx=pflx_out,
        pr=pr,
        ps=ps,
        pi=pi,
        pg=pg,
        pre=pre,
    )

    # Verification
    print("\nOutput verification:")
    print(f"Temperature range: {out.t.min():.1f} - {out.t.max():.1f} K")
    print(f"Temperature change: {(out.t - inp.t).mean():.2e} K")
    print(f"Max precipitation flux: {out.pflx.max():.2e}")

    # Write output
    print(f"\nWriting output to: {args.output_file}")
    out.write(args.output_file)
    print("Done!")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
