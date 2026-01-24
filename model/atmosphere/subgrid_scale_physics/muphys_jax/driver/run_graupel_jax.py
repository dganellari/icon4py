#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
JAX graupel driver compatible with GT4Py run_graupel_only.py interface.
Uses the same NetCDF input files and produces comparable output.

Usage:
    python run_graupel_jax.py -o output.nc -b xla input.nc 10 30.0 100.0
"""

import argparse
import pathlib
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from muphys_jax.core.definitions import Q
from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.implementations.graupel_allinone_fused import graupel_allinone_fused_run
from muphys_jax.implementations.graupel_baseline import graupel_run as graupel_baseline_run

# --- CUDA context warmup for Triton/JAX interop ---
try:
    import torch
    if torch.cuda.is_available():
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
except Exception:
    pass


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


class GraupelInput(NamedTuple):
    """Input data for graupel scheme."""

    ncells: int
    nlev: int
    dz: jnp.ndarray
    p: jnp.ndarray
    rho: jnp.ndarray
    t: jnp.ndarray
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
                v=jnp.array(load_var("hus")),  # specific humidity (vapor)
                c=jnp.array(load_var("clw")),  # cloud liquid water
                r=jnp.array(load_var("qr")),  # rain
                s=jnp.array(load_var("qs")),  # snow
                i=jnp.array(load_var("cli")),  # cloud ice
                g=jnp.array(load_var("qg")),  # graupel
            )

            return cls(
                ncells=ncells,
                nlev=nlev,
                dz=jnp.array(dz),
                t=jnp.array(load_var("ta")),
                p=jnp.array(load_var("pfull")),
                rho=jnp.array(load_var("rho")),
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
        description="Run JAX graupel scheme (compatible with GT4Py run_graupel_only.py)"
    )
    parser.add_argument(
        "-o", metavar="output_file", dest="output_file", help="output filename", default="output.nc"
    )
    parser.add_argument(
        "-b", metavar="backend", dest="backend", help="JAX backend (xla or iree)", default="xla"
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
        "--fused",
        help="use fused scans (90 kernel launches instead of 180)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--allinone-fused",
        help="use all-in-one fused scan (single JAX scan, experimental)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tiled",
        help="use tiled scans (process multiple levels per iteration)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tile-size",
        help="number of levels per tiled scan iteration",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--no-layout-opt",
        help="disable memory layout optimization",
        action="store_true",
        default=False,  # Enable layout opt by default
    )
    parser.add_argument(
        "--unrolled",
        help="use unrolled loop (SINGLE KERNEL, target DaCe performance)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pallas",
        help="use Pallas GPU kernel (requires jax[cuda12_pallas])",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--triton",
        help="use Triton CUDA kernel (requires triton, jax-triton) - TARGET DACE PERF",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mlir",
        help="use MLIR GPU kernel (requires mlir-python-bindings) - TARGET DACE PERF",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--baseline",
        help="use baseline implementation (vmap-batched, 90 kernels)",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Set JAX backend
    import os

    os.environ["JAX_BACKEND"] = args.backend
    print(f"Using JAX backend: {args.backend}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    # Load input data
    print(f"Loading input from: {args.input_file}")
    inp = GraupelInput.load(pathlib.Path(args.input_file))
    print(f"Grid size: {inp.ncells} cells × {inp.nlev} levels")
    print(f"Temperature range: {np.array(inp.t).min():.1f} - {np.array(inp.t).max():.1f} K")
    print(f"Timestep: {args.dt} s")
    print(f"Cloud droplet concentration: {args.qnc} m^-3")

    # Warmup compilation
    print("\nWarming up (JIT compilation)...")
    if args.baseline:
        print("Mode: BASELINE (vmap-batched, 90 kernels)")
    elif args.mlir:
        print("Mode: MLIR (GPU kernel via MLIR dialects - TARGET DACE PERF)")
    elif args.triton:
        print("Mode: TRITON (custom CUDA kernel - TARGET DACE PERF)")
    elif args.pallas:
        print("Mode: PALLAS (vmap + fori_loop)")
    elif args.unrolled:
        print("Mode: UNROLLED (static unroll)")
    elif args.tiled:
        print(f"Mode: TILED (tile_size={args.tile_size}, {90//args.tile_size} iterations)")
    elif args.allinone_fused:
        print("Mode: ALL-IN-ONE FUSED SCAN (single JAX scan, experimental)")
    elif args.fused:
        print("Mode: FUSED SCANS (90 kernels)")
    else:
        print("Mode: DEFAULT (180 kernels)")
    print(f"Layout optimization: {'DISABLED' if args.no_layout_opt else 'ENABLED'}")

    # Choose which implementation to use
    if args.baseline:
        run_func = graupel_baseline_run
        run_kwargs = {}  # Baseline doesn't accept any optimization flags
    elif args.allinone_fused:
        run_func = graupel_allinone_fused_run
        run_kwargs = {}  # All-in-one fused doesn't accept optimization flags
    else:
        run_func = graupel_run
        run_kwargs = dict(
            use_fused_scans=args.fused, use_tiled_scans=args.tiled, tile_size=args.tile_size,
            optimize_layout=not args.no_layout_opt, use_unrolled=args.unrolled, use_pallas=args.pallas,
            use_triton=args.triton, use_mlir=args.mlir
        )

    t_out, q_out, pflx, pr, ps, pi, pg, pre = run_func(
        inp.dz, inp.t, inp.p, inp.rho, inp.q, args.dt, args.qnc, **run_kwargs
    )
    t_out.block_until_ready()
    print("Compilation complete!")

    # Run iterations with timing
    start_time = None
    num_iters = int(args.itime)

    for iteration in range(num_iters + 1):
        if iteration == 1:
            start_time = time.time()
        t_out, q_out, pflx, pr, ps, pi, pg, pre = run_func(
            inp.dz, inp.t, inp.p, inp.rho, inp.q, args.dt, args.qnc, **run_kwargs
        )

    t_out.block_until_ready()
    end_time = time.time()

    if start_time is not None:
        elapsed_time = end_time - start_time
        print(f"\nFor {num_iters} iterations it took {elapsed_time:.4f} seconds!")
        print(f"Time per iteration: {elapsed_time / num_iters:.4f} seconds")

    print("\nConverting outputs to numpy arrays...")
    out = GraupelOutput(
        t=np.array(t_out),
        q=Q(
            v=np.array(q_out.v),
            c=np.array(q_out.c),
            r=np.array(q_out.r),
            s=np.array(q_out.s),
            i=np.array(q_out.i),
            g=np.array(q_out.g),
        ),
        pflx=np.array(pflx),
        pr=np.array(pr),
        ps=np.array(ps),
        pi=np.array(pi),
        pg=np.array(pg),
        pre=np.array(pre),
    )

    # Verification
    print("\nOutput verification:")
    print(f"Temperature range: {out.t.min():.1f} - {out.t.max():.1f} K")
    print(f"Temperature change: {(out.t - np.array(inp.t)).mean():.2e} K")
    print(f"Max precipitation flux: {out.pflx.max():.2e}")

    # Write output
    print(f"\nWriting output to: {args.output_file}")
    out.write(args.output_file)
    print("Done!")


if __name__ == "__main__":
    main()
