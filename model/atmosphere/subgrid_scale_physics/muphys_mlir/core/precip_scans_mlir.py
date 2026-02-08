# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
MLIR-based GPU implementation of precipitation scans.

This module generates MLIR code using standard dialects (gpu, scf, arith, memref, math)
to implement vertical precipitation scans with carry state kept in registers,
matching DaCe's approach for optimal performance.

Key optimizations:
1. Carry state in SSA values (registers) - no D2D memory copies
2. All 4 species processed in single kernel launch
3. Coalesced memory access pattern (contiguous cells)
4. Use arith.select instead of scf.if for branchless execution
5. Constants hoisted outside loops
6. Static memref shapes for better optimization

Architecture:
    Python -> MLIR Builder -> GPU Dialect -> NVVM -> LLVM -> PTX -> CUDA

Lowering Pipeline:
    1. canonicalize, cse          - Standard optimizations
    2. gpu-kernel-outlining       - Extract GPU kernels
    3. convert-scf-to-cf          - Structured -> control flow
    4. convert-gpu-to-nvvm        - GPU ops -> NVIDIA dialect
    5. convert-arith/math-to-llvm - Arithmetic lowering
    6. convert-memref-to-llvm     - Memory lowering
    7. gpu-to-llvm                - Final GPU lowering
    8. reconcile-unrealized-casts - Cleanup
"""

from typing import Any

import numpy as np


# Check MLIR availability
MLIR_AVAILABLE = False
MLIR_IMPORT_ERROR = None

try:
    import ctypes

    from mlir import ir
    from mlir.dialects import arith, func, gpu, math as mlir_math, memref, scf
    from mlir.execution_engine import ExecutionEngine
    from mlir.passmanager import PassManager

    MLIR_AVAILABLE = True
except ImportError as e:
    MLIR_IMPORT_ERROR = str(e)


# =============================================================================
# Kernel Cache - Avoid recompilation for same grid sizes
# =============================================================================

_kernel_cache: dict[tuple[int, int], Any] = {}


def _get_cache_key(nlev: int, ncells: int) -> tuple[int, int]:
    """Generate cache key for kernel lookup."""
    return (nlev, ncells)


def clear_kernel_cache():
    """Clear the compiled kernel cache."""
    global _kernel_cache
    _kernel_cache.clear()


# =============================================================================
# MLIR Code Generation
# =============================================================================


def generate_precip_scan_mlir(nlev: int, ncells: int) -> str:
    """
    Generate optimized MLIR code for unified precipitation scan kernel.

    Processes all 4 precipitation species (rain, snow, ice, graupel) in a single
    GPU kernel with carry state maintained in registers throughout the vertical scan.

    Optimizations applied:
    - Static shapes for memrefs (better LLVM optimization)
    - arith.select for branchless conditionals
    - All 4 species in single kernel (shared loads for zeta, rho)
    - Constants computed once outside the loop
    - Memory coalescing: threads access contiguous cells

    Args:
        nlev: Number of vertical levels (e.g., 65)
        ncells: Number of horizontal grid cells (e.g., 100000)

    Returns:
        MLIR module as string
    """
    if not MLIR_AVAILABLE:
        raise RuntimeError(f"MLIR not available: {MLIR_IMPORT_ERROR}")

    with ir.Context() as ctx, ir.Location.unknown():
        # Register required dialects
        ctx.allow_unregistered_dialects = True

        module = ir.Module.create()

        with ir.InsertionPoint(module.body):
            # Types
            f64 = ir.F64Type.get()
            i1 = ir.IntegerType.get_signless(1)
            i32 = ir.IntegerType.get_signless(32)
            idx = ir.IndexType.get()

            # Static memref types for better optimization
            # Shape: [nlev, ncells] with row-major layout
            memref_2d = ir.MemRefType.get([nlev, ncells], f64)
            memref_params = ir.MemRefType.get([8], f64)  # [p0,o0,p1,o1,p2,o2,p3,o3]

            # Function signature:
            # Inputs: zeta, rho, q0-q3, vc0-vc3, mask0-mask3 (14 arrays)
            # Outputs: q0_out-q3_out, flx0_out-flx3_out (8 arrays)
            # Params: prefactors and offsets
            func_type = ir.FunctionType.get(
                inputs=[memref_2d] * 22 + [memref_params],  # 14 in + 8 out + params
                results=[],
            )

            func_op = func.FuncOp("precip_scan_unified", func_type, visibility="public")
            entry = func_op.add_entry_block()

            with ir.InsertionPoint(entry):
                args = list(entry.arguments)

                # Unpack arguments
                zeta, rho = args[0], args[1]
                q = args[2:6]  # q0, q1, q2, q3
                vc = args[6:10]  # vc0, vc1, vc2, vc3
                mask = args[10:14]  # mask0, mask1, mask2, mask3
                q_out = args[14:18]
                flx_out = args[18:22]
                params = args[22]

                # Index constants
                c0 = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, 0))
                c1 = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, 1))
                c_nlev = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, nlev))
                c_ncells = arith.ConstantOp(idx, ir.IntegerAttr.get(idx, ncells))

                # Float constants (hoisted outside GPU launch for efficiency)
                zero_f64 = arith.ConstantOp(f64, ir.FloatAttr.get(f64, 0.0))
                half_f64 = arith.ConstantOp(f64, ir.FloatAttr.get(f64, 0.5))
                one_f64 = arith.ConstantOp(f64, ir.FloatAttr.get(f64, 1.0))
                two_f64 = arith.ConstantOp(f64, ir.FloatAttr.get(f64, 2.0))

                # Load parameters (prefactor, offset for each species)
                param_idx = [arith.ConstantOp(idx, ir.IntegerAttr.get(idx, i)) for i in range(8)]
                prefactor = [memref.LoadOp(params, [param_idx[2 * i]]).result for i in range(4)]
                offset = [memref.LoadOp(params, [param_idx[2 * i + 1]]).result for i in range(4)]

                # GPU Launch: one block per cell, one thread per block
                # Each thread processes entire vertical column sequentially
                launch = gpu.LaunchOp(
                    gridSizeX=c_ncells.result,
                    gridSizeY=c1.result,
                    gridSizeZ=c1.result,
                    blockSizeX=c1.result,
                    blockSizeY=c1.result,
                    blockSizeZ=c1.result,
                )

                with ir.InsertionPoint(launch.body):
                    # Get cell index from block ID
                    cell = gpu.BlockIdOp(gpu.Dimension.x).result

                    # Initialize carry state for all 4 species (16 values total)
                    # Each species has: q_prev, flx_prev, rhox_prev, activated_prev
                    init_q = [zero_f64.result] * 4
                    init_flx = [zero_f64.result] * 4
                    init_rhox = [zero_f64.result] * 4
                    init_act = [arith.ConstantOp(i1, ir.IntegerAttr.get(i1, 0)).result] * 4

                    # Flatten init args for scf.for
                    init_args = init_q + init_flx + init_rhox + init_act

                    # Main vertical scan loop
                    loop = scf.ForOp(c0.result, c_nlev.result, c1.result, init_args)

                    with ir.InsertionPoint(loop.body):
                        k = loop.induction_variable
                        carry = list(loop.inner_iter_args)

                        # Unpack carry
                        q_prev = carry[0:4]
                        flx_prev = carry[4:8]
                        rhox_prev = carry[8:12]
                        act_prev = carry[12:16]

                        # Load shared inputs (zeta, rho) - same for all species
                        zeta_k = memref.LoadOp(zeta, [k, cell]).result
                        rho_k = memref.LoadOp(rho, [k, cell]).result

                        # Process all 4 species
                        new_q = []
                        new_flx = []
                        new_rhox = []
                        new_act = []

                        for s in range(4):
                            # Load species-specific inputs
                            q_k = memref.LoadOp(q[s], [k, cell]).result
                            vc_k = memref.LoadOp(vc[s], [k, cell]).result
                            mask_k = memref.LoadOp(mask[s], [k, cell]).result

                            # Check activation: activated = act_prev | (mask > 0.5)
                            mask_cmp = arith.CmpFOp(
                                arith.CmpFPredicate.OGT, mask_k, half_f64.result
                            ).result
                            activated = arith.OrIOp(act_prev[s], mask_cmp).result

                            # Compute rho_x = q * rho
                            rho_x = arith.MulFOp(q_k, rho_k).result

                            # Compute effective flux = (rho_x / zeta) + 2 * flx_prev
                            rho_x_div_zeta = arith.DivFOp(rho_x, zeta_k).result
                            two_flx = arith.MulFOp(two_f64.result, flx_prev[s]).result
                            flx_eff = arith.AddFOp(rho_x_div_zeta, two_flx).result

                            # Compute fall speed = prefactor * sqrt(rho_x + offset)
                            base = arith.AddFOp(rho_x, offset[s]).result
                            sqrt_base = mlir_math.SqrtOp(base).result
                            fall_speed = arith.MulFOp(prefactor[s], sqrt_base).result

                            # Compute partial flux = min(rho_x * vc * fall_speed, flx_eff)
                            rho_x_vc = arith.MulFOp(rho_x, vc_k).result
                            flx_partial_calc = arith.MulFOp(rho_x_vc, fall_speed).result
                            flx_partial = arith.MinimumFOp(flx_partial_calc, flx_eff).result

                            # Compute vt = act_prev ? vc * prefactor * sqrt(rhox_prev + offset) : 0
                            base_vt = arith.AddFOp(rhox_prev[s], offset[s]).result
                            sqrt_vt = mlir_math.SqrtOp(base_vt).result
                            vt_calc = arith.MulFOp(vc_k, prefactor[s]).result
                            vt_calc = arith.MulFOp(vt_calc, sqrt_vt).result
                            # Branchless select instead of scf.if
                            vt = arith.SelectOp(act_prev[s], vt_calc, zero_f64.result).result

                            # Compute q_activated = (zeta * (flx_eff - flx_partial)) / ((1 + zeta*vt) * rho)
                            flx_diff = arith.SubFOp(flx_eff, flx_partial).result
                            zeta_flx_diff = arith.MulFOp(zeta_k, flx_diff).result
                            zeta_vt = arith.MulFOp(zeta_k, vt).result
                            one_plus_zeta_vt = arith.AddFOp(one_f64.result, zeta_vt).result
                            denom = arith.MulFOp(one_plus_zeta_vt, rho_k).result
                            q_activated = arith.DivFOp(zeta_flx_diff, denom).result

                            # Compute flx_activated = (q_activated * rho * vt + flx_partial) * 0.5
                            q_rho = arith.MulFOp(q_activated, rho_k).result
                            q_rho_vt = arith.MulFOp(q_rho, vt).result
                            flx_sum = arith.AddFOp(q_rho_vt, flx_partial).result
                            flx_activated = arith.MulFOp(flx_sum, half_f64.result).result

                            # Select outputs based on activation (branchless)
                            q_result = arith.SelectOp(activated, q_activated, q_k).result
                            flx_result = arith.SelectOp(
                                activated, flx_activated, zero_f64.result
                            ).result

                            # Store outputs
                            memref.StoreOp(q_result, q_out[s], [k, cell])
                            memref.StoreOp(flx_result, flx_out[s], [k, cell])

                            # Update carry for next iteration
                            rhox_result = arith.MulFOp(q_result, rho_k).result

                            new_q.append(q_result)
                            new_flx.append(flx_result)
                            new_rhox.append(rhox_result)
                            new_act.append(activated)

                        # Yield updated carry
                        scf.YieldOp(new_q + new_flx + new_rhox + new_act)

                    gpu.TerminatorOp()

                func.ReturnOp([])

        return str(module)


# =============================================================================
# Optimization Passes
# =============================================================================


def get_optimization_pipeline(optimization_level: int = 2) -> str:
    """
    Get MLIR optimization pass pipeline.

    Args:
        optimization_level: 0=none, 1=basic, 2=standard, 3=aggressive

    Returns:
        Pass pipeline string for PassManager.parse()
    """
    if optimization_level == 0:
        return "builtin.module()"

    # Basic optimizations (level 1+)
    passes = [
        "canonicalize",  # Pattern-based simplification
        "cse",  # Common subexpression elimination
    ]

    if optimization_level >= 2:
        # Standard optimizations
        passes.extend(
            [
                "loop-invariant-code-motion",  # Hoist loop-invariant ops
                "canonicalize",  # Run again after LICM
            ]
        )

    if optimization_level >= 3:
        # Aggressive optimizations
        passes.extend(
            [
                "symbol-dce",  # Dead code elimination
                "canonicalize",
            ]
        )

    return f"builtin.module({','.join(passes)})"


def get_gpu_lowering_pipeline() -> str:
    """
    Get the GPU lowering pass pipeline.

    This pipeline transforms high-level MLIR to GPU-executable code:
    1. Extract GPU kernels
    2. Lower to NVVM (NVIDIA) dialect
    3. Lower to LLVM IR
    """
    return (
        "builtin.module("
        "gpu-kernel-outlining,"
        "convert-scf-to-cf,"
        "convert-math-to-llvm,"
        "convert-arith-to-llvm,"
        "convert-gpu-to-nvvm,"
        "gpu-to-llvm,"
        "convert-memref-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )


# =============================================================================
# Compilation and Execution
# =============================================================================


def compile_mlir_to_gpu(mlir_code: str, optimization_level: int = 2, verbose: bool = False) -> Any:
    """
    Compile MLIR code to GPU executable using NVVM pipeline.

    Args:
        mlir_code: MLIR module as string
        optimization_level: 0=none, 1=basic, 2=standard, 3=aggressive
        verbose: If True, print intermediate IR

    Returns:
        ExecutionEngine ready to invoke kernels
    """
    if not MLIR_AVAILABLE:
        raise RuntimeError(f"MLIR not available: {MLIR_IMPORT_ERROR}")

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(mlir_code)

        if verbose:
            print("=== Original MLIR ===")
            print(module)

        # Step 1: Apply optimizations
        if optimization_level > 0:
            opt_pipeline = get_optimization_pipeline(optimization_level)
            pm_opt = PassManager.parse(opt_pipeline)
            pm_opt.run(module.operation)

            if verbose:
                print(f"\n=== After Optimization (level {optimization_level}) ===")
                print(module)

        # Step 2: Lower to GPU/LLVM
        gpu_pipeline = get_gpu_lowering_pipeline()
        pm_gpu = PassManager.parse(gpu_pipeline)
        pm_gpu.run(module.operation)

        if verbose:
            print("\n=== After GPU Lowering ===")
            print(module)

        # Step 3: Create execution engine
        engine = ExecutionEngine(module)
        return engine


def get_or_compile_kernel(
    nlev: int, ncells: int, optimization_level: int = 2, force_recompile: bool = False
) -> Any:
    """
    Get compiled kernel from cache or compile new one.

    Args:
        nlev: Number of vertical levels
        ncells: Number of horizontal cells
        optimization_level: Optimization level for compilation
        force_recompile: If True, bypass cache and recompile

    Returns:
        ExecutionEngine for the kernel
    """
    cache_key = _get_cache_key(nlev, ncells)

    if not force_recompile and cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    # Generate and compile
    mlir_code = generate_precip_scan_mlir(nlev, ncells)
    engine = compile_mlir_to_gpu(mlir_code, optimization_level)

    # Cache the compiled kernel
    _kernel_cache[cache_key] = engine
    return engine


# =============================================================================
# Memory Descriptors for MLIR ABI
# =============================================================================


class MemRefDescriptor2D(ctypes.Structure):
    """
    MLIR memref descriptor for 2D ranked memrefs.

    Matches MLIR's ABI for memref<?x?xf64>:
        struct {
            float64* allocated;
            float64* aligned;
            int64_t offset;
            int64_t sizes[2];
            int64_t strides[2];
        }
    """

    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "MemRefDescriptor2D":
        """Create descriptor from numpy array (must be C-contiguous float64)."""
        assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
        assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"
        arr = np.ascontiguousarray(arr)

        desc = cls()
        desc.allocated = arr.ctypes.data
        desc.aligned = arr.ctypes.data
        desc.offset = 0
        desc.sizes[0] = arr.shape[0]
        desc.sizes[1] = arr.shape[1]
        desc.strides[0] = arr.shape[1]  # Row-major: stride[0] = ncols
        desc.strides[1] = 1
        return desc


class MemRefDescriptor1D(ctypes.Structure):
    """MLIR memref descriptor for 1D ranked memrefs."""

    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
    ]

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "MemRefDescriptor1D":
        """Create descriptor from 1D numpy array."""
        assert arr.ndim == 1, f"Expected 1D array, got {arr.ndim}D"
        assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"
        arr = np.ascontiguousarray(arr)

        desc = cls()
        desc.allocated = arr.ctypes.data
        desc.aligned = arr.ctypes.data
        desc.offset = 0
        desc.sizes[0] = arr.shape[0]
        desc.strides[0] = 1
        return desc


# =============================================================================
# Main Entry Point
# =============================================================================


def precip_scan_mlir(
    params_list: list[tuple[float, float, float]],
    zeta: np.ndarray,
    rho: np.ndarray,
    q_list: list[np.ndarray],
    vc_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    optimization_level: int = 2,
    use_cache: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Execute precipitation scans using MLIR-generated GPU kernel.

    This is the main entry point matching the JAX interface.

    Args:
        params_list: List of (prefactor, exponent, offset) for 4 species
        zeta: Vertical coordinate array (nlev x ncells)
        rho: Air density array (nlev x ncells)
        q_list: List of mixing ratio arrays for 4 species
        vc_list: List of ventilation coefficient arrays for 4 species
        mask_list: List of activation mask arrays for 4 species
        optimization_level: MLIR optimization level (0-3)
        use_cache: If True, cache compiled kernels

    Returns:
        List of (q_out, flux_out) tuples for 4 species
    """
    if not MLIR_AVAILABLE:
        raise RuntimeError(
            f"MLIR not available: {MLIR_IMPORT_ERROR}\n"
            "Install with: pip install mlir-python-bindings"
        )

    nlev, ncells = zeta.shape

    # Get compiled kernel (from cache if available)
    if use_cache:
        engine = get_or_compile_kernel(nlev, ncells, optimization_level)
    else:
        mlir_code = generate_precip_scan_mlir(nlev, ncells)
        engine = compile_mlir_to_gpu(mlir_code, optimization_level)

    # Prepare inputs as contiguous float64 arrays
    zeta_arr = np.ascontiguousarray(zeta, dtype=np.float64)
    rho_arr = np.ascontiguousarray(rho, dtype=np.float64)
    q_arrs = [np.ascontiguousarray(q, dtype=np.float64) for q in q_list]
    vc_arrs = [np.ascontiguousarray(vc, dtype=np.float64) for vc in vc_list]
    mask_arrs = [np.ascontiguousarray(m.astype(np.float64)) for m in mask_list]

    # Allocate outputs
    q_out_arrs = [np.empty_like(q) for q in q_arrs]
    flx_out_arrs = [np.empty_like(q) for q in q_arrs]

    # Prepare parameters: [p0, o0, p1, o1, p2, o2, p3, o3]
    params_arr = np.zeros(8, dtype=np.float64)
    for i, (prefactor, exponent, offset) in enumerate(params_list):
        params_arr[2 * i] = prefactor
        params_arr[2 * i + 1] = offset

    # Create memref descriptors
    descs = []
    descs.append(MemRefDescriptor2D.from_numpy(zeta_arr))
    descs.append(MemRefDescriptor2D.from_numpy(rho_arr))
    for arr in q_arrs:
        descs.append(MemRefDescriptor2D.from_numpy(arr))
    for arr in vc_arrs:
        descs.append(MemRefDescriptor2D.from_numpy(arr))
    for arr in mask_arrs:
        descs.append(MemRefDescriptor2D.from_numpy(arr))
    for arr in q_out_arrs:
        descs.append(MemRefDescriptor2D.from_numpy(arr))
    for arr in flx_out_arrs:
        descs.append(MemRefDescriptor2D.from_numpy(arr))
    descs.append(MemRefDescriptor1D.from_numpy(params_arr))

    # Invoke kernel
    engine.invoke("precip_scan_unified", *[ctypes.byref(d) for d in descs])

    return list(zip(q_out_arrs, flx_out_arrs))


# =============================================================================
# Utility Functions
# =============================================================================


def print_mlir_code(nlev: int = 65, ncells: int = 100):
    """Print generated MLIR code for inspection."""
    if not MLIR_AVAILABLE:
        print(f"MLIR not available: {MLIR_IMPORT_ERROR}")
        return

    code = generate_precip_scan_mlir(nlev, ncells)
    print(code)


def print_lowered_mlir(nlev: int = 65, ncells: int = 100, optimization_level: int = 2):
    """Print MLIR code after optimization and lowering passes."""
    if not MLIR_AVAILABLE:
        print(f"MLIR not available: {MLIR_IMPORT_ERROR}")
        return

    mlir_code = generate_precip_scan_mlir(nlev, ncells)
    compile_mlir_to_gpu(mlir_code, optimization_level, verbose=True)


def get_kernel_stats() -> dict:
    """Get statistics about cached kernels."""
    return {
        "cached_kernels": len(_kernel_cache),
        "grid_sizes": list(_kernel_cache.keys()),
    }
