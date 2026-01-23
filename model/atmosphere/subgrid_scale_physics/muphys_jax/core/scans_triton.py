# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Triton GPU kernels for vertical scans in graupel microphysics.
Uses direct Triton compilation (bypassing jax-triton due to API compatibility issues).
"""

import jax
import jax.numpy as jnp

# Check Triton availability
TRITON_AVAILABLE = False
TRITON_IMPORT_ERROR = None

try:
    import torch
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    TRITON_IMPORT_ERROR = str(e)
    TORCH_CUDA_AVAILABLE = False
except Exception as e:
    TRITON_IMPORT_ERROR = str(e)
    TORCH_CUDA_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _precip_scan_unified_kernel(
        # Inputs (pointers) - all 4 species
        zeta_ptr, rho_ptr,
        q0_ptr, q1_ptr, q2_ptr, q3_ptr,
        vc0_ptr, vc1_ptr, vc2_ptr, vc3_ptr,
        mask0_ptr, mask1_ptr, mask2_ptr, mask3_ptr,
        # Outputs (pointers) - all 4 species
        q0_out_ptr, q1_out_ptr, q2_out_ptr, q3_out_ptr,
        flx0_out_ptr, flx1_out_ptr, flx2_out_ptr, flx3_out_ptr,
        # Parameters - 4 species (prefactor, offset for each)
        prefactor0, offset0,
        prefactor1, offset1,
        prefactor2, offset2,
        prefactor3, offset3,
        # Dimensions
        ncells, nlev,
        # Strides
        stride_lev, stride_cell,
    ):
        """Unified Triton kernel for all 4 precipitation species.

        Uses sqrt() instead of pow() since all species have exponent=0.5.
        """
        cell_idx = tl.program_id(0)

        if cell_idx >= ncells:
            return

        # Initialize carry state - load first element to get dtype, multiply by 0
        first_idx = cell_idx * stride_cell
        zero = tl.load(zeta_ptr + first_idx) * 0.0
        q_prev_0, flx_prev_0, rhox_prev_0, activated_prev_0 = zero, zero, zero, 0
        q_prev_1, flx_prev_1, rhox_prev_1, activated_prev_1 = zero, zero, zero, 0
        q_prev_2, flx_prev_2, rhox_prev_2, activated_prev_2 = zero, zero, zero, 0
        q_prev_3, flx_prev_3, rhox_prev_3, activated_prev_3 = zero, zero, zero, 0

        # Process all vertical levels sequentially
        for k in range(nlev):
            idx = k * stride_lev + cell_idx * stride_cell

            # Load shared inputs
            zeta_k = tl.load(zeta_ptr + idx)
            rho_k = tl.load(rho_ptr + idx)

            # Load species-specific inputs
            q0_k = tl.load(q0_ptr + idx)
            q1_k = tl.load(q1_ptr + idx)
            q2_k = tl.load(q2_ptr + idx)
            q3_k = tl.load(q3_ptr + idx)

            vc0_k = tl.load(vc0_ptr + idx)
            vc1_k = tl.load(vc1_ptr + idx)
            vc2_k = tl.load(vc2_ptr + idx)
            vc3_k = tl.load(vc3_ptr + idx)

            mask0_bool = tl.load(mask0_ptr + idx) > 0.5
            mask1_bool = tl.load(mask1_ptr + idx) > 0.5
            mask2_bool = tl.load(mask2_ptr + idx) > 0.5
            mask3_bool = tl.load(mask3_ptr + idx) > 0.5

            # === Process species 0 ===
            activated_0 = (activated_prev_0 > 0) | mask0_bool
            rho_x_0 = q0_k * rho_k
            flx_eff_0 = (rho_x_0 / zeta_k) + 2.0 * flx_prev_0
            base0 = rho_x_0 + offset0
            fall_speed_0 = prefactor0 * tl.sqrt(base0)
            flx_partial_0 = tl.minimum(rho_x_0 * vc0_k * fall_speed_0, flx_eff_0)
            base0_vt = rhox_prev_0 + offset0
            vt_0 = tl.where(activated_prev_0 > 0, vc0_k * prefactor0 * tl.sqrt(base0_vt), zero)
            q_activated_0 = (zeta_k * (flx_eff_0 - flx_partial_0)) / ((1.0 + zeta_k * vt_0) * rho_k)
            flx_activated_0 = (q_activated_0 * rho_k * vt_0 + flx_partial_0) * 0.5
            q0_out = tl.where(activated_0, q_activated_0, q0_k)
            flx0_out = tl.where(activated_0, flx_activated_0, zero)

            # === Process species 1 ===
            activated_1 = (activated_prev_1 > 0) | mask1_bool
            rho_x_1 = q1_k * rho_k
            flx_eff_1 = (rho_x_1 / zeta_k) + 2.0 * flx_prev_1
            base1 = rho_x_1 + offset1
            fall_speed_1 = prefactor1 * tl.sqrt(base1)
            flx_partial_1 = tl.minimum(rho_x_1 * vc1_k * fall_speed_1, flx_eff_1)
            base1_vt = rhox_prev_1 + offset1
            vt_1 = tl.where(activated_prev_1 > 0, vc1_k * prefactor1 * tl.sqrt(base1_vt), zero)
            q_activated_1 = (zeta_k * (flx_eff_1 - flx_partial_1)) / ((1.0 + zeta_k * vt_1) * rho_k)
            flx_activated_1 = (q_activated_1 * rho_k * vt_1 + flx_partial_1) * 0.5
            q1_out = tl.where(activated_1, q_activated_1, q1_k)
            flx1_out = tl.where(activated_1, flx_activated_1, zero)

            # === Process species 2 ===
            activated_2 = (activated_prev_2 > 0) | mask2_bool
            rho_x_2 = q2_k * rho_k
            flx_eff_2 = (rho_x_2 / zeta_k) + 2.0 * flx_prev_2
            base2 = rho_x_2 + offset2
            fall_speed_2 = prefactor2 * tl.sqrt(base2)
            flx_partial_2 = tl.minimum(rho_x_2 * vc2_k * fall_speed_2, flx_eff_2)
            base2_vt = rhox_prev_2 + offset2
            vt_2 = tl.where(activated_prev_2 > 0, vc2_k * prefactor2 * tl.sqrt(base2_vt), zero)
            q_activated_2 = (zeta_k * (flx_eff_2 - flx_partial_2)) / ((1.0 + zeta_k * vt_2) * rho_k)
            flx_activated_2 = (q_activated_2 * rho_k * vt_2 + flx_partial_2) * 0.5
            q2_out = tl.where(activated_2, q_activated_2, q2_k)
            flx2_out = tl.where(activated_2, flx_activated_2, zero)

            # === Process species 3 ===
            activated_3 = (activated_prev_3 > 0) | mask3_bool
            rho_x_3 = q3_k * rho_k
            flx_eff_3 = (rho_x_3 / zeta_k) + 2.0 * flx_prev_3
            base3 = rho_x_3 + offset3
            fall_speed_3 = prefactor3 * tl.sqrt(base3)
            flx_partial_3 = tl.minimum(rho_x_3 * vc3_k * fall_speed_3, flx_eff_3)
            base3_vt = rhox_prev_3 + offset3
            vt_3 = tl.where(activated_prev_3 > 0, vc3_k * prefactor3 * tl.sqrt(base3_vt), zero)
            q_activated_3 = (zeta_k * (flx_eff_3 - flx_partial_3)) / ((1.0 + zeta_k * vt_3) * rho_k)
            flx_activated_3 = (q_activated_3 * rho_k * vt_3 + flx_partial_3) * 0.5
            q3_out = tl.where(activated_3, q_activated_3, q3_k)
            flx3_out = tl.where(activated_3, flx_activated_3, zero)

            # Store outputs
            tl.store(q0_out_ptr + idx, q0_out)
            tl.store(q1_out_ptr + idx, q1_out)
            tl.store(q2_out_ptr + idx, q2_out)
            tl.store(q3_out_ptr + idx, q3_out)
            tl.store(flx0_out_ptr + idx, flx0_out)
            tl.store(flx1_out_ptr + idx, flx1_out)
            tl.store(flx2_out_ptr + idx, flx2_out)
            tl.store(flx3_out_ptr + idx, flx3_out)

            # Update carry (stays in registers)
            q_prev_0, flx_prev_0, rhox_prev_0 = q0_out, flx0_out, q0_out * rho_k
            activated_prev_0 = tl.where(activated_0, 1, 0)
            q_prev_1, flx_prev_1, rhox_prev_1 = q1_out, flx1_out, q1_out * rho_k
            activated_prev_1 = tl.where(activated_1, 1, 0)
            q_prev_2, flx_prev_2, rhox_prev_2 = q2_out, flx2_out, q2_out * rho_k
            activated_prev_2 = tl.where(activated_2, 1, 0)
            q_prev_3, flx_prev_3, rhox_prev_3 = q3_out, flx3_out, q3_out * rho_k
            activated_prev_3 = tl.where(activated_3, 1, 0)


    def _jax_to_torch_dlpack(jax_array):
        """Convert JAX array to PyTorch tensor via DLPack (zero-copy on GPU)."""
        dlpack_capsule = jax.dlpack.to_dlpack(jax_array)
        return torch.utils.dlpack.from_dlpack(dlpack_capsule)


    def _torch_to_jax_dlpack(torch_tensor):
        """Convert PyTorch tensor to JAX array via DLPack (zero-copy on GPU)."""
        dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
        return jax.dlpack.from_dlpack(dlpack_capsule)


    def _triton_kernel_host(zeta, rho, q0, q1, q2, q3, vc0, vc1, vc2, vc3,
                           mask0_f, mask1_f, mask2_f, mask3_f,
                           p0, o0, p1, o1, p2, o2, p3, o3):
        """Host function that launches Triton kernel via PyTorch."""
        import time

        nlev, ncells = q0.shape

        # Convert JAX arrays to PyTorch (zero-copy via DLPack)
        t0 = time.perf_counter()
        zeta_t = _jax_to_torch_dlpack(zeta)
        rho_t = _jax_to_torch_dlpack(rho)
        q0_t, q1_t, q2_t, q3_t = [_jax_to_torch_dlpack(q) for q in [q0, q1, q2, q3]]
        vc0_t, vc1_t, vc2_t, vc3_t = [_jax_to_torch_dlpack(vc) for vc in [vc0, vc1, vc2, vc3]]
        mask0_t, mask1_t, mask2_t, mask3_t = [_jax_to_torch_dlpack(m) for m in [mask0_f, mask1_f, mask2_f, mask3_f]]
        t1 = time.perf_counter()

        # Allocate outputs
        q0_out_t = torch.empty_like(q0_t)
        q1_out_t = torch.empty_like(q1_t)
        q2_out_t = torch.empty_like(q2_t)
        q3_out_t = torch.empty_like(q3_t)
        flx0_out_t = torch.empty_like(q0_t)
        flx1_out_t = torch.empty_like(q1_t)
        flx2_out_t = torch.empty_like(q2_t)
        flx3_out_t = torch.empty_like(q3_t)
        t2 = time.perf_counter()

        # Synchronize before kernel (only if CUDA available)
        if TORCH_CUDA_AVAILABLE:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Launch kernel
        grid = (ncells,)
        _precip_scan_unified_kernel[grid](
            zeta_t, rho_t, q0_t, q1_t, q2_t, q3_t,
            vc0_t, vc1_t, vc2_t, vc3_t,
            mask0_t, mask1_t, mask2_t, mask3_t,
            q0_out_t, q1_out_t, q2_out_t, q3_out_t,
            flx0_out_t, flx1_out_t, flx2_out_t, flx3_out_t,
            p0, o0, p1, o1, p2, o2, p3, o3,
            ncells, nlev, ncells, 1,
            num_warps=4
        )

        # Synchronize after kernel to get accurate timing
        if TORCH_CUDA_AVAILABLE:
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        # Convert outputs back to JAX
        results = [_torch_to_jax_dlpack(t) for t in [
            q0_out_t, flx0_out_t, q1_out_t, flx1_out_t,
            q2_out_t, flx2_out_t, q3_out_t, flx3_out_t
        ]]
        t5 = time.perf_counter()

        # Print timing breakdown (only first few calls)
        if not hasattr(_triton_kernel_host, '_call_count'):
            _triton_kernel_host._call_count = 0
        _triton_kernel_host._call_count += 1

        if _triton_kernel_host._call_count <= 3:
            print(f"\n[Triton Call {_triton_kernel_host._call_count}] Timing breakdown:")
            print(f"  DLPack input conversion:  {(t1-t0)*1000:.2f} ms")
            print(f"  Output allocation:        {(t2-t1)*1000:.2f} ms")
            print(f"  Pre-kernel sync:          {(t3-t2)*1000:.2f} ms")
            print(f"  Kernel + sync:            {(t4-t3)*1000:.2f} ms  ← KERNEL TIME")
            print(f"  DLPack output conversion: {(t5-t4)*1000:.2f} ms")
            print(f"  Total:                    {(t5-t0)*1000:.2f} ms")

        return results


def precip_scan_triton(params_list, zeta, rho, q_list, vc_list, mask_list):
    """Process 4 precipitation scans using Triton kernel via pure_callback."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton not available: {TRITON_IMPORT_ERROR}")

    if not TORCH_CUDA_AVAILABLE:
        raise RuntimeError(
            "PyTorch with CUDA support is required!\n"
            "Install with: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    q0, q1, q2, q3 = q_list
    vc0, vc1, vc2, vc3 = vc_list
    mask0, mask1, mask2, mask3 = mask_list
    nlev, ncells = q0.shape
    dtype = q0.dtype

    # Convert bool masks to float
    mask0_f = mask0.astype(dtype)
    mask1_f = mask1.astype(dtype)
    mask2_f = mask2.astype(dtype)
    mask3_f = mask3.astype(dtype)

    # Extract parameters (only prefactor and offset, exponent is always 0.5 for sqrt)
    p0, _, o0 = params_list[0]  # Ignore exponent
    p1, _, o1 = params_list[1]
    p2, _, o2 = params_list[2]
    p3, _, o3 = params_list[3]

    # Call via jax.pure_callback
    result_shapes = tuple([jax.ShapeDtypeStruct((nlev, ncells), dtype) for _ in range(8)])

    results = jax.pure_callback(
        lambda *args: _triton_kernel_host(*args),
        result_shapes,
        zeta, rho, q0, q1, q2, q3, vc0, vc1, vc2, vc3,
        mask0_f, mask1_f, mask2_f, mask3_f,
        p0, o0, p1, o1, p2, o2, p3, o3
    )

    q0_out, flx0_out, q1_out, flx1_out, q2_out, flx2_out, q3_out, flx3_out = results
    return [(q0_out, flx0_out), (q1_out, flx1_out), (q2_out, flx2_out), (q3_out, flx3_out)]
