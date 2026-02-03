#!/usr/bin/env python3
"""
Minimal IREE test to identify what triggers the consteval error.
Run with: python tests/test_iree_minimal.py
"""

import os
import sys

# Set IREE backend BEFORE any imports
os.environ["JAX_PLATFORMS"] = "iree_cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# NOTE: IREE CUDA was built with f64->f32 demotion, so we use f32
# os.environ["JAX_ENABLE_X64"] = "true"

# Try to configure local-task executor for consteval
os.environ["IREE_TASK_EXECUTOR_THREAD_COUNT"] = "4"
os.environ["IREE_TASK_TOPOLOGY_GROUP_COUNT"] = "1"
os.environ["IREE_PJRT_CONSTEVAL_BACKENDS"] = "local-sync"

import jax
# jax.config.update("jax_enable_x64", True)  # Disabled - IREE demotes f64 to f32

import jax.numpy as jnp
from jax import lax

print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Test 1: Basic scan (this works according to user)
print("\n=== Test 1: Basic scan ===")
def fun(carry, x):
    return carry + x, carry

@jax.jit
def basic_scan(ins):
    carry, outputs = lax.scan(fun, jnp.zeros((10,), dtype=jnp.float32), ins)
    return outputs

ins = jnp.ones((10, 10), dtype=jnp.float32)
try:
    result = basic_scan(ins)
    print(f"Basic scan: PASSED, shape={result.shape}")
except Exception as e:
    print(f"Basic scan: FAILED - {e}")

# Test 2: Import definitions only
print("\n=== Test 2: Import definitions ===")
try:
    from muphys_jax.core.definitions import Q, TempState
    print("Import definitions: PASSED")
except Exception as e:
    print(f"Import definitions: FAILED - {e}")

# Test 3: Import constants
print("\n=== Test 3: Import constants ===")
try:
    from muphys_jax.core.common import constants as const
    print(f"Import constants: PASSED (qmin={const.qmin})")
except Exception as e:
    print(f"Import constants: FAILED - {e}")

# Test 4: Import thermo functions
print("\n=== Test 4: Import thermo ===")
try:
    from muphys_jax.core import thermo
    print("Import thermo: PASSED")
except Exception as e:
    print(f"Import thermo: FAILED - {e}")

# Test 5: Call a thermo function
print("\n=== Test 5: Call thermo function ===")
try:
    @jax.jit
    def test_thermo(t, rho):
        return thermo.qsat_rho(t, rho)
    
    t = jnp.array([273.15, 280.0, 290.0], dtype=jnp.float32)
    rho = jnp.array([1.2, 1.1, 1.0], dtype=jnp.float32)
    result = test_thermo(t, rho)
    print(f"Call thermo: PASSED, result={result}")
except Exception as e:
    print(f"Call thermo: FAILED - {e}")

# Test 6: Import graupel baseline
print("\n=== Test 6: Import graupel_baseline ===")
try:
    from muphys_jax.implementations.graupel_baseline import graupel_run
    print("Import graupel_baseline: PASSED")
except Exception as e:
    print(f"Import graupel_baseline: FAILED - {e}")

# Test 6b: Test larger lax.scan with multiple outputs
print("\n=== Test 6b: Larger scan with multiple outputs ===")
try:
    def complex_scan_fn(carry, x):
        a, b = carry
        new_a = a + x[0]
        new_b = b * 0.99 + x[1] * 0.01
        return (new_a, new_b), (new_a, new_b)
    
    @jax.jit
    def complex_scan(x1, x2):
        init = (jnp.zeros((4,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.float32))
        final, outputs = lax.scan(complex_scan_fn, init, (x1, x2))
        return outputs
    
    x1 = jnp.ones((5, 4), dtype=jnp.float32) * 0.1
    x2 = jnp.ones((5, 4), dtype=jnp.float32) * 273.0
    out = complex_scan(x1, x2)
    out[0].block_until_ready()
    print(f"Complex scan: PASSED, shapes={out[0].shape}, {out[1].shape}")
except Exception as e:
    print(f"Complex scan: FAILED - {e}")

# Test 7: Import data loading and load data
print("\n=== Test 7: Load data ===")
try:
    from muphys_jax.utils import load_graupel_inputs
    import pathlib
    
    # Use relative path from muphys_jax
    data_path = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "testdata/muphys_graupel_data/mini/input.nc"
    if data_path.exists():
        # load_graupel_inputs returns tuple: (dz, t, p, rho, q, dt, qnc, ncells, nlev)
        dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(data_path)
        print(f"Load data: PASSED, dz shape={dz.shape}, ncells={ncells}, nlev={nlev}")
    else:
        print(f"Load data: SKIPPED (file not found: {data_path})")
except Exception as e:
    print(f"Load data: FAILED - {e}")

# Test 8: Run graupel baseline with small data
print("\n=== Test 8: Run graupel_baseline ===")
try:
    from muphys_jax.core.definitions import Q
    
    # Create tiny test data with STATIC shapes - USE FLOAT32 for IREE CUDA
    ncells, nlev = 4, 5
    
    # Create arrays directly as JAX arrays (simpler path)
    import numpy as np
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    p = jnp.ones((ncells, nlev), dtype=jnp.float32) * 80000.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    q = Q(
        v=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.005,
        c=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
    )
    dt = 30.0
    qnc = 100.0
    
    print(f"  Input shapes: dz={dz.shape}, t={t.shape}, q.v={q.v.shape}")
    print(f"  Input dtypes: dz={dz.dtype}, t={t.dtype}")
    
    result = graupel_run(dz, t, p, rho, q, dt, qnc)
    result[0].block_until_ready()
    print(f"Run graupel_baseline: PASSED, t_out shape={result[0].shape}")
except Exception as e:
    print(f"Run graupel_baseline: FAILED - {e}")

# Test 8b: Test just q_t_update (phase transitions without precipitation)
print("\n=== Test 8b: Run q_t_update only ===")
try:
    from muphys_jax.implementations.graupel_baseline import q_t_update
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    p = jnp.ones((ncells, nlev), dtype=jnp.float32) * 80000.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    q = Q(
        v=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.005,
        c=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
    )
    
    @jax.jit
    def test_qt(t, p, rho, q):
        return q_t_update(t, p, rho, q, 30.0, 100.0)
    
    q_out, t_out = test_qt(t, p, rho, q)
    t_out.block_until_ready()
    print(f"q_t_update: PASSED, t_out shape={t_out.shape}")
except Exception as e:
    print(f"q_t_update: FAILED - {e}")

# Test 8c: Test precipitation scan only
print("\n=== Test 8c: Run precip_scan_batched only ===")
try:
    from muphys_jax.core.scans_baseline import precip_scan_batched
    from muphys_jax.core import properties as props
    from muphys_jax.core.common import constants as const
    
    ncells, nlev = 4, 5
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    qr = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001
    
    # Get velocity scale factor (correct function name)
    xrho = jnp.sqrt(const.rho_00 / rho)
    vc_r = props.vel_scale_factor_default(xrho)
    kmin_r = qr > const.qmin
    zeta = dz / 30.0
    
    # params_list: [(a, b, c)] for rain
    params_list = [(14.58, 0.111, 1.0e-12)]
    
    @jax.jit
    def test_precip(params_list, zeta, rho, q_list, vc_list, mask_list):
        return precip_scan_batched(params_list, zeta, rho, q_list, vc_list, mask_list)
    
    results = test_precip(params_list, zeta, rho, [qr], [vc_r], [kmin_r])
    # results is list of (q, pflx) tuples
    q_out, pflx = results[0]
    q_out.block_until_ready()
    print(f"precip_scan: PASSED, q_out shape={q_out.shape}")
except Exception as e:
    print(f"precip_scan: FAILED - {e}")

# Test 8d: Test full precipitation_effects (combines all species precip scans)
# NOTE: This test FAILS on IREE CUDA due to >16 bindings limit
print("\n=== Test 8d: Run precipitation_effects ===")
print("  SKIPPED - IREE CUDA has 16 binding limit, precipitation_effects needs 17+")

# Test 8d-alt: Test with packed data to reduce bindings
print("\n=== Test 8d-alt: Run precip with packed kmin masks ===")
try:
    from muphys_jax.core.scans_baseline import precip_scan_batched
    from muphys_jax.core import properties as props
    from muphys_jax.core.common import constants as const
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    qr = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001
    qs = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001
    qi = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001
    qg = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001
    
    xrho = jnp.sqrt(const.rho_00 / rho)
    vc_r = props.vel_scale_factor_default(xrho)
    vc_s = props.vel_scale_factor_snow(xrho, rho, t, qs)
    vc_i = props.vel_scale_factor_ice(xrho)
    vc_g = props.vel_scale_factor_default(xrho)
    
    kmin_r = qr > const.qmin
    kmin_s = qs > const.qmin
    kmin_i = qi > const.qmin
    kmin_g = qg > const.qmin
    
    zeta = 30.0 / (2.0 * dz)
    
    params_list = [
        (14.58, 0.111, 1.0e-12),  # rain
        (57.80, 0.16666666666666666, 1.0e-12),  # snow
        (1.25, 0.160, 1.0e-12),  # ice
        (12.24, 0.217, 1.0e-08),  # graupel
    ]
    
    # This still may exceed binding limit with 4 species
    @jax.jit
    def test_all_precip(zeta, rho, qr, qs, qi, qg, vc_r, vc_s, vc_i, vc_g, kmin_r, kmin_s, kmin_i, kmin_g):
        return precip_scan_batched(
            params_list, zeta, rho,
            [qr, qs, qi, qg],
            [vc_r, vc_s, vc_i, vc_g],
            [kmin_r, kmin_s, kmin_i, kmin_g]
        )
    
    results = test_all_precip(zeta, rho, qr, qs, qi, qg, vc_r, vc_s, vc_i, vc_g, kmin_r, kmin_s, kmin_i, kmin_g)
    qr_out, _ = results[0]
    qr_out.block_until_ready()
    print(f"all_precip batched: PASSED, qr_out shape={qr_out.shape}")
except Exception as e:
    print(f"all_precip batched: FAILED - {e}")

# Test 8e: Test with STACKED arrays to reduce binding count
print("\n=== Test 8e: Run precip with stacked arrays (fewer bindings) ===")
try:
    from muphys_jax.core import properties as props
    from muphys_jax.core.common import constants as const
    from muphys_jax.core.scans_baseline import _single_species_scan
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    
    # Single species q
    qr = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001
    
    xrho = jnp.sqrt(const.rho_00 / rho)
    vc_r = props.vel_scale_factor_default(xrho)
    kmin_r = qr > const.qmin
    
    zeta = 30.0 / (2.0 * dz)
    
    # Rain params as array
    params = jnp.array([14.58, 0.111, 1.0e-12], dtype=jnp.float32)
    
    @jax.jit
    def test_single_precip(params, zeta, rho, q, vc, kmin):
        return _single_species_scan(params, zeta, rho, q, vc, kmin)
    
    # Test just rain (single species) - should have ~8 bindings
    qr_out, pflx = test_single_precip(params, zeta, rho, qr, vc_r, kmin_r)
    qr_out.block_until_ready()
    print(f"single species precip: PASSED, qr_out shape={qr_out.shape}")
except Exception as e:
    print(f"stacked precip: FAILED - {e}")

# Test 8f: Combine q_t_update + precipitation in SAME JIT (like graupel_run does)
print("\n=== Test 8f: Combined q_t_update + precip in same JIT ===")
try:
    from muphys_jax.implementations.graupel_baseline import q_t_update
    from muphys_jax.core.scans_baseline import precip_scan_batched
    from muphys_jax.core import properties as props
    from muphys_jax.core.common import constants as const
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    p = jnp.ones((ncells, nlev), dtype=jnp.float32) * 80000.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    q = Q(
        v=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.005,
        c=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
    )
    
    @jax.jit
    def combined_step(t, p, rho, dz, q, dt, qnc):
        # Step 1: phase transitions
        q_mid, t_mid = q_t_update(t, p, rho, q, dt, qnc)
        
        # Step 2: precipitation scan (simplified - just rain)
        xrho = jnp.sqrt(const.rho_00 / rho)
        vc_r = props.vel_scale_factor_default(xrho)
        kmin_r = q_mid.r > const.qmin
        zeta = dt / (2.0 * dz)
        params_list = [(14.58, 0.111, 1.0e-12)]
        
        results = precip_scan_batched(params_list, zeta, rho, [q_mid.r], [vc_r], [kmin_r])
        qr_out, pflx = results[0]
        
        return t_mid, qr_out
    
    t_out, qr_out = combined_step(t, p, rho, dz, q, 30.0, 100.0)
    t_out.block_until_ready()
    print(f"combined q_t + precip: PASSED, t_out shape={t_out.shape}")
except Exception as e:
    print(f"combined q_t + precip: FAILED - {e}")

# Test 8g: Full 4-species precipitation after q_t_update in same JIT
print("\n=== Test 8g: Combined q_t_update + 4-species precip ===")
try:
    @jax.jit
    def combined_full(t, p, rho, dz, q, dt, qnc):
        # Step 1: phase transitions
        q_mid, t_mid = q_t_update(t, p, rho, q, dt, qnc)
        
        # Step 2: all 4 species precipitation
        xrho = jnp.sqrt(const.rho_00 / rho)
        vc_r = props.vel_scale_factor_default(xrho)
        vc_s = props.vel_scale_factor_snow(xrho, rho, t_mid, q_mid.s)
        vc_i = props.vel_scale_factor_ice(xrho)
        vc_g = props.vel_scale_factor_default(xrho)
        
        kmin_r = q_mid.r > const.qmin
        kmin_s = q_mid.s > const.qmin
        kmin_i = q_mid.i > const.qmin
        kmin_g = q_mid.g > const.qmin
        
        zeta = dt / (2.0 * dz)
        params_list = [
            (14.58, 0.111, 1.0e-12),
            (57.80, 0.16666666666666666, 1.0e-12),
            (1.25, 0.160, 1.0e-12),
            (12.24, 0.217, 1.0e-08),
        ]
        
        results = precip_scan_batched(
            params_list, zeta, rho,
            [q_mid.r, q_mid.s, q_mid.i, q_mid.g],
            [vc_r, vc_s, vc_i, vc_g],
            [kmin_r, kmin_s, kmin_i, kmin_g]
        )
        
        return t_mid, results[0][0]  # t_out, qr_out
    
    t_out, qr_out = combined_full(t, p, rho, dz, q, 30.0, 100.0)
    t_out.block_until_ready()
    print(f"combined full: PASSED, t_out shape={t_out.shape}")
except Exception as e:
    print(f"combined full: FAILED - {e}")

# Test 8h: temperature_update_scan only
print("\n=== Test 8h: temperature_update_scan only ===")
try:
    from muphys_jax.core.scans_baseline import temperature_scan_step
    from muphys_jax.core.definitions import TempState
    from muphys_jax.core import thermo
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    qv = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.005
    qliq = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0002
    qice = jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0003
    pr = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1e-6
    pflx = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1e-6
    
    # Compute energy
    ei_old = thermo.internal_energy(t, qv, qliq, qice, rho, dz)
    
    # Shifted temperature
    t_kp1 = jnp.concatenate([t[:, 1:], t[:, -1:]], axis=1)
    mask = jnp.ones((ncells, nlev), dtype=bool)
    dt = 30.0
    
    @jax.jit
    def test_temp_scan(t, t_kp1, ei_old, pr, pflx, qv, qliq, qice, rho, dz, dt, mask):
        init_state = TempState(
            t=jnp.zeros(ncells, dtype=jnp.float32),
            eflx=jnp.zeros(ncells, dtype=jnp.float32),
            activated=jnp.zeros(ncells, dtype=bool)
        )
        inputs = (t.T, t_kp1.T, ei_old.T, pr.T, pflx.T, qv.T, qliq.T, qice.T, rho.T, dz.T, 
                  jnp.broadcast_to(dt, (nlev, ncells)), mask.T)
        final_state, outputs = jax.lax.scan(temperature_scan_step, init_state, inputs)
        return outputs.t.T
    
    t_out = test_temp_scan(t, t_kp1, ei_old, pr, pflx, qv, qliq, qice, rho, dz, dt, mask)
    t_out.block_until_ready()
    print(f"temp_scan: PASSED, t_out shape={t_out.shape}")
except Exception as e:
    print(f"temp_scan: FAILED - {e}")

# Test 8i: Full q_t + precip + temp_scan combined
print("\n=== Test 8i: Full combined (q_t + precip + temp_scan) ===")
print("  SKIPPED - Single JIT with all 3 stages exceeds IREE CUDA memory/allocation limits")

# Test 8j: Split JIT approach (IREE workaround)
print("\n=== Test 8j: Split JIT approach (separate boundaries) ===")
try:
    from muphys_jax.implementations.graupel_baseline import q_t_update
    from muphys_jax.core.scans_baseline import precip_scan_batched, temperature_scan_step
    from muphys_jax.core.definitions import TempState
    from muphys_jax.core import thermo, properties as props
    from muphys_jax.core.common import constants as const
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    p = jnp.ones((ncells, nlev), dtype=jnp.float32) * 80000.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    q = Q(
        v=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.005,
        c=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
    )
    dt = 30.0
    qnc = 100.0
    
    # JIT 1: Phase transitions
    @jax.jit
    def step1_qt(t, p, rho, q, dt, qnc):
        return q_t_update(t, p, rho, q, dt, qnc)
    
    q_mid, t_mid = step1_qt(t, p, rho, q, dt, qnc)
    t_mid.block_until_ready()
    print(f"  Step 1 (q_t_update): PASSED")
    
    # JIT 2: Precipitation (all 4 species)
    @jax.jit
    def step2_precip(t_mid, rho, dz, q_mid, dt):
        xrho = jnp.sqrt(const.rho_00 / rho)
        vc_r = props.vel_scale_factor_default(xrho)
        vc_s = props.vel_scale_factor_snow(xrho, rho, t_mid, q_mid.s)
        vc_i = props.vel_scale_factor_ice(xrho)
        vc_g = props.vel_scale_factor_default(xrho)
        
        kmin_r = q_mid.r > const.qmin
        kmin_s = q_mid.s > const.qmin
        kmin_i = q_mid.i > const.qmin
        kmin_g = q_mid.g > const.qmin
        
        zeta = dt / (2.0 * dz)
        params_list = [
            (14.58, 0.111, 1.0e-12),
            (57.80, 0.16666666666666666, 1.0e-12),
            (1.25, 0.160, 1.0e-12),
            (12.24, 0.217, 1.0e-08),
        ]
        
        results = precip_scan_batched(
            params_list, zeta, rho,
            [q_mid.r, q_mid.s, q_mid.i, q_mid.g],
            [vc_r, vc_s, vc_i, vc_g],
            [kmin_r, kmin_s, kmin_i, kmin_g]
        )
        (qr, pr), (qs, ps), (qi, pi), (qg, pg) = results
        kmin_any = kmin_r | kmin_s | kmin_i | kmin_g
        return qr, qs, qi, qg, pr, ps, pi, pg, kmin_any
    
    qr, qs, qi, qg, pr, ps, pi, pg, kmin_any = step2_precip(t_mid, rho, dz, q_mid, dt)
    qr.block_until_ready()
    print(f"  Step 2 (precipitation): PASSED")
    
    # JIT 3: Temperature correction
    @jax.jit
    def step3_temp(t_mid, q_mid, qr, qs, qi, qg, pr, ps, pi, pg, kmin_any, rho, dz, dt):
        qliq = q_mid.c + qr
        qice = qs + qi + qg
        pflx_tot = ps + pi + pg
        ei_old = thermo.internal_energy(t_mid, q_mid.v, qliq, qice, rho, dz)
        
        ncells, nlev = t_mid.shape
        t_kp1 = jnp.concatenate([t_mid[:, 1:], t_mid[:, -1:]], axis=1)
        
        init_state = TempState(
            t=jnp.zeros(ncells, dtype=jnp.float32),
            eflx=jnp.zeros(ncells, dtype=jnp.float32),
            activated=jnp.zeros(ncells, dtype=bool)
        )
        inputs = (t_mid.T, t_kp1.T, ei_old.T, pr.T, pflx_tot.T, q_mid.v.T, qliq.T, qice.T, 
                  rho.T, dz.T, jnp.broadcast_to(dt, (nlev, ncells)), kmin_any.T)
        final_state, outputs = jax.lax.scan(temperature_scan_step, init_state, inputs)
        return outputs.t.T
    
    t_out = step3_temp(t_mid, q_mid, qr, qs, qi, qg, pr, ps, pi, pg, kmin_any, rho, dz, dt)
    t_out.block_until_ready()
    print(f"  Step 3 (temp correction): PASSED")
    print(f"Split JIT approach: PASSED, t_out shape={t_out.shape}")
except Exception as e:
    print(f"Split JIT: FAILED - {e}")

# Test 8k: Use graupel_run_split (IREE-compatible implementation)
print("\n=== Test 8k: graupel_run_split (IREE-compatible) ===")
try:
    from muphys_jax.implementations.graupel_baseline import graupel_run_split
    
    ncells, nlev = 4, 5
    t = jnp.ones((ncells, nlev), dtype=jnp.float32) * 270.0
    p = jnp.ones((ncells, nlev), dtype=jnp.float32) * 80000.0
    rho = jnp.ones((ncells, nlev), dtype=jnp.float32) * 1.0
    dz = jnp.ones((ncells, nlev), dtype=jnp.float32) * 100.0
    q = Q(
        v=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.005,
        c=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        r=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        s=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        i=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
        g=jnp.ones((ncells, nlev), dtype=jnp.float32) * 0.0001,
    )
    
    t_out, q_out, pflx, prr, prs, pri, prg, eflx = graupel_run_split(dz, t, p, rho, q, 30.0, 100.0)
    t_out.block_until_ready()
    print(f"graupel_run_split: PASSED, t_out shape={t_out.shape}, q_out.r shape={q_out.r.shape}")
except Exception as e:
    print(f"graupel_run_split: FAILED - {e}")

print("\n=== All tests complete ===")
