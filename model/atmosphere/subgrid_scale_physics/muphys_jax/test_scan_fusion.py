"""Test and benchmark scan fusion optimization."""
import sys
import time
from pathlib import Path
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import netCDF4

# Enable float64
jax.config.update("jax_enable_x64", True)

# Setup path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))
os.chdir(current_dir)

from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.core.definitions import Q


def _calc_dz(z: np.ndarray) -> np.ndarray:
    """Calculate layer thickness from geometric height."""
    ksize = z.shape[0]
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


def load_real_data(filename="/capstor/store/cscs/userlab/d126/muphys_grids/inputs/atm_R2B06.nc"):
    """Load real atmospheric data from NetCDF file."""
    with netCDF4.Dataset(filename, mode="r") as ncfile:
        # Get dimensions
        try:
            ncells = len(ncfile.dimensions["cell"])
        except KeyError:
            ncells = len(ncfile.dimensions["ncells"])
        nlev = len(ncfile.dimensions["height"])

        # Calculate layer thickness
        dz = _calc_dz(ncfile.variables["zg"])
        dz = np.transpose(dz)

        def load_var(varname: str) -> np.ndarray:
            var = ncfile.variables[varname]
            if var.dimensions[0] == "time":
                var = var[0, :, :]
            return np.transpose(var).astype(np.float64)

        q = Q(
            v=jnp.array(load_var("hus")),
            c=jnp.array(load_var("clw")),
            r=jnp.array(load_var("qr")),
            s=jnp.array(load_var("qs")),
            i=jnp.array(load_var("cli")),
            g=jnp.array(load_var("qg")),
        )

        return (
            jnp.array(dz),
            jnp.array(load_var("ta")),
            jnp.array(load_var("pfull")),
            jnp.array(load_var("rho")),
            q,
            30.0,
            100.0,
        )


def test_correctness():
    """Test that fused and unfused scans produce identical results."""
    print("=" * 80)
    print("CORRECTNESS TEST: Fused vs Unfused Scans")
    print("=" * 80)
    
    # Load test data
    print("\nLoading REAL atmospheric data...")
    dz, te, p, rho, q_in, dt, qnc = load_real_data()
    ncells = te.shape[0]
    nlevels = te.shape[1]
    print(f"Grid: {ncells} cells × {nlevels} levels")
    print(f"Temperature range: {np.array(te).min():.1f} - {np.array(te).max():.1f} K")
    
    # Note: graupel_run is already JIT-compiled with @partial(jax.jit, static_argnames=['use_fused_scans'])
    print("\nCompiling and running UNFUSED version (180 kernel launches)...")
    result_unfused = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=False)
    result_unfused[0].block_until_ready()
    
    # Run fused version
    print("Compiling and running FUSED version (90 kernel launches)...")
    result_fused = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
    result_fused[0].block_until_ready()
    
    # Compare results
    print("\n" + "=" * 80)
    print("CORRECTNESS COMPARISON")
    print("=" * 80)
    
    t_unfused, q_unfused = result_unfused[:2]
    t_fused, q_fused = result_fused[:2]
    
    fields = [
        ("qr (rain)", q_unfused.r, q_fused.r),
        ("qs (snow)", q_unfused.s, q_fused.s),
        ("qi (ice)", q_unfused.i, q_fused.i),
        ("qg (graupel)", q_unfused.g, q_fused.g),
        ("t (temperature)", t_unfused, t_fused),
    ]
    
    all_close = True
    for name, unfused, fused in fields:
        abs_diff = jnp.abs(unfused - fused)
        max_diff = jnp.max(abs_diff)
        mean_diff = jnp.mean(abs_diff)
        rel_diff = jnp.max(abs_diff / (jnp.abs(unfused) + 1e-10))
        
        # Check if close
        is_close = jnp.allclose(unfused, fused, rtol=1e-12, atol=1e-14)
        status = "✓ PASS" if is_close else "✗ FAIL"
        all_close = all_close and is_close
        
        print(f"\n{name:20s} {status}")
        print(f"  Max abs diff:  {max_diff:.2e}")
        print(f"  Mean abs diff: {mean_diff:.2e}")
        print(f"  Max rel diff:  {rel_diff:.2e}")
    
    print("\n" + "=" * 80)
    if all_close:
        print("✓ ALL TESTS PASSED - Results are identical!")
    else:
        print("✗ TESTS FAILED - Results differ!")
    print("=" * 80)
    
    return all_close


def benchmark():
    """Benchmark fused vs unfused scans."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK: Fused vs Unfused Scans")
    print("=" * 80)
    
    # Load test data
    print("\nLoading REAL atmospheric data...")
    dz, te, p, rho, q_in, dt, qnc = load_real_data()
    ncells = te.shape[0]
    nlevels = te.shape[1]
    print(f"Grid: {ncells} cells × {nlevels} levels")
    print(f"Temperature range: {np.array(te).min():.1f} - {np.array(te).max():.1f} K")
    
    # Note: graupel_run is already JIT-compiled
    print("\nBenchmarking UNFUSED version...")
    n_warmup = 10
    n_iterations = 100
    
    # Warmup (no copy needed - buffer donation removed)
    for _ in range(n_warmup):
        result = graupel_run(
            dz, te, p, rho, q_in, dt, qnc, use_fused_scans=False
        )
        result[0].block_until_ready()

    # Benchmark
    times_unfused = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = graupel_run(
            dz, te, p, rho, q_in, dt, qnc, use_fused_scans=False
        )
        result[0].block_until_ready()
        elapsed = time.perf_counter() - start
        times_unfused.append(elapsed * 1000)  # Convert to ms
    
    mean_unfused = np.mean(times_unfused)
    std_unfused = np.std(times_unfused)
    
    print(f"  Mean: {mean_unfused:.2f} ms")
    print(f"  Std:  {std_unfused:.2f} ms")
    
    # Benchmark fused version
    print("\nBenchmarking FUSED version...")

    # Warmup (no copy needed - buffer donation removed)
    for _ in range(n_warmup):
        result = graupel_run(
            dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True
        )
        result[0].block_until_ready()

    # Benchmark
    times_fused = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = graupel_run(
            dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True
        )
        result[0].block_until_ready()
        elapsed = time.perf_counter() - start
        times_fused.append(elapsed * 1000)  # Convert to ms
    
    mean_fused = np.mean(times_fused)
    std_fused = np.std(times_fused)
    
    print(f"  Mean: {mean_fused:.2f} ms")
    print(f"  Std:  {std_fused:.2f} ms")
    
    # Summary
    speedup = mean_unfused / mean_fused
    improvement = (1 - mean_fused / mean_unfused) * 100
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Unfused (baseline):  {mean_unfused:.2f} ± {std_unfused:.2f} ms  (180 kernel launches)")
    print(f"Fused (optimized):   {mean_fused:.2f} ± {std_fused:.2f} ms  (90 kernel launches)")
    print(f"\nSpeedup:      {speedup:.3f}x")
    print(f"Improvement:  {improvement:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    # Run correctness test first
    correctness_ok = test_correctness()
    
    if correctness_ok:
        # Run benchmark
        benchmark()
    else:
        print("\n⚠️  Skipping benchmark due to correctness test failures!")
        sys.exit(1)
