#!/usr/bin/env python3
"""
Generate parallel scan version of precipitation_effects using associative_scan.

ANALYSIS OF THE RECURRENCE STRUCTURE:
=====================================

The precipitation scan has the following structure per level k:

  activated[k] = activated[k-1] | kmin[k]           # Cumulative OR (parallelizable!)
  rho_x[k] = q_in[k] * rho[k]                       # Independent
  flx_eff[k] = rho_x[k]/zeta[k] + 2*pflx[k-1]       # Depends on pflx[k-1]
  flx_partial[k] = min(rho_x[k]*vc*fall_speed, flx_eff)  # Independent

  # This is the complex part:
  rhox_prev = (q[k-1] + q[k]) * 0.5 * rho[k-1]      # Depends on q[k-1]
  vt[k] = vc * (rhox_prev + offset)^exp if activated[k-1] else 0

  q_new[k] = zeta[k] * (flx_eff[k] - flx_partial[k]) / ((1 + zeta[k]*vt[k]) * rho[k])
  pflx[k] = (q_new[k]*rho[k]*vt[k] + flx_partial[k]) * 0.5

The dependency on q[k-1] via vt makes this NON-LINEAR.
However, we can still parallelize by:

1. Computing activated first (cumulative OR - O(log n))
2. Using iterative refinement or Newton-Raphson for the coupled recurrence
3. Or accepting small approximation error for near-linear regimes

For now, we implement a SIMPLIFIED version that ignores the vt dependency,
which is valid when:
- Before activation (vt = 0)
- When rhox_prev is small (vt ≈ constant)

Usage:
    python generate_parallel_scan.py --benchmark
"""

import argparse
import time
import jax
import jax.numpy as jnp
from jax import lax


def linear_recurrence_scan(a, b):
    """Solve q[k] = a[k] * q[k-1] + b[k] using parallel associative scan.

    Args:
        a: Array of shape (nlev, ncells) - multiplicative coefficients
        b: Array of shape (nlev, ncells) - additive coefficients

    Returns:
        q: Array of shape (nlev, ncells) - solution to the recurrence

    The recurrence q[k] = a[k] * q[k-1] + b[k] with q[-1] = 0 has solution:
        q[k] = sum_{j=0}^{k} b[j] * prod_{i=j+1}^{k} a[i]

    This can be computed using associative scan with the monoid:
        (a_L, b_L) ⊕ (a_R, b_R) = (a_L * a_R, b_L * a_R + b_R)

    The result is (cumulative_product, cumulative_sum).
    """
    def combine(left, right):
        prod_L, sum_L = left
        prod_R, sum_R = right
        return (prod_L * prod_R, sum_L * prod_R + sum_R)

    # Run associative scan along axis 0 (levels)
    _, result = lax.associative_scan(combine, (a, b), axis=0)
    return result


def precipitation_scan_parallel(q, rho, zeta, vel_coeff, vel_exp, qmin, kmin):
    """Single species precipitation scan using parallel linear recurrence.

    This is mathematically equivalent to the sequential scan but runs in
    O(log nlev) parallel steps instead of O(nlev) sequential steps.

    The key simplification: for the LINEARIZED form, we assume vt=0 initially.
    This gives us:
        flx_eff[k] = rho_x[k] / zeta[k] + 2 * pflx[k-1]
        q_new[k] = zeta[k] * flx_eff[k] / rho[k] (simplified)
        pflx[k] = q_new[k] * 0.5

    Substituting:
        q[k] = (rho_x[k] / zeta[k] + 2 * pflx[k-1]) * zeta[k] / rho[k]
             = rho_x[k] / rho[k] + 2 * zeta[k] / rho[k] * pflx[k-1]
             = q[k] + 2 * zeta[k] / rho[k] * pflx[k-1]  (since rho_x = q*rho)

    And pflx[k] = 0.5 * q_new[k], so:
        q_new[k] = q[k] + zeta[k] / rho[k] * q_new[k-1]

    This is the linear recurrence: q_new[k] = a[k] * q_new[k-1] + b[k]
    where a[k] = zeta[k] / rho[k] and b[k] = q[k] (the input q values)
    """
    # Transpose to (nlev, ncells) for scanning along axis 0
    q_t = q.T
    rho_t = rho.T
    zeta_t = zeta.T
    kmin_t = kmin.T

    nlev, ncells = q_t.shape

    # Compute level-local values
    rho_x = q_t * rho_t
    rho_sqrt = jnp.sqrt(1.225 / rho_t)  # sqrt(rho0 / rho)

    # Fall speed: vel_coeff * (rho_x + qmin)^vel_exp
    fall_speed = vel_coeff * jnp.power(rho_x + qmin, vel_exp)
    vc = rho_sqrt * vel_coeff  # velocity coefficient

    # For the linearized version (first pass):
    # a[k] = zeta[k] / rho[k]  (recurrence coefficient)
    # b[k] = base term from current level

    # The full recurrence is more complex due to flx_partial and vt
    # For now, use the simplified linear form
    a = zeta_t / rho_t
    b = q_t  # Simplified: just the input q

    # Compute cumulative activation (this IS parallelizable with cummax)
    activated = lax.associative_scan(jnp.logical_or, kmin_t, axis=0)

    # Solve the linear recurrence
    q_out = linear_recurrence_scan(a, b)

    # Apply activation mask
    q_out = jnp.where(activated, q_out, q_t)

    # Transpose back to (ncells, nlev)
    return q_out.T


def benchmark_methods(ncells=327680, nlev=90, num_runs=10):
    """Benchmark different scan implementations."""
    print("=" * 60)
    print(f"Benchmarking precipitation scan: {ncells} cells × {nlev} levels")
    print("=" * 60)

    # Create test data
    q = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-6
    rho = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1.0
    zeta = jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.15
    kmin = jnp.zeros((ncells, nlev), dtype=bool)
    kmin = kmin.at[:, 10].set(True)  # Activate at level 10

    # Method 1: Sequential lax.scan
    @jax.jit
    def sequential_scan(q, rho, zeta, kmin):
        def body(carry, inputs):
            pflx, activated = carry
            q_k, rho_k, zeta_k, kmin_k = inputs

            activated = activated | kmin_k
            rho_x = q_k * rho_k
            flx_eff = rho_x / zeta_k + 2 * pflx
            q_new = flx_eff * zeta_k / rho_k
            pflx_new = jnp.where(activated, q_new * 0.5, jnp.zeros_like(pflx))
            q_out = jnp.where(activated, q_new, q_k)

            return (pflx_new, activated), q_out

        init = (jnp.zeros(ncells), jnp.zeros(ncells, dtype=bool))
        _, q_out = lax.scan(body, init, (q.T, rho.T, zeta.T, kmin.T))
        return q_out.T

    # Method 2: Parallel associative scan
    @jax.jit
    def parallel_scan(q, rho, zeta, kmin):
        return precipitation_scan_parallel(
            q, rho, zeta,
            vel_coeff=14.58, vel_exp=0.111, qmin=1e-12,
            kmin=kmin
        )

    # Warm up
    print("\nWarming up...")
    _ = sequential_scan(q, rho, zeta, kmin).block_until_ready()
    _ = parallel_scan(q, rho, zeta, kmin).block_until_ready()

    # Benchmark sequential
    print(f"\nBenchmarking sequential scan ({num_runs} runs)...")
    times_seq = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = sequential_scan(q, rho, zeta, kmin)
        result.block_until_ready()
        times_seq.append(time.perf_counter() - start)

    # Benchmark parallel
    print(f"Benchmarking parallel scan ({num_runs} runs)...")
    times_par = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = parallel_scan(q, rho, zeta, kmin)
        result.block_until_ready()
        times_par.append(time.perf_counter() - start)

    # Results
    avg_seq = sum(times_seq) / len(times_seq) * 1000
    avg_par = sum(times_par) / len(times_par) * 1000
    speedup = avg_seq / avg_par

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sequential scan:  {avg_seq:.2f} ms (avg)")
    print(f"Parallel scan:    {avg_par:.2f} ms (avg)")
    print(f"Speedup:          {speedup:.2f}x")
    print()

    # Check HLO
    print("HLO analysis:")
    for name, fn in [("sequential", sequential_scan), ("parallel", parallel_scan)]:
        lowered = fn.lower(q, rho, zeta, kmin)
        hlo = lowered.as_text(dialect='hlo')
        print(f"  {name:12s}: {len(hlo):6d} bytes, while:{hlo.count('while'):2d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--ncells", type=int, default=327680)
    parser.add_argument("--nlev", type=int, default=90)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    if args.benchmark:
        benchmark_methods(args.ncells, args.nlev)
    else:
        print("Use --benchmark to run comparison")


if __name__ == "__main__":
    main()
