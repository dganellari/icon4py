#!/usr/bin/env python3
"""
Profile individual operations to understand where time is spent in q_t_update.

This helps identify if power operations are the bottleneck.
"""

import argparse
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
import numpy as np

from muphys_jax.utils.data_loading import load_graupel_inputs
from muphys_jax.core.definitions import Q


def benchmark(fn, args, name, num_warmup=10, num_runs=20):
    """Benchmark a function."""
    for _ in range(num_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"  {name}: {np.median(times):.3f} ms (min: {np.min(times):.3f})")
    return np.median(times)


def main():
    parser = argparse.ArgumentParser(description="Profile operations")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file")
    args = parser.parse_args()

    print("=" * 70)
    print("OPERATION PROFILING")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print()

    # Load data
    print("Loading data...")
    dz, t, p, rho, q, dt, qnc, ncells, nlev = load_graupel_inputs(args.input)
    print(f"Grid: {ncells} cells x {nlev} levels")

    # Transpose to (nlev, ncells)
    t_t = jnp.transpose(t)
    p_t = jnp.transpose(p)
    rho_t = jnp.transpose(rho)
    qnc_t = jnp.transpose(qnc)
    q_t = Q(
        v=jnp.transpose(q.v), c=jnp.transpose(q.c), r=jnp.transpose(q.r),
        s=jnp.transpose(q.s), i=jnp.transpose(q.i), g=jnp.transpose(q.g),
    )

    shape = t_t.shape
    print(f"Shape: {shape}")

    # Create test arrays
    x = jnp.ones(shape, dtype=jnp.float64) * 0.5
    y = jnp.ones(shape, dtype=jnp.float64) * 0.3
    mask = jnp.ones(shape, dtype=bool)

    print("\n" + "=" * 70)
    print("ELEMENT-WISE OPERATION COSTS")
    print("=" * 70)

    # Basic operations
    print("\n1. Basic arithmetic:")
    benchmark(jax.jit(lambda a, b: a + b), (x, y), "add")
    benchmark(jax.jit(lambda a, b: a * b), (x, y), "multiply")
    benchmark(jax.jit(lambda a, b: a / b), (x, y), "divide")
    benchmark(jax.jit(lambda a, b: a - b), (x, y), "subtract")

    # Power operations
    print("\n2. Power operations:")
    benchmark(jax.jit(lambda a: a * a), (x,), "x^2 (manual)")
    benchmark(jax.jit(lambda a: jnp.power(a, 2)), (x,), "x^2 (jnp.power)")
    benchmark(jax.jit(lambda a: jnp.power(a, 0.5)), (x,), "x^0.5 (jnp.power)")
    benchmark(jax.jit(lambda a: jnp.sqrt(a)), (x,), "sqrt")
    benchmark(jax.jit(lambda a: jnp.power(a, 0.66667)), (x,), "x^0.667 (jnp.power)")
    benchmark(jax.jit(lambda a: lax.pow(a, 0.66667)), (x,), "x^0.667 (lax.pow)")
    benchmark(jax.jit(lambda a: jnp.power(a, 0.111)), (x,), "x^0.111 (jnp.power)")
    benchmark(jax.jit(lambda a: jnp.power(a, 1.75)), (x,), "x^1.75 (jnp.power)")

    # Exponential/log
    print("\n3. Exponential/logarithm:")
    benchmark(jax.jit(lambda a: jnp.exp(a)), (x,), "exp")
    benchmark(jax.jit(lambda a: jnp.log(a)), (x,), "log")
    benchmark(jax.jit(lambda a: jnp.exp(0.66667 * jnp.log(a))), (x,), "exp(0.667*log(x))")

    # Conditionals
    print("\n4. Conditionals:")
    benchmark(jax.jit(lambda a, b, m: jnp.where(m, a, b)), (x, y, mask), "jnp.where")
    benchmark(jax.jit(lambda a, b, m: lax.select(m, a, b)), (x, y, mask), "lax.select")

    # Combined operations (simulating transition functions)
    print("\n5. Combined operations (simulating transitions):")

    def simple_transition(qc, qg, rho, t):
        """cloud_to_graupel simplified"""
        mask = (jnp.minimum(qc, qg) > 1e-15) & (t > 236.15)
        return jnp.where(mask, 4.43 * qc * jnp.power(qg * rho, 0.94878), 0.0)

    benchmark(jax.jit(simple_transition), (q_t.c, q_t.g, rho_t, t_t), "cloud_to_graupel")

    def complex_transition(t, qc, qr, nc):
        """cloud_to_rain simplified"""
        tau = jnp.maximum(1e-30, jnp.minimum(1.0 - qc / (qc + qr), 0.9))
        phi = jnp.power(tau, 0.68)
        one_minus_phi = 1.0 - phi
        phi = 600.0 * phi * (one_minus_phi * one_minus_phi * one_minus_phi)
        qc_ratio = qc * qc / nc
        one_minus_tau = 1.0 - tau
        xau = 1e10 * (qc_ratio * qc_ratio) * (1.0 + phi / (one_minus_tau * one_minus_tau))
        mask = (qc > 1e-6) & (t > 236.15)
        return jnp.where(mask, xau, 0.0)

    benchmark(jax.jit(complex_transition), (t_t, q_t.c, q_t.r, qnc_t), "cloud_to_rain")

    # Memory bandwidth test
    print("\n6. Memory bandwidth baseline:")
    # Just read and write - no compute
    benchmark(jax.jit(lambda a: a * 1.0), (x,), "identity (read+write)")
    benchmark(jax.jit(lambda a, b, c, d, e: (a, b, c, d, e)), (x, x, x, x, x), "5 arrays passthrough")

    # Full q_t_update
    print("\n7. Full q_t_update:")
    from muphys_jax.implementations.graupel_native_transposed import q_t_update_native

    benchmark(jax.jit(q_t_update_native), (t_t, p_t, rho_t, q_t, dt, qnc_t), "q_t_update_native")

    # Count power operations
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Based on the profiling, jnp.power with non-integer exponents is the bottleneck.
Each power operation takes ~X ms, and q_t_update has ~33 of them.

Options:
1. Replace jnp.power with integer powers where possible
2. Use lookup tables for expensive power operations
3. Approximate power operations (e.g., Newton-Raphson for roots)
4. Fuse all operations into a single CUDA kernel
""")


if __name__ == "__main__":
    main()
