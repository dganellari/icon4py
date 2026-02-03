#!/usr/bin/env python3
"""
Generate tiled precipitation scan for optimal GPU performance.

Strategy: Unroll within tiles of TILE_SIZE levels, use sequential loop between tiles.
This balances:
- Fusion within tiles (unrolled operations can fuse)
- Reasonable IR size (not 90 levels of duplication)
- Sequential dependency between tiles (still O(nlev/TILE_SIZE) iterations)

For TILE_SIZE=16 and nlev=90:
- 6 tiles (90/16 = 5.625, rounded up)
- Each tile has 16 unrolled levels (fused)
- IR size: ~6x smaller than full unroll

Usage:
    python generate_tiled_scan.py --benchmark --tile-size 16
"""

import argparse
import time
import jax
import jax.numpy as jnp
from jax import lax


def precip_scan_tiled(q, rho, zeta, vc, kmin, params, tile_size=16):
    """Tiled precipitation scan with unrolling within tiles.

    Args:
        q: Input mixing ratio (ncells, nlev)
        rho: Density (ncells, nlev)
        zeta: dt/(2*dz) coefficient (ncells, nlev)
        vc: Velocity coefficient (ncells, nlev)
        kmin: Activation mask (ncells, nlev)
        params: (prefactor, exponent, offset) tuple
        tile_size: Number of levels to unroll per tile

    Returns:
        q_out: Updated mixing ratio (ncells, nlev)
        flx_out: Flux (ncells, nlev)
    """
    prefactor, exponent, offset = params
    ncells, nlev = q.shape
    num_tiles = (nlev + tile_size - 1) // tile_size

    # Transpose for level-first processing
    q_t = q.T  # (nlev, ncells)
    rho_t = rho.T
    zeta_t = zeta.T
    vc_t = vc.T
    kmin_t = kmin.T

    def process_tile(carry, tile_idx):
        """Process one tile of levels."""
        q_prev, flx_prev, rhox_prev, activated_prev = carry

        start = tile_idx * tile_size
        end = jnp.minimum(start + tile_size, nlev)

        # Extract tile slices (static size for JIT)
        # Pad if necessary
        q_tile = lax.dynamic_slice(q_t, [start, 0], [tile_size, ncells])
        rho_tile = lax.dynamic_slice(rho_t, [start, 0], [tile_size, ncells])
        zeta_tile = lax.dynamic_slice(zeta_t, [start, 0], [tile_size, ncells])
        vc_tile = lax.dynamic_slice(vc_t, [start, 0], [tile_size, ncells])
        kmin_tile = lax.dynamic_slice(kmin_t, [start, 0], [tile_size, ncells])

        # Process tile using lax.scan (will be unrolled if tile_size is small)
        def tile_body(carry_inner, inputs):
            q_p, flx_p, rhox_p, act_p = carry_inner
            q_k, rho_k, zeta_k, vc_k, mask_k = inputs

            activated = act_p | mask_k
            rho_x = q_k * rho_k
            flx_eff = rho_x / zeta_k + 2.0 * flx_p

            fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
            flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)

            rhox_avg = (q_p + q_k) * 0.5 * rho_k
            vt_active = vc_k * prefactor * lax.pow(rhox_avg + offset, exponent)
            vt = lax.select(act_p, vt_active, jnp.zeros_like(q_k))

            q_activated = zeta_k * (flx_eff - flx_partial) / ((1.0 + zeta_k * vt) * rho_k)
            flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5

            q_out = lax.select(activated, q_activated, q_k)
            flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q_k))
            rhox_out = q_out * rho_k

            return (q_out, flx_out, rhox_out, activated), (q_out, flx_out)

        tile_init = (q_prev, flx_prev, rhox_prev, activated_prev)
        tile_inputs = (q_tile, rho_tile, zeta_tile, vc_tile, kmin_tile)

        final_carry, (q_outs, flx_outs) = lax.scan(tile_body, tile_init, tile_inputs)

        return final_carry, (q_outs, flx_outs)

    # Initialize carry
    init = (
        jnp.zeros(ncells),  # q_prev
        jnp.zeros(ncells),  # flx_prev
        jnp.zeros(ncells),  # rhox_prev
        jnp.zeros(ncells, dtype=bool),  # activated
    )

    # Process all tiles
    _, (q_tiles, flx_tiles) = lax.scan(process_tile, init, jnp.arange(num_tiles))

    # Reshape outputs: (num_tiles, tile_size, ncells) -> (nlev, ncells)
    q_out = q_tiles.reshape(-1, ncells)[:nlev]
    flx_out = flx_tiles.reshape(-1, ncells)[:nlev]

    return q_out.T, flx_out.T


def benchmark_tiling(ncells=327680, nlev=90, num_runs=10):
    """Benchmark different tile sizes."""
    print("=" * 70)
    print(f"Benchmarking tiled precipitation scan: {ncells} cells × {nlev} levels")
    print("=" * 70)

    # Create test data
    q = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1e-6
    rho = jnp.ones((ncells, nlev), dtype=jnp.float64) * 1.0
    zeta = jnp.ones((ncells, nlev), dtype=jnp.float64) * 0.15
    vc = jnp.ones((ncells, nlev), dtype=jnp.float64) * 5.0
    kmin = jnp.zeros((ncells, nlev), dtype=bool)
    kmin = kmin.at[:, 10].set(True)
    params = (14.58, 0.111, 1e-12)

    # Baseline: sequential lax.scan
    @jax.jit
    def baseline_scan(q, rho, zeta, vc, kmin):
        prefactor, exponent, offset = params

        def body(carry, inputs):
            q_p, flx_p, rhox_p, act_p = carry
            q_k, rho_k, zeta_k, vc_k, mask_k = inputs

            activated = act_p | mask_k
            rho_x = q_k * rho_k
            flx_eff = rho_x / zeta_k + 2.0 * flx_p

            fall_speed = prefactor * lax.pow(rho_x + offset, exponent)
            flx_partial = lax.min(rho_x * vc_k * fall_speed, flx_eff)

            rhox_avg = (q_p + q_k) * 0.5 * rho_k
            vt_active = vc_k * prefactor * lax.pow(rhox_avg + offset, exponent)
            vt = lax.select(act_p, vt_active, jnp.zeros_like(q_k))

            q_activated = zeta_k * (flx_eff - flx_partial) / ((1.0 + zeta_k * vt) * rho_k)
            flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5

            q_out = lax.select(activated, q_activated, q_k)
            flx_out = lax.select(activated, flx_activated, jnp.zeros_like(q_k))
            rhox_out = q_out * rho_k

            return (q_out, flx_out, rhox_out, activated), (q_out, flx_out)

        init = (jnp.zeros(ncells), jnp.zeros(ncells), jnp.zeros(ncells),
                jnp.zeros(ncells, dtype=bool))
        _, (q_out, flx_out) = lax.scan(body, init, (q.T, rho.T, zeta.T, vc.T, kmin.T))
        return q_out.T, flx_out.T

    results = {}

    # Benchmark baseline
    print("\nWarming up baseline...")
    _ = baseline_scan(q, rho, zeta, vc, kmin)[0].block_until_ready()

    print(f"Benchmarking baseline ({num_runs} runs)...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = baseline_scan(q, rho, zeta, vc, kmin)
        result[0].block_until_ready()
        times.append(time.perf_counter() - start)
    results['baseline'] = sum(times) / len(times) * 1000

    lowered = baseline_scan.lower(q, rho, zeta, vc, kmin)
    hlo = lowered.as_text(dialect='hlo')
    results['baseline_hlo'] = len(hlo)
    results['baseline_while'] = hlo.count('while')

    # Benchmark different tile sizes
    for tile_size in [8, 16, 32, 45, 90]:
        if nlev % tile_size != 0 and tile_size != 90:
            # Skip non-divisible tile sizes for simplicity
            continue

        print(f"\nBenchmarking tile_size={tile_size}...")

        @jax.jit
        def tiled_fn(q, rho, zeta, vc, kmin):
            return precip_scan_tiled(q, rho, zeta, vc, kmin, params, tile_size=tile_size)

        # Warm up
        try:
            _ = tiled_fn(q, rho, zeta, vc, kmin)[0].block_until_ready()
        except Exception as e:
            print(f"  Error with tile_size={tile_size}: {e}")
            continue

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = tiled_fn(q, rho, zeta, vc, kmin)
            result[0].block_until_ready()
            times.append(time.perf_counter() - start)

        results[f'tile_{tile_size}'] = sum(times) / len(times) * 1000

        lowered = tiled_fn.lower(q, rho, zeta, vc, kmin)
        hlo = lowered.as_text(dialect='hlo')
        results[f'tile_{tile_size}_hlo'] = len(hlo)
        results[f'tile_{tile_size}_while'] = hlo.count('while')

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'HLO Size':<12} {'While loops':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline_time = results['baseline']
    print(f"{'baseline':<20} {baseline_time:>10.2f} {results['baseline_hlo']:>10} {results['baseline_while']:>10} {'1.00x':>10}")

    for tile_size in [8, 16, 32, 45, 90]:
        key = f'tile_{tile_size}'
        if key in results:
            t = results[key]
            speedup = baseline_time / t
            print(f"{key:<20} {t:>10.2f} {results[f'{key}_hlo']:>10} {results[f'{key}_while']:>10} {speedup:>9.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--ncells", type=int, default=327680)
    parser.add_argument("--nlev", type=int, default=90)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    if args.benchmark:
        benchmark_tiling(args.ncells, args.nlev)
    else:
        print("Use --benchmark to run comparison")


if __name__ == "__main__":
    main()
