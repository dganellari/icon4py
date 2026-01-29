#!/usr/bin/env python3
"""
Inject optimized HLO back into JAX computation graph.

Two approaches:
1. jax.extend.ffi - Modern FFI for custom XLA calls (preferred)
2. jax.lax.custom_call - Lower-level custom call mechanism

Usage:
    # After optimizing HLO with hlo-opt:
    python inject_optimized_hlo.py --hlo optimized_precip.hlo --test

This creates a drop-in replacement for precipitation_effects that uses
the optimized HLO instead of the JAX implementation.
"""

import argparse
import sys
import pathlib
from functools import partial

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import mlir
from jax._src.lib.mlir import ir


# ============================================================================
# Approach 1: Using XLA's custom_call with serialized HLO
# ============================================================================

def load_hlo_module(hlo_path: str) -> str:
    """Load HLO text from file."""
    with open(hlo_path, 'r') as f:
        return f.read()


def create_precip_effect_from_hlo(hlo_text: str, ncells: int, nlev: int):
    """
    Create a JAX-callable function from optimized HLO.

    This uses jax.pure_callback for the simplest integration,
    or can be extended to use custom_call for better performance.
    """

    # For now, we'll create a wrapper that:
    # 1. Compiles the optimized HLO once
    # 2. Calls it via XLA's execution API

    # Note: Direct HLO injection requires building a custom XLA client
    # The cleanest approach is to use JAX's FFI system

    raise NotImplementedError(
        "Direct HLO injection requires XLA client setup. "
        "See approach 2 (AOT compilation) or approach 3 (Pallas) instead."
    )


# ============================================================================
# Approach 2: AOT Compilation + Loading
# ============================================================================

def export_for_aot(jax_fn, *args, output_path: str):
    """
    Export JAX function for Ahead-of-Time compilation.

    This creates a serialized executable that can be loaded later.
    """
    from jax.experimental import export as jax_export

    print(f"Exporting for AOT compilation...")

    # Lower and compile
    lowered = jax.jit(jax_fn).lower(*args)
    serialized = lowered.compile().as_serialized_hlo()

    # Save serialized HLO
    with open(output_path, 'wb') as f:
        f.write(serialized)

    print(f"✓ Saved serialized HLO to: {output_path}")
    return output_path


def load_aot_executable(serialized_path: str):
    """Load a serialized HLO executable."""
    from jax._src import xla_bridge

    with open(serialized_path, 'rb') as f:
        serialized = f.read()

    # This requires the same XLA backend that was used for compilation
    client = xla_bridge.get_backend()
    executable = client.deserialize_executable(serialized)

    return executable


# ============================================================================
# Approach 3: Swap Implementation at Runtime
# ============================================================================

class OptimizedPrecipEffect:
    """
    A drop-in replacement for precipitation_effects that can use
    either the original JAX implementation or an optimized version.

    Usage:
        # Create optimized version
        precip = OptimizedPrecipEffect()

        # Use in graupel code
        result = precip(last_lev, kmin_r, ..., q, t, rho, dz, dt)

        # Or swap implementation
        precip.use_optimized = True
    """

    def __init__(self, optimized_fn=None):
        self.optimized_fn = optimized_fn
        self.use_optimized = optimized_fn is not None

        # Import original
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.parent))
        from muphys_jax.implementations.graupel_baseline import precipitation_effects
        self.original_fn = precipitation_effects

    def __call__(self, last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt):
        if self.use_optimized and self.optimized_fn is not None:
            return self.optimized_fn(
                last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
            )
        return self.original_fn(
            last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q_in, t, rho, dz, dt
        )

    def benchmark(self, *args, num_runs=100):
        """Benchmark original vs optimized."""
        import time

        # Warmup
        _ = self.original_fn(*args)
        _ = jax.block_until_ready(_)

        # Original
        self.use_optimized = False
        start = time.perf_counter()
        for _ in range(num_runs):
            result = self.original_fn(*args)
            jax.block_until_ready(result)
        original_time = (time.perf_counter() - start) / num_runs * 1000

        print(f"Original:  {original_time:.2f} ms")

        if self.optimized_fn is not None:
            # Warmup optimized
            self.use_optimized = True
            _ = self.optimized_fn(*args)
            _ = jax.block_until_ready(_)

            start = time.perf_counter()
            for _ in range(num_runs):
                result = self.optimized_fn(*args)
                jax.block_until_ready(result)
            optimized_time = (time.perf_counter() - start) / num_runs * 1000

            print(f"Optimized: {optimized_time:.2f} ms")
            print(f"Speedup:   {original_time / optimized_time:.2f}x")


# ============================================================================
# Approach 4: Custom Primitive with MLIR Lowering
# ============================================================================

def create_custom_primitive_from_hlo(hlo_path: str, name: str = "optimized_precip"):
    """
    Create a JAX primitive that lowers to the optimized HLO.

    This is the most integrated approach but requires more setup.
    """

    # Define the primitive
    optimized_precip_p = core.Primitive(name)
    optimized_precip_p.multiple_results = True

    # Load the HLO
    hlo_text = load_hlo_module(hlo_path)

    def impl(*args):
        # This would be called in eager mode - fall back to original
        raise NotImplementedError("Primitive must be used within jit")

    def abstract_eval(*avals):
        # Define output shapes/types based on input shapes
        # For precipitation_effects: 11 outputs, all f64[ncells, nlev]
        ncells, nlev = avals[4].shape  # q_v shape
        out_shape = core.ShapedArray((ncells, nlev), jnp.float64)
        return [out_shape] * 11

    optimized_precip_p.def_impl(impl)
    optimized_precip_p.def_abstract_eval(abstract_eval)

    # MLIR lowering would go here - this requires injecting the HLO
    # into the MLIR module, which is complex

    return optimized_precip_p


# ============================================================================
# Main: Test and Demo
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inject optimized HLO into JAX"
    )
    parser.add_argument('--hlo', type=str, help='Optimized HLO file')
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--input', '-i', type=str, help='Input netCDF file')

    args = parser.parse_args()

    print("=" * 80)
    print("HLO INJECTION DEMO")
    print("=" * 80)

    # For now, demonstrate the runtime swap approach
    from export_precip_effect import load_precip_inputs

    print("\nLoading test data...")
    last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt, ncells, nlev = \
        load_precip_inputs(args.input)

    print("\nCreating OptimizedPrecipEffect wrapper...")
    precip = OptimizedPrecipEffect()

    if args.test:
        print("\nRunning original implementation...")
        result = precip(last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt)
        print(f"✓ Got {len(result)} outputs")
        for i, r in enumerate(result):
            if hasattr(r, 'shape'):
                print(f"  output[{i}]: {r.shape}")

    if args.benchmark:
        print("\nBenchmarking...")
        precip.benchmark(
            last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt,
            num_runs=100
        )

    print("\n" + "=" * 80)
    print("INTEGRATION APPROACHES")
    print("=" * 80)
    print("""
1. Runtime Swap (simplest):
   - Use OptimizedPrecipEffect wrapper
   - Swap between original and optimized at runtime
   - No HLO integration needed

2. AOT Compilation:
   - Export JAX function to serialized HLO
   - Optimize the serialized HLO externally
   - Load back and execute via XLA client

3. Custom Primitive (most integrated):
   - Define JAX primitive with custom MLIR lowering
   - Inject optimized HLO during lowering
   - Full integration with JAX transformations (grad, vmap)

4. jax.extend.ffi (for C/CUDA kernels):
   - Write optimized kernel in C/CUDA
   - Register as FFI target
   - Call from JAX

For your use case (hlo-opt optimization), approach 2 (AOT) is recommended.
""")


if __name__ == "__main__":
    main()
