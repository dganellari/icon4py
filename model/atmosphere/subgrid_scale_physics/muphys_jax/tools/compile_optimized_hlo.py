#!/usr/bin/env python3
"""
Compile transformed/optimized HLO back to a serialized executable.

This is step 3 in the optimization pipeline:
1. Export: precipitation_effects → HLO
2. Transform: hlo-opt (unroll while, fuse ops, etc.)
3. Compile: optimized HLO → serialized executable (THIS SCRIPT)
4. Inject: serialized executable → custom primitive in JAX

Usage:
    # Compile optimized StableHLO to serialized executable
    python tools/compile_optimized_hlo.py \
        --input optimized_precip.stablehlo \
        --output optimized_precip.serialized

    # Compile HLO text format
    python tools/compile_optimized_hlo.py \
        --input optimized_precip.hlo \
        --format hlo \
        --output optimized_precip.serialized
"""

import argparse
import sys
import pathlib

import jax
from jax._src import xla_bridge
from jax._src.lib import xla_client


def compile_stablehlo_to_executable(stablehlo_path: str, output_path: str):
    """Compile StableHLO MLIR to a serialized XLA executable."""
    print(f"Loading StableHLO from: {stablehlo_path}")

    with open(stablehlo_path, 'r') as f:
        stablehlo_text = f.read()

    print(f"  Size: {len(stablehlo_text) / 1024 / 1024:.2f} MB")

    # Get the XLA client
    backend = xla_bridge.get_backend()
    print(f"  Backend: {backend.platform}")

    # Parse StableHLO and compile
    print("\nCompiling...")

    try:
        # Use XLA's MLIR parser to load StableHLO
        # This requires the stablehlo dialect to be registered
        computation = xla_client.mlir_to_xla_computation(stablehlo_text)

        # Compile to executable
        compile_options = xla_client.CompileOptions()
        executable = backend.compile(computation, compile_options)

        # Serialize
        serialized = executable.serialize()

        with open(output_path, 'wb') as f:
            f.write(serialized)

        print(f"✓ Compiled and serialized to: {output_path}")
        print(f"  Size: {len(serialized) / 1024 / 1024:.2f} MB")

        return output_path

    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        print("\nTrying alternative compilation method...")
        return compile_via_jax_import(stablehlo_text, output_path)


def compile_hlo_to_executable(hlo_path: str, output_path: str):
    """Compile HLO text to a serialized XLA executable."""
    print(f"Loading HLO from: {hlo_path}")

    with open(hlo_path, 'r') as f:
        hlo_text = f.read()

    print(f"  Size: {len(hlo_text) / 1024 / 1024:.2f} MB")

    backend = xla_bridge.get_backend()
    print(f"  Backend: {backend.platform}")

    print("\nCompiling...")

    try:
        # Parse HLO text and compile
        computation = xla_client.hlo_module_proto_to_computation(
            xla_client.hlo_text_to_proto(hlo_text)
        )

        compile_options = xla_client.CompileOptions()
        executable = backend.compile(computation, compile_options)

        serialized = executable.serialize()

        with open(output_path, 'wb') as f:
            f.write(serialized)

        print(f"✓ Compiled and serialized to: {output_path}")
        print(f"  Size: {len(serialized) / 1024 / 1024:.2f} MB")

        return output_path

    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        raise


def compile_via_jax_import(mlir_text: str, output_path: str):
    """
    Alternative: Import MLIR into JAX and compile through JAX's pipeline.

    This uses JAX's internal MLIR import which may handle more dialects.
    """
    try:
        from jax._src.interpreters import mlir as jax_mlir
        from jax.experimental import export as jax_export

        print("Attempting JAX MLIR import...")

        # This is experimental - JAX doesn't officially support importing arbitrary MLIR
        # But we can try using the internal APIs

        raise NotImplementedError(
            "Direct MLIR import not fully supported. "
            "Consider using run_hlo_module for benchmarking, "
            "or manually writing a JAX wrapper."
        )

    except Exception as e:
        print(f"JAX import failed: {e}")
        raise


def verify_executable(serialized_path: str):
    """Verify that the serialized executable can be loaded."""
    print(f"\nVerifying executable...")

    try:
        backend = xla_bridge.get_backend()

        with open(serialized_path, 'rb') as f:
            serialized = f.read()

        executable = backend.deserialize_executable(serialized)
        print(f"✓ Executable loaded successfully")
        print(f"  Name: {executable.name()}")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compile optimized HLO to serialized executable"
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input HLO/StableHLO file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output serialized executable')
    parser.add_argument('--format', '-f', choices=['stablehlo', 'hlo', 'auto'],
                       default='auto',
                       help='Input format (auto-detect by extension)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the compiled executable can be loaded')

    args = parser.parse_args()

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    # Auto-detect format
    input_path = pathlib.Path(args.input)
    if args.format == 'auto':
        if input_path.suffix == '.stablehlo':
            format = 'stablehlo'
        elif input_path.suffix == '.hlo':
            format = 'hlo'
        else:
            print(f"Cannot auto-detect format for {input_path.suffix}, assuming stablehlo")
            format = 'stablehlo'
    else:
        format = args.format

    print("=" * 80)
    print(f"COMPILING: {args.input} ({format})")
    print("=" * 80)

    try:
        if format == 'stablehlo':
            compile_stablehlo_to_executable(args.input, args.output)
        else:
            compile_hlo_to_executable(args.input, args.output)

        if args.verify:
            verify_executable(args.output)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print(f"""
To benchmark the optimized executable:

1. Benchmark with run_hlo_module:
   run_hlo_module --platform=cuda --num_runs=100 {args.input}

2. Compare against original (use export_precip_effect.py to get original .hlo):
   run_hlo_module --platform=cuda --num_runs=100 <original>.hlo
""")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        print("\nAlternative: Use run_hlo_module for benchmarking:")
        print(f"  run_hlo_module --platform=cuda --num_runs=100 {args.input}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
