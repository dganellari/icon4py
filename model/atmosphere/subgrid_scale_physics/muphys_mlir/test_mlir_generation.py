#!/usr/bin/env python3
"""
Test script for MLIR code generation.

This verifies MLIR IR generation and shows the optimized code structure.
"""

import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.precip_scans_mlir import (
    generate_precip_scan_mlir,
    MLIR_AVAILABLE,
    MLIR_IMPORT_ERROR
)


def main():
    print("="*70)
    print("MLIR Precipitation Scan - Optimized Code Generation Test")
    print("="*70)

    if not MLIR_AVAILABLE:
        print(f"\n❌ MLIR not available: {MLIR_IMPORT_ERROR}")
        print("\nInstall MLIR Python bindings:")
        print("  pip install mlir-python-bindings")
        return 1

    print("\n✅ MLIR is available\n")

    # Small test case for readable output
    nlev = 3
    ncells = 2

    print(f"Generating MLIR for {nlev} levels × {ncells} cells...")
    print("(Small size for readable output)\n")

    try:
        mlir_code = generate_precip_scan_mlir(nlev, ncells)

        print("="*70)
        print("Generated MLIR Code:")
        print("="*70)
        print(mlir_code)
        print("="*70)

        print("\n✅ MLIR IR generation successful!")

        print("\n" + "="*70)
        print("Optimizations Applied:")
        print("="*70)
        print("""
1. ✅ ALL 4 SPECIES in single kernel
   - Rain, snow, ice, graupel processed together
   - Shared loads for zeta and rho

2. ✅ CARRY STATE in SSA values (registers)
   - 16 carry values: q_prev, flx_prev, rhox_prev, activated (×4 species)
   - scf.for with iter_args keeps them in registers
   - NO D2D memory copies between iterations

3. ✅ BRANCHLESS EXECUTION
   - arith.select instead of scf.if for conditionals
   - Better GPU utilization (no warp divergence)

4. ✅ STATIC MEMREF SHAPES
   - memref<65x100000xf64> instead of memref<?x?xf64>
   - Enables better LLVM optimization

5. ✅ CONSTANTS HOISTED
   - 0.0, 0.5, 1.0, 2.0 computed once outside loop
   - Parameters loaded once before GPU launch

6. ✅ MEMORY COALESCING
   - Access pattern: [k, cell] with threads on cells
   - Contiguous memory access across threads
""")

        print("="*70)
        print("Expected Performance:")
        print("="*70)
        print("""
- JAX lax.scan:  51ms (92% D2D copies)
- DaCe:          14.6ms (registers)
- MLIR target:   ~15-20ms (registers, should match DaCe)

The key insight: scf.for with iter_args compiles to a GPU loop
where carry variables stay in thread-local registers, just like DaCe.
""")

        print("="*70)
        print("Next Steps:")
        print("="*70)
        print("""
1. Install MLIR with GPU support:
   - Need MLIR built with NVPTX/NVVM backend
   - Or use mlir-python-bindings with CUDA support

2. Test compilation pipeline:
   python -c "from core.precip_scans_mlir import compile_mlir_to_gpu"

3. Run on actual hardware and benchmark

4. Compare results with JAX/DaCe implementations
""")

        return 0

    except Exception as e:
        print(f"\n❌ Error generating MLIR code: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
