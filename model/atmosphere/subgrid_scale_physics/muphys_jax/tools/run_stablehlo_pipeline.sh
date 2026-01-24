#!/bin/bash
# Complete StableHLO transformation pipeline for graupel optimization
#
# This script runs the full workflow:
# 1. Export StableHLO IR from JAX
# 2. Analyze IR structure
# 3. Transform to eliminate D2D copies
# 4. Apply MLIR optimization passes
# 5. Lower to GPU
# 6. Benchmark performance

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "StableHLO Transformation Pipeline"
echo "=============================================="
echo "Working directory: $BASE_DIR"
echo

# Step 1: Export baseline StableHLO from simple scan
echo "Step 1: Exporting baseline StableHLO from simple scan..."
cd "$BASE_DIR"
python tools/export_stablehlo.py

if [ ! -f "stablehlo_scan_baseline.mlir" ]; then
    echo "ERROR: Failed to generate stablehlo_scan_baseline.mlir"
    exit 1
fi

echo "✓ Generated stablehlo_scan_baseline.mlir"
echo "  Size: $(wc -c < stablehlo_scan_baseline.mlir) bytes"
echo

# Step 2: Export full graupel StableHLO
echo "Step 2: Exporting StableHLO from full graupel physics..."
python tools/export_graupel_stablehlo.py

if [ ! -f "stablehlo_graupel_full.mlir" ]; then
    echo "ERROR: Failed to generate stablehlo_graupel_full.mlir"
    exit 1
fi

echo "✓ Generated stablehlo_graupel_full.mlir"
echo "  Size: $(wc -c < stablehlo_graupel_full.mlir) bytes"
echo

# Step 3: Transform simple scan (v1 transformer)
echo "Step 3: Transforming simple scan with v1 transformer..."
python tools/transform_stablehlo.py stablehlo_scan_baseline.mlir stablehlo_scan_unrolled.mlir

if [ ! -f "stablehlo_scan_unrolled.mlir" ]; then
    echo "ERROR: Failed to generate stablehlo_scan_unrolled.mlir"
    exit 1
fi

echo "✓ Generated stablehlo_scan_unrolled.mlir"
echo

# Step 4: Analyze full graupel with v2 transformer
echo "Step 4: Analyzing full graupel IR with v2 transformer..."
python tools/transform_stablehlo_v2.py stablehlo_graupel_full.mlir stablehlo_graupel_analyzed.mlir 100

if [ ! -f "stablehlo_graupel_analyzed.mlir" ]; then
    echo "ERROR: Failed to generate stablehlo_graupel_analyzed.mlir"
    exit 1
fi

echo "✓ Generated stablehlo_graupel_analyzed.mlir"
echo

# Step 5: Check for MLIR tools
echo "Step 5: Checking for MLIR optimization tools..."

if command -v mlir-opt &> /dev/null; then
    echo "✓ mlir-opt found: $(which mlir-opt)"

    echo
    echo "Applying MLIR optimization passes to simple scan..."
    mlir-opt \
        --canonicalize \
        --cse \
        --symbol-dce \
        stablehlo_scan_unrolled.mlir \
        -o stablehlo_scan_optimized.mlir 2>&1 | head -50

    if [ -f "stablehlo_scan_optimized.mlir" ]; then
        echo "✓ Generated stablehlo_scan_optimized.mlir"
    else
        echo "⚠ mlir-opt failed, but continuing..."
    fi
else
    echo "⚠ mlir-opt not found - skipping optimization passes"
    echo "  Install with: pip install mlir"
fi

echo
echo "=============================================="
echo "Pipeline Summary"
echo "=============================================="
echo
echo "Generated files:"
ls -lh stablehlo*.mlir | awk '{print "  " $9 " (" $5 ")"}'

echo
echo "Next steps:"
echo "  1. Review analysis in stablehlo_graupel_analyzed.mlir"
echo "  2. Implement full body transformation in transform_stablehlo_v2.py"
echo "  3. Apply GPU lowering passes"
echo "  4. Compile and benchmark"
echo
echo "For GPU lowering:"
echo "  mlir-opt --convert-stablehlo-to-linalg \\"
echo "           --linalg-fuse-elementwise-ops \\"
echo "           --convert-linalg-to-gpu \\"
echo "           --gpu-kernel-outlining \\"
echo "           stablehlo_optimized.mlir"
echo
echo "=============================================="
