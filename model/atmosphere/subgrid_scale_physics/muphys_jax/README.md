# muphys_jax — JAX Graupel Microphysics

JAX implementation of the ICON graupel microphysics scheme with StableHLO injection
for GPU optimization. Self-contained — no dependency on the GT4Py `muphys` package.

## Performance

| Implementation                                         | Time (ms) | Notes                                        |
| ------------------------------------------------------ | --------- | -------------------------------------------- |
| Baseline (vmap-batched)                                | ~51       | `(ncells, nlev)` layout, internal transposes |
| Native transposed + Combined HLO (q_t_update + precip) | ~32       | Single HLO module for the full computation   |

Measured on NVIDIA GH200, R2B06 grid (327680 cells, 90 levels), float64.

## Architecture

### Core (`core/`)

| Module                 | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| `common/constants.py`  | Physical constants (self-contained, no GT4Py dependency)                     |
| `common/backend.py`    | JIT compilation wrapper                                                      |
| `definitions.py`       | `Q` namedtuple (v, c, r, s, i, g) and `TempState`                            |
| `properties.py`        | Fall speed, velocity scale factors                                           |
| `transitions.py`       | Phase transition rates (cloud-to-rain, ice nucleation, etc.)                 |
| `thermo.py`            | Thermodynamic functions (saturation, internal energy)                        |
| `scans.py`             | `lax.scan`-based precipitation and temperature update scans (all layouts)    |
| `optimized_precip.py`  | Custom JAX primitive for HLO injection of precipitation_effects              |
| `optimized_graupel.py` | Custom JAX primitive for HLO injection of full graupel (q_t_update + precip) |

### Implementations (`implementations/`)

| Module                         | Layout           | Description                                                                                                                                                   |
| ------------------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `graupel_baseline.py`          | `(ncells, nlev)` | Canonical baseline. Contains `q_t_update` (all phase transitions) and `temperature_update_scan`. All other implementations import shared functions from here. |
| `graupel.py`                   | `(ncells, nlev)` | Variant with tiled/unrolled scan options. Imports `q_t_update` and `temperature_update_scan` from baseline.                                                   |
| `graupel_native_transposed.py` | `(nlev, ncells)` | Zero-transpose implementation. Three modes: (1) Full-graupel HLO, (2) Precip-only HLO + JAX q_t_update, (3) Pure JAX fallback.                                |
| `q_t_update_fused.py`          | any              | Monolithic q_t_update with inlined physics (`lax.select`/`lax.pow`) for better GPU kernel fusion. Used by native transposed mode 2.                           |
| `graupel_iree.py`              | `(ncells, nlev)` | IREE backend variant (`fori_loop` scans).                                                                                                                     |

### Driver (`driver/`)

| Module               | Description                                                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_graupel_jax.py` | CLI driver compatible with GT4Py `run_graupel_only.py`. Supports `--baseline`, `--split`, `--iree-optimized`, `--tiled`, `--unrolled` modes. |

### Utils (`utils/`)

| Module            | Description                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `data_loading.py` | Shared NetCDF loading: `load_graupel_inputs()`, `load_graupel_reference()`, `load_precip_inputs()`, `calc_dz()`. |

## Pipeline: StableHLO Injection

Trace JAX functions to StableHLO IR, then inject the pre-compiled HLO back into the
JAX computation via custom primitives at MLIR lowering time.

### Step 1: Export StableHLO

```bash
# Export precipitation_effects (transposed layout)
JAX_ENABLE_X64=1 python tools/export_precip_transposed.py --input data.nc

# Export q_t_update
JAX_ENABLE_X64=1 python tools/generate_qt_update_stablehlo.py --input data.nc

# Generate unrolled transposed precipitation scan
JAX_ENABLE_X64=1 python tools/generate_unrolled_transposed.py --input data.nc

# Combine q_t_update + precip into a single module
JAX_ENABLE_X64=1 python tools/generate_combined_graupel.py --input data.nc
```

### Step 2: Run with HLO injection

```bash
# Precip-only injection
JAX_ENABLE_X64=1 python tools/run_graupel_optimized.py \
    --input data.nc \
    --optimized-hlo stablehlo/precip_transposed_x64_lowered.stablehlo \
    --mode native-transposed

# Full graupel injection (best performance)
JAX_ENABLE_X64=1 python tools/run_graupel_optimized.py \
    --input data.nc \
    --graupel-hlo stablehlo/graupel_combined.stablehlo \
    --mode native-transposed
```

### Step 3: Programmatic usage

```python
from muphys_jax.core.optimized_graupel import configure_optimized_graupel
from muphys_jax.implementations.graupel_native_transposed import graupel_run_native_transposed

# Enable HLO injection
configure_optimized_graupel(hlo_path="stablehlo/graupel_combined.stablehlo", use_optimized=True)

# Run (automatically uses injected HLO)
result = graupel_run_native_transposed(dz_t, t_t, p_t, rho_t, q_t, dt, qnc_t)
```

## Tools (`tools/`)

| Script                            | Description                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| `export_precip_transposed.py`     | Export `precipitation_effects_native_transposed` to StableHLO |
| `generate_qt_update_stablehlo.py` | Export `q_t_update_fused` to StableHLO                        |
| `generate_unrolled_stablehlo.py`  | Generate unrolled precipitation scan StableHLO                |
| `generate_unrolled_transposed.py` | Generate unrolled transposed precipitation scan StableHLO     |
| `generate_combined_graupel.py`    | Combine q_t_update + precip into single StableHLO module      |
| `run_graupel_optimized.py`        | Benchmark driver with HLO injection support                   |
| `benchmark_stablehlo.py`          | Benchmark StableHLO export and compilation                    |

## Tests (`tests/`)

```bash
# Run all tests (requires test data in testdata/)
JAX_ENABLE_X64=1 pytest tests/ -v -m datatest

# Run standalone baseline test
JAX_ENABLE_X64=1 pytest tests/test_graupel_standalone.py -v

# Run HLO injection test
JAX_ENABLE_X64=1 pytest tests/test_graupel_hlo_direct.py -v

# Run native transposed validation (CLI tool with benchmark)
JAX_ENABLE_X64=1 python tests/test_graupel_native_transposed.py --input data.nc --reference ref.nc
```

| Test                                | What it validates                                                    |
| ----------------------------------- | -------------------------------------------------------------------- |
| `test_graupel_standalone.py`        | Baseline graupel vs Fortran reference (pytest)                       |
| `test_graupel_hlo_direct.py`        | End-to-end HLO export + injection vs reference (pytest)              |
| `test_graupel_native_transposed.py` | Native transposed vs baseline + optional reference (CLI + benchmark) |
| `test_q_t_update_fused.py`          | Fused q_t_update vs original (correctness + benchmark)               |
| `test_iree_minimal.py`              | IREE backend basic functionality                                     |
| `test_iree_optimized.py`            | IREE scan comparison                                                 |

## Execution Modes in `graupel_native_transposed`

The native transposed implementation selects its mode automatically:

1. **Full-graupel HLO** (best performance): When `configure_optimized_graupel()` is called.
   Single HLO module replaces both q_t_update and precipitation_effects.

2. **Precip-only HLO**: When `configure_optimized_precip(transposed=True)` is called.
   q_t_update runs via JAX (fused version), precipitation via injected HLO.

3. **Pure JAX fallback**: When no HLO is configured.
   Everything runs through JAX `lax.scan` — no external dependencies.

## Test Data

Test data is expected in:

- `testdata/muphys/graupel_only/{mini,R2B05}/input.nc` — graupel input
- `testdata/muphys_graupel_data/{mini,R2B05}/reference.nc` — Fortran reference output
