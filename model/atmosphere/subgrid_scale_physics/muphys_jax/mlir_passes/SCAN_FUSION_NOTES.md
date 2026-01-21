# Scan Fusion Implementation Notes

## Problem
- Current: 2 separate scans (precipitation + temperature) = 180 kernel launches
- Target: 1 fused scan = 90 kernel launches (~1.4x speedup)

## Challenge
Temperature scan depends on precipitation outputs:
```python
# Scan 1: Precipitation
qr, qs, qi, qg, pr, ps, pi, pg = precip_scan(...)

# Scan 2: Temperature (needs qr, qs, qi, qg from scan 1)
t_new = temp_scan(..., qr, qs, qi, qg, pr, ps, pi, pg)
```

## Solution Approach

### Option 1: Single Fused Scan (Complex but Optimal)
Combine both scans into one with larger carry state.

**Status:** Implemented in `graupel_fused.py` but needs testing/debugging

**Issues:**
- Complex carry state management
- Need to carefully handle precipitation batching (4 species)
- Input preparation is tricky (need to align dimensions)

**Next Steps:**
1. Debug the fused scan implementation
2. Test that it produces identical results
3. Benchmark performance

### Option 2: Sequential Computation per Level (Simpler)
Instead of 2 full scans, do both operations at each level:

```python
def combined_step(carry, inputs):
    # Step 1: Do precipitation for this level
    q_updated, flux = precip_step(carry.precip, inputs.precip)
    
    # Step 2: Immediately do temperature for this level
    t_updated = temp_step(carry.temp, inputs.temp, q_updated, flux)
    
    return new_carry, (q_updated, flux, t_updated)

# Single scan!
results = lax.scan(combined_step, init_carry, inputs)
```

**Pros:**
- Simpler to implement
- Easier to debug
- Still reduces to 90 kernel launches

**Cons:**
- Lose vmap parallelization of 4 precipitation species
- May be slower than separate batched scans if overhead is high

### Option 3: Pipeline with XLA Async (Advanced)
Let XLA overlap the two scans:

```python
# Start precipitation scan
precip_future = jax.jit(precip_scan, donate_argnums=(0,))(*precip_inputs)

# Temperature scan can start as soon as first levels are ready
# XLA compiler may automatically pipeline
temp_result = temp_scan(..., precip_future)
```

**Status:** Requires XLA compiler tuning, may not work

## Recommendation

**Short term:** Stick with current 2-scan implementation (correct baseline)

**Medium term:** Implement Option 2 (sequential per-level)
- Clearer code
- Easier to verify correctness
- Still gets the 180→90 reduction

**Long term:** Optimize Option 1 (fully fused)
- Maximum performance
- More complex to maintain

## Performance Target

| Implementation | Kernel Launches | Expected Time |
|----------------|-----------------|---------------|
| Current (2 scans) | 180 | 53.4 ms (baseline) |
| Option 2 (sequential) | 90 | ~40-45 ms (1.2-1.3x) |
| Option 1 (fused+batched) | 90 | ~35-38 ms (1.4-1.5x) |
| Custom CUDA | 1 | ~20-25 ms (2-3x) |

## Files

- `graupel.py` - Current implementation (2 scans) ✅
- `graupel_fused.py` - Fused scan attempt (needs debugging) ⚠️
- `scans.py` - Core scan implementations

## Next Actions

1. **Test correctness** of current implementation
2. **Implement Option 2** (sequential per-level fusion)
3. **Benchmark** to validate speedup
4. **Iterate** based on results

---

**Bottom Line:** Scan fusion is the right approach but needs careful implementation. The current 2-scan version is the correct baseline to optimize from.
