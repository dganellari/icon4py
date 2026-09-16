# Solver fusion: a measured additional improvement

MI300A job 639284, nid002926, regional/120. The solver change reduces granule
device time by **3.28% on top of compiler-fused theta-rho**. Both arms use the
same theta-rho implementation. This is a separate, additional optimisation;
it does not replace the theta change. It falls below the 5% target but passes
both the paired statistical checks and the conservative A/A noise screen.

| Measurement | Theta fusion only (A) | Theta + solver fusion (B) | Time reduction |
|---|---:|---:|---:|
| Granule device | 5.519307 ms | 5.338015 ms | **3.28%** |
| Granule host wall | 6.577827 ms | 6.384864 ms | **2.93%** |
| Both solvers, device | 1.985850 ms | 1.820405 ms | **8.33%** |
| Predictor solver | 0.983048 ms | 0.902123 ms | 8.23% |
| Corrector solver | 1.002801 ms | 0.918282 ms | 8.43% |

The granule device metric sums separately recorded program device times. The
wall measurement times the whole granule call. Neither includes diffusion or
constitutes a full weather-model timestep.

## Why this result is credible

Both arms pass whole-granule validation across 148 state fields, with zero
observed finite-value error and matching nonfinite patterns. All 1,969 source
hashes match before and after; all 21 experiment files match the local bundle.
The restored Rayleigh flag is zero. COMPLETE and the completed status report
are present.

There are 12 balanced ABBA/BAAB quartets and 12 interleaved A/A controls. Every
intervention quartet improves both solver time and granule device/wall time.
No arm-order dependence is detected. Independent reconstruction from the raw
samples reproduces all block medians, quartet contrasts and control-adjusted
confidence intervals (INDEPENDENT_CHECK.json).

Granule device saving is **0.181292 ms**, raw 95% CI **[0.164057, 0.198526]**.
Its conservative noise threshold (absolute mean A/A bias plus twice the A/A
quartet standard deviation) is 0.073838 ms, well below the saving. The matched
control-adjusted interval is also positive: [0.167836, 0.220683] ms.

Theta-rho time differs by only 0.001765 ms, with CI [-0.001795, 0.005326]: no
resolved change, as expected with the same implementation. The two solvers
supply 0.165445 ms, about 91% of the observed total device saving.

## What the compiler actually changed

Every matched solver specialization has one fewer GPU kernel and three fewer
coefficient-sized global temporary arrays (39788×119 doubles). That is a net
113.6 MB reduction in logical intermediate array storage per solver variant.
The isolated prototype removed four coefficient arrays; the full pipeline's
net reduction is three. It is the full generated-code result that matters.

The predictor goes from 12 to 11 kernels; corrector variants go from 10/11 to
9/10. Thread blocks remain 256×1×1, four-level blocking is unchanged, and the
shared theta program remains five kernels. This supports retaining the solver
change as an optimisation candidate. It does not separately quantify the
benefit of fewer launches versus reduced memory traffic; no hardware counters
were collected, and array capacity is not measured HBM traffic.

## Total relative to completely original code

This job directly measures the additional 3.28% against the already theta-fused
baseline. It does not include a completely unmodified arm. Combining the earlier
2.13% theta reduction with this result multiplicatively suggests about 5.34%
less device time than original, but that is a cross-run estimate from different
nodes, not a directly measured combined speedup. Do not simply add percentages
or attach the older Python-fusion gain as another independent improvement.

GH200 remains untested for this solver change. The current result is one paired
MI300A allocation, and does not establish how much the vendor gap closes.
