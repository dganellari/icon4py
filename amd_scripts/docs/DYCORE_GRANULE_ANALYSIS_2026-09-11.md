> Historical snapshot from September 11. For current results, see [the main analysis](DYCORE_GRANULE_ANALYSIS.md).

# Why the regional grid is slower on MI300A

Updated September 11, 2026.

**MI300A takes about 11% more GPU execution time than GH200 on the global grid,
but about 48% more on the regional grid.** The difference has appeared in several
runs. We have identified promising parts of the code to improve, but have not
yet proved what causes the full gap or validated an optimisation.

This report covers `solve_nonhydro`, one group of calculations in the dynamical
core. We call this group a *granule*. It does not include diffusion or a complete
weather-model timestep.

## What the timing numbers mean

The main comparison uses 120 vertical levels on both grids:

| Grid | MI300A GPU execution time | GH200 GPU execution time | AMD/GH200 |
| --- | ---: | ---: | ---: |
| Global | 35.216 ms | 31.812 ms | **1.107×** |
| Regional | 5.735 ms | 3.877 ms | **1.479×** |

**1.479× means MI300A takes 47.9% more time for this measurement.** It does not
mean the complete model takes 47.9% longer.

We measure time in two ways:

- **GPU execution time:** time the repeatedly called programs separately, then
  add their typical times. We use the median—the middle result from repeated
  measurements—to reduce the influence of unusually slow or fast samples. This
  sum focuses on the GPU calculations, but is not one continuous stopwatch
  measurement of the whole call.
- **Whole-call time:** use a stopwatch around the complete granule call,
  including the work needed to launch and finish the GPU calculations. On
  regional, this is **6.413 ms versus 4.789 ms**, or **1.339×: 33.9% more time**.

Both measures show the regional disadvantage, but they give different sizes for
it. The table comes from the completed cache-measurement runs. A later control
run reproduced the regional GPU ratio at **1.485×**; its global ratio was
**1.128×**, with a timing-bias warning explained below.

## Why a smaller grid can still expose a larger gap

Global has **327,680 cells**. The MeteoSwiss operational regional grid has
**44,528 cells**, roughly one seventh as many. Regional therefore takes much less
absolute time on both GPUs. The finding is that GH200 benefits more from the
change: the *relative* gap between the GPUs grows.

The grids also differ in structure. Regional has a boundary and extra rules for
handling the edge of the domain. Its connections between cells and the ranges
of cells or edges processed by the code differ too. We cannot treat it as an
otherwise identical global problem made seven times smaller.

Two memory effects can happen together:

- **More scattered accesses.** Values belonging to neighbouring cells can be far
  apart in computer memory. Fetching them may require more separate memory
  accesses. Our ordering analysis finds more scatter on regional, including
  away from the boundary. It does not support blaming the halo alone—the extra
  cells used to handle boundaries or communication.
- **More reuse of a smaller data set.** A *cache* is small, fast memory inside the
  GPU that keeps data available for reuse. L1 is the cache closest to the
  calculation units; L2 is another, larger cache layer. A smaller problem may
  reuse more data from cache even when individual accesses are more scattered.

The latest control runs show better L2 cache reuse on regional for both GPUs:

| Measurement | Global | Regional |
| --- | ---: | ---: |
| AMD: fraction of all L2 requests counted as hits | 29.22% | 37.99% |
| GH200: fraction of read/write data pieces counted as L2 hits | 29.34% | 54.94% |
| GH200: the same measure for reads only | 8.81% | 42.84% |

A *hit* means the cache reports that it can supply a request. **The vendors count
these events differently:** AMD counts requests; NVIDIA counts fixed-size pieces
of data. Use the table to see how each GPU changes between grids. It does not
establish an exact ratio of their cache benefits.

Clearing GH200's caches before each GPU operation still left its regional read
hit rate at about 42%. This suggests substantial reuse happens *within* each
operation, rather than relying on data left by the previous operation. It does
not prove that the entire regional problem fits in cache.

## Where we will try to improve the code

**Theta-rho is our first target.** This part of the code computes density and
temperature-related quantities, pressure gradients and wind updates. On regional
it takes **2.19×** GH200's GPU time and accounts for **about 30.5% of the total
regional GPU-time gap**. On global its ratio is only 1.159×.

A *kernel* is one operation launched on the GPU. Theta-rho becomes **six kernels
on regional versus three on global**, on both GPUs. Some calculations that run
together on global run separately on regional. Combining compatible calculations
is called *fusion*: it may reduce launches and avoid writing temporary results
out and reading them back. It can also change how much GPU storage each kernel
needs, so an improvement is not guaranteed.

**Corrector vertical momentum is the second target.** Together, these two
programs account for most of the extra read traffic relative to a simple
size-based reference:

| Program | Regional reads beyond L2 | Extra reads above the size-based reference |
| --- | ---: | ---: |
| Theta-rho | 2.178 GB per granule call | **0.829 GB** |
| Corrector vertical momentum | 1.152 GB per granule call | **0.459 GB** |

The reference asks: how much would each program read if traffic fell in exact
proportion to its number of cells or edges? Across programs present on both
grids, positive differences add up to **1.626 GB per call**. The two targets
contribute **79.2% of that amount**. Other programs read less than this reference;
including their reductions leaves **0.740 GB of net extra reads**.

This is a way to choose where to investigate. It is **not 79.2% of the timing
gap**, and the extra reads are not necessarily avoidable: boundary rules and the
amount of work per cell or edge also change.

## What the waiting-time measurements tell us

GPUs keep many groups of threads ready so they can do other work while some
threads wait for data. On AMD these groups are called *waves*. **Occupancy**
measures how many waves are resident on a calculation unit; **latency** measures
how long an operation remains outstanding.

Averaged across the measured kernels, AMD's regional run has fewer resident waves
and a longer wait for memory instructions to finish. That makes waiting for memory a
reasonable explanation to investigate. But theta-rho shows why we cannot yet
call it the cause:

| AMD theta-rho measurement | Global | Regional |
| --- | ---: | ---: |
| Resident waves per calculation unit | 12.7 | **16.3** |
| Memory-instruction completion time, GPU clock ticks | 1,373 | **3,302** |
| L2 cycles with requests pending | 98.6% | **68.6%** |

These figures average the individual kernels, giving more weight to kernels
that ran longer during the diagnostic measurements.

Theta-rho has **more** resident waves on regional, not fewer. Those waves could
provide more useful overlap, or remain resident longer while waiting. Occupancy
alone does not tell us which. The latency measurement supports investigating
waiting, but does not identify the exact cache or memory component responsible.

Likewise, 68.6% L2 activity does not mean the other 31.4% is recoverable runtime.
It describes requests pending at L2, not how much useful work the whole GPU or
memory system could do. We must change the code and measure the result.

## Why the bandwidth test did not settle the cause

*Bandwidth* is how much data can be moved per second. A separate test using
simple copy and arithmetic operations measured **2,951 GB/s on MI300A** and
**3,743 GB/s on GH200**. These are results for those particular operations, not
proven maximum rates for every program.

There is also a difference in what the tools measure. Our AMD application
counters count traffic leaving L2. Some of it may be served by another cache,
called **MALL**, before reaching the main GPU memory, called **HBM**. NVIDIA's
traffic measurement is taken at HBM. Comparing those byte counts as if both were
main-memory traffic can give a misleading explanation of the timing gap.

We therefore cannot use this bandwidth test to assign a percentage of the gap
to memory speed. **We can still measure whether an optimisation saves time.**
That does not require a new bandwidth test first.

## What the latest runs established, and what comes next

The latest runs were controls: compare unchanged code with itself before trusting
an optimisation comparison. This is an **A/A test**. An **A/B test** instead
compares the original code, A, with a changed version, B.

- **The numerical comparison now passes on both GPUs and grids.** The two A/A
  executions produced zero observed difference across 148 collected model
  arrays. The earlier numerical mismatch no longer appears in these runs.
- **Regional timing passed our noise check.** The apparent change in the vendor
  gap, despite unchanged code, was about **0.0065 ms**, compared with a real gap
  of **1.859 ms**. The statistical check supports proceeding with regional tests;
  it does not guarantee that every small improvement will be detectable.
- **Global timing needs more caution.** Unchanged code produced an apparent
  change of **0.227 ms** in the vendor gap. Small global improvements could be
  measurement bias rather than a code benefit.
- **There is a file-check caveat.** Both jobs failed a final check because the
  source-file inventory had changed. A later check found all 1,599 original
  files unchanged, plus two new analysis scripts—enough to trigger the failure.
  That supports using the collected results, but the later check cannot exclude
  temporary changes during the runs.

The next comparison is **case 5: theta-rho fusion**, on both GPUs and both grids.
The scripts are prepared and transferred to capstor; **no fusion result has yet
been reviewed**. They check that the results remain correct, that the compiler
actually combines the intended calculations, and whether time and traffic fall.
Each optimisation run now includes its own A/A timing control on the same node.
Thirty local software tests pass; GPU correctness still has to pass in the run.

After reviewing case 5, case 4 changes the order in which GPU blocks visit data
without changing the calculations. Case 8 combines that change with fusion, so
we measure their combined benefit instead of assuming their savings add up.

The practical question is now concrete: **can these changes reduce the regional
execution time, and do they help MI300A enough to narrow its gap to GH200?** The
experiments can answer that even if several hardware effects contribute at once.

## Evidence and run instructions

- [Main timing comparison](../cache_runs/20260910T103804Z_mi300a_nid002728_631263/RUN_REVIEW.md):
  AMD job 631263 on nid002728, GH200 job 855892 on nid005083.
- [Latest A/A timings and full counter tables](../causal_runs/pilot_review_2026-09-11/PILOT_REVIEW.md):
  AMD job 632411 on nid002480, GH200 job 858442 on nid005241. Includes the
  statistical calculations and file-check caveat.
- [Detailed cache measurements](../cache_runs/20260910T103804Z_mi300a_nid002728_631263/COUNTER_REVIEW.md)
  and [earlier grid-ordering analysis](DYCORE_GRANULE_ANALYSIS_DETAIL.md).
- [Experiment design and submission instructions](../DYCORE_CAUSAL_EXPERIMENT.md).

Earlier analysis and handoff documents retain historical results; use this page
and the current experiment instructions for the latest reviewed status.
