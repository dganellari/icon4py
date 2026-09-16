# Read the changes in this order

## 1. The solver: move arithmetic to where its results are consumed

Start with [forward_sweep_fused.py](code/forward_sweep_fused.py). This is a
formatted copy of the measured prototype; it is not a third optimisation.

Before, the field operator computed four arrays over cells and vertical levels:

```python
a = ...
b = ...
c = ...
d = ...
q, w = tridiagonal_forward_sweep_for_w(a, b, c, d)
```

Those arrays were written by a GPU kernel and read by the sweep kernel. In the
new `_coefficient_forward_scan`, `a`, `b`, `c` and `d` are scalar values calculated
for the current level immediately before the recurrence uses them:

```python
# Calculate the coefficients for this level here.
normalization = 1 / (b + a * previous_q)
q = -c * normalization
w = (d - a * previous_w) * normalization
```

The snippet omits precision casts for readability; the implementation retains
them. The order of levels, recurrence and boundary initialization are unchanged.
Neighbouring-level inputs are supplied by the field operator below the scan.
The public `solve_tridiagonal_matrix_for_w_forward_sweep` program is unchanged.

Both the predictor and corrector solvers call this forward sweep. The experiment's
[`solver_program.py`](../solver_scan_fusion/solver_program.py) creates isolated
solver modules whose forward-sweep import points at the new implementation.
It changes the import and the experiment's program name, not the rest of the
solver calculations. The original model modules are never overwritten.

The standalone graph eliminates four arrays. In the full solver, other inputs
still need intermediate storage, so the measured net reduction is **three arrays**
and **one kernel** per solver specialization. Do not describe this as a measured
HBM byte reduction: the run collected timings and code evidence, not counters.

## 2. Theta-rho: teach the compiler to fuse the original program

Theta-rho produces rho/theta over levels 0:120. Its pressure/wind consumers use
two vertical bands, 0:27 and 27:120. To join them, the compiler first needs to
split the producer at level 27.

The existing transformation refused this when theta was an external output.
Unlike a temporary, that array must remain available to later calculations.
The [GT4Py patch](patches/03-shared-output-fusion-gt4py.patch) adds
`allow_shared_data=False` and two guarded helpers to `VerticalSplitMapRange`:

- `_shared_access_partition`: checks that reads/writes can be partitioned safely.
- `_split_shared_access`: separates graph access paths without removing the
  external array or its stores.

Read the rejection checks before the mutation code. They exclude aliases, views,
overlapping or shifted accesses, reductions, unsupported strides and consumers
crossing the partition. The existing transient-only path is retained.

The call path is:

```text
original Icon4Py program
  → GT4Py lowers it to a DaCe dataflow graph
  → top-level optimizer calls VerticalSplitMapRange
  → the experiment callback opts in only for the selected theta partition
  → existing DaCe vertical map fusion joins matching producer/consumer pieces
  → normal GPU blocking and HIP code generation
```

The new code is in **GT4Py's DaCe transformation layer**, not a new patch to
DaCe itself. Selection uses the existing optimizer callback
`TopLevelDataFlowVerticalSplitCallBack`; see
[`compiler_hook.py`](../theta_shared_timing/compiler_hook.py) and
[`configure_options`](../theta_shared_timing/theta_compiler.py).
The flag is reset for every candidate because the matcher reuses its object.

This differs from the earlier Python-domain rewrite: the original source stays
unchanged, six kernels become five, and the four gradient buffers are retained.
The older Python rewrite produced seven kernels. These are alternative theta
implementations, so their speedups must not be added.

## 3. How the two changes are compared

[`dycore_causal_plugin.py`](../solver_scan_fusion/dycore_causal_plugin.py),
`ProgramPair.__call__`, selects original or experimental solver programs.
For theta-rho it always chooses the same compiler-fused baseline object in A and
B. This makes the solver's 3.28% result an additional gain over theta fusion.

[`measure.py`](../solver_scan_fusion/measure.py) makes a private copy of the
installed GT4Py package, applies the pinned compiler module, then launches the
granule test. It checks outputs, generated code, source hashes and paired timings.
It does not change the installed compiler or submit jobs itself.

The readable prototype and compiler patch are the product changes to assess.
The program cloning, temporary compiler copy and timing controls are experimental
infrastructure, not a proposed permanent user-facing API.
