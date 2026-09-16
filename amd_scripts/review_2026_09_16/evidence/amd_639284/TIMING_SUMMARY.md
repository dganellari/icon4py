# MI300A regional/120 solver-fusion timing

Positive saving means solver fusion is faster. These are same-node granule measurements.
Raw intervals do not automatically establish a benefit: review the interleaved A/A controls and order diagnostics in JSON.

| Comparison | Metric | A ms | Variant ms | Saved ms | Raw 95% CI ms |
|---|---|---:|---:|---:|---|
| solver_coefficients_in_scan | granule_device | 5.519307 | 5.338015 | 0.181292 | [0.164057, 0.198526] |
| solver_coefficients_in_scan | solvers_device | 1.985850 | 1.820405 | 0.165445 | [0.158236, 0.172654] |
| solver_coefficients_in_scan | theta_device | 0.761984 | 0.760219 | 0.001765 | [-0.001795, 0.005326] |
| solver_coefficients_in_scan | granule_wall | 6.577827 | 6.384864 | 0.192963 | [0.171971, 0.213955] |
