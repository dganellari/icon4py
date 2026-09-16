# MI300A regional/120 compiler-fusion timing

Positive saving means compiler fusion is faster. These are same-node granule measurements.
Raw intervals do not automatically establish a benefit: review the interleaved A/A controls and order diagnostics in JSON.

| Comparison | Metric | A ms | Compiler ms | Saved ms | Raw 95% CI ms |
|---|---|---:|---:|---:|---|
| compiler_vs_native | granule_device | 5.511207 | 5.393807 | 0.117400 | [0.105326, 0.129475] |
| compiler_vs_native | theta_device | 0.872057 | 0.747106 | 0.124951 | [0.122181, 0.127722] |
| compiler_vs_native | granule_wall | 6.412382 | 6.330604 | 0.081778 | [0.064218, 0.099338] |
| compiler_vs_python | granule_device | 5.405009 | 5.359334 | 0.045676 | [0.032822, 0.058530] |
| compiler_vs_python | theta_device | 0.781927 | 0.744248 | 0.037679 | [0.035035, 0.040322] |
| compiler_vs_python | granule_wall | 6.314351 | 6.265489 | 0.048862 | [0.032062, 0.065662] |
