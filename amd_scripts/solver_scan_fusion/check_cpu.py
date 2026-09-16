"""Compile both forward sweeps on CPU and inspect their optimized graphs."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'model/common/src'), str(ROOT / 'model/atmosphere/dycore/src')]

import dace
import gt4py.next as gtx
import numpy as np
from gt4py.next.program_processors.runners.dace import make_dace_backend, transformations as tx
from icon4py.model.common import dimension as dims
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep as native,
)
from forward_sweep_fused import solve_tridiagonal_matrix_for_w_forward_sweep as fused

OUT = Path(__file__).resolve().parent
report = {'graphs': {}, 'comparisons': []}


def backend(arm):
    def capture(sdfg):
        sdfg.save(str(OUT / f'{arm}.sdfgz'), compress=True)
        arrays = {
            n: dict(shape=list(map(str, d.shape)), storage=str(d.storage))
            for n, d in sdfg.arrays.items()
            if d.transient and isinstance(d, dace.data.Array) and len(d.shape) == 2
        }
        report['graphs'][arm] = arrays
        print(arm, '2D intermediates', arrays, flush=True)

    return make_dace_backend(
        gpu=False, async_sdfg_call=False, use_metrics=False,
        unstructured_horizontal_has_unit_stride=True, use_zero_origin=True,
        optimization_args={'optimization_hooks': {tx.GT4PyAutoOptHook.TopLevelDataFlowPost: capture}},
    )


programs = {'native': native.with_backend(backend('native')), 'fused': fused.with_backend(backend('fused'))}
for seed in (41, 42, 43):
    rng = np.random.default_rng(seed)
    names = ('theta_v_ic', 'ddqz_z_half', 'z_alpha', 'z_beta', 'z_w_expl', 'z_exner_expl')
    inputs = {n: gtx.as_field((dims.CellDim, dims.KDim), np.asfortranarray(rng.uniform(0.5, 1.5, (8, 17)))) for n in names}
    inputs['vwind_impl_wgt'] = gtx.as_field((dims.CellDim,), rng.uniform(0.1, 0.2, 8))
    expected = None
    for arm, program in programs.items():
        outputs = {n: gtx.as_field((dims.CellDim, dims.KDim), np.full((8, 17), -71., order='F')) for n in ('z_q', 'w')}
        program(**inputs, **outputs, dtime=0.01, cpd=10., horizontal_start=1,
                horizontal_end=7, vertical_start=1, vertical_end=16, offset_provider={})
        result = {n: np.asarray(a.ndarray).copy() for n, a in outputs.items()}
        if arm == 'native':
            expected = result
        else:
            for name in result:
                np.testing.assert_allclose(result[name], expected[name], rtol=1e-11, atol=1e-12)
                assert np.isfinite(result[name]).all()
                report['comparisons'].append(dict(seed=seed, field=name,
                    max_abs_error=float(np.max(np.abs(result[name]-expected[name])))))
    print('passed seed', seed, flush=True)
report['status'] = 'passed'
(OUT / 'LOCAL_CHECK.json').write_text(json.dumps(report, indent=2)+'\n')
