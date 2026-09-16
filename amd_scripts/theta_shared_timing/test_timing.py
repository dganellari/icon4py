"""Local checks for compiler selection, fail-closed provenance and timing summaries."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import measure
import pytest
from dycore_causal_core import (
    THETA,
    alias,
    kernel_launches,
    kernel_spans,
    selected,
    vertical_block_factors,
)
from theta_compiler import check_graphs, comparisons, configure_options, graph_evidence


def test_only_compiler_arm_selects_hook():
    from compiler_hook import select_theta_split
    from gt4py.next.program_processors.runners.dace import transformations as tx

    for spec in comparisons():
        for name in (THETA, "unrelated__causal_x"):
            options = {}
            configure_options(name, spec, options)
            assert options == {}
        if spec.get("baseline"):
            options = {}
            configure_options(alias(THETA, spec["baseline"]), spec, options)
            assert options == {}
        options = {"optimization_args": {"blocking_size": 4}}
        configure_options(alias(THETA, spec), spec, options)
        assert selected(THETA, spec) and not selected("unrelated", spec)
        assert options["optimization_args"]["blocking_size"] == 4
        assert (
            options["optimization_args"]["optimization_hooks"][
                tx.GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack
            ]
            is select_theta_split
        )
        with pytest.raises(ValueError, match="overwritten"):
            configure_options(alias(THETA, spec), spec, options)


def overlay_fixture(tmp_path):
    bundle, source, output = [tmp_path / name for name in ("bundle", "installed", "output")]
    for p in (bundle, source, output):
        p.mkdir()
    module = Path("next/program_processors/runners/dace/transformations/map_fusion_extended.py")
    (source / module).parent.mkdir(parents=True)
    (source / module).write_text("original")
    domain = source / "next/iterator/ir_utils/domain_utils.py"
    domain.parent.mkdir(parents=True)
    domain.write_text("float fix")
    (bundle / "map_fusion_extended.py").write_text("patched")
    h = lambda value: hashlib.sha256(value.encode()).hexdigest()
    (bundle / "PATCH_MANIFEST.json").write_text(
        json.dumps(
            dict(
                module=str(module),
                accepted_installed_sha256=[h("original"), h("patched")],
                patched_sha256=h("patched"),
                domain_utils_sha256=h("float fix"),
            )
        )
    )
    return bundle, source, output, module


def test_overlay_preserves_installation(tmp_path):
    bundle, source, out, module = overlay_fixture(tmp_path)
    overlay = measure.build_overlay(source, out, bundle)
    assert (source / module).read_text() == "original"
    assert (overlay / "gt4py" / module).read_text() == "patched"


def test_real_overlay_import_in_fresh_process(tmp_path):
    import gt4py

    out = tmp_path / "run"
    out.mkdir()
    overlay = measure.build_overlay(Path(gt4py.__file__).parent, out)
    env = dict(os.environ, PYTHONPATH=str(overlay) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    command = (
        "import gt4py; from pathlib import Path; import sys; "
        "from gt4py.next.program_processors.runners.dace.transformations.map_fusion_extended "
        "import VerticalSplitMapRange; "
        "assert Path(gt4py.__file__).resolve().parent == Path(sys.argv[1]).resolve(); "
        "assert VerticalSplitMapRange(allow_shared_data=True).allow_shared_data"
    )
    subprocess.run(
        [sys.executable, "-c", command, str(overlay / "gt4py")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("target", ["installed", "patch", "domain"])
def test_overlay_rejects_mismatched_sources(tmp_path, target):
    bundle, source, out, module = overlay_fixture(tmp_path)
    path = {
        "installed": source / module,
        "patch": bundle / "map_fusion_extended.py",
        "domain": source / "next/iterator/ir_utils/domain_utils.py",
    }[target]
    path.write_text("unexpected change")
    with pytest.raises(ValueError):
        measure.build_overlay(source, out, bundle)
    assert not (out / "compiler_overlay").exists()


def test_actual_saved_graphs_pass_audit():
    import dace

    probe = Path(__file__).parent / "fixtures"
    if not probe.exists():
        pytest.skip("Saved graph fixtures are only needed for local integration validation.")
    spec = comparisons()[0]
    audits = []
    for arm in ("native", "shared"):
        graph = dace.SDFG.from_file(str(probe / f"{arm}.hip.sdfgz"))
        code = (probe / f"{arm}.hip.cpp").read_text()
        fused = [
            name
            for name, begin, _ in kernel_spans(code)
            if all(
                token in code[code.rfind("__global__", 0, begin) : begin]
                for token in (
                    "rho_at_edges_on_model_levels",
                    "theta_v_at_edges_on_model_levels",
                    "horizontal_pressure_gradient",
                    "next_vn",
                )
            )
        ]
        audits.append(
            dict(
                program=THETA if arm == "native" else alias(THETA, spec),
                launches=kernel_launches(code),
                vertical_block_factors=vertical_block_factors(code),
                fused_theta_pressure_kernels=fused,
                theta_graph=graph_evidence(graph),
            )
        )
    assert len(check_graphs(spec, audits)) == 2
    audits[1]["launches"] = audits[0]["launches"]
    with pytest.raises(ValueError, match="kernel count"):
        check_graphs(spec, audits)


def test_summary_retains_placebo_bias():
    def blocks(saved):
        result = []
        for q in range(12):
            order = "ABBA" if q % 2 == 0 else "BAAB"
            for arm in order:
                time = 5.0 - (saved if arm == "B" else 0)
                result.append(
                    dict(
                        arm=arm,
                        device_ms=time,
                        wall_ms=time + 1,
                        programs={THETA: time / 5},
                        quartet=q,
                    )
                )
        return result

    report = dict(
        validation={"status": "passed"},
        hostname="test",
        blocks=blocks(0.1),
        placebo={"blocks": blocks(0.04)},
    )
    result = measure.summarize(report)["metrics"]["granule_device"]
    assert result["saved_ms"] == pytest.approx(0.1)
    assert result["placebo"]["saved_ms"] == pytest.approx(0.04)
    assert result["order_review"]["control_adjusted"]["mean_difference_ms"] == pytest.approx(0.06)
