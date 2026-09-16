"""Pytest adapter for validated, paired, in-granule interventions.

This module is experimental infrastructure. It never edits installed model or
backend files. Both variants use the same allocated state and ordinary GT4Py
in-SDFG device timers. Counter collection is a separate execution mode.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import math
import numbers
import os
from pathlib import Path
import re
import socket
import statistics
import sys
import time

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import dycore_profile_window as window
from dycore_causal_core import (
    THETA,
    alias,
    original_name,
    paired_summary,
    timing_schedule,
    program_source,
    reorder_kernels,
    require,
    selected,
    sha256,
    kernel_spans,
    kernel_launches,
    arm_spec,
    program_spec,
    vertical_block_factors,
)


pytest_collection_modifyitems = window.pytest_collection_modifyitems
pytest_runtest_protocol = window.pytest_runtest_protocol
pytest_sessionfinish = window.pytest_sessionfinish
STATE = dict(arm="A", build=None, programs={}, audits=[], patches=[], aliases={})


def case():
    return json.loads(os.environ["ICON4PY_CAUSAL_CASE"])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def make_program(program, spec, bounds):
    module = sys.modules[program.definition.__module__]
    source = Path(module.__file__).read_text()
    name = alias(program.__name__, spec)
    text = program_source(
        source,
        program.__name__,
        name,
        fusion=spec.get("fusion", False) and program.__name__ == THETA,
        bounds=bounds,
    )
    path = Path(os.environ["ICON4PY_CAUSAL_OUTPUT"]) / "programs" / (name + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    namespace = dict(vars(module))
    namespace["__file__"] = str(path)
    exec(compile(text, str(path), "exec"), namespace)
    STATE["aliases"][name] = program.__name__
    return namespace[name]


class ProgramPair:
    def __init__(self, setup, kwargs):
        self.name = kwargs["program"].__name__
        self.kwargs = kwargs
        self.setup = setup
        STATE["programs"][self.name] = self
        self.native = setup(**kwargs)
        self.variant = None
        self.checked = False
        self.baseline = None
        self.baseline_checked = False

    def __call__(self, *args, **kwargs):
        spec = arm_spec(case(), STATE["arm"])
        if spec is None or not selected(self.name, spec):
            return self.native(*args, **kwargs)
        attribute = "baseline" if STATE["arm"] == "A" else "variant"
        checked_attribute = "baseline_checked" if STATE["arm"] == "A" else "checked"
        if getattr(self, attribute) is None:
            options = dict(self.kwargs)
            options["backend"] = copy.deepcopy(options["backend"])
            options["program"] = make_program(
                options["program"], spec, options.get("horizontal_sizes", {})
            )
            STATE["build"] = spec
            try:
                setattr(self, attribute, self.setup(**options))
            finally:
                STATE["build"] = None
        # Code generation may be deferred until this first call.
        implementation = getattr(self, attribute)
        STATE["build"] = spec
        try:
            if (
                self.name == THETA
                and not getattr(self, checked_attribute)
                and os.environ.get("ICON4PY_CAUSAL_MODE") == "timing"
            ):
                import cupy as cp
                import numpy as np

                names = (
                    "rho_at_edges_on_model_levels",
                    "theta_v_at_edges_on_model_levels",
                    "horizontal_pressure_gradient",
                    "next_vn",
                )
                outputs = [kwargs[n].ndarray for n in names]
                before = [cp.asnumpy(a) for a in outputs]
                self.native(*args, **kwargs)
                expected = [cp.asnumpy(a) for a in outputs]
                for a, saved in zip(outputs, before):
                    a[...] = cp.asarray(saved)
                implementation(*args, **kwargs)
                bounds = self.kwargs["horizontal_sizes"]
                vertical = self.kwargs["vertical_sizes"]
                ranges = [(bounds["horizontal_start"], bounds["horizontal_end"])] * 2 + [
                    (bounds["start_edge_nudging_level_2"], bounds["end_edge_local"]),
                    (bounds["start_edge_lateral_boundary"], bounds["end_edge_local"]),
                ]
                for name, a, ref, (lo, hi) in zip(names, outputs, expected, ranges):
                    sl = (
                        slice(int(lo), int(hi)),
                        slice(int(vertical["vertical_start"]), int(vertical["vertical_end"])),
                    )
                    observed = cp.asnumpy(a)[sl]
                    reference = ref[sl]
                    require(
                        np.isfinite(reference).all() and np.isfinite(observed).all(),
                        f"Theta-rho '{name}' has nonfinite active outputs; validation cannot establish equivalence.",
                    )
                    require(
                        np.allclose(observed, reference, rtol=1e-11, atol=1e-12),
                        f"Theta-rho '{name}' active outputs differ.",
                    )
                setattr(self, checked_attribute, True)
                return None
            return implementation(*args, **kwargs)
        finally:
            STATE["build"] = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    from dace.codegen import compiler
    from gt4py.next import config as gt_config
    from icon4py.model.common import model_options

    require(
        gt_config.BUILD_JOBS <= 0 or gt_config.BUILD_JOBS_MODE is gt_config.BuildJobsMode.SERIAL,
        "Causal code-generation hooks require serial compilation; set GT4PY_BUILD_JOBS=0.",
    )

    require(
        os.environ.get("GT4PY_COLLECT_METRICS_LEVEL") == "10", "Device metrics level must be 10."
    )
    native_setup, native_options = model_options.setup_program, model_options.get_dace_options
    native_folder = compiler.generate_program_folder

    def setup(**kwargs):
        return ProgramPair(native_setup, kwargs)

    def options(program_name, backend_config, **descriptor):
        result = native_options(original_name(program_name), backend_config, **descriptor)
        if "__causal_" in program_name:
            spec = program_spec(program_name, case())
            opts = result.setdefault("optimization_args", {})
            if selected(original_name(program_name), spec) and "block_2d" in spec:
                opts["gpu_block_size_2d"] = tuple(spec["block_2d"])
            if original_name(program_name) == THETA:
                if "vlb" in spec:
                    from icon4py.model.common import dimension

                    opts["blocking_dims"] = (
                        list(dimension.vertical_dims()) if spec["vlb"] > 1 else []
                    )
                    opts["blocking_size"] = spec["vlb"]
                if "block" in spec:
                    opts["gpu_block_size_1d"] = tuple(spec["block"])
                    opts["gpu_block_size_2d"] = tuple(spec["block"])
        return result

    def folder(sdfg, code_objects, *args, **kwargs):
        owner = original_name(sdfg.name)
        if owner in STATE["programs"] or sdfg.name in STATE["aliases"]:
            for obj in code_objects:
                if not kernel_spans(obj.code):
                    continue
                before = obj.code
                traversal = []
                if "__causal_" in sdfg.name and case().get("order_width"):
                    obj.code, traversal = reorder_kernels(
                        obj.code, case()["order_width"], os.environ["ICON4PY_CAUSAL_PLATFORM"]
                    )
                mapping = {}
                for name, _, _ in kernel_spans(obj.code):
                    if "fieldop" in name:
                        unique = "dc_" + sha256(sdfg.name)[:12] + "_" + name
                        mapping[name] = unique
                for name, unique in mapping.items():
                    obj.code = re.sub(r"\b" + re.escape(name) + r"\b", unique, obj.code)
                audit = dict(
                    program=sdfg.name,
                    original_program=owner,
                    before_sha256=sha256(before),
                    after_sha256=sha256(obj.code),
                    kernels=mapping,
                    traversal=traversal,
                    launches=kernel_launches(obj.code),
                    vertical_block_factors=vertical_block_factors(obj.code),
                )
                audit["fused_theta_pressure_kernels"] = [
                    name
                    for name, begin, _ in kernel_spans(obj.code)
                    if all(
                        token in obj.code[obj.code.rfind("__global__", 0, begin) : begin]
                        for token in (
                            "rho_at_edges_on_model_levels",
                            "theta_v_at_edges_on_model_levels",
                            "horizontal_pressure_gradient",
                            "next_vn",
                        )
                    )
                ]
                STATE["audits"].append(audit)
                # Persist beside the build, so cache hits retain proof of intervention.
                destination = Path(args[0]) if args else Path(kwargs["out_path"])
                save_json(destination / "causal_code.json", audit)
        return native_folder(sdfg, code_objects, *args, **kwargs)

    for obj, name, replacement in (
        (model_options, "setup_program", setup),
        (model_options, "get_dace_options", options),
        (compiler, "generate_program_folder", folder),
    ):
        STATE["patches"].append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)
    # Handle modules imported by installed pytest entry points before configuration.
    for module in list(sys.modules.values()):
        if module and getattr(module, "__name__", "").startswith("icon4py."):
            if getattr(module, "setup_program", None) is native_setup:
                STATE["patches"].append((module, "setup_program", native_setup))
                module.setup_program = setup


def pytest_unconfigure(config):
    for obj, name, value in reversed(STATE["patches"]):
        setattr(obj, name, value)


def collect_arrays(roots, scalar_state=None):
    """Collect model fields, preserving views and avoiding backend/workspace internals."""
    import cupy as cp
    import numpy as np

    visited, arrays = set(), {}

    def visit(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, cp.ndarray):
            key = (value.data.ptr, value.shape, value.strides, str(value.dtype))
            arrays.setdefault(key, value)
        elif hasattr(value, "ndarray"):
            visit(value.ndarray)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, functools.partial):
            visit(value.func)
            visit(value.args)
            visit(value.keywords)
        elif inspect.ismethod(value):
            visit(value.__self__)
        elif inspect.isfunction(value):
            for cell in value.__closure__ or ():
                visit(cell.cell_contents)
        elif not isinstance(value, type) and (
            dataclasses.is_dataclass(value) or type(value).__module__.startswith("icon4py.model.")
        ):
            attributes = {}
            if dataclasses.is_dataclass(value):
                attributes.update(
                    (f.name, getattr(value, f.name)) for f in dataclasses.fields(value)
                )
            # Include runtime attributes that are not declared dataclass fields.
            attributes.update(vars(value) if hasattr(value, "__dict__") else {})
            if type(value).__module__.startswith("icon4py.model.") and scalar_state is not None:
                for name, item in attributes.items():
                    if item is None or isinstance(
                        item, (str, bytes, bool, numbers.Number, enum.Enum, np.generic)
                    ):
                        scalar_state.append((value, name, item))
            for item in attributes.values():
                visit(item)

    for root in roots:
        visit(root)
    require(arrays, "No model state fields found for paired restoration.")
    return list(arrays.values())


def scalar_record(obj, name, value):
    if isinstance(value, numbers.Real) and not math.isfinite(value):
        value = str(value)
    return dict(owner=type(obj).__qualname__, attribute=name, initial_value=value)


def verify_scalar_inventory(roots, snapshot):
    current = []
    collect_arrays(roots, scalar_state=current)
    before = {(id(obj), name) for obj, name, _ in snapshot}
    after = {(id(obj), name) for obj, name, _ in current}
    require(
        before == after,
        "Primitive model-state inventory changed; new, removed, or retyped attributes "
        "would escape paired restoration. Added: "
        + str(sorted(after - before))
        + "; removed: "
        + str(sorted(before - after)),
    )


def metric_offsets():
    from gt4py.next.instrumentation import metrics

    return {
        key: len(src.metrics.get("compute").samples) if src.metrics.get("compute") else 0
        for key, src in metrics.sources.items()
    }


def device_samples(before):
    from gt4py.next.instrumentation import metrics

    result = {}
    for key, src in metrics.sources.items():
        metric = src.metrics.get("compute")
        samples = metric.samples[before.get(key, 0) :] if metric else []
        if samples:
            name = original_name(src.metadata["name"])
            result[name] = result.get(name, 0.0) + sum(samples) * 1000
    require(THETA in result, "Missing theta-rho in-SDFG device timer.")
    require(all(0 <= t < 1e6 for t in result.values()), "Invalid ordinary device timings.")
    return result


class CausalBenchmark(BenchmarkFixture):
    def __init__(self, request):
        self.request = request
        self.name = request.node.name
        self.extra_info = {}
        self.disabled = self.skipped = self.has_error = self._called = False

    def __call__(self, callback, *args, **kwargs):
        import cupy as cp
        import numpy as np

        require(not self._called, "Benchmark callback invoked twice.")
        self._called = True
        mode = os.environ.get("ICON4PY_CAUSAL_MODE", "timing")
        report = dict(
            schema_version=1,
            case=case(),
            hostname=socket.gethostname(),
            platform=os.environ["ICON4PY_CAUSAL_PLATFORM"],
            grid=self.request.config.getoption("grid"),
            status="running",
            blocks=[],
            mode=mode,
            gpu_properties=cp.cuda.runtime.getDeviceProperties(0),
            seed_state=self.request.node._icon4py_profile_rng,
        )
        output = os.environ["ICON4PY_PROFILE_REPORT"].replace("{pid}", str(os.getpid()))
        warmup = int(os.environ.get("ICON4PY_PROFILE_WARMUP", "5"))
        rounds = int(os.environ.get("ICON4PY_PROFILE_ROUNDS", "3"))
        try:
            scalar_state = []
            roots = [callback, args, kwargs] + [
                p.kwargs.get("constant_args", {}) for p in STATE["programs"].values()
            ]
            arrays = collect_arrays(roots, scalar_state=scalar_state)
            report["restored_scalar_state"] = [
                scalar_record(obj, name, value) for obj, name, value in scalar_state
            ]
            report["state_arrays"] = [
                dict(shape=a.shape, strides=a.strides, dtype=str(a.dtype), bytes=a.nbytes)
                for a in arrays
            ]
            # Host snapshots avoid doubling GPU allocations, particularly on GH200.
            saved = [cp.asnumpy(a) for a in arrays]
            fingerprint = hashlib.sha256()
            for array in saved:
                fingerprint.update(str((array.shape, str(array.dtype))).encode())
                fingerprint.update(np.ascontiguousarray(array).tobytes())
            report["initial_state_sha256"] = fingerprint.hexdigest()
            report["initial_scalar_state_sha256"] = sha256(
                json.dumps(
                    report["restored_scalar_state"], sort_keys=True, default=str, allow_nan=False
                )
            )
            if hasattr(self.request, "getfixturevalue"):
                mesh = self.request.getfixturevalue("grid_manager").grid
                report["grid_dimensions"] = dict(
                    cells=mesh.num_cells,
                    edges=mesh.num_edges,
                    vertices=mesh.num_vertices,
                    levels=mesh.num_levels,
                    limited_area=mesh.limited_area,
                )

            def restore():
                verify_scalar_inventory(roots, scalar_state)
                for obj, name, value in scalar_state:
                    current = getattr(obj, name)
                    # Leave unchanged frozen dataclass configuration alone.
                    if current is not value and current != value:
                        setattr(obj, name, value)
                for target, source in zip(arrays, saved):
                    target[...] = cp.asarray(source)
                cp.cuda.runtime.deviceSynchronize()

            def invoke():
                return callback(*args, **kwargs)

            # Build both arms before timing/capture. GPU validation runs once in
            # the ordinary timing process, on all collected state fields.
            expected = None
            for arm in ("A", "B"):
                STATE["arm"] = arm
                restore()
                invoke()
                cp.cuda.runtime.deviceSynchronize()
                verify_scalar_inventory(roots, scalar_state)
                if mode == "timing":
                    if arm == "A":
                        expected = [cp.asnumpy(a) for a in arrays]
                    else:
                        finite_values, max_error = 0, 0.0
                        for index, (array, reference) in enumerate(zip(arrays, expected)):
                            actual = cp.asnumpy(array)
                            require(
                                np.array_equal(np.isfinite(actual), np.isfinite(reference)),
                                f"Nonfinite pattern differs in state field {index}.",
                            )
                            if np.issubdtype(actual.dtype, np.inexact):
                                mask = np.isfinite(reference)
                                finite_values += int(mask.sum())
                                if mask.any():
                                    max_error = max(
                                        max_error,
                                        float(np.max(np.abs(actual[mask] - reference[mask]))),
                                    )
                                require(
                                    np.allclose(
                                        actual, reference, rtol=1e-11, atol=1e-12, equal_nan=True
                                    ),
                                    f"GPU numerical validation failed in state field {index}.",
                                )
                            else:
                                require(
                                    np.array_equal(actual, reference),
                                    f"Integer state field {index} differs.",
                                )
                        require(
                            finite_values > 0,
                            "Validation contained no finite floating-point values.",
                        )
                        report["validation"] = dict(
                            status="passed",
                            fields=len(arrays),
                            finite_values=finite_values,
                            max_abs_error=max_error,
                            rtol=1e-11,
                            atol=1e-12,
                        )
                        del expected
            report["scalar_inventory_validation"] = "passed_after_both_arms"
            report["rayleigh_reset_evidence"] = [
                dict(
                    initial_value=value,
                    requested_dtime=getattr(callback, "keywords", {}).get("dtime"),
                    note="Initial zero shows the stale-flag mechanism was possible; a passing A/A alone does not prove historical causation.",
                )
                for _, name, value in scalar_state
                if name == "_dtime_previous_substep"
            ]

            if mode == "timing":
                seed = int(os.environ.get("CAUSAL_ORDER_SEED", "20260912"))
                schedule = timing_schedule(
                    int(os.environ.get("CAUSAL_QUARTETS", "6")),
                    seed,
                    placebo=case()["name"] != "native",
                )
                report["timing_design"] = dict(seed=seed, schedule=schedule, interleaved=True)
                phase_data = {"intervention": report["blocks"], "placebo": []}
                for phase, quartet, order in schedule:
                    phase_blocks = phase_data[phase]
                    for arm in order:
                        STATE["arm"] = "A" if phase == "placebo" else arm
                        restore()
                        for _ in range(warmup):
                            invoke()
                        samples = []
                        for _ in range(rounds):
                            before = metric_offsets()
                            start = time.perf_counter_ns()
                            invoke()
                            wall_ms = (time.perf_counter_ns() - start) / 1e6
                            programs = device_samples(before)
                            samples.append(
                                dict(
                                    device_ms=sum(programs.values()),
                                    wall_ms=wall_ms,
                                    programs=programs,
                                )
                            )
                        names = set(samples[0]["programs"])
                        require(
                            all(set(s["programs"]) == names for s in samples),
                            "Hot-loop program coverage changed.",
                        )
                        phase_blocks.append(
                            dict(
                                arm=arm,
                                quartet=quartet,
                                order=order,
                                samples=samples,
                                device_ms=sum(
                                    statistics.median(s["programs"][n] for s in samples)
                                    for n in names
                                ),
                                wall_ms=statistics.median(s["wall_ms"] for s in samples),
                                programs={
                                    n: statistics.median(s["programs"][n] for s in samples)
                                    for n in names
                                },
                            )
                        )
                if phase_data["placebo"]:
                    report["placebo"] = dict(
                        blocks=phase_data["placebo"], paired=paired_summary(phase_data["placebo"])
                    )
                report["paired"] = paired_summary(report["blocks"])
            else:
                require(mode in ("hip", "cuda"), "Unknown counter mode.")
                proof = json.loads(Path(os.environ["ICON4PY_CAUSAL_PROOF"]).read_text())
                require(
                    proof["status"] == "complete"
                    and proof["validation"]["status"] == "passed"
                    and proof["case"] == case(),
                    "Missing or mismatched numerical validation proof.",
                )
                require(
                    proof["grid"] == report["grid"]
                    and proof["initial_state_sha256"] == report["initial_state_sha256"],
                    "Profiler inputs differ from the validated timing run.",
                )
                require(
                    proof.get("initial_scalar_state_sha256")
                    == report["initial_scalar_state_sha256"],
                    "Profiler primitive model state differs from the validated timing run.",
                )
                capture_arm = os.environ.get("ICON4PY_CAUSAL_COUNTER_ARM", "B")
                require(capture_arm in ("A", "B"), "Invalid counter arm.")
                report["counter_arm"] = capture_arm
                STATE["arm"] = capture_arm
                restore()
                for _ in range(warmup):
                    invoke()
                cp.cuda.runtime.deviceSynchronize()
                control = getattr(self.request.node, "_icon4py_hip_control", None)
                if mode == "hip":
                    control.resume()
                else:
                    cp.cuda.runtime.profilerStart()
                try:
                    for _ in range(rounds):
                        invoke()
                    cp.cuda.runtime.deviceSynchronize()
                finally:
                    if mode == "hip":
                        control.pause()
                    else:
                        cp.cuda.runtime.profilerStop()
                report["captured_calls"] = rounds
            verify_scalar_inventory(roots, scalar_state)
            report["aliases"] = STATE["aliases"]
            report["status"] = "complete"
            self.request.config._icon4py_profile_completed = True
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            self.has_error = True
            raise
        finally:
            save_json(output, report)
            self.extra_info["causal_report"] = output


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.argname == "benchmark" and window._is_target(request.node):
        fixture = CausalBenchmark(request)
        fixturedef.cached_result = (fixture, fixturedef.cache_key(request), None)
        return fixture
