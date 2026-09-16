"""Fixed-call profiling of the existing dycore granule benchmark.

Load with ``python -m pytest -p dycore_profile_window --benchmark-disable`` and
select only ``test_benchmark_solve_nonhydro[False-False]``. The original test
still creates its fields, grid, compiled programs and synchronized callback.
Only its ``benchmark`` fixture is replaced: no calibration or adaptive rounds.

Environment:
  ICON4PY_PROFILE_WARMUP: complete granule calls before capture (default 5).
  ICON4PY_PROFILE_ROUNDS: complete granule calls inside capture (default 3).
  ICON4PY_PROFILE_MODE: ``cuda``, ``hip`` (ROCTx control), or ``timing``.
  ICON4PY_PROFILE_SEED: seed for unseeded NumPy default_rng calls (default 20260909).
  ICON4PY_PROFILE_REPORT: JSON path; ``{pid}`` expands to the application PID.

For Nsight Compute application replay, use --profile-from-start off and keep
warmup, rounds and seed identical in every run. Prefer a report path containing
{pid} so replay applications do not overwrite one another's metadata. For
Nsight Systems, use --capture-range=cudaProfilerApi. Timings from an instrumented
run include profiler overhead; use mode=timing for standalone wall timings.

CuPy profiler API: https://docs.cupy.dev/en/v13.5.1/reference/cuda.html
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import platform
import socket
import statistics
import sys
import time
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


TARGET = (
    "model/atmosphere/dycore/tests/dycore/integration_tests/"
    "test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]"
)


def _integer(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        result = int(value)
    except ValueError as error:
        raise pytest.UsageError(f"'{name}' must be an integer, got '{value}'.") from error
    if result < minimum:
        raise pytest.UsageError(f"'{name}' must be at least {minimum}, got {result}.")
    return result


def _is_target(item: Any) -> bool:
    path, name = TARGET.split("::")
    return item.nodeid.replace("\\", "/").endswith(TARGET) or (
        str(getattr(item, "path", "")).replace("\\", "/").endswith(path)
        and item.name == name
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    config._icon4py_profile_completed = False
    if len(items) != 1 or not _is_target(items[0]):
        raise pytest.UsageError(
            "'dycore_profile_window' requires exactly one selected test: " + TARGET
            + f"; selected node IDs: {[item.nodeid for item in items]}."
        )
    if config.getoption("numprocesses", default=0):
        raise pytest.UsageError("'dycore_profile_window' requires sequential pytest execution (-n0).")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: Any, nextitem: Any):
    if not _is_target(item):
        yield
        return

    import numpy as np

    seed = _integer("ICON4PY_PROFILE_SEED", 20260909, minimum=0)
    sequence = np.random.SeedSequence(seed)
    original_default_rng = np.random.default_rng
    original_metrics_path = os.environ.get("GT4PY_METRICS_OUTPUT_PATH")
    if original_metrics_path is not None:
        expanded_metrics_path = original_metrics_path.replace("{pid}", str(os.getpid()))
        pathlib.Path(expanded_metrics_path).parent.mkdir(parents=True, exist_ok=True)
        os.environ["GT4PY_METRICS_OUTPUT_PATH"] = expanded_metrics_path
    state = {"seed": seed, "unseeded_rng_calls": 0}
    item._icon4py_profile_rng = state
    if os.environ.get("ICON4PY_PROFILE_MODE") == "hip":
        from amd_cache_counters import ProfilerControl

        item._icon4py_hip_control = ProfilerControl()
        # --selected-regions starts the trace inactive; Pause here aborts SDK 1.1.0.

    def deterministic_default_rng(seed: Any = None):
        if seed is None:
            state["unseeded_rng_calls"] += 1
            seed = sequence.spawn(1)[0]
        return original_default_rng(seed)

    np.random.default_rng = deterministic_default_rng
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng
        if original_metrics_path is not None:
            os.environ["GT4PY_METRICS_OUTPUT_PATH"] = original_metrics_path


class FixedGranuleBenchmark(BenchmarkFixture):
    """A callable fixture for one fixed sequence of full granule invocations."""

    def __init__(self, request: Any):
        self.request = request
        self.name = request.node.name
        self.extra_info: dict[str, Any] = {}
        # pytest-benchmark's report hook requires a BenchmarkFixture instance.
        # Its runner and finalizer are not used by this intercepted fixture.
        self.disabled = False
        self.skipped = False
        self.has_error = False
        self._called = False

    def __call__(self, callback: Any, *args: Any, **kwargs: Any) -> Any:
        if self._called:
            raise RuntimeError("The profiling benchmark may only be invoked once per test.")
        self._called = True
        warmup = _integer("ICON4PY_PROFILE_WARMUP", 5, minimum=1)
        rounds = _integer("ICON4PY_PROFILE_ROUNDS", 3, minimum=1)
        mode = os.environ.get("ICON4PY_PROFILE_MODE", "cuda")
        if mode not in ("cuda", "hip", "timing"):
            raise pytest.UsageError("'ICON4PY_PROFILE_MODE' must be 'cuda', 'hip' or 'timing'.")

        import cupy as cp

        runtime = cp.cuda.runtime
        if runtime.is_hip and mode == "cuda":
            raise RuntimeError("This Nsight profiling window requires a CUDA device.")
        control = getattr(self.request.node, "_icon4py_hip_control", None)
        if mode == "hip" and (not runtime.is_hip or control is None):
            raise RuntimeError("HIP capture requires a HIP device and a selected-regions ROCTx controller.")
        device = runtime.getDevice()
        device_name = runtime.getDeviceProperties(device)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        report = {
            "schema_version": 1,
            "nodeid": self.request.node.nodeid,
            "target": TARGET,
            "grid": self.request.config.getoption("grid", default=None),
            "backend": self.request.config.getoption("backend", default=None),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "cupy": cp.__version__,
            "cuda_runtime": None if runtime.is_hip else runtime.runtimeGetVersion(),
            "gpu_api": "hip" if runtime.is_hip else "cuda",
            "gpu_runtime_version": runtime.runtimeGetVersion(),
            "profiler_control_library": control.path if control is not None else None,
            "device_id": device,
            "device_name": device_name,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode,
            "warmup_requested": warmup,
            "rounds_requested": rounds,
            "warmup_completed": 0,
            "rounds_completed": 0,
            "input_rng_at_entry": dict(getattr(self.request.node, "_icon4py_profile_rng", {})),
            "state_reset_between_calls": False,
            "gt4py_metrics_level": os.environ.get("GT4PY_COLLECT_METRICS_LEVEL"),
            "gt4py_metrics_output": os.environ.get("GT4PY_METRICS_OUTPUT_PATH"),
            "wall_times_ms": [],
            "status": "running",
            "timing_note": "Instrumented-run wall timings include profiler overhead.",
        }
        result = None
        try:
            runtime.deviceSynchronize()
            for _ in range(warmup):
                result = callback(*args, **kwargs)
                report["warmup_completed"] += 1
            runtime.deviceSynchronize()

            profiling_started = False
            try:
                if mode == "cuda":
                    runtime.profilerStart()
                    profiling_started = True
                elif mode == "hip":
                    control.resume()
                    profiling_started = True
                for _ in range(rounds):
                    start = time.perf_counter_ns()
                    result = callback(*args, **kwargs)
                    # The original benchmark callback already synchronizes each call.
                    report["wall_times_ms"].append((time.perf_counter_ns() - start) / 1e6)
                    report["rounds_completed"] += 1
            finally:
                try:
                    runtime.deviceSynchronize()
                finally:
                    if profiling_started:
                        if mode == "hip":
                            control.pause()
                        else:
                            runtime.profilerStop()

            report["median_wall_ms"] = statistics.median(report["wall_times_ms"])
            report["status"] = "complete"
            self.request.config._icon4py_profile_completed = True
            return result
        except BaseException as error:
            self.has_error = True
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            report["input_rng_at_exit"] = dict(
                getattr(self.request.node, "_icon4py_profile_rng", {})
            )
            self.extra_info["fixed_profile_window"] = report
            output = os.environ.get("ICON4PY_PROFILE_REPORT")
            if output:
                path = pathlib.Path(output.replace("{pid}", str(os.getpid())))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                "ICON4PY_PROFILE_WINDOW "
                + json.dumps({k: report[k] for k in (
                    "status", "mode", "warmup_completed", "rounds_completed", "pid"
                )}),
                flush=True,
            )


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: Any, request: Any):
    if fixturedef.argname != "benchmark" or not _is_target(request.node):
        return None
    fixture = FixedGranuleBenchmark(request)
    # pytest's default setup hook normally sets this. Returning a replacement from
    # the first-result hook requires populating its normal fixture cache ourselves.
    fixturedef.cached_result = (fixture, fixturedef.cache_key(request), None)
    return fixture


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    if exitstatus == 0 and not getattr(session.config, "_icon4py_profile_completed", False):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print("ERROR: The requested fixed profiling window did not complete.", flush=True)
