#!/usr/bin/env python3
"""Run the real CMake with usable child-exit signals and a bounded wait."""

import os
import signal
import subprocess
import sys


def main():
    previous = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    print(f"[causal cmake] inherited SIGCHLD blocked: {signal.SIGCHLD in previous}", flush=True)
    process = subprocess.Popen(
        [os.environ["ICON4PY_CAUSAL_REAL_CMAKE"], *sys.argv[1:]], start_new_session=True
    )
    try:
        return process.wait(timeout=float(os.environ.get("CAUSAL_CMAKE_TIMEOUT_SECONDS", "1200")))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print("[causal cmake] Timed out; killed CMake and its compiler children.", file=sys.stderr)
        return 124


if __name__ == "__main__":
    sys.exit(main())
