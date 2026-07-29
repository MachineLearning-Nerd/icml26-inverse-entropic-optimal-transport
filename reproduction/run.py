"""Fixed entrypoint for the cumulative reproduction suite."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from reproduction.claim_1 import verify_claim_1
from reproduction.swiss_calibration import run_swiss_calibration
from reproduction.verifiers import verify_claim_2, verify_claim_3


ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _affinity_count() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return None


def main() -> int:
    started = time.perf_counter()
    print("=== EBiEOT cumulative reproduction suite ===", flush=True)
    print("fixed_command=uv run --frozen python -m reproduction.run", flush=True)
    print("selected_backend=hf", flush=True)
    print("selected_flavor=cpu-upgrade", flush=True)
    print("selected_image=ghcr.io/astral-sh/uv:python3.12-bookworm-slim", flush=True)
    print("estimated_required_cores=48", flush=True)
    print("estimate_reason=six concurrent exact-iteration Swiss-Roll fits at eight PyTorch threads each", flush=True)

    results = [
        verify_claim_1(),
        verify_claim_2(),
        verify_claim_3(),
        run_swiss_calibration(),
    ]
    runtime_seconds = time.perf_counter() - started
    provenance = {
        "git_sha": _git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity_count": _affinity_count(),
        "runtime_seconds": runtime_seconds,
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "selected_image": "ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        "estimated_required_cores": 48,
    }
    summary = {
        "suite": "released-linear-swiss-roll-architecture-calibration",
        "paper": "arXiv:2410.02628v5",
        "all_passed": all(result["passed"] for result in results),
        "claims": results,
        "provenance": provenance,
    }
    print("=== MACHINE_READABLE_RESULT ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("=== END_MACHINE_READABLE_RESULT ===", flush=True)
    if not summary["all_passed"]:
        print("FAIL: at least one verifier or negative control did not meet its contract", flush=True)
        return 1
    print(
        "PASS: released-linear Swiss calibration completed and Claims 1-3 remain verified",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
