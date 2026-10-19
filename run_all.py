"""
run_all.py
==========
Top-level orchestration script.

Runs every offline experiment in sequence:
  1. Component A offline algorithms (occupancy grid, Dijkstra, A*, PID tuning)
  2. Component B offline experiments (consensus, PSO, neural-network training)

Webots simulations must be started manually — see README.txt for instructions.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def run_script(script: Path, label: str) -> bool:
    """Run a Python script as a subprocess and report success/failure."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Script : {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        print(f"  [PASS] {label}")
        return True
    else:
        print(f"  [FAIL] {label} — exit code {result.returncode}")
        return False


def main() -> None:
    start = time.time()
    results: list[tuple[str, bool]] = []

    scripts = [
        (ROOT / "run_component_a.py", "Component A — offline experiments"),
        (ROOT / "run_component_b.py", "Component B — offline experiments"),
    ]

    for script, label in scripts:
        ok = run_script(script, label)
        results.append((label, ok))

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for label, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    failed = sum(1 for _, ok in results if not ok)
    if failed:
        print(f"\n  {failed} script(s) failed. Check output above for details.")
        sys.exit(1)
    else:
        print("\n  All offline scripts completed successfully.")


if __name__ == "__main__":
    main()
