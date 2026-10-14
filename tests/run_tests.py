"""Run each test module in its own process, several at a time.

Separate processes keep process-wide state (for example the Pillow limits the
indexer raises) from leaking between modules.

Usage: python tests/run_tests.py [name ...]
With names, only modules whose file name contains one of them are run.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
MAX_PARALLEL = 8


def run_module(module: Path) -> tuple[Path, subprocess.CompletedProcess, float]:
    started = perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", module.name],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return module, result, perf_counter() - started


def main(names: list[str]) -> int:
    modules = sorted(
        module for module in (ROOT / "tests").glob("test_*.py")
        if not names or any(name in module.name for name in names)
    )
    failures = []
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, os.cpu_count() or 1)) as pool:
        for module, result, seconds in pool.map(run_module, modules):
            summary = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no output"
            print(f"{'ok  ' if result.returncode == 0 else 'FAIL'} {module.name} ({seconds:.1f}s): {summary}", flush=True)
            if result.returncode:
                failures.append(module.name)
                print(result.stdout + result.stderr, flush=True)
    print(f"\n{len(modules) - len(failures)}/{len(modules)} modules passed in {perf_counter() - started:.1f}s", flush=True)
    if failures:
        print("Failed: " + ", ".join(failures))
    return 1 if failures or not modules else 0


if __name__ == "__main__":
    # Module output is UTF-8; a redirected console may not be able to encode all of it.
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.exit(main(sys.argv[1:]))
