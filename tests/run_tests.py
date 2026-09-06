"""Run each test module in a fresh process so legacy ML stubs cannot leak.

Usage: python tests/run_tests.py
"""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    failures = []
    modules = sorted((root / "tests").glob("test_*.py"))
    for module in modules:
        print(f"\n--- {module.name} ---", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", module.name, "-v"],
            cwd=root,
        )
        if result.returncode:
            failures.append(module.name)
    print(f"\n{len(modules) - len(failures)}/{len(modules)} modules passed", flush=True)
    if failures:
        print("Failed: " + ", ".join(failures))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
