"""Run every notebook as a plain script and report pass/fail - a smoke test for the whole repo.

Usage:
    uv run python run_all.py              # all sections
    uv run python run_all.py 04 06        # only sections whose folder starts with 04 or 06
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
TIMEOUT_S = 300


def main() -> int:
    prefixes = sys.argv[1:]
    scripts = sorted(
        path
        for folder in ROOT.glob("[0-9][0-9]_*")
        if not prefixes or folder.name.startswith(tuple(prefixes))
        for path in folder.glob("[0-9][0-9]_*.py")
    )
    failures = []
    for script in scripts:
        name = script.relative_to(ROOT)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S
            )
            ok, detail = proc.returncode == 0, proc.stderr.strip().splitlines()[-1:] if proc.returncode else []
        except subprocess.TimeoutExpired:
            ok, detail = False, [f"timeout after {TIMEOUT_S}s"]
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} ({time.perf_counter() - start:.0f}s) {' '.join(detail)[:150]}", flush=True)
        if not ok:
            failures.append(name)
    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
