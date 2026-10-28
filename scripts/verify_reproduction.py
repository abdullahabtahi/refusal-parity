"""Rerun the analysis and check that every reported number is reproduced byte for byte.

    make data GATED=1 && make verify

Hashes the generated outputs, regenerates them (both pre-registered reports, then the paper's numbers and the
controls table), and compares. Exit code 1 lists every file that changed; `git diff` then shows how.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("runs/arms_report.json", "runs/controls_report.json", "paper/numbers.tex", "paper/table_controls.tex")
STEPS = ("arms_report.py", "controls_report.py", "paper_numbers.py")


def digests():
    missing = [f for f in OUTPUTS if not (ROOT / f).exists()]
    if missing:
        sys.exit(f"missing {missing}: fetch the released data first (make data GATED=1)")
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in OUTPUTS}


def main():
    before = digests()
    for step in STEPS:
        print(f"running {step} ...", flush=True)
        subprocess.run([sys.executable, step], cwd=ROOT / "src", check=True, stdout=subprocess.DEVNULL)
    changed = [f for f, h in digests().items() if h != before[f]]
    if changed:
        sys.exit("NOT reproduced:\n  " + "\n  ".join(changed))
    print(f"reproduced byte for byte: {', '.join(OUTPUTS)}")


if __name__ == "__main__":
    main()
