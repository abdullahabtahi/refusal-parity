"""Fetch the released data into the layout the analysis code reads (data/, runs/).

    .venv/bin/python scripts/fetch_data.py                   # open repo
    .venv/bin/python scripts/fetch_data.py --gated           # + gated repos (accept their terms on the Hub first)
    .venv/bin/python scripts/fetch_data.py --from hf/release # a local copy of the release instead of the Hub

Open repo: prompts, per-response labels (as runs/<run>/judged.jsonl) and every summary file the paper is built from;
enough for analyze.py, arms_report.py and the per-language tables. --gated adds model responses, judge-validation
items, activations and steering arms; enough to rebuild runs/controls_report.json and paper/numbers.tex exactly.

Existing files are never overwritten without --force, so running this in a checkout that already holds results
changes nothing.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import read_jsonl, write_jsonl  # noqa: E402

OWNER = "abdullahabtahi"
OPEN = ("refusal-parity", "dataset")
GATED = (("refusal-parity-responses", "dataset"), ("refusal-parity-steering", "model"))
JUDGE_PREFIXES = ("q3g", "gem_", "kw_", "resp_lang")  # judged.jsonl minus these = the generation record


def source(name, repo_type, local, revision):
    if local:
        path = Path(local) / name
        if not path.is_dir():
            raise SystemExit(f"{path} not found (build it with scripts/build_release.py)")
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(f"{OWNER}/{name}", repo_type=repo_type, revision=revision))


class Writer:
    """Copies into the checkout, skipping files that exist unless force; counts what it did."""

    def __init__(self, force):
        self.force, self.wrote, self.kept = force, 0, 0

    def _free(self, dst):
        if dst.exists() and not self.force:
            self.kept += 1
            return False
        self.wrote += 1
        return True

    def copy(self, src, dst):
        if self._free(dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    def rows(self, rows, dst):
        if self._free(dst):
            write_jsonl(dst, rows)


def fetch_open(src, w, labels):
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_file() and rel.parts[0] in ("data", "runs"):
            w.copy(p, ROOT / rel)
    if labels:  # with --gated the full judged rows (a superset of the labels) are written instead
        for p in sorted((src / "labels").glob("*.jsonl")):
            w.copy(p, ROOT / "runs" / p.stem / "judged.jsonl")


def fetch_responses(src, w):
    """judged.jsonl as released, and responses.jsonl rebuilt from it as the generation record (the released tok_*
    are generation usage, as in the original responses.jsonl)."""
    for p in sorted((src / "responses").glob("*.jsonl")):
        rows = read_jsonl(p)
        w.copy(p, ROOT / "runs" / p.stem / "judged.jsonl")
        w.rows([{k: v for k, v in r.items() if not k.startswith(JUDGE_PREFIXES)} for r in rows],
               ROOT / "runs" / p.stem / "responses.jsonl")  # fmt: skip
    for p in sorted((src / "judge_validation").glob("*.jsonl")):
        w.copy(p, ROOT / "runs" / "pgp_val" / p.name)


def fetch_steering(src, w):
    for p in sorted((src / "activations").rglob("*")):
        if p.is_file():
            w.copy(p, ROOT / "runs" / p.relative_to(src / "activations"))
    for p in sorted((src / "arms").glob("*")):
        w.copy(p, ROOT / "runs" / "steering_arms" / p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gated", action="store_true", help="also fetch the gated responses and steering repos")
    ap.add_argument("--from", dest="local", help="a local release folder (hf/release) instead of the Hub")
    ap.add_argument("--revision", default="main", help="Hub revision (branch, tag or commit) for every repo")
    ap.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = ap.parse_args()
    w = Writer(args.force)
    fetch_open(source(*OPEN, args.local, args.revision), w, labels=not args.gated)
    if args.gated:
        (resp, rtype), (steer, stype) = GATED
        fetch_responses(source(resp, rtype, args.local, args.revision), w)
        fetch_steering(source(steer, stype, args.local, args.revision), w)
    print(f"wrote {w.wrote} files, kept {w.kept} existing (use --force to overwrite)")


if __name__ == "__main__":
    main()
