"""Invariants the paired design and the calibration split rely on (skipped when data/eval.jsonl is absent)."""

from collections import defaultdict

import pytest

from common import DATA, LANGS, read_jsonl

EVAL = DATA / "eval.jsonl"
pytestmark = pytest.mark.skipif(not EVAL.exists(), reason="data/eval.jsonl not present (fetch with `make data`)")


def _ids():
    out = defaultdict(list)  # (kind, split, lang) -> ids in file order, the order mech.py stacks activations in
    for r in read_jsonl(EVAL):
        if r["kind"] in ("harmful", "benign"):
            out[r["kind"], r["split"], r["lang"]].append(r["id"])
    return out


def test_ids_are_parallel_and_identically_ordered_in_every_language():
    """arms.split_calibration and every paired test assume row i is the same prompt (or its translation) in
    every language; a reorder would silently pair different prompts."""
    ids = _ids()
    for kind in ("harmful", "benign"):
        for split in ("calib", "test"):
            ref = ids[kind, split, LANGS[0]]
            assert ref and all(ids[kind, split, l] == ref for l in LANGS), (kind, split)


def test_calibration_and_test_ids_are_disjoint():
    ids = _ids()
    for kind in ("harmful", "benign"):
        assert not set(ids[kind, "calib", "en"]) & set(ids[kind, "test", "en"]), kind
