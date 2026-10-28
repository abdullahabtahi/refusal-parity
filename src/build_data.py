"""Assemble the evaluation corpus and pass it through CuratorKIT (clean + exact dedup + provenance).

Inputs : MultiJail (harmful, parallel x10 langs), translated XSTest-safe (benign), aya_redteaming (OOD harmful)
Outputs: data/eval.jsonl with fields id, lang, tier, kind (harmful|benign|ood), split (calib|test), prompt
         data/curatorkit/  (CuratorKIT manifest, rejected.jsonl, provenance, checksums)
"""

import random
import sys
from collections import Counter

from curatorkit import Curator, CuratorConfig
from datasets import load_dataset

from common import DATA, LANGS, REVISIONS, SEED, TIERS, read_jsonl, write_jsonl

# Translation QA gates for the benign set. Constrained-LID probabilities are spread over near-variant
# labels, so the LID gate is a label match (LID_MIN = 0) and meaning preservation is carried by chrF.
LID_MIN, CHRF_MIN = 0.0, 25.0
N_CALIB_HARM, N_CALIB_BENIGN = 105, 90
AYA_LANGS = {
    "arabic": "ar",
    "english": "en",
    "hindi": "hi",
    "filipino": "tl",
    "french": "fr",
    "russian": "ru",
    "serbian": "sr",
    "spanish": "es",
}
N_OOD = 100


def harmful_rows():
    mj = load_dataset("DAMO-NLP-SG/MultiJail", split="train", revision=REVISIONS["DAMO-NLP-SG/MultiJail"])
    return [dict(id=f"mj{r['id']}", lang=l, kind="harmful", prompt=r[l]) for r in mj for l in LANGS if r[l]]


def benign_rows(label="safe", kind="benign"):
    rows = read_jsonl(DATA / "raw" / f"xstest_{label}_ml.jsonl")
    # A translation passes if fastText agrees on the language and the back-translation keeps meaning.
    # Keep an id only if it passes in *every* language, so the benign set stays parallel.
    ok = Counter(
        r["id"] for r in rows if r["lid"] == r["lang"] and r["lid_conf"] >= LID_MIN and r["bt_chrf"] >= CHRF_MIN
    )
    keep = {i for i, c in ok.items() if c == len(LANGS)}
    print(f"{kind}: {len(keep)}/{len(ok)} ids pass translation QA in all {len(LANGS)} languages")
    return [
        dict(id=r["id"], lang=r["lang"], kind=kind, prompt=r["prompt"], xs_type=r["xs_type"])
        for r in rows
        if r["id"] in keep
    ]


def ood_rows():
    rng = random.Random(SEED)
    out = []
    for split, lang in AYA_LANGS.items():
        d = list(
            load_dataset("CohereLabs/aya_redteaming", split=split, revision=REVISIONS["CohereLabs/aya_redteaming"])
        )
        for i, r in enumerate(rng.sample(d, N_OOD)):
            out.append(
                dict(
                    id=f"aya_{lang}{i}",
                    lang=lang,
                    kind="ood",
                    prompt=r["prompt"],
                    harm=r["harm_category"],
                    scope=r["global_or_local"],
                )
            )
    return out


def curate(rows, name="curatorkit"):
    """Run every row through CuratorKIT; returns the rows it kept (matched back by row index)."""
    raw = DATA / "raw" / f"corpus_{name}.jsonl" if name != "curatorkit" else DATA / "raw" / "corpus.jsonl"
    write_jsonl(raw, [dict(instruction=r["prompt"], row=i) for i, r in enumerate(rows)])
    res = Curator(
        CuratorConfig(
            dataset=str(raw),
            format="prompt_only",
            min_tokens=1,
            max_tokens=4096,
            dedup="exact",
            clean=True,
            export_formats=["messages"],
            output_dir=str(DATA / name),
        )
    ).run()
    res.print_summary()
    kept = {}
    for s in res.passed:
        kept[int(s.metadata["row"])] = s.instruction
    print(f"CuratorKIT kept {len(kept)}/{len(rows)}")
    return [dict(r, prompt=kept[i]) for i, r in enumerate(rows) if i in kept]


def assign_splits(rows):
    rng = random.Random(SEED)
    for kind, n_cal in (("harmful", N_CALIB_HARM), ("benign", N_CALIB_BENIGN)):
        ids = sorted({r["id"] for r in rows if r["kind"] == kind})
        calib = set(rng.sample(ids, n_cal))
        for r in rows:
            if r["kind"] == kind:
                r["split"] = "calib" if r["id"] in calib else "test"
    for r in rows:
        r.setdefault("split", "test")
        r["tier"] = TIERS.get(r["lang"], "OOD")
    return rows


def xs_unsafe():
    """XSTest's unsafe contrast prompts: a gate diagnostic only (P4, pre-registered 2026-10-28), never fitted on.
    Same translation QA and CuratorKIT pass as the benign set; ids kept only if parallel in every language."""
    rows = curate(benign_rows("unsafe", "xs_unsafe"), name="curatorkit_xs_unsafe")
    n = Counter(r["id"] for r in rows)
    rows = [dict(r, split="test") for r in rows if n[r["id"]] == len(LANGS)]
    write_jsonl(DATA / "xs_unsafe.jsonl", rows)
    print(f"xs_unsafe: {len(rows) // len(LANGS)} ids x {len(LANGS)} languages")


def main():
    if "--xs_unsafe" in sys.argv:
        return xs_unsafe()
    rows = harmful_rows() + benign_rows() + ood_rows()
    rows = curate(rows)
    # Dedup can remove one language's copy of a prompt; drop such ids so paired tests stay fully parallel.
    langs_per_id = Counter(r["id"] for r in rows if r["kind"] != "ood")
    ragged = {i for i, c in langs_per_id.items() if c != len(LANGS)}
    print("dropping ragged ids:", sorted(ragged))
    rows = assign_splits([r for r in rows if r["id"] not in ragged])
    write_jsonl(DATA / "eval.jsonl", rows)
    print(Counter((r["kind"], r["split"]) for r in rows))


if __name__ == "__main__":
    main()
