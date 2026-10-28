"""Judge-validation set from PolyGuardPrompts (human-verified refusal and harm labels).

Samples parallel ids, stratified on the English labels (harmful response / safe refusal / safe compliance),
in the PolyGuardPrompts languages that overlap the study. Output: data/pgp_val.jsonl
"""

import random

import pandas as pd
from datasets import load_dataset

from refusal_parity.common import DATA, REVISIONS, SEED, write_jsonl

LANG = {"English": "en", "Chinese": "zh", "Italian": "it", "Arabic": "ar", "Korean": "ko", "Thai": "th", "Hindi": "hi"}
PER_STRATUM = {"harmful": 70, "refusal": 65, "compliance": 65}


def stratum(r):
    if r.response_harm_label == "harmful":
        return "harmful"
    return "refusal" if r.response_refusal_label == "refusal" else "compliance"


def main():
    df = load_dataset(
        "ToxicityPrompts/PolyGuardPrompts", split="test", revision=REVISIONS["ToxicityPrompts/PolyGuardPrompts"]
    ).to_pandas()
    df = df[
        df.language.isin(LANG)
        & df.response.notna()
        & (df.response.str.len() > 0)
        & df.response_harm_label.notna()
        & df.response_refusal_label.notna()
    ]
    complete = df.groupby("id").language.nunique().pipe(lambda s: s[s == len(LANG)].index)
    en = df[(df.language == "English") & df.id.isin(complete)].copy()
    en["stratum"] = en.apply(stratum, axis=1)
    rng = random.Random(SEED)
    ids = [i for s, n in PER_STRATUM.items() for i in rng.sample(sorted(en[en.stratum == s].id), n)]
    sel = df[df.id.isin(ids)]
    rows = [
        dict(
            id=f"pgp{r.id}",
            lang=LANG[r.language],
            kind="pgp",
            prompt=r.prompt,
            response=r.response,
            h_refusal=r.response_refusal_label == "refusal",
            h_harmful=r.response_harm_label == "harmful",
            h_refusal_agree=r.response_refusal_agreement,
            h_harm_agree=r.response_harm_agreement,
            adversarial=r.adversarial,
        )
        for r in sel.itertuples()
    ]
    write_jsonl(DATA / "pgp_val.jsonl", rows)
    print(len(rows), "rows;", pd.DataFrame(rows).groupby("lang")[["h_refusal", "h_harmful"]].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
