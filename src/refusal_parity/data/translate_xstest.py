"""Translate XSTest safe prompts into the 9 non-English MultiJail languages with NLLB,
then QA each translation with fastText LID and back-translation chrF.

    python -m refusal_parity.data.translate_xstest [--label unsafe]

Output: data/raw/xstest_<label>_ml.jsonl  (one row per prompt x language). The unsafe half is XSTest's
matched-surface contrast set, used only as a gate diagnostic (harm vs dataset style), never for fitting.
"""

import argparse

import torch
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from refusal_parity.common import DATA, LANGS, NLLB_CODES, REVISIONS, detect_lang, load_lid, write_jsonl

MODEL = "facebook/nllb-200-distilled-600M"
BATCH = 32


def translate(model, tok, texts, src, tgt, device):
    tok.src_lang = NLLB_CODES[src]
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i : i + BATCH], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODES[tgt]), max_new_tokens=128, num_beams=4
            )
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="safe", choices=["safe", "unsafe"])
    label = ap.parse_args().label
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISIONS[MODEL])
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.float16, revision=REVISIONS[MODEL]).to(device).eval()
    )
    lid, chrf = load_lid(), CHRF()

    xs = [
        r for r in load_dataset("Paul/XSTest", split="train", revision=REVISIONS["Paul/XSTest"]) if r["label"] == label
    ]
    en = [r["prompt"] for r in xs]
    rows = [
        dict(id=f"xs{r['id']}", lang="en", prompt=r["prompt"], xs_type=r["type"], lid="en", lid_conf=1.0, bt_chrf=100.0)
        for r in xs
    ]

    for lang in LANGS[1:]:
        fwd = translate(model, tok, en, "en", lang, device)
        back = translate(model, tok, fwd, lang, "en", device)
        for r, src, t, b in zip(xs, en, fwd, back, strict=True):
            code, conf = detect_lang(lid, t)
            rows.append(
                dict(
                    id=f"xs{r['id']}",
                    lang=lang,
                    prompt=t,
                    xs_type=r["type"],
                    lid=code,
                    lid_conf=round(conf, 3),
                    bt_chrf=round(chrf.sentence_score(b, [src]).score, 1),
                )
            )
        print(lang, "done", flush=True)

    write_jsonl(DATA / "raw" / f"xstest_{label}_ml.jsonl", rows)
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
