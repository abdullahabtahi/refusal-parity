"""Last-token activations at the gate layers for gate-level diagnostics (pre-registered 2026-10-28 06:40, P3/P4).

    python gate_acts.py --model CohereLabs/tiny-aya-earth --out ../runs/gate_acts_earth

Rows (all unseen by any fit): the test split (harmful, benign, Aya red-team OOD), the same harmful and benign
prompts with the "Good morning. " prefix, and XSTest's unsafe contrast prompts in 10 languages
(data/xs_unsafe.jsonl from `build_data.py --xs_unsafe`: the benign set's translation QA + CuratorKIT).
Tokenisation and batching match generate(): rendered chat template, left padding, batches of 16 sorted by
length, decoder only (no 262k-vocab logits), last token. Cohere2's position ids depend on left padding, so
each group runs on its own, in run_arm's row order, to reproduce generation's batches: test (harmful+benign),
greeted test, then OOD and XSTest-unsafe. Activations stay in the model dtype; consumers compute
acts.float() @ v, as GateModel does. Agreement with the gate inside generation is up to batch composition
(bf16 rounding), so rows within rounding of a threshold can differ. Writes acts_L<layer>.pt ({"acts": (N, d)})
per layer and rows.jsonl (row order = tensor order).

    python gate_acts.py --calib --out ../runs/gate_acts_earth_calib   # post hoc, 2026-10-28

--calib recomputes the calibration prompts twice: as mech.py did (one (kind, language) group at a time,
batches of 4, file order) and as generation does (batches of 16 sorted by length). Against the stored Mac (MPS)
calibration activations this separates a hardware effect from a batching effect on the gate score.
"""

import argparse
from pathlib import Path

import torch
from safetune._refusal_helpers import _get_decoder_layers

from common import DATA, RUNS, read_jsonl, write_jsonl
from generate import DEVICE, load_model, render, select

GREETING = "Good morning. "


def gate_groups():
    """Row groups, each batched on its own (see module docstring)."""
    test = [dict(r, variant="orig") for r in select(["harmful", "benign"], "test")]
    greet = [dict(r, prompt=GREETING + r["prompt"], variant="greet") for r in test]
    extra = [dict(r, variant="orig") for r in select(["ood"], "test") + read_jsonl(DATA / "xs_unsafe.jsonl")]
    return [test, greet, extra]


def calib_groups():
    """[(rows, batch, sort)]: mech.py's procedure per (kind, language), then generation-style batching."""
    from common import LANGS

    rows = select(["harmful", "benign"], "calib")
    mech = [
        ([dict(r, variant="mech") for r in rows if r["kind"] == k and r["lang"] == lang], 4, False)
        for k in ("harmful", "benign")
        for lang in LANGS
    ]
    return [*mech, ([dict(r, variant="gen") for r in rows], 16, True)]


def last_token_acts(tok, model, texts, layers, batch=16, sort=True):
    """{layer: (N, d) in the model dtype}, in input order; sort=True batches by length, as generate() does."""
    got, out = {}, {l: [None] * len(texts) for l in layers}
    hooks = [
        _get_decoder_layers(model)[l].register_forward_hook(
            lambda _m, _i, o, l=l: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -1, :].cpu())
        )
        for l in layers
    ]
    try:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])) if sort else list(range(len(texts)))
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
            with torch.no_grad():
                model.model(**enc)
            for l in layers:
                for k, j in enumerate(idx):
                    out[l][j] = got[l][k]
    finally:
        for h in hooks:
            h.remove()
    return {l: torch.stack(v) for l, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-earth")
    ap.add_argument("--layers", default="13,14")  # 13: split-fit gate layer; 14: the α = 1 arms' gate layer
    ap.add_argument("--out", default=str(RUNS / "gate_acts_earth"))
    ap.add_argument("--calib", action="store_true", help="calibration prompts, mech.py- and generation-style")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = calib_groups() if a.calib else [(g, 16, True) for g in gate_groups()]
    layers = [int(x) for x in a.layers.split(",")]
    tok, model = load_model(a.model)
    parts = [
        last_token_acts(tok, model, [render(tok, r["prompt"], "default") for r in g], layers, batch, sort)
        for g, batch, sort in groups
    ]
    rows, acts = [r for g, _, _ in groups for r in g], {l: torch.cat([p[l] for p in parts]) for l in layers}
    for l, x in acts.items():
        torch.save({"acts": x, "layer": l, "model": a.model}, out / f"acts_L{l}.pt")
    write_jsonl(out / "rows.jsonl", [{k: r[k] for k in ("id", "lang", "kind", "variant", "prompt")} for r in rows])
    print(f"{len(rows)} rows x layers {list(acts)} -> {out}")


if __name__ == "__main__":
    main()
