"""Detection vs action, per language and layer (H3).

    python mech.py --model CohereLabs/tiny-aya-global --out runs/mech_global

For each layer l and language L, using last-token prompt activations (chat template, default preamble):
  r_pool[l]       pooled harmful-minus-benign diff-in-means over calibration prompts of all languages (unit)
  r_L[l]          the same direction from language L alone
  auroc[L, l]     AUROC of <h, r_pool> separating harmful vs benign *test* prompts in L     (detection)
  proj[L, l]      mean <h, r_pool> of harmful test prompts in L, minus benign mean            (action margin)
  cos[L, L', l]   cosine(r_L, r_L')                                                          (shared direction?)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import DATA, LANGS, read_jsonl
from generate import load_model, render

BATCH = 4  # small batches: on a 16 GB Mac, larger ones pushed the system into swap thrash


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_test", type=int, default=100)
    a = ap.parse_args()

    tok, model = load_model(a.model)
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers))
    blocks = model.model.layers
    rows = [r for r in read_jsonl(DATA / "eval.jsonl") if r["kind"] in ("harmful", "benign")]

    def acts(kind, split, lang, n=None):
        """{layer: (N, d) last-prompt-token residual stream}, batched; same rendering as generation."""
        ps = [
            render(tok, r["prompt"], "default")
            for r in rows
            if r["kind"] == kind and r["split"] == split and r["lang"] == lang
        ][:n]
        out = {l: [] for l in layers}
        # Hooks keep only the last (left-padded) token per layer; model.model skips the 262k-vocab lm_head,
        # whose all-position logits alone were ~3.8 GB per batch.
        hooks = [
            blocks[l].register_forward_hook(
                lambda _m, _i, o, l=l: out[l].append((o[0] if isinstance(o, tuple) else o)[:, -1, :].float().cpu())
            )
            for l in layers
        ]
        try:
            for i in range(0, len(ps), BATCH):
                enc = tok(ps[i : i + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(
                    model.device
                )
                with torch.no_grad():
                    model.model(**enc)
        finally:
            for h in hooks:
                h.remove()
        print(f"  {kind}/{split}/{lang}: {len(ps)}", flush=True)
        return {l: torch.cat(v) for l, v in out.items()}

    cal = {(k, L): acts(k, "calib", L) for k in ("harmful", "benign") for L in LANGS}
    if a.n_test:
        test, fit = {(k, L): acts(k, "test", L, a.n_test) for k in ("harmful", "benign") for L in LANGS}, cal
    else:  # calibration-only: fit directions on the even half, score on the held-out odd half
        fit = {key: {l: v[l][::2] for l in layers} for key, v in cal.items()}
        test = {key: {l: v[l][1::2] for l in layers} for key, v in cal.items()}

    auroc = np.zeros((len(LANGS), n_layers))
    proj = np.zeros_like(auroc)
    cos = np.zeros((len(LANGS), len(LANGS), n_layers))
    for l in layers:
        r_lang = [unit(fit["harmful", L][l].mean(0) - fit["benign", L][l].mean(0)) for L in LANGS]
        r_pool = unit(torch.stack(r_lang).mean(0))
        for i, L in enumerate(LANGS):
            sh, sb = test["harmful", L][l] @ r_pool, test["benign", L][l] @ r_pool
            auroc[i, l] = roc_auc_score([1] * len(sh) + [0] * len(sb), torch.cat([sh, sb]).numpy())
            proj[i, l] = float(sh.mean() - sb.mean())
            for j in range(len(LANGS)):
                cos[i, j, l] = float(r_lang[i] @ r_lang[j])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mech.npz", auroc=auroc, proj=proj, cos=cos, langs=np.array(LANGS))
    # Calibration activations (N, layers, d) in fp16: steer.py fits directions/gates/thresholds from these offline.
    torch.save(
        {
            f"{k}|{L}": torch.stack([cal[k, L][l] for l in layers], 1).half()
            for k in ("harmful", "benign")
            for L in LANGS
        },
        out / "calib_acts.pt",
    )
    if a.n_test:
        torch.save(
            {
                f"{k}|{L}": torch.stack([test[k, L][l] for l in layers], 1).half()
                for k in ("harmful", "benign")
                for L in LANGS
            },
            out / "test_acts.pt",
        )
    best = int(auroc.mean(0).argmax())
    summary = dict(
        model=a.model,
        best_layer_by_mean_auroc=best,
        auroc_at_best={L: round(float(auroc[i, best]), 3) for i, L in enumerate(LANGS)},
        mean_offdiag_cos_at_best=round(float(cos[:, :, best][~np.eye(len(LANGS), dtype=bool)].mean()), 3),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
