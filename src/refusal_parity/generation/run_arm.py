"""Phase 1 of an arm: AuditKIT ak.generate with a Tiny Aya (+ SafeTune steering) callable.

    python -m refusal_parity.generation.run_arm --arm B0 --model CohereLabs/tiny-aya-global --preamble default
    python -m refusal_parity.generation.run_arm --arm EORS --acts runs/mech_global/calib_acts.pt --alpha 1.0 --eps 0.05

Writes runs/<arm>[_<tag>]/responses.jsonl and arm_config.json. Then: python -m refusal_parity.scoring.audit score <that file>.
"""

import argparse
import gc
import json
from pathlib import Path

import auditkit as ak
import torch
from auditkit.model import CallableModel

from refusal_parity.common import RUNS, lid_langs, write_jsonl
from refusal_parity.generation.generate import generate, load_model, render, select
from refusal_parity.scoring.audit import to_samples
from refusal_parity.steering import arms

STEER_ARMS = {"S-EN", "S-POOL", "NEG", "CAST-EN", "G-EN", "G-POOL", "G-CAL", "EORS", "EORS-S", "CPOOL-S"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--preamble", default="default", choices=["default", "none", "strict"])
    ap.add_argument("--kinds", default="harmful,benign")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acts", default=str(RUNS / "mech_global" / "calib_acts.pt"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--behavior_layers", default=None, help="comma list; default = cond_layer-2..cond_layer+2")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)  # 32 pushed a 16 GB Mac (with a browser open) into swap
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--prefix",
        default="",
        help="text prepended to every prompt before language ID and rendering "
        "(deployment-realistic framing, e.g. a greeting; deviations log 2026-10-27 23:45)",
    )
    ap.add_argument("--resta_alpha", type=float, default=0.25)  # SafeTune docs: Tiny Aya breaks at >=0.5
    ap.add_argument("--resta_base", default="CohereLabs/tiny-aya-base")
    ap.add_argument("--resta_aligned", default="CohereLabs/tiny-aya-global")
    a = ap.parse_args()

    out = RUNS / (a.arm + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows = select(a.kinds.split(","), a.split, limit=a.limit)
    rows = [dict(r, prompt=a.prefix + r["prompt"]) for r in rows] if a.prefix else rows  # new dicts: no mutation
    tok, model = load_model(a.model)
    cfg = vars(a).copy()

    gen_model, unconditional, on_batch = model, None, None
    if a.arm == "RESTA":  # SafeTune's own apply_resta, base/aligned streamed lazily from disk
        from safetune.interventions.recover.resta import apply_resta

        from refusal_parity.steering.lazy_sd import LazyModel

        apply_resta(
            model,
            base=LazyModel(a.resta_base),
            aligned=LazyModel(a.resta_aligned),
            alpha=a.resta_alpha,
            dare=True,
            dare_drop_rate=0.3,
            dare_seed=0,
        )
        cfg.update(resta="drifted + alpha*DARE(aligned-base), p=0.3, seed=0 (safetune.apply_resta)")
    if a.arm in STEER_ARMS:
        acts = arms.load_acts(Path(a.acts))
        n_layers = model.config.num_hidden_layers
        cands = range(n_layers // 4, 3 * n_layers // 4)
        cond_layer, auc = arms.pick_condition_layer(acts, cands)
        beh = (
            [int(x) for x in a.behavior_layers.split(",")]
            if a.behavior_layers
            else list(range(max(0, cond_layer - 2), min(n_layers, cond_layer + 3)))
        )
        if a.arm in arms.SPLIT_ARMS:  # fit fold -> gate layer and direction; conformal fold -> thresholds
            cond_layer, auc, specs = arms.fit_split_arms(acts, beh, a.eps, cands)
            spec = specs[a.arm]
            cfg.update(split=dict(seed=arms.SPLIT_SEED, frac_fit=arms.SPLIT_FIT))
        else:
            spec = arms.fit_arms(acts, beh, cond_layer, a.eps)[a.arm]
        wrapper = arms.build(model, spec, cond_layer, a.alpha)
        cfg.update(
            cond_layer=cond_layer, cond_layer_calib_auroc=auc, behavior_layers=beh, thresholds=spec["thresholds"]
        )
        if spec["cond"] is None:
            unconditional = wrapper
        else:
            gen_model = wrapper
            if isinstance(spec["thresholds"], dict):  # per-language gate: language from GlotLID at inference
                langs = lid_langs([r["prompt"] for r in rows])
                cfg["lid_accuracy_on_eval"] = sum(l == r["lang"] for l, r in zip(langs, rows, strict=True)) / len(rows)

                def on_batch(idx):
                    wrapper.row_langs = [langs[i] for i in idx]

    (out / "arm_config.json").write_text(json.dumps(cfg, indent=1, default=str))

    texts = {r["prompt"]: render(tok, r["prompt"], a.preamble) for r in rows}

    def tiny_aya(prompts):
        langs_cb = (lambda idx: on_batch([prompt_row[prompts[i]] for i in idx])) if on_batch else None
        return generate(
            tok,
            gen_model,
            [texts[p] for p in prompts],
            max_new_tokens=a.max_new_tokens,
            batch=a.batch,
            on_batch=langs_cb,
            unconditional=unconditional,
        )

    prompt_row = {r["prompt"]: i for i, r in enumerate(rows)}
    # An explicit name carrying the full arm config: AuditKIT's default callable name ignores mutable
    # captures (cfg, model, prompt table), so every arm would share one cache key and E0 would silently
    # replay B0's cached answers. max_retries=0: a crash must not re-run hours of generation 3 more times.
    model_spec = CallableModel(tiny_aya, name="tiny-aya:" + json.dumps(cfg, sort_keys=True, default=str))
    answers = ak.generate(
        to_samples([dict(r, response=None) for r in rows]),
        model=model_spec,
        config=ak.RunConfig(batch_size=len(rows), max_retries=0),
    )
    for r, s in zip(rows, answers, strict=True):
        r.update(response=s.actual_output or "", model=a.model, preamble=a.preamble, arm=a.arm)
    empty = sum(not r["response"].strip() for r in rows)
    assert empty <= 0.02 * len(rows), f"{empty}/{len(rows)} empty responses: generation failed inside AuditKIT?"
    write_jsonl(out / "responses.jsonl", rows)
    print(f"{a.arm}: {len(rows)} responses -> {out}/responses.jsonl")
    model = gen_model = None  # release the model before the caller loads anything else
    gc.collect()
    torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()  # Colab: CUDA


if __name__ == "__main__":
    main()
