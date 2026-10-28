"""Activation-level read on the gates before any generation (held-out test activations).

    python gate_proxy.py runs/mech_global [--eps 0.05]

Per language: each gated arm's TPR (fires on harmful) and FPR (fires on benign); H6 check (CAST vs projection
AUROC) and the share of prompts with negative projection (the regime where Prop. 1 predicts inversion).
Writes <dir>/gate_proxy.json and prints tables.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import arms
from common import LANGS, TIERS

GATED = ["CAST-EN", "G-EN", "G-POOL", "G-CAL", "EORS"]


def matched_thresholds(cal, layer, eps):
    """Pooled projection direction v, EORS per-language thresholds tau (budget eps), and the single pooled
    threshold t whose *calibration* mean FPR matches EORS's."""
    v = arms.unit(arms.diff_means(cal, LANGS, layer))
    sb = {l: arms.proj_scores(cal, "benign", l, layer, v) for l in LANGS}
    tau = {l: arms.conformal_threshold(sb[l], eps) for l in LANGS}
    target = np.mean([np.mean(sb[l] > tau[l]) for l in LANGS])
    cands = np.sort(np.concatenate(list(sb.values())))
    t = min(cands, key=lambda c: abs(np.mean([np.mean(sb[l] > c) for l in LANGS]) - target))
    return v, tau, t


def matched_comparison(cal, test, layer, eps):
    """EORS at budget eps vs a single pooled threshold chosen so its *calibration* mean FPR matches EORS's.
    Returns {gate: {lang: (tpr, fpr)}} on held-out test activations (projection gate, pooled direction)."""
    v, tau, t = matched_thresholds(cal, layer, eps)
    s = {(k, l): arms.proj_scores(test, k, l, layer, v) for k in ("harmful", "benign") for l in LANGS}
    rate = lambda k, l, thr: float(np.mean(s[k, l] > thr))
    return {
        "single threshold": {l: (rate("harmful", l, t), rate("benign", l, t)) for l in LANGS},
        "EORS": {l: (rate("harmful", l, tau[l]), rate("benign", l, tau[l])) for l in LANGS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()
    d = Path(a.dir)
    cal, test = arms.load_acts(d / "calib_acts.pt"), arms.load_acts(d / "test_acts.pt")
    n_layers = next(iter(cal.values())).shape[1]
    layer, auc = arms.pick_condition_layer(cal, range(n_layers // 4, 3 * n_layers // 4))
    specs = arms.fit_arms(cal, [layer], layer, a.eps)

    rows = []
    for arm in GATED:
        par = arms.gate_parity(test, specs[arm], layer)
        for l in LANGS:
            rows.append(dict(arm=arm, lang=l, tier=TIERS[l], **par[l]))
    t = pd.DataFrame(rows)
    summary = (
        t.groupby("arm")
        .agg(
            mean_tpr=("tpr", "mean"),
            worst_tpr=("tpr", "min"),
            max_fpr=("fpr", "max"),
            fpr_gap=("fpr", lambda x: x.max() - x.min()),
        )
        .reindex(GATED)
    )

    v_en = specs["CAST-EN"]["cond"]
    h6 = []
    for l in LANGS:
        y = [1] * len(test[f"harmful|{l}"]) + [0] * len(test[f"benign|{l}"])
        p = np.r_[arms.proj_scores(test, "harmful", l, layer, v_en), arms.proj_scores(test, "benign", l, layer, v_en)]
        c = np.r_[arms.cast_scores(test, "harmful", l, layer, v_en), arms.cast_scores(test, "benign", l, layer, v_en)]
        h6.append(
            dict(
                lang=l,
                tier=TIERS[l],
                auroc_proj=roc_auc_score(y, p),
                auroc_cast=roc_auc_score(y, c),
                neg_proj_harmful=float((p[: sum(y)] < 0).mean()),
                neg_proj_benign=float((p[sum(y) :] < 0).mean()),
            )
        )
    h6 = pd.DataFrame(h6)

    pd.set_option("display.width", 200, "display.precision", 3)
    print(f"condition layer {layer} (held-out calib AUROC {auc:.3f}), eps={a.eps}\n")
    print("TPR (gate fires on harmful) by language:")
    print(t.pivot(index="arm", columns="lang", values="tpr").reindex(GATED)[LANGS])
    print("\nFPR (gate fires on benign) by language:")
    print(t.pivot(index="arm", columns="lang", values="fpr").reindex(GATED)[LANGS])
    print("\nSummary:\n", summary)
    print("\nH6: CAST vs projection separability (English-fit direction):\n", h6)
    (d / "gate_proxy.json").write_text(
        json.dumps(
            dict(
                layer=layer,
                calib_auroc=auc,
                eps=a.eps,
                per_lang=rows,
                summary=summary.reset_index().to_dict("records"),
                h6=h6.to_dict("records"),
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
