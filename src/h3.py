"""H3 (preregistered): is Earth's worst-language deficit vs Global an action failure rather than a detection failure?

    python h3.py                     # needs runs/mech_{global,earth}/calib_acts.pt and runs/{B0,E0}/judged.jsonl

Detection and action are recomputed for BOTH models with one protocol (mech.py's calibration-only mode): fit the
pooled harmful-minus-benign direction on the even calibration half, score the odd half. (Global's mech.npz used
the test split, so its stored AUROCs are not comparable with Earth's.) Layer: Global's best mean-AUROC layer from
this protocol, plus each model's own best as a sensitivity check.
Deficit languages: the largest SRR drops E0 vs B0 under each judge (union). H3 holds if AUROC drops <= 0.03 in every
deficit language while the projection gap (action margin) shrinks; it fails if AUROC drops > 0.05 in any of them.
The standardised margin (Cohen's d) is reported alongside, because raw projection units differ between models.
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

import analyze
import arms
from common import LANGS, RUNS

N_DEFICIT = 2


def detect_act(acts):
    """{lang: (auroc[layer], gap[layer])} on the odd calibration half, direction from the even half."""
    n_layers = next(iter(acts.values())).shape[1]
    fit = {k: v[::2] for k, v in acts.items()}
    ev = {k: v[1::2] for k, v in acts.items()}
    out = {l: (np.zeros(n_layers), np.zeros(n_layers), np.zeros(n_layers)) for l in LANGS}
    for layer in range(n_layers):
        v = arms.unit(arms.diff_means(fit, LANGS, layer))
        for l in LANGS:
            sh, sb = arms.proj_scores(ev, "harmful", l, layer, v), arms.proj_scores(ev, "benign", l, layer, v)
            out[l][0][layer] = roc_auc_score([1] * len(sh) + [0] * len(sb), np.r_[sh, sb])
            out[l][1][layer] = sh.mean() - sb.mean()
            # Standardised margin (Cohen's d): raw projection units are not comparable across two models
            out[l][2][layer] = (sh.mean() - sb.mean()) / np.sqrt((sh.var(ddof=1) + sb.var(ddof=1)) / 2)
    return out


def srr(cond, judge):
    t = analyze.summary(analyze.load(cond, judge=judge))[0].set_index("lang")
    return t.SRR.to_dict()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="q3g4b,gem", help="provisional runs may use a subset, e.g. gem")
    ap.add_argument("--e0", default="E0", help="run dir holding E0's judged.jsonl (relative to runs/)")
    a = ap.parse_args()
    judges = a.judges.split(",")
    g = detect_act(arms.load_acts(RUNS / "mech_global" / "calib_acts.pt"))
    e = detect_act(arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt"))
    best = {m: int(np.mean([d[l][0] for l in LANGS], axis=0).argmax()) for m, d in (("global", g), ("earth", e))}
    drops = {j: {l: srr("B0", j)[l] - srr(a.e0, j)[l] for l in LANGS} for j in judges}
    deficit = sorted({l for d in drops.values() for l in sorted(d, key=d.get, reverse=True)[:N_DEFICIT]})
    res = {}
    for name, layer in (("global_best", best["global"]), ("earth_best", best["earth"])):
        per = {
            l: dict(
                auroc_global=float(g[l][0][layer]),
                auroc_earth=float(e[l][0][layer]),
                auroc_drop=float(g[l][0][layer] - e[l][0][layer]),
                gap_global=float(g[l][1][layer]),
                gap_earth=float(e[l][1][layer]),
                gap_change=float(e[l][1][layer] - g[l][1][layer]),
                d_global=float(g[l][2][layer]),
                d_earth=float(e[l][2][layer]),
                d_change=float(e[l][2][layer] - g[l][2][layer]),
            )
            for l in LANGS
        }
        holds = all(per[l]["auroc_drop"] <= 0.03 and per[l]["gap_change"] < 0 for l in deficit)
        fails = any(per[l]["auroc_drop"] > 0.05 for l in deficit)
        res[name] = dict(layer=layer, per_lang=per, h3_holds=holds, h3_fails=fails)
    out = dict(
        judges=judges,
        provisional=judges != ["q3g4b", "gem"],
        deficit_langs=deficit,
        srr_drop_e0_vs_b0=drops,
        best_layer=best,
        results=res,
    )
    name = "h3.json" if not out["provisional"] else "h3_provisional_" + "_".join(judges) + ".json"
    (RUNS / "E0" / name).write_text(json.dumps(out, indent=1))
    print(
        "deficit languages:",
        deficit,
        "| SRR drops:",
        {j: {l: round(v, 3) for l, v in d.items()} for j, d in drops.items()},
    )
    for name, r in res.items():
        print(f"\n{name} (layer {r['layer']}): holds={r['h3_holds']} fails={r['h3_fails']}")
        for l in LANGS:
            p = r["per_lang"][l]
            print(
                f"  {l}{'*' if l in deficit else ' '} AUROC {p['auroc_global']:.3f}->{p['auroc_earth']:.3f} "
                f"(drop {p['auroc_drop']:+.3f})  gap {p['gap_global']:.2f}->{p['gap_earth']:.2f}  "
                f"d {p['d_global']:.2f}->{p['d_earth']:.2f} ({p['d_change']:+.2f})"
            )


if __name__ == "__main__":
    main()
