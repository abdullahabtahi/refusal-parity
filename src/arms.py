"""Fit every steering arm offline from saved calibration activations, then build it on SafeTune's CASTModel.

Gate scores: projection h.v for the G-* arms and EORS (steer.GateModel); SafeTune's own CAST similarity for
the faithful CAST-EN baseline. Layer choice and thresholds use calibration data only.

Arms (see research/00_research_plan.md):
  S-EN / S-POOL  English / pooled CAA vectors, unconditional     NEG  random vectors, S-EN norms (control)
  CAST-EN   faithful SafeTune CAST: English vectors + cosine-tanh gate, comparator and threshold by max F1
  G-EN      English vectors + projection gate, one max-F1 threshold
  G-POOL    pooled vectors + projection gate, one max-F1 threshold
  G-CAL     pooled, per-language max-F1 threshold (Aziz et al. 2026-style calibration)
  EORS      pooled, per-language tau_l = (1-eps) quantile of benign calibration scores (equal benign FPR)
  EORS-S    as EORS, but split: layer and direction from a fit fold, thresholds from a disjoint conformal fold
            (Mondrian split-conformal; the guarantee needs a score fitted independently of the calibration fold)
  CPOOL-S   the matched-budget control: same split, direction and eps, one conformal threshold on pooled scores
"""

import numpy as np
import torch
from safetune.interventions.steer.cast import CASTCondition, CASTModel, _cast_similarity
from sklearn.metrics import roc_auc_score

from common import LANGS, SEED, SPLIT_SEED
from steer import GateModel

SPLIT_FIT = 1 / 3  # pre-registered 2026-10-28 06:40 (research/00_research_plan.md); seed: common.SPLIT_SEED
SPLIT_ARMS = {"EORS-S", "CPOOL-S"}


def load_acts(path):
    return torch.load(path, weights_only=True)  # {"harmful|en": (N, L, d) fp16, ...}


def diff_means(acts, langs, layer):
    h = torch.cat([acts[f"harmful|{l}"][:, layer].float() for l in langs]).mean(0)
    b = torch.cat([acts[f"benign|{l}"][:, layer].float() for l in langs]).mean(0)
    return h - b


def unit(v):
    return v / v.norm()


def proj_scores(acts, kind, lang, layer, v):
    return (acts[f"{kind}|{lang}"][:, layer].float() @ v).numpy()


def cast_scores(acts, kind, lang, layer, v):
    return np.array([_cast_similarity(x, v) for x in acts[f"{kind}|{lang}"][:, layer].float()])


def pick_condition_layer(acts, candidates):
    """Layer whose pooled direction best separates harmful vs benign, scored on a held-out half of calibration."""
    best, best_auc = None, -1.0
    fit, held = {k: v[::2] for k, v in acts.items()}, {k: v[1::2] for k, v in acts.items()}
    for layer in candidates:
        v = unit(diff_means(fit, LANGS, layer))
        aucs = [
            roc_auc_score(
                [1] * len(held[f"harmful|{l}"]) + [0] * len(held[f"benign|{l}"]),
                np.r_[proj_scores(held, "harmful", l, layer, v), proj_scores(held, "benign", l, layer, v)],
            )
            for l in LANGS
        ]
        if np.mean(aucs) > best_auc:
            best, best_auc = layer, float(np.mean(aucs))
    return best, best_auc


def max_f1_threshold(s_h, s_b):
    """Threshold t maximising F1 of 'fire if score > t' (harmful = positive)."""
    cands = np.unique(np.r_[s_h, s_b])
    f1 = [2 * (s_h > t).sum() / (2 * (s_h > t).sum() + (s_b > t).sum() + (s_h <= t).sum()) for t in cands]
    return float(cands[int(np.argmax(f1))]), float(max(f1))


def fit_arms(acts, behavior_layers, cond_layer, eps):
    """Returns {arm: dict(vecs, cond, thresholds, gate)}; gate is 'proj', 'cast' or None (unconditional)."""
    pooled = {l: diff_means(acts, LANGS, l) for l in behavior_layers}
    english = {l: diff_means(acts, ["en"], l) for l in behavior_layers}
    rng = torch.Generator().manual_seed(SEED)
    negative = {l: unit(torch.randn(v.shape, generator=rng)) * v.norm() for l, v in english.items()}
    v_en, v_pool = unit(diff_means(acts, ["en"], cond_layer)), unit(diff_means(acts, LANGS, cond_layer))

    sh = {l: proj_scores(acts, "harmful", l, cond_layer, v_pool) for l in LANGS}
    sb = {l: proj_scores(acts, "benign", l, cond_layer, v_pool) for l in LANGS}
    thr_en, _ = max_f1_threshold(
        proj_scores(acts, "harmful", "en", cond_layer, v_en), proj_scores(acts, "benign", "en", cond_layer, v_en)
    )
    thr_pool, _ = max_f1_threshold(np.concatenate(list(sh.values())), np.concatenate(list(sb.values())))

    # Faithful CAST (SafeTune/IBM): cosine-tanh similarity; pick comparator + threshold by F1 on English.
    ch, cb = cast_scores(acts, "harmful", "en", cond_layer, v_en), cast_scores(acts, "benign", "en", cond_layer, v_en)
    t_s, f_s = max_f1_threshold(ch, cb)  # comparator "smaller": fires when sim > t
    t_l, f_l = max_f1_threshold(-ch, -cb)  # comparator "larger":  fires when sim < t
    cast = ("smaller", t_s) if f_s >= f_l else ("larger", -t_l)

    return {
        "S-EN": dict(vecs=english, cond=None, thresholds=None, gate=None),
        "S-POOL": dict(vecs=pooled, cond=None, thresholds=None, gate=None),
        "NEG": dict(vecs=negative, cond=None, thresholds=None, gate=None),
        "CAST-EN": dict(vecs=english, cond=v_en, thresholds=cast[1], comparator=cast[0], gate="cast"),
        "G-EN": dict(vecs=english, cond=v_en, thresholds=thr_en, gate="proj"),
        "G-POOL": dict(vecs=pooled, cond=v_pool, thresholds=thr_pool, gate="proj"),
        "G-CAL": dict(
            vecs=pooled, cond=v_pool, thresholds={l: max_f1_threshold(sh[l], sb[l])[0] for l in LANGS}, gate="proj"
        ),
        "EORS": dict(
            vecs=pooled, cond=v_pool, thresholds={l: conformal_threshold(sb[l], eps) for l in LANGS}, gate="proj"
        ),
    }


def conformal_threshold(benign_scores, eps):
    """Split-conformal threshold: the ceil((1-eps)(n+1))-th smallest benign score. For an exchangeable new
    benign prompt, P(score > tau) <= eps (Vovk et al. 2005). Falls back to the max when n is too small."""
    s = np.sort(benign_scores)
    k = int(np.ceil((1 - eps) * (len(s) + 1)))
    return float(s[min(k, len(s)) - 1])


def split_calibration(acts, frac_fit=SPLIT_FIT, seed=SPLIT_SEED):
    """Disjoint fit / conformal folds by prompt index. Ids are parallel across languages (same order per
    language), so a prompt and its translations always land in the same fold."""
    rng, ids = np.random.default_rng(seed), {"fit": {}, "cal": {}}
    for kind in ("harmful", "benign"):
        perm = rng.permutation(acts[f"{kind}|{LANGS[0]}"].shape[0])
        m = round(len(perm) * frac_fit)
        ids["fit"][kind], ids["cal"][kind] = np.sort(perm[:m]), np.sort(perm[m:])
    fold = lambda f: {k: v[torch.as_tensor(ids[f][k.split("|")[0]])] for k, v in acts.items()}
    return fold("fit"), fold("cal"), ids


def fit_split_arms(acts, behavior_layers, eps, candidates, seed=SPLIT_SEED):
    """EORS-S and CPOOL-S. Steering vectors and behaviour layers are the caller's (the same as every other arm:
    they never touch the gate decision). Returns (cond_layer, fit-fold AUROC, {arm: spec})."""
    fit, cal, _ = split_calibration(acts, seed=seed)
    layer, auc = pick_condition_layer(fit, candidates)
    vecs = {l: diff_means(acts, LANGS, l) for l in behavior_layers}
    v = unit(diff_means(fit, LANGS, layer))
    sb = {l: proj_scores(cal, "benign", l, layer, v) for l in LANGS}
    return (
        layer,
        auc,
        {
            "EORS-S": dict(
                vecs=vecs, cond=v, thresholds={l: conformal_threshold(sb[l], eps) for l in LANGS}, gate="proj"
            ),
            "CPOOL-S": dict(
                vecs=vecs, cond=v, thresholds=conformal_threshold(np.concatenate(list(sb.values())), eps), gate="proj"
            ),
        },
    )


def build(model, spec, cond_layer, alpha):
    """SafeTune wrapper for one arm. Unconditional arms are applied with `with wrapper:` (hooks always on)."""
    if spec["gate"] is None:
        dummy = CASTCondition(condition_vector=torch.zeros(1), condition_layer=cond_layer, threshold=0.0, pool="last")
        return CASTModel(model, spec["vecs"], condition=dummy, alpha=alpha)
    cond = CASTCondition(
        condition_vector=spec["cond"],
        condition_layer=cond_layer,
        threshold=spec["thresholds"] if not isinstance(spec["thresholds"], dict) else 0.0,
        comparator=spec.get("comparator", "smaller"),
        pool="last",
    )
    if spec["gate"] == "cast":
        return CASTModel(model, spec["vecs"], condition=cond, alpha=alpha, per_prompt_gate=True)
    return GateModel(model, spec["vecs"], cond, spec["thresholds"], alpha)


def gate_parity(acts, spec, cond_layer):
    """Activation-level proxy: per-language gate TPR (harmful) and FPR (benign), no generation needed."""
    score = cast_scores if spec["gate"] == "cast" else proj_scores
    flip = spec["gate"] == "cast" and spec.get("comparator") == "larger"
    out = {}
    for l in LANGS:
        t = spec["thresholds"][l] if isinstance(spec["thresholds"], dict) else spec["thresholds"]
        fires = (lambda s, t=t: s < t) if flip else (lambda s, t=t: s > t)
        out[l] = dict(
            tpr=float(fires(score(acts, "harmful", l, cond_layer, spec["cond"])).mean()),
            fpr=float(fires(score(acts, "benign", l, cond_layer, spec["cond"])).mean()),
        )
    return out
