"""Synthetic check of the arm-fitting logic (no model): EORS equalises benign false-fire rates across
languages whose activations are shifted differently; a single pooled threshold does not."""

import numpy as np
import torch

from refusal_parity.common import LANGS
from refusal_parity.steering import arms


def synthetic_acts(n=200, layers=4, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    harm_dir = torch.randn(d, generator=g)
    acts = {}
    for i, lang in enumerate(LANGS):
        shift = -0.15 * i * harm_dir  # later "languages" look less harmful: the LRL score shift
        base = torch.randn(d, generator=g) * 0.3  # language identity offset
        for kind, sign in (("harmful", 1.0), ("benign", 0.0)):
            x = torch.randn(n, layers, d, generator=g) + base + sign * harm_dir + shift
            acts[f"{kind}|{lang}"] = x.half()
    return acts


def test_eors_equalises_benign_fpr():
    acts = synthetic_acts()
    layer, auc = arms.pick_condition_layer(acts, range(4))
    assert auc > 0.7
    specs = arms.fit_arms(acts, [layer], layer, eps=0.05)
    eors = arms.gate_parity(acts, specs["EORS"], layer)
    pool = arms.gate_parity(acts, specs["G-POOL"], layer)
    eors_fpr = np.array([eors[l]["fpr"] for l in LANGS])
    pool_fpr = np.array([pool[l]["fpr"] for l in LANGS])
    assert np.all(np.abs(eors_fpr - 0.05) <= 0.02), eors_fpr  # quantile threshold: FPR ~ eps per language
    assert pool_fpr.max() - pool_fpr.min() > eors_fpr.max() - eors_fpr.min()  # single threshold is unequal
    assert set(specs) >= {"S-EN", "S-POOL", "NEG", "CAST-EN", "G-EN", "G-POOL", "G-CAL", "EORS"}
    neg, en = specs["NEG"]["vecs"][layer], specs["S-EN"]["vecs"][layer]
    assert torch.isclose(neg.norm(), en.norm(), rtol=1e-4)  # negative control matches norm


def test_cast_similarity_inverts_on_shifted_languages():
    """The finding behind GateModel: CAST's cosine-tanh score is sign-dependent, so a downward-shifted
    language can rank harmful below benign while the plain projection still separates perfectly."""
    from sklearn.metrics import roc_auc_score

    acts = synthetic_acts()
    v = arms.unit(arms.diff_means(acts, LANGS, 0))
    y = [1] * 200 + [0] * 200
    auc = lambda f, l: roc_auc_score(y, np.r_[f(acts, "harmful", l, 0, v), f(acts, "benign", l, 0, v)])
    assert auc(arms.proj_scores, "jv") > 0.99 and auc(arms.proj_scores, "en") > 0.99
    assert auc(arms.cast_scores, "en") > 0.99 and auc(arms.cast_scores, "jv") < 0.5


def test_split_arms_keep_folds_apart():
    """EORS-S / CPOOL-S: layer and direction from the fit fold, thresholds from the conformal fold only, and the
    same prompt ids in the same fold for every language (a translation never straddles folds)."""
    acts = synthetic_acts()
    fit, cal, ids = arms.split_calibration(acts)
    assert not set(ids["fit"]["benign"]) & set(ids["cal"]["benign"])
    assert len(ids["fit"]["benign"]) + len(ids["cal"]["benign"]) == 200
    assert all(fit[f"benign|{l}"].shape[0] == len(ids["fit"]["benign"]) for l in LANGS)
    layer, _, specs = arms.fit_split_arms(acts, [0], eps=0.05, candidates=range(4))
    eors, pool = specs["EORS-S"], specs["CPOOL-S"]
    assert isinstance(eors["thresholds"], dict) and isinstance(pool["thresholds"], float)
    assert torch.equal(eors["cond"], pool["cond"])  # same direction: only the threshold rule differs
    # thresholds are conformal quantiles of conformal-fold benign scores under the fit-fold direction
    for l in LANGS:
        s = arms.proj_scores(cal, "benign", l, layer, eors["cond"])
        assert eors["thresholds"][l] == arms.conformal_threshold(s, 0.05)
    pooled = np.concatenate([arms.proj_scores(cal, "benign", l, layer, eors["cond"]) for l in LANGS])
    assert pool["thresholds"] == arms.conformal_threshold(pooled, 0.05)
    # perturbing conformal-fold harmful prompts changes nothing (they are not used by the gate)
    bumped = {k: v.clone() for k, v in acts.items()}
    for l in LANGS:
        bumped[f"harmful|{l}"][ids["cal"]["harmful"]] += 5
    _, _, specs2 = arms.fit_split_arms(bumped, [0], eps=0.05, candidates=range(4))
    assert specs2["EORS-S"]["thresholds"] == eors["thresholds"]
    # conformal-fold benign prompts shape thresholds only: layer and direction stay put
    shifted = {k: v.clone() for k, v in acts.items()}
    for l in LANGS:
        shifted[f"benign|{l}"][ids["cal"]["benign"]] += 5
    layer3, _, specs3 = arms.fit_split_arms(shifted, [0], eps=0.05, candidates=range(4))
    assert layer3 == layer and torch.equal(specs3["EORS-S"]["cond"], eors["cond"])


if __name__ == "__main__":
    test_eors_equalises_benign_fpr()
    test_cast_similarity_inverts_on_shifted_languages()
    test_split_arms_keep_folds_apart()
    print("ok")
