"""Gate-level and mechanistic analyses, natural drift and CircuitKIT (paper §4, appendices B-D)."""

import json

import numpy as np
import pandas as pd

from refusal_parity.analysis import analyze, arms_report, gate_proxy
from refusal_parity.common import HRL, LANGS, ROOT, RUNS
from refusal_parity.reporting import figures_diagnostics
from refusal_parity.reporting.paper_numbers.core import macro, pct, pct1, spp1
from refusal_parity.steering import arms


def natural_drift():
    """Natural drift: Earth (E0) vs Global (B0), paired, both judges (preregistered comparison; H3 below)"""
    for j, J in (("q3g4b", "Q"), ("gem", "G")):
        a0, e0 = analyze.load("B0", judge=j), analyze.load("E0", judge=j)
        t, boot = analyze.compare(a0, e0)
        src = f"analyze.compare(B0, E0, judge={j}) paired bootstrap / McNemar-Holm"
        for k, name in (
            ("d_mean_SRR", "Srr"),
            ("d_worst_SRR", "WorstSrr"),
            ("d_mean_HRR", "Hrr"),
            ("d_max_BRR", "MaxBrr"),
        ):
            mean, lo, hi = boot[k]
            macro(f"Drift{J}{name}", mean, src, spp1)
            macro(f"Drift{J}{name}Lo", lo, src, spp1)
            macro(f"Drift{J}{name}Hi", hi, src, spp1)
        macro(f"Drift{J}NSig", int(((t.metric == "unsafe") & (t.p_holm < 0.05)).sum()), src, "{:d}")
        macro(f"Drift{J}HarmBn", t[(t.metric == "unsafe") & (t.lang == "bn")].delta.iloc[0], src, spp1)


def tiny_aya_report_values():
    """External reference values (not our measurements): Salamanca et al. 2026, Tiny Aya report, Table 7, MultiJail safe"""
    # response rate across 10 languages, Command A judge in contextual safety mode. arXiv 2603.11510v1, read 2026-10-27.
    src = "Salamanca et al. 2026 (arXiv 2603.11510v1) Table 7, read 2026-10-27"
    TINY_AYA_T7 = dict(Global=(87.0, 91.1), Earth=(77.5, 87.8))  # (min, mean) safe response rate, %
    macro("ReportMinSafeGlobal", TINY_AYA_T7["Global"][0], src, "{:.1f}")
    macro("ReportMinSafeEarth", TINY_AYA_T7["Earth"][0], src, "{:.1f}")
    macro("ReportMeanSafeGlobal", TINY_AYA_T7["Global"][1], src, "{:.1f}")
    macro("ReportMeanSafeEarth", TINY_AYA_T7["Earth"][1], src, "{:.1f}")
    macro("ReportMinSafeDrop", TINY_AYA_T7["Global"][0] - TINY_AYA_T7["Earth"][0], src, "{:.1f}")


def drift_power():
    """Power (Connor 1987 McNemar approximation): the smallest paired per-language change detectable with 80% power, using"""
    # the observed B0/E0 discordance under the primary judge. alpha 0.05 unadjusted, and 0.005 (Holm's first step, 10 tests).
    pw = arms_report.power(analyze.load("B0"), analyze.load("E0"))
    src = "analyze.load(B0, E0, judge=q3g4b) discordance + Connor (1987) McNemar MDE, 80% power"
    for key, tag in (("harmful_unsafe", "Harm"), ("benign_refuse", "Brr")):
        macro(f"PowerN{tag}", pw[key]["n"], src, "{:d}")
        macro(f"PowerPsi{tag}", 100 * pw[key]["psi"], src, pct1)  # share of paired prompts whose label differs
        for mde, suffix in (
            (pw[key]["mde"], ""),
            (pw[key]["mde_holm"], "Holm"),
        ):  # nan: |change| <= psi caps any effect
            macro(f"PowerMde{tag}{suffix}", "not attainable" if mde != mde else f"{100 * mde:.1f} points", src)

    h3 = json.loads((RUNS / "E0" / "h3.json").read_text())
    assert not h3["provisional"], "H3 macros need both judges"
    pl = h3["results"]["global_best"]["per_lang"]
    src = "runs/E0/h3.json (global_best layer, odd calibration half)"
    macro("HThreeLayer", h3["results"]["global_best"]["layer"], src, "{:d}")
    macro("HThreeDeficit", ", ".join(figures_diagnostics.NAMES[l] for l in h3["deficit_langs"]), src)
    macro("HThreeAurocDropMax", max(pl[l]["auroc_drop"] for l in h3["deficit_langs"]), src, "{:.3f}")
    macro("HThreeDBnChange", pl["bn"]["d_change"], src, "{:+.2f}")
    macro("HThreeDSwChange", pl["sw"]["d_change"], src, "{:+.2f}")
    macro("HThreeDEnChange", pl["en"]["d_change"], src, "{:+.2f}")


def circuitkit_parity():
    """CircuitKIT parity gate (Tiny Aya Global in TransformerLens vs Hugging Face, fp16, Kaggle T4)"""
    ck = json.loads((RUNS / "circuitkit" / "parity" / "parity_kaggle_v7_patched.json").read_text())
    src = "runs/circuitkit/parity/parity_kaggle_v7_patched.json"
    macro("CkParityMaxKl", "{:.1e}".format(ck["max_kl"]).replace("e-0", "\\times10^{-") + "}", src)
    macro("CkParityTopOne", 100 * ck["top1_agree"], src, pct)
    macro("CkParityN", len(ck["per_prompt"]), src, "{:d}")


def detection_vs_action():
    """Detection vs action (Tiny Aya Global)"""
    s = json.loads((RUNS / "mech_global" / "summary.json").read_text())
    src = "runs/mech_global/summary.json"
    a = s["auroc_at_best"]
    macro("BestLayer", s["best_layer_by_mean_auroc"], src, "{:d}")
    macro("AurocMinNonJv", min(v for k, v in a.items() if k != "jv"), src)
    macro("AurocMax", max(a.values()), src)
    macro("AurocJv", a["jv"], src)
    macro("CosMean", s["mean_offdiag_cos_at_best"], src)


def cast_gate_level():
    """CAST sign defect (H6)"""
    gp = json.loads((RUNS / "mech_global" / "gate_proxy.json").read_text())
    h6 = pd.DataFrame(gp["h6"]).set_index("lang")
    src = "runs/mech_global/gate_proxy.json (h6)"
    x, y = h6.neg_proj_benign.values, (h6.auroc_proj - h6.auroc_cast).values
    macro("CastPearson", float(np.corrcoef(x, y)[0, 1]), src)
    macro("CastAurocSwProj", h6.loc["sw", "auroc_proj"], src)
    macro("CastAurocSwCast", h6.loc["sw", "auroc_cast"], src)
    macro("NegBenignZh", 100 * h6.loc["zh", "neg_proj_benign"], src, pct)
    macro("NegBenignSw", 100 * h6.loc["sw", "neg_proj_benign"], src, pct)
    summ = pd.DataFrame(gp["summary"]).set_index("arm")
    macro("CastFprMin", 100 * min(r["fpr"] for r in gp["per_lang"] if r["arm"] == "CAST-EN"), src, pct)
    macro("CastFprMax", 100 * summ.loc["CAST-EN", "max_fpr"], src, pct)
    macro("CondLayer", gp["layer"], src, "{:d}")


def gate_parity_matched_cost():
    """Gate parity at matched cost (post hoc), eps = 0.10"""
    d = RUNS / "mech_global"
    gp = json.loads((d / "gate_proxy.json").read_text())
    cal, test = arms.load_acts(d / "calib_acts.pt"), arms.load_acts(d / "test_acts.pt")
    res = gate_proxy.matched_comparison(cal, test, gp["layer"], 0.10)
    src = "gate_proxy.matched_comparison(eps=0.10) on runs/mech_global"
    st, eo = res["single threshold"], res["EORS"]
    fpr = lambda r: np.array([r[l][1] for l in LANGS])
    tpr = lambda r: np.array([r[l][0] for l in LANGS])
    macro("MatchMeanFprSingle", 100 * fpr(st).mean(), src, pct1)
    macro("MatchMeanFprEors", 100 * fpr(eo).mean(), src, pct1)
    macro("MatchMaxFprSingle", 100 * fpr(st).max(), src, pct)
    macro("MatchMaxFprEors", 100 * fpr(eo).max(), src, pct)
    macro("MatchGapSingle", 100 * (fpr(st).max() - fpr(st).min()), src, "{:.0f}")
    macro("MatchGapEors", 100 * (fpr(eo).max() - fpr(eo).min()), src, "{:.0f}")
    macro("MatchThFprSingle", 100 * st["th"][1], src, pct)
    macro("MatchThFprEors", 100 * eo["th"][1], src, pct)
    macro("MatchThTprSingle", tpr(st)[LANGS.index("th")], src)
    macro("MatchThTprEors", tpr(eo)[LANGS.index("th")], src)
    hrl = list(HRL)
    macro("MatchHrlFprSingleMax", 100 * max(st[l][1] for l in hrl), src, pct)
    macro("MatchHrlFprEorsMin", 100 * min(eo[l][1] for l in hrl), src, pct)
    macro("MatchHrlFprEorsMax", 100 * max(eo[l][1] for l in hrl), src, pct)
    macro("MatchMeanTprSingle", tpr(st).mean(), src)
    macro("MatchMeanTprEors", tpr(eo).mean(), src)


def gate_brittleness():
    """Gate brittleness to benign prefixes (post hoc; runs/spoof/{spoof,control}.json)"""
    spf = [
        r
        for r in json.loads((RUNS / "spoof" / "spoof.json").read_text())["summary"]
        + json.loads((RUNS / "spoof" / "control.json").read_text())["summary"]
        if r["kind"] == "benign"
    ]
    src = "runs/spoof/{spoof,control}.json (benign, EORS eps .10, ar/th/jv/bn/sw)"
    for cond, name in (("orig", "Orig"), ("native_prefix", "Native"), ("en_prefix", "En"), ("greet_prefix", "Greet")):
        v = [r["fire_rate"] for r in spf if r["cond"] == cond]
        macro(f"Brit{name}Mean", 100 * np.mean(v), src, pct)
        macro(f"Brit{name}Min", 100 * min(v), src, pct)
        macro(f"Brit{name}Max", 100 * max(v), src, pct)
    hm = [
        r
        for r in json.loads((RUNS / "spoof" / "spoof.json").read_text())["summary"]
        if r["kind"] == "harmful" and r["cond"] == "en_prefix"
    ]
    macro("BritLidFlipMax", 100 * max(r["lid_flip"] for r in hm), src, pct)


def cast_linear_regime():
    """Remark 1's linear regime on Earth's calibration activations: |s| * max_i |v_i| (CAST-EN direction, layer 14)"""
    _ce = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    _v = arms.unit(arms.diff_means(_ce, ["en"], 14))
    _reg = {
        k: [float(np.median(np.abs(arms.proj_scores(_ce, k, l, 14, _v))) * _v.abs().max().item()) for l in LANGS]
        for k in ("harmful", "benign")
    }
    src = "runs/mech_earth/calib_acts.pt (layer 14, English condition direction; median over prompts, range over languages)"
    macro("CastRegimeHarmMin", min(_reg["harmful"]), src, "{:.1f}")
    macro("CastRegimeHarmMax", max(_reg["harmful"]), src, "{:.1f}")
    macro("CastRegimeBenMin", min(_reg["benign"]), src)
    macro("CastRegimeBenMax", max(_reg["benign"]), src)


def circuitkit_appendix():
    """CircuitKIT tooling appendix (length-matched pairs, A100; runs/circuitkit/colab_a100_matched/)"""
    ckd = RUNS / "circuitkit" / "colab_a100_matched"
    fr = json.loads((ckd / "faithfulness.json").read_text())
    diag = lambda keep: [
        r["patching_raw"] for r in fr if r["src"] == r["tgt"] and abs((1 - r["sparsity"]) - keep) < 1e-6
    ]
    src = "runs/circuitkit/colab_a100_matched/faithfulness.json (diagonal, raw patching faithfulness)"
    macro("CkFaithTwentyMax", max(diag(0.20)), src)
    macro("CkFaithFortyMin", min(diag(0.40)), src)
    macro("CkFaithFortyMax", max(diag(0.40)), src)
    macro("CkFaithSixtyMin", min(diag(0.60)), src)
    macro("CkFaithSixtyMax", max(diag(0.60)), src)
    macro("CkFaithFull", min(diag(1.0)), src)
    peaks = [v["peak_gpu_gib"] for v in json.loads((ckd / "discover_summary.json").read_text()).values()]
    macro("CkPeakMin", min(peaks), str((ckd / "discover_summary.json").relative_to(ROOT)), "{:.1f}")
    macro("CkPeakMax", max(peaks), str((ckd / "discover_summary.json").relative_to(ROOT)), "{:.1f}")
    nodes = json.loads((ckd / "refusal_en_scores.json").read_text())["node_scores"]
    macro("CkNodes", len(nodes), "runs/circuitkit/colab_a100_matched/refusal_en_scores.json (scored nodes)", "{:d}")
