"""Matched-budget controls pre-registered 2026-10-28 (paper §5.1, §5.4, Table 2)."""

import numpy as np
import pandas as pd

from refusal_parity.analysis import analyze
from refusal_parity.common import HRL, LANGS, LRL, ROOT
from refusal_parity.reporting import figures_diagnostics
from refusal_parity.reporting.paper_numbers.core import cell, controls, macro, pair, pct, pct1, spp1, triple


def controls_primary():
    """Controls pre-registered 2026-10-28 06:40 (runs/controls_report.json; Qwen3Guard only)"""
    ctl = controls()
    src = "runs/controls_report.json (q3g4b; EORS-S vs CPOOL-S, matched budget)"
    ptxt = lambda p: "$p<.001$" if p < 0.001 else f"$p={p:.3f}$"
    for e, E in (("0.05", "Five"), ("0.1", "Ten")):
        x = ctl["end_to_end"][e]
        b = x["boot"]
        for k, n in (
            ("d_max_BRR", "MaxBrr"),
            ("d_mean_BRR", "MeanBrr"),
            ("d_range_BRR", "Range"),
            ("d_mean_SRR", "Srr"),
            ("d_worst_SRR", "WorstSrr"),
            ("d_mean_HRR", "Hrr"),
        ):
            triple(f"Ctl{E}{n}", b[k], src)
        triple(f"Ctl{E}LrlHrr", [x["lrl_hrr"][k] for k in ("delta", "lo", "hi")], src)
        triple(f"Ctl{E}LrlUnsafe", [x["lrl_unsafe"][k] for k in ("delta", "lo", "hi")], src)
        macro(f"Ctl{E}PoneHolm", ptxt(x["P1_max_BRR_p_holm"]), src)
        macro(f"Ctl{E}PtwoHolm", ptxt(x["P2_lrl_HRR_p_holm"]), src)
        for arm, n in (("CPOOL-S", "Pool"), ("EORS-S", "Eors")):
            brr = pd.Series(x["per_arm"][arm]["BRR"])
            macro(f"Ctl{E}{n}WorstBrr", 100 * brr.max(), src, pct1)
            macro(f"Ctl{E}{n}WorstBrrLang", figures_diagnostics.NAMES[brr.idxmax()], src)
            macro(f"Ctl{E}{n}MeanBrrAbs", 100 * brr.mean(), src, pct1)
            macro(f"Ctl{E}{n}Ratio", brr.max() / brr.mean(), src, "{:.2f}")
            macro(f"Ctl{E}{n}ExcessMax", 100 * max(x["excess_brr"][arm].values()), src, "{:.1f}")
        dh = pd.Series(x["per_arm"]["EORS-S"]["HRR"]) - pd.Series(x["per_arm"]["CPOOL-S"]["HRR"])
        macro(f"Ctl{E}HrrLossLang", figures_diagnostics.NAMES[dh.idxmin()], src)
        macro(f"Ctl{E}HrrLossPts", dh.min(), src, spp1)  # spp1 scales to points
        macro(f"Ctl{E}HrrGainLang", figures_diagnostics.NAMES[dh.idxmax()], src)
        macro(f"Ctl{E}HrrGainPts", dh.max(), src, spp1)
        v = ctl["gate"]["P3"][e]
        macro(f"Ctl{E}ValidAbove", v["n_above"], src, "{:d}")
        macro(f"Ctl{E}ValidMeanFpr", 100 * np.mean(list(v["fpr"].values())), src, pct1)
        macro(f"Ctl{E}ValidMaxFpr", 100 * max(v["fpr"].values()), src, pct1)
        macro(f"Ctl{E}Band", 100 * v["upper"]["en"], src, pct1)
    macro("CtlNCal", ctl["gate"]["P3"]["0.05"]["n_cal"], src, "{:d}")


def controls_style_check():
    """P4: harm vs dataset style, and out-of-distribution firing (gate level)"""
    ctl = controls()
    p4 = ctl["gate"]["P4"]
    src = "runs/controls_report.json gate.P4 (XSTest-unsafe vs XSTest-safe test prompts, EORS-S direction)"
    macro("CtlStyleAuc", p4["pooled_auc"], src)
    auc = pd.Series(p4["auc_xs_unsafe_vs_safe"])
    macro("CtlStyleAucMin", auc.min(), src)
    macro("CtlStyleAucMinLang", figures_diagnostics.NAMES[auc.idxmin()], src)
    macro("CtlMjAuc", np.mean(list(p4["auc_multijail_vs_safe"].values())), src)
    macro("CtlXsN", p4["n_xs_unsafe"]["en"], src, "{:d}")
    macro("CtlOodEn", 100 * p4["ood_fire_rate_eps05"]["en"], src, pct)
    macro("CtlOodAr", 100 * p4["ood_fire_rate_eps05"]["ar"], src, pct)
    macro("CtlMjFireEn", 100 * p4["multijail_fire_rate_eps05"]["en"], src, pct)
    macro("CtlMjFireAr", 100 * p4["multijail_fire_rate_eps05"]["ar"], src, pct)
    _other = [v for k, v in p4["ood_fire_rate_eps05"].items() if k not in LANGS]
    macro("CtlOodOtherMin", 100 * min(_other), src, pct)
    macro("CtlOodOtherMax", 100 * max(_other), src, pct)


def controls_greeting():
    """Greeting difference-in-differences per threshold rule (pre-registered secondary)"""
    ctl = controls()
    src = "runs/controls_report.json greeting (q3g4b; eps .10; prefix 'Good morning. ')"
    for rule, n in (("EORS-S", "Eors"), ("CPOOL-S", "Pool")):
        g = ctl["greeting"][rule]
        triple(f"CtlGreetDid{n}", [g["pooled"][k] for k in ("delta", "lo", "hi")], src)
        macro(f"CtlGreetBrr{n}", 100 * g["components"]["eors"], src, pct)
        macro(f"CtlGreetBrr{n}Greet", 100 * g["components"]["eors_greet"], src, pct)
        worst = analyze.summary(analyze.load(f"{rule}_e10_greet_a100"))[0].BRR.max()
        macro(f"CtlGreetWorst{n}", 100 * worst, f"analyze.summary({rule}_e10_greet_a100)", pct)


def controls_frontier():
    """Gate-level frontier across budgets (exploratory)"""
    ctl = controls()
    src = "runs/controls_report.json frontier (gate level, test activations; exploratory)"
    fr = {(f["eps"], f["rule"]): f for f in ctl["frontier"]}
    macro("CtlFrontMaxPool", 100 * fr[0.04, "pooled"]["max_fpr"], src, pct1)
    macro("CtlFrontMaxMond", 100 * fr[0.04, "Mondrian"]["max_fpr"], src, pct1)
    tier = lambda r, ls: 100 * np.mean([r["tpr"][l] for l in ls])
    for rule, n in (("pooled", "Pool"), ("Mondrian", "Mond")):
        macro(f"CtlFrontMeanFprTen{n}", 100 * fr[0.1, rule]["mean_fpr"], src, pct1)
        macro(f"CtlFrontLrlTprTen{n}", tier(fr[0.1, rule], LRL), src, pct)
        macro(f"CtlFrontHrlTprTen{n}", tier(fr[0.1, rule], HRL), src, pct)
    macro("CtlFrontTwentyMond", 100 * fr[0.2, "Mondrian"]["mean_fpr"], src, pct)
    _exc = sorted((f["eps"], f["n_above"]) for f in ctl["frontier"] if f["rule"] == "Mondrian")
    _from = min(e for e, n in _exc if n >= 2)  # smallest budget from which two or more languages exceed the band
    macro("CtlFrontExceedEps", f"{_from:.2f}".lstrip("0"), src)
    macro("CtlFrontExceedMin", min(n for e, n in _exc if e >= _from), src, "{:d}")
    macro("CtlFrontExceedMax", max(n for e, n in _exc if e >= _from), src, "{:d}")


def controls_calibration_shift():
    """Hardware vs batching vs prompt sampling (post hoc)"""
    ctl = controls()
    cs_ = ctl["calibration_shift"]
    src = "runs/controls_report.json calibration_shift (post hoc)"
    macro("CtlHwSd", cs_["hardware"]["sd"], src)
    macro("CtlHwMax", cs_["hardware"]["max_abs"], src)
    macro("CtlBatchSd", cs_["batching"]["sd"], src)
    macro("CtlBatchMax", cs_["batching"]["max_abs"], src)
    macro("CtlMedianGap", np.mean(list(cs_["benign_median_gap_test_minus_cal"].values())), src)
    macro("CtlScoreSd", cs_["benign_score_sd"], src, "{:.1f}")


def controls_review_round2():
    """review round 2 (2026-10-28): tier shape of the cost, LID routing, split resampling, judge gap, Holm by language"""
    ctl = controls()
    fr = {(f["eps"], f["rule"]): f for f in ctl["frontier"]}
    src = "runs/controls_report.json (q3g4b)"
    for e, E in (("0.05", "Five"), ("0.1", "Ten")):
        x = ctl["end_to_end"][e]
        for arm, n in (("CPOOL-S", "Pool"), ("EORS-S", "Eors")):
            macro(f"Ctl{E}{n}HrrRange", 100 * x["hrr_shape"][arm]["range"], src, "{:.1f}")
            macro(f"Ctl{E}{n}TierGap", 100 * x["hrr_shape"][arm]["tier_gap"], src, "{:.1f}")
        dh = pd.Series(x["per_arm"]["EORS-S"]["HRR"]) - pd.Series(x["per_arm"]["CPOOL-S"]["HRR"])
        for lang in ("th", "bn", "sw"):
            macro(f"Ctl{E}Hrr{lang.capitalize()}", dh[lang], src, spp1)
        holm = [
            r_ for r_ in x["per_lang"] if r_["kind"] == "benign" and r_["metric"] == "refuse" and r_["p_holm"] < 0.05
        ]
        macro(f"Ctl{E}HolmBenignN", len(holm), src, "{:d}")
        macro(f"Ctl{E}HolmBenignLangs", ", ".join(figures_diagnostics.NAMES[r_["lang"]] for r_ in holm), src)
        v = ctl["gate"]["P3"][e]
        macro(f"Ctl{E}ValidAboveLid", v["n_above_lid"], src, "{:d}")
        macro(f"Ctl{E}InsampleMaxFpr", 100 * max(v["insample_fpr"].values()), src, pct1)
    sr = ctl["split_resampling"]
    src = "runs/controls_report.json split_resampling (exploratory; eps .05)"
    macro("CtlSplitN", sr["n_splits"], src, "{:d}")
    for k, n in (("d_max_fpr", "DMax"), ("d_lrl_tpr", "DLrl")):
        macro(f"CtlSplit{n}", sr[k]["median"], src, spp1)
        macro(f"CtlSplit{n}Lo", sr[k]["q025"], src, spp1)
        macro(f"CtlSplit{n}Hi", sr[k]["q975"], src, spp1)
        macro(f"CtlSplit{n}Share", 100 * sr[k]["share_negative"], src, pct)
    macro("CtlSplitAboveMax", sr["n_above"]["max"], src, "{:d}")
    macro("CtlSplitAboveZero", 100 * sr["n_above"]["share_zero"], src, pct)
    _d = sr["draws"]
    _share = lambda f: 100 * float(np.mean([f(d) for d in _d]))
    _gap = lambda a: a["hrl_tpr"] - a["lrl_tpr"]
    macro("CtlSplitTierGapShare", _share(lambda d: _gap(d["EORS-S"]) > _gap(d["CPOOL-S"])), src, pct)
    macro("CtlSplitMeanLowerShare", _share(lambda d: d["EORS-S"]["mean_fpr"] < d["CPOOL-S"]["mean_fpr"]), src, pct)
    macro(
        "CtlSplitDMean", float(np.median([d["EORS-S"]["mean_fpr"] - d["CPOOL-S"]["mean_fpr"] for d in _d])), src, spp1
    )
    jg = ctl["judge_gap"]
    src = "runs/controls_report.json judge_gap (post hoc; harmful test prompts, 10 languages)"
    macro("CtlJudgeGapGem", 100 * jg["gap_gem"], src, "{:.1f}")
    macro("CtlJudgeGapEZero", 100 * jg["gap_e0"], src, "{:.1f}")
    triple("CtlJudgeDid", [jg["did"], jg["lo"], jg["hi"]], src)
    fr_m = [f for f in ctl["frontier"] if f["rule"] == "Mondrian"]
    p04 = fr[0.04, "pooled"]
    near = min(fr_m, key=lambda f: abs(f["mean_fpr"] - p04["mean_fpr"]))
    src = "runs/controls_report.json frontier (matched mean false-firing; exploratory)"
    macro("CtlFrontMatchMeanPool", 100 * p04["mean_fpr"], src, pct1)
    macro("CtlFrontMatchMeanMond", 100 * near["mean_fpr"], src, pct1)
    macro("CtlFrontMatchMaxPool", 100 * p04["max_fpr"], src, pct1)
    macro("CtlFrontMatchMaxMond", 100 * near["max_fpr"], src, pct1)
    gains = [
        pair("E0_a100", f"{arm}_e{e}_a100")["boot"]["d_mean_SRR"][0]
        for e in ("05", "10")
        for arm in ("CPOOL-S", "EORS-S")
    ]
    macro("CtlSrrGainMin", min(gains), "runs/arms_report.json (vs E0, controls)", spp1)
    macro("CtlSrrGainMax", max(gains), "runs/arms_report.json (vs E0, controls)", spp1)


def table_controls():
    """Table: the matched-budget controls (generated; never typed)"""
    ROWSC = [
        ("E0_a100", "CPOOL-S_e05_a100", r"CPOOL-S $\varepsilon{=}.05$", r"vs.\ E0"),
        ("E0_a100", "EORS-S_e05_a100", r"EORS-S $\varepsilon{=}.05$", r"vs.\ E0"),
        ("E0_a100", "CPOOL-S_e10_a100", r"CPOOL-S $\varepsilon{=}.10$", r"vs.\ E0"),
        ("E0_a100", "EORS-S_e10_a100", r"EORS-S $\varepsilon{=}.10$", r"vs.\ E0"),
        ("CPOOL-S_e05_a100", "EORS-S_e05_a100", r"EORS-S $\varepsilon{=}.05$", r"vs.\ CPOOL-S"),
        ("CPOOL-S_e10_a100", "EORS-S_e10_a100", r"EORS-S $\varepsilon{=}.10$", r"vs.\ CPOOL-S"),
    ]
    tex = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Arm & Reference & $\Delta$ mean SRR & $\Delta$ worst-language BRR & $\Delta$ mean BRR & $\Delta$ HRR, low-resource tier \\",
        r"\midrule",
    ]
    for i, (a, b_, name, what) in enumerate(ROWSC):
        r_ = pair(a, b_)
        bo, lr = r_["boot"], [r_["lrl_hrr"][k] for k in ("delta", "lo", "hi")]
        tex.append(
            f"{name} & {what} & {cell(bo['d_mean_SRR'])} & {cell(bo['d_max_BRR'])} & {cell(bo['d_mean_BRR'])} & {cell(lr)} \\\\"
        )
        if i == 3:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "table_controls.tex").write_text(
        "% AUTO-GENERATED by src/refusal_parity/reporting/paper_numbers/ from runs/arms_report.json (q3g4b)\n"
        + "\n".join(tex)
        + "\n"
    )
