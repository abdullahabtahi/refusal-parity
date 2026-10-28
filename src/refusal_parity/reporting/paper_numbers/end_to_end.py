"""End-to-end steering arms and the pre-registered follow-ups (paper §5.1-5.2, Tables 1 and 3)."""

import json

from refusal_parity.analysis import analyze
from refusal_parity.common import ROOT, RUNS
from refusal_parity.reporting import figures_diagnostics
from refusal_parity.reporting.paper_numbers.core import (
    cell,
    macro,
    pair,
    pct,
    pct1,
    per_lang,
    spp1,
    triple,
)


def arms_end_to_end():
    """End-to-end repair arms (pre-registered 2026-10-27 18:50; runs/arms_report.json; Earth on A100)"""

    for j, J in (("q3g4b", "Q"), ("gem", "G")):
        src = f"runs/arms_report.json ({j})"
        e = pair("G-POOL_a100", "EORS_a100", j)
        triple(f"Eors{J}MaxBrr", e["boot"]["d_max_BRR"], src)
        triple(f"Eors{J}Hrr", e["boot"]["d_mean_HRR"], src)
        triple(f"Eors{J}Srr", e["boot"]["d_mean_SRR"], src)
        triple(f"EorsTen{J}MaxBrr", pair("G-POOL_a100", "EORS_e10_a100", j)["boot"]["d_max_BRR"], src)
        h = pair("G-EN_a100", "CAST-EN_a100", j)
        triple(f"Cast{J}LrlHrr", [h["lrl_hrr"][k] for k in ("delta", "lo", "hi")], src)
        triple(f"Cast{J}LrlUnsafe", [h["lrl_unsafe"][k] for k in ("delta", "lo", "hi")], src)
        triple(f"Pre{J}Srr", pair("B0_a100", "B1_a100", j)["boot"]["d_mean_SRR"], src)
        triple(f"Pre{J}Hrr", pair("B0_a100", "B1_a100", j)["boot"]["d_mean_HRR"], src)
        triple(f"Resta{J}Srr", pair("E0_a100", "RESTA_a100", j)["boot"]["d_mean_SRR"], src)
        triple(f"Resta{J}MaxBrr", pair("E0_a100", "RESTA_a100", j)["boot"]["d_max_BRR"], src)
        triple(f"Neg{J}Srr", pair("E0_a100", "NEG_a100", j)["boot"]["d_mean_SRR"], src)
        gains = {
            a: pair("E0_a100", a, j)["boot"]["d_mean_SRR"][0]
            for a in ("G-POOL_a100", "G-EN_a100", "CAST-EN_a100", "G-CAL_a100", "EORS_a100", "EORS_e10_a100")
        }
        macro(f"Arms{J}SrrGainMin", min(gains.values()), src, spp1)
        macro(f"Arms{J}SrrGainMax", max(gains.values()), src, spp1)
        worst = {
            a: pair("E0_a100", a, j)["boot"]["d_max_BRR"][0]
            for a in ("G-POOL_a100", "G-EN_a100", "CAST-EN_a100", "G-CAL_a100")
        }
        macro(f"Arms{J}MaxBrrMin", min(worst.values()), src, spp1)
        macro(f"Arms{J}MaxBrrMax", max(worst.values()), src, spp1)
        macro(f"EorsVsE{J}MaxBrr", pair("E0_a100", "EORS_a100", j)["boot"]["d_max_BRR"][0], src, spp1)
        hp = h["power"]["harmful_refuse"]
        if j == "q3g4b":
            macro("CastPowerPsi", 100 * hp["psi"], src, pct1)
            macro(
                "CastPowerMde",
                "no per-language effect is detectable with 80\\% power"
                if hp["mde"] != hp["mde"]
                else f"per-language effects below about {100 * hp['mde']:.1f} points are undetectable",
                src,
            )
            macro(
                "CastNSig",
                sum(r["p_holm"] < 0.05 for r in h["per_lang"] if r["kind"] == "harmful" and r["metric"] == "refuse"),
                src,
                "{:d}",
            )

    # absolute worst-language BRR and where the EORS cost falls (Qwen3Guard)
    tab = {a: analyze.summary(analyze.load(a))[0].set_index("lang") for a in ("G-POOL_a100", "EORS_a100", "E0_a100")}
    src = "analyze.summary(*_a100, judge=q3g4b)"
    macro("GpoolWorstBrr", 100 * tab["G-POOL_a100"].BRR.max(), src, pct)
    macro("GpoolWorstBrrLang", figures_diagnostics.NAMES[tab["G-POOL_a100"].BRR.idxmax()], src)
    macro("EorsWorstBrr", 100 * tab["EORS_a100"].BRR.max(), src, pct)
    cost = tab["G-POOL_a100"].HRR - tab["EORS_a100"].HRR
    macro("EorsCostLang", figures_diagnostics.NAMES[cost.idxmax()], src)
    macro("EorsCostLangPts", 100 * cost.max(), src, "{:.0f}")
    macro("EZeroAhundredSrr", 100 * tab["E0_a100"].SRR.mean(), src, pct1)
    macro("EorsSrrAbs", 100 * tab["EORS_a100"].SRR.mean(), src, pct1)
    macro(
        "CastRho",
        json.loads((ROOT / "paper" / "figs" / "fig3_cast_end_to_end.json").read_text())["provenance"]["rho"],
        "paper/figs/fig3_cast_end_to_end.json",
        "{:.2f}",
    )


def arms_level_vs_spread():
    """level vs spread (review 2026-10-28): a stricter rule lowers every language; the ratio shows whether it is fairer"""
    src = "analyze.summary(*_a100, judge=q3g4b)"
    tab = {a: per_lang(a) for a in ("G-POOL_a100", "EORS_a100", "EORS_e10_a100", "G-CAL_a100", "E0_a100")}
    for a, n in (
        ("G-POOL_a100", "Gpool"),
        ("EORS_a100", "Eors"),
        ("EORS_e10_a100", "EorsTen"),
        ("G-CAL_a100", "Gcal"),
        ("E0_a100", "EZero"),
    ):
        macro(f"{n}MeanBrr", 100 * tab[a].BRR.mean(), src, pct1)
        macro(f"{n}BrrRatio", tab[a].BRR.max() / tab[a].BRR.mean(), src, "{:.2f}")
    # the originally pre-registered H5 (2026-10-26): every language's BRR within E0 + eps, and a smaller SRR gap
    excess = tab["EORS_a100"].BRR - tab["E0_a100"].BRR
    macro("EorsExcessMax", 100 * excess.max(), src, "{:.1f}")
    macro("EorsExcessMaxLang", figures_diagnostics.NAMES[excess.idxmax()], src)
    macro("EorsExcessNAbove", int((excess > 0.05).sum()), src, "{:d}")
    triple("EorsQGapSrr", pair("G-POOL_a100", "EORS_a100")["boot"]["d_gap_SRR"], "runs/arms_report.json (q3g4b)")
    triple("EorsQMeanBrr", pair("G-POOL_a100", "EORS_a100")["boot"]["d_mean_BRR"], "runs/arms_report.json (q3g4b)")
    h = pair("G-EN_a100", "CAST-EN_a100")
    triple("CastQMaxBrr", h["boot"]["d_max_BRR"], "runs/arms_report.json (q3g4b)")
    triple("CastQMeanBrr", h["boot"]["d_mean_BRR"], "runs/arms_report.json (q3g4b)")
    macro(
        "CastNSigBrr",
        sum(r["p_holm"] < 0.05 for r in h["per_lang"] if r["kind"] == "benign" and r["metric"] == "refuse"),
        "runs/arms_report.json",
        "{:d}",
    )
    macro(
        "CastNNeg",
        sum(r["delta"] <= 0 for r in h["per_lang"] if r["kind"] == "harmful" and r["metric"] == "refuse"),
        "runs/arms_report.json",
        "{:d}",
    )


def table_arms():
    """Table 1: every repair arm against its paired reference (generated; never typed)"""
    ROWS = [
        ("E0_a100", "G-POOL_a100", "G-POOL", "single pooled threshold"),
        ("E0_a100", "G-EN_a100", "G-EN", "English vector, projection gate"),
        ("E0_a100", "CAST-EN_a100", "CAST-EN", "English vector, CAST gate"),
        ("E0_a100", "G-CAL_a100", "G-CAL", "per-language max-F1"),
        ("E0_a100", "EORS_e10_a100", r"EORS $\varepsilon{=}.10$", "per-language conformal"),
        ("E0_a100", "EORS_a100", r"EORS $\varepsilon{=}.05$", "per-language conformal"),
        ("E0_a100", "RESTA_a100", "ReSta", "weight arithmetic"),
        ("E0_a100", "NEG_a100", "NEG", "negative control"),
        ("B0_a100", "B1_a100", "B1 (no preamble)", r"vs.\ B0, Tiny Aya Global"),
    ]
    for judge, fname in (("q3g4b", "table_arms.tex"), ("gem", "table_arms_gem.tex")):
        tex = [
            r"\begin{tabular}{@{}llccc@{}}",
            r"\toprule",
            r"Arm & Gate / change & $\Delta$ mean SRR & $\Delta$ worst-language BRR & $\Delta$ mean HRR \\",
            r"\midrule",
        ]
        for a, b, name, what in ROWS:
            bo = pair(a, b, judge)["boot"]
            tex.append(
                f"{name} & {what} & {cell(bo['d_mean_SRR'])} & {cell(bo['d_max_BRR'])} & {cell(bo['d_mean_HRR'])} \\\\"
            )
            if name == "NEG":
                tex.append(r"\midrule")
        tex += [r"\bottomrule", r"\end{tabular}"]
        (ROOT / "paper" / fname).write_text(
            f"% AUTO-GENERATED by src/refusal_parity/reporting/paper_numbers/ from runs/arms_report.json ({judge})\n"
            + "\n".join(tex)
            + "\n"
        )


def eors_vs_gcal():
    """EORS vs the closest prior method (per-language max-F1, after Aziz et al.), paired; post hoc (not in the 18:50 plan)"""
    for j, J in (("q3g4b", "Q"), ("gem", "G")):
        _, bo = analyze.compare(analyze.load("G-CAL_a100", judge=j), analyze.load("EORS_a100", judge=j))
        src = f"analyze.compare(G-CAL_a100, EORS_a100, judge={j}) post hoc"
        triple(f"EorsCal{J}MaxBrr", bo["d_max_BRR"], src)
        triple(f"EorsCal{J}Srr", bo["d_mean_SRR"], src)


def followups_alpha_half_and_greeting():
    """Follow-ups pre-registered 2026-10-27 23:45 (Qwen3Guard only; runs/arms_report.json)"""
    src = "runs/arms_report.json (q3g4b; alpha .5 follow-up)"
    e = pair("G-POOL_a05", "EORS_a05")
    triple("AhEorsMaxBrr", e["boot"]["d_max_BRR"], src)
    triple("AhEorsSrr", e["boot"]["d_mean_SRR"], src)
    triple("AhEorsHrr", e["boot"]["d_mean_HRR"], src)
    h = pair("G-EN_a05", "CAST-EN_a05")
    triple("AhCastLrlHrr", [h["lrl_hrr"][k] for k in ("delta", "lo", "hi")], src)
    triple("AhCastLrlUnsafe", [h["lrl_unsafe"][k] for k in ("delta", "lo", "hi")], src)
    gains = [pair("E0_a100", a)["boot"]["d_mean_SRR"][0] for a in ("G-POOL_a05", "EORS_a05", "G-EN_a05", "CAST-EN_a05")]
    macro("AhSrrGainMin", min(gains), src, spp1)
    macro("AhSrrGainMax", max(gains), src, spp1)
    tab05 = {a: analyze.summary(analyze.load(a))[0].set_index("lang") for a in ("G-POOL_a05", "EORS_a05")}
    macro("AhGpoolWorstBrr", 100 * tab05["G-POOL_a05"].BRR.max(), src, pct)
    macro("AhGpoolWorstBrrLang", figures_diagnostics.NAMES[tab05["G-POOL_a05"].BRR.idxmax()], src)
    macro("AhEorsWorstBrr", 100 * tab05["EORS_a05"].BRR.max(), src, pct)

    gd = json.loads((RUNS / "arms_report.json").read_text())["greeting_did"]
    src = "runs/arms_report.json greeting_did (q3g4b; EORS eps .10 vs E0, prefix 'Good morning. ')"
    b = gd["benign_refuse"]
    triple("GreetDid", [b["pooled"][k] for k in ("delta", "lo", "hi")], src)
    macro("GreetDidLangMin", min(v["delta"] for v in b["per_lang"].values()), src, spp1)
    macro("GreetDidLangMax", max(v["delta"] for v in b["per_lang"].values()), src, spp1)
    macro("GreetDidNPos", sum(v["lo"] > 0 for v in b["per_lang"].values()), src, "{:d}")
    for k, name in (("e0", "EZero"), ("eors", "Eors"), ("e0_greet", "EZeroGreet"), ("eors_greet", "EorsGreet")):
        macro(f"GreetBrr{name}", 100 * b["components"][k], src, pct)
    triple("GreetDidUnsafe", [gd["harmful_unsafe"]["pooled"][k] for k in ("delta", "lo", "hi")], src)
    triple("GreetDidHrr", [gd["harmful_refuse"]["pooled"][k] for k in ("delta", "lo", "hi")], src)


def table_arms_alpha_half():
    """Appendix table: the alpha .5 replication, same layout as Table 1 (Qwen3Guard only)"""
    ROWS05 = [
        ("E0_a100", "G-POOL_a05", "G-POOL", "vs.\\ E0"),
        ("E0_a100", "EORS_a05", r"EORS $\varepsilon{=}.05$", "vs.\\ E0"),
        ("E0_a100", "G-EN_a05", "G-EN", "vs.\\ E0"),
        ("E0_a100", "CAST-EN_a05", "CAST-EN", "vs.\\ E0"),
        ("G-POOL_a05", "EORS_a05", r"EORS $\varepsilon{=}.05$", "vs.\\ G-POOL"),
        ("G-EN_a05", "CAST-EN_a05", "CAST-EN", "vs.\\ G-EN"),
    ]
    tex = [
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Arm & Reference & $\Delta$ mean SRR & $\Delta$ worst-language BRR & $\Delta$ mean HRR \\",
        r"\midrule",
    ]
    for i, (a, b_, name, what) in enumerate(ROWS05):
        bo = pair(a, b_)["boot"]
        tex.append(
            f"{name} & {what} & {cell(bo['d_mean_SRR'])} & {cell(bo['d_max_BRR'])} & {cell(bo['d_mean_HRR'])} \\\\"
        )
        if i == 3:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "table_arms_a05.tex").write_text(
        "% AUTO-GENERATED by src/refusal_parity/reporting/paper_numbers/ from runs/arms_report.json (q3g4b)\n"
        + "\n".join(tex)
        + "\n"
    )


def steering_strength():
    """Steering strength (alpha rule fixed before the sweep; runs/alpha_sweep.json, S-POOL on Earth, calibration)"""
    al = json.loads((RUNS / "alpha_sweep.json").read_text())
    src = "runs/alpha_sweep.json (Gemini labels; ungated S-POOL, calibration subset)"
    macro("AlphaPick", float(al["pick"]), src, "{:.1f}")
    macro("AlphaOneHrr", 100 * al["per_alpha"]["1.0"]["HRR"], src, pct)
    macro("AlphaOneBrr", 100 * al["per_alpha"]["1.0"]["BRR"], src, pct)
    macro("AlphaTwoOnLang", 100 * al["per_alpha"]["2.0"]["on_lang"], src, pct)
