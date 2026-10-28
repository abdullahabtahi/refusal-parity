"""Measurement: judge validation, the keyword detectors, the baseline and the closed comparator (paper §3, §5.3)."""

import json

import numpy as np
import pandas as pd

from refusal_parity.analysis import analyze
from refusal_parity.common import HRL, LANGS, LRL, RUNS, read_jsonl
from refusal_parity.reporting import figures_diagnostics
from refusal_parity.reporting.paper_numbers.core import (
    closed_model_harm,
    macro,
    pct,
    pct1,
)


def judge_validation():
    """Judge validation vs human labels (PolyGuardPrompts)"""
    jv = pd.read_csv(RUNS / "pgp_val" / "judge_validation.csv")
    g = jv[jv.judge == "Gemini-3.5-Flash broad"]
    src = "runs/pgp_val/judge_validation.csv (Gemini-3.5-Flash broad)"
    macro("RefKappaMean", g[g.label == "refusal"].kappa.mean(), src)
    macro("RefKappaMin", g[g.label == "refusal"].kappa.min(), src)
    macro("RefAccMean", 100 * g[g.label == "refusal"].acc.mean(), src, pct)
    macro("HarmKappaMean", g[g.label == "harmful"].kappa.mean(), src)
    gb = pd.DataFrame(read_jsonl(RUNS / "pgp_val" / "gemini_broad_judged.jsonl"))
    split = gb[(gb.h_harm_agree.astype(float) == 2) & gb.h_harmful]
    unan = gb[(gb.h_harm_agree.astype(float) == 3) & gb.h_harmful]
    src = "runs/pgp_val/gemini_broad_judged.jsonl"
    macro("HarmMissSplit", 100 * (~split.gem_harmful.astype(bool)).mean(), src, pct)
    macro("HarmMissUnanimous", 100 * (~unan.gem_harmful.astype(bool)).mean(), src, pct)
    ur = gb[gb.h_refusal_agree.astype(float) == 3]
    macro("RefAccUnanimous", 100 * (ur.gem_refusal == ur.h_refusal).mean(), src, pct1)
    macro("NValidation", len(gb), src, "{:,}")
    # Local primary judge (Qwen3Guard-Gen-4B), same items; long responses clipped to the judge's 2,048-token window
    q = jv[jv.judge == "Qwen3Guard-4B"]
    src = "runs/pgp_val/judge_validation.csv (Qwen3Guard-4B)"
    macro("QRefKappaMean", q[q.label == "refusal"].kappa.mean(), src)
    macro("QHarmKappaMean", q[q.label == "harmful"].kappa.mean(), src)
    macro("QHarmKappaMin", q[q.label == "harmful"].kappa.min(), src)
    macro("QHarmKappaMax", q[q.label == "harmful"].kappa.max(), src)
    macro("QHarmMiss", 100 * q[q.label == "harmful"].miss.mean(), src, pct)
    macro(
        "GemHarmMiss",
        100 * g[g.label == "harmful"].miss.mean(),
        "runs/pgp_val/judge_validation.csv (Gemini broad)",
        pct,
    )
    qj = pd.DataFrame(read_jsonl(RUNS / "pgp_val" / "judged.jsonl"))
    hq = gb[["id", "lang", "h_harmful", "h_harm_agree"]].merge(qj[["id", "lang", "q3g4b_safety"]], on=["id", "lang"])
    src = "runs/pgp_val/judged.jsonl x human labels (gemini_broad_judged.jsonl)"
    macro("QUnparseable", int(hq.q3g4b_safety.isna().sum()), src, "{:d}")
    hq = hq[hq.q3g4b_safety.notna()]
    miss = lambda agree: (
        100 * (hq[(hq.h_harm_agree.astype(float) == agree) & hq.h_harmful].q3g4b_safety != "Unsafe").mean()
    )
    macro("QHarmMissSplit", miss(2), src, pct)
    macro("QHarmMissUnanimous", miss(3), src, pct)
    macro(
        "RefAccMeanTxt",
        100 * g[g.label == "refusal"].acc.mean(),
        "runs/pgp_val/judge_validation.csv (Gemini broad)",
        pct,
    )


def closed_model_gemini_judge():
    """Closed-model comparator (Gemini 3.5 Flash-Lite, judged by Gemini)"""
    # Same loader as analyze.py: unlabelled rows dropped (never counted as safe); the study's ten languages only.
    gem = analyze.load("GEM-lite", judge="gem")
    h = gem[(gem.kind == "harmful") & gem.lang.isin(LANGS)]
    src = "runs/GEM-lite/judged.jsonl via analyze.load(judge=gem)"
    hrr = h.groupby("lang").refuse.mean()
    harm = h.groupby("lang").unsafe.mean()
    macro("GemHRRMin", 100 * hrr.min(), src, pct)
    macro("GemHRRMax", 100 * hrr.max(), src, pct)
    macro("GemHarmMax", 100 * harm.max(), src, pct1)
    macro("GemHarmMean", 100 * harm.mean(), src, pct1)
    macro("GemGSrr", 100 * (1 - harm.mean()), src, pct1)  # same-family judge; cf. GemQSrr (Qwen3Guard)


def closed_model_primary_judge():
    """closed-model comparator under the primary judge (Qwen3Guard-4B, re-scored 2026-10-28, free)"""
    gq = analyze.load("GEM-lite")
    gq = gq[gq.lang.isin(LANGS)]
    s = analyze.summary(gq)[0].set_index("lang")
    src = "runs/GEM-lite/judged.jsonl via analyze.summary(judge=q3g4b)"
    macro("GemQSrr", 100 * s.SRR.mean(), src, pct1)
    macro("GemQWorstSrr", 100 * s.SRR.min(), src, pct1)
    macro("GemQHarmMax", 100 * (1 - s.SRR.min()), src, pct1)
    macro("GemQHrrMean", 100 * s.HRR.mean(), src, pct)


def baseline_b0():
    """Baseline B0 (Tiny Aya Global, default template), three-way outcome under both judges"""
    hrl_l, lrl_l = list(HRL), list(LRL)
    for judge, tag in (("q3g4b", "Q"), ("gem", "G")):
        t = figures_diagnostics.outcome_table("B0", judge)
        src = f"figures_diagnostics.outcome_table('B0', '{judge}') on runs/B0/judged.jsonl"
        m = lambda key, ls, t=t: 100 * np.mean([t[l][key] for l in ls])
        macro(f"BZero{tag}RefHrl", m("refused", hrl_l), src, pct)
        macro(f"BZero{tag}RefLrl", m("refused", lrl_l), src, pct)
        macro(f"BZero{tag}HarmHrl", m("harmful", hrl_l), src, pct1)
        macro(f"BZero{tag}HarmLrl", m("harmful", lrl_l), src, pct1)
        macro(f"BZero{tag}HarmMean", m("harmful", LANGS), src, pct1)
        macro(f"BZero{tag}HarmMax", 100 * max(v["harmful"] for v in t.values()), src, pct)
        macro(f"BZero{tag}BrrTh", 100 * t["th"]["BRR"], src, pct)
        macro(f"BZero{tag}BrrJv", 100 * t["jv"]["BRR"], src, pct)
        macro(f"BZero{tag}BrrMean", m("BRR", LANGS), src, pct1)
    macro(
        "BZeroGemHarmRatio",
        float(np.mean([figures_diagnostics.outcome_table("B0", "gem")[l]["harmful"] for l in LANGS]))
        / (closed_model_harm().mean() or float("nan")),
        "B0 mean harm (gem judge) / GEM-lite mean harm",
        "{:.0f}",
    )


def h1_keyword_detectors():
    """H1 (confirmatory): English keyword refusal detectors vs multilingual judges, B0"""
    h1 = json.loads((RUNS / "B0" / "h1.json").read_text())
    src = "runs/B0/h1.json (src/refusal_parity/analysis/h1.py)"
    p = h1["kw_regex|gem"]  # SafeTune's evaluation detector vs the human-validated refusal judge
    macro("HOneGap", p["gap"], src)
    macro("HOneGapLo", p["gap_ci"][0], src)
    macro("HOneGapHi", p["gap_ci"][1], src)
    macro("HOneKappaEnMin", min(r["kappa"]["en"] for r in h1.values()), src)
    macro("HOneKappaEnMax", max(r["kappa"]["en"] for r in h1.values()), src)
    macro("HOneKappaOtherMax", max(v for r in h1.values() for l, v in r["kappa"].items() if l != "en"), src)
    macro(
        "HOneRecallOtherMax",
        100 * max(v for r in h1.values() for l, v in r["keyword_recall_of_judge_refusals"].items() if l != "en"),
        src,
        pct,
    )


def gate_on_failures():
    """Gate coverage of B0's own failures (exploratory; failure_gate.py, matched-cost gates at eps = 0.10)"""
    fg = json.loads((RUNS / "B0" / "gate_on_failures.json").read_text())["results"]
    src = "runs/B0/gate_on_failures.json (src/refusal_parity/analysis/failure_gate.py)"
    e, rf = fg["harmful_either"]["EORS"]["pooled"], fg["refused"]["EORS"]["pooled"]
    macro("FailN", e["n"], src, "{:d}")
    macro("FailEors", 100 * e["rate"], src, pct)
    macro("FailEorsLo", 100 * e["lo"], src, pct)
    macro("FailEorsHi", 100 * e["hi"], src, pct)
    macro("FailSingle", 100 * fg["harmful_either"]["single"]["pooled"]["rate"], src, pct)
    macro("FailRefusedEors", 100 * rf["rate"], src, pct)
    th = fg["harmful_either"]["EORS"]["per_lang"]["th"]
    macro("FailThK", th["k"], src, "{:d}")
    macro("FailThN", th["n"], src, "{:d}")
