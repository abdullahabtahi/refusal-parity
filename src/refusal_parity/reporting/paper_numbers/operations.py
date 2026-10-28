"""Cost and compute: ROI against the closed model, hardware agreement, compute per arm."""

import json

import numpy as np

from refusal_parity.analysis import analyze
from refusal_parity.common import RUNS, read_jsonl
from refusal_parity.reporting.paper_numbers.core import macro, pct, pct1, triple


def roi():
    """ROI: closed API vs self-hosted Tiny Aya (measured usage; list price is a documented input)"""
    # Gemini 3.5 Flash-Lite, Gemini API standard paid tier, USD per 1M tokens (output includes thinking):
    # https://ai.google.dev/gemini-api/docs/pricing, "Last updated 2026-10-26 UTC", accessed 2026-10-27.
    PRICE_IN, PRICE_OUT = 0.30, 2.50
    gl = read_jsonl(RUNS / "GEM-lite" / "responses.jsonl")  # generation usage (judged.jsonl tok_* are overwritten)
    src = "runs/GEM-lite/responses.jsonl usage_metadata x list price"
    tin = sum(int(r.get("tok_in") or 0) for r in gl)
    tout = sum(int(r.get("tok_out") or 0) + int(r.get("tok_think") or 0) for r in gl)
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
    macro("RoiGemN", len(gl), src, "{:,}")
    macro("RoiGemCost", cost, src, "\\${:.2f}")
    macro("RoiGemPerK", 1000 * cost / len(gl), src, "\\${:.2f}")
    macro("RoiGemMeanOut", tout / len(gl), src, "{:.0f}")
    cap_in = tin / len(gl)  # same prompts, output capped at Tiny Aya's 96 new tokens: an upper bound at equal length
    macro("RoiGemPerKCap", 1000 * (cap_in * PRICE_IN + 96 * PRICE_OUT) / 1e6, src + " (96-token cap)", "\\${:.2f}")
    # Tiny Aya on one M1 Pro (bf16, MPS, batch 16, <=96 new tokens): wall time from arm_config.json (written after model
    # load) to responses.jsonl (written at the end), for both full arms.
    timing = json.loads((RUNS / "roi_timing.json").read_text())["runs"]  # frozen from file mtimes right after E0
    rates = [r["n"] / r["minutes"] for r in timing.values()]
    src = "runs/roi_timing.json (B0, E0 generation wall time)"
    macro("RoiTinyPerMin", np.mean(rates), src, "{:.0f}")
    macro("RoiTinyHoursPerM", 1e6 / np.mean(rates) / 60, src, "{:,.0f}")
    glt = analyze.summary(analyze.load("GEM-lite", judge="gem"))[0]
    macro("GemBrrMean", 100 * glt.BRR.mean(), "analyze.summary(GEM-lite, judge=gem), ten languages", pct1)


def hardware_check():
    """Hardware check (A100 vs M1 MPS, E0)"""
    hw = json.loads((RUNS / "E0_a100" / "hardware_check.json").read_text())
    src = "runs/E0_a100/hardware_check.json"
    triple("HwSrr", hw["d_mean_SRR"], src)
    macro("HwRefKappa", hw["refusal"]["kappa"], src)
    macro("HwHarmKappa", hw["harm"]["kappa"], src)
    macro("HwTextSame", 100 * hw["text_identical"], src, pct)


def compute_per_arm():
    """compute per arm (A100; recorded by scripts/colab_gen.sh for the six control arms)"""
    _ctl_runs = [f"{a}_{tag}" for a in ("EORS-S", "CPOOL-S") for tag in ("e05_a100", "e10_a100", "e10_greet_a100")]
    _gen = [json.loads((RUNS / r / "colab_timing.json").read_text())["seconds"] for r in _ctl_runs]
    _score = [json.loads((RUNS / r / "colab" / "colab_timing.json").read_text())["seconds"] for r in _ctl_runs]
    src = "runs/{EORS-S,CPOOL-S}_*_a100/{,colab/}colab_timing.json (median over six arms)"
    macro("ComputeGenMin", np.median(_gen) / 60, src, "{:.0f}")
    macro("ComputeScoreMin", np.median(_score) / 60, src, "{:.0f}")
