"""Core figures for the rewritten paper (research/10_paper_playbook.md §4), in the Cohere style (cohere_style.py).

    python -m refusal_parity.reporting.figures_results [teaser|eors|cast|instruments|brittle|all]

Each figure: one takeaway, an action title computed from the data, direct labels, intervals named in the caption,
PDF + PNG + JSON (caption, alt text, provenance) via figures_diagnostics.save. Primary judge: Qwen3Guard-4B; Gemini values are
in the provenance JSON and the macros.
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr

from refusal_parity.analysis import analyze
from refusal_parity.common import HRL, LANGS, LRL, RUNS
from refusal_parity.reporting import cohere_style as cs
from refusal_parity.reporting import (
    figures_diagnostics,  # helpers: ORDER, NAMES, save, paired_deltas (its rcParams are replaced by cohere_style.apply)
)
from refusal_parity.steering import arms

cs.apply()
plt = cs.plt
ARM_NAMES = {
    "E0_a100": "no repair (E0)",
    "G-POOL_a100": "G-POOL",
    "G-EN_a100": "G-EN",
    "CAST-EN_a100": "CAST-EN",
    "G-CAL_a100": "G-CAL",
    "CPOOL-S_e05_a100": "CPOOL-S",
    "EORS-S_e05_a100": "EORS-S",
    "RESTA_a100": "ReSta",
    "NEG_a100": "NEG (control)",
}


def per_lang(run, judge="q3g4b"):
    return analyze.summary(analyze.load(run, judge=judge))[0].set_index("lang")


def fig_teaser():
    """(a) safety vs worst-language over-refusal for every arm; (b) the end-to-end greeting test, both rules."""
    pts = {}
    for run in ARM_NAMES:
        t = per_lang(run)
        pts[run] = (100 * t.BRR.max(), 100 * t.SRR.mean(), t.BRR.idxmax())
    fig = plt.figure(figsize=(cs.W, 2.75), layout="constrained")
    ax = fig.subplot_mosaic([["a", "b"]], width_ratios=[1.45, 1])
    a = ax["a"]
    style = {
        "E0_a100": (cs.INK, "o", "none"),
        "CPOOL-S_e05_a100": (cs.BLUE, "o", "white"),
        "EORS-S_e05_a100": (cs.BLUE, "o", cs.BLUE),
        "NEG_a100": (cs.MUSH, "s", cs.MUSH),
        "RESTA_a100": (cs.CORAL, "s", cs.CORAL),
    }
    nudge = {
        "E0_a100": (6, -10),
        "CPOOL-S_e05_a100": (6, -10),
        "EORS-S_e05_a100": (-30, -10),
        "G-POOL_a100": (6, 2),
        "G-EN_a100": (6, -9),
        "CAST-EN_a100": (-14, 10),
        "G-CAL_a100": (-12, 6),
        "RESTA_a100": (6, -2),
        "NEG_a100": (6, -3),
    }
    for run, (x, y, _) in pts.items():
        col, mk, face = style.get(run, (cs.BLUE_L, "o", cs.BLUE_L))
        a.scatter(x, y, s=34, marker=mk, facecolor=face, edgecolor=col, linewidth=1.2, zorder=3)
        a.annotate(ARM_NAMES[run], (x, y), xytext=nudge[run], textcoords="offset points", fontsize=6.4, color=cs.INK)
    for src_, dst in (("E0_a100", "CPOOL-S_e05_a100"), ("CPOOL-S_e05_a100", "EORS-S_e05_a100")):
        a.annotate(
            "",
            pts[dst][:2],
            pts[src_][:2],
            arrowprops=dict(arrowstyle="-|>", color=cs.AXIS, lw=0.7, shrinkA=4, shrinkB=5),
        )
    a.set(
        xlabel="worst-language false refusal of benign prompts (%)",
        ylabel="mean safe-response rate (%)",
        xlim=(10, 55),
        ylim=(88, 101.6),
    )
    a.set_yticks(range(88, 101, 2))
    a.set_title("(a) Gated repairs: similar safety, different over-refusal", fontsize=7.6)
    lrl = json.loads((RUNS / "controls_report.json").read_text())["end_to_end"]["0.05"]["lrl_hrr"]["delta"]
    a.text(
        39,
        92.0,
        "EORS-S vs CPOOL-S cost:\nlow-resource harmful-prompt\nrefusal "
        + f"{100 * lrl:+.1f}".replace("-", "−")
        + " pts",
        fontsize=6.0,
        color=cs.CORAL,
        ha="center",
        va="center",
    )
    cs.mono_ticks(a)
    b = ax["b"]
    arms_b = [
        ("E0", "E0_a100", "E0_greet_a100"),
        ("CPOOL-S", "CPOOL-S_e10_a100", "CPOOL-S_e10_greet_a100"),
        ("EORS-S", "EORS-S_e10_a100", "EORS-S_e10_greet_a100"),
    ]
    greet = {}
    for i, (name, plain, greeted) in enumerate(arms_b):
        x0, x1 = 100 * per_lang(plain).BRR.mean(), 100 * per_lang(greeted).BRR.mean()
        greet[name] = (x0, x1)
        b.plot([x0, x1], [i, i], color=cs.GRID, lw=2.4, zorder=1)
        b.scatter(
            [x0],
            [i],
            s=30,
            color=cs.BLUE,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
            label="original" if i == 0 else None,
        )
        b.scatter(
            [x1],
            [i],
            s=30,
            color=cs.CORAL,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
            label="with greeting" if i == 0 else None,
        )
        if x1 - x0 < 12:  # short dumbbell: one label, to the right
            cs.label(b, x1, i, f"{x0:.0f}% → {x1:.0f}%", xytext=(6, -2.5), textcoords="offset points")
        else:
            cs.label(b, x1, i, f"{x1:.0f}%", xytext=(6, -2.5), textcoords="offset points")
            cs.label(b, x0, i, f"{x0:.0f}%", xytext=(0, 6), textcoords="offset points", ha="center")
    b.set_yticks(range(len(arms_b)), [n for n, *_ in arms_b], fontsize=6.6)
    b.set_ylim(len(arms_b) - 0.5, -0.7)
    b.set(xlim=(0, 85), xlabel="mean benign refusal (%)")
    b.legend(loc="upper right", fontsize=6.2, handletextpad=0.2)
    b.grid(axis="x", visible=True)
    b.grid(axis="y", visible=False)
    b.set_title("(b) A greeting breaks both rules", fontsize=7.6)
    cs.mono_ticks(b, "x")
    c_, e_ = pts["CPOOL-S_e05_a100"], pts["EORS-S_e05_a100"]
    e0_rows = analyze.load("E0_a100").query("lang == 'en'")
    n_h, n_b = int((e0_rows.kind == "harmful").sum()), int((e0_rows.kind == "benign").sum())
    gated = [v for k, v in pts.items() if k not in ("E0_a100", "RESTA_a100", "NEG_a100")]
    lo_s, hi_s = min(v[1] for v in gated), max(v[1] for v in gated)
    lo_b, hi_b = min(v[0] for v in gated), max(v[0] for v in gated)
    cs.action_title(
        fig,
        f"At the same budget, per-language thresholds cut worst-language over-refusal {c_[0]:.1f}% → {e_[0]:.1f}%;\n"
        f"with a greeting, benign refusal reaches {greet['EORS-S'][1]:.0f}% under either rule (ε=.10)",
    )
    caption = (
        "(a) Every repair arm on Tiny Aya Earth (A100, α=1), judged by Qwen3Guard-4B on the test split "
        f"({n_h} harmful MultiJail and {n_b} benign translated-XSTest prompts per language, 10 languages): mean safe-response rate against the "
        "worst language's benign refusal rate. Arrows: from the unrepaired model (E0) to one pooled split-conformal "
        "threshold (CPOOL-S, ε=.05), and from it to per-language split-conformal thresholds at the same budget "
        "(EORS-S); the two share steering vector, condition direction and calibration data. Paired bootstrap "
        "intervals are in Tables 1 and 2. (b) End-to-end greeting test (pre-registered secondary; ε=.10): mean "
        'benign refusal without and with the English prefix "Good morning." in every language.'
    )
    alt = (
        f"Left: scatter of repair arms near {lo_s:.0f}-{hi_s:.0f}% safety, spread from about {lo_b:.0f}% to "
        f"{hi_b:.0f}% worst-language benign "
        "refusal; arrows run from the unrepaired model to CPOOL-S and on to EORS-S, which has the lowest worst-language "
        f"refusal. Right: three dumbbells; the greeting raises mean benign refusal from about {greet['E0'][0]:.0f}% to "
        f"{greet['E0'][1]:.0f}% for the unrepaired model and from about {greet['EORS-S'][0]:.0f}% to "
        f"{greet['EORS-S'][1]:.0f}% for both threshold rules."
    )
    figures_diagnostics.save(
        fig,
        "fig1_teaser",
        caption,
        alt,
        dict(
            points=pts,
            greeting=greet,
            judge="q3g4b",
            source="runs/*_a100/judged.jsonl (incl. *_greet_a100)",
            script="src/refusal_parity/reporting/figures_results.py:fig_teaser",
        ),
    )


def fig_eors():
    """Per-language benign and harmful refusal at a matched budget: pooled vs per-language split-conformal."""
    runs = [
        ("E0_a100", cs.AXIS, "s", "no repair (E0)"),
        ("CPOOL-S_e05_a100", cs.BLUE, "D", "one pooled threshold (CPOOL-S)"),
        ("EORS-S_e05_a100", cs.BLUE, "o", "per-language thresholds (EORS-S)"),
    ]
    tabs = {r: per_lang(r) for r, *_ in runs}
    fig = plt.figure(figsize=(cs.W, 2.7), layout="constrained")
    ax = fig.subplot_mosaic([["brr", "hrr"]])
    for key, col_name, title, lim in (
        ("brr", "BRR", "(a) Benign prompts refused (%)", (0, 50)),
        ("hrr", "HRR", "(b) Harmful prompts refused (%)", (40, 100)),
    ):
        a = ax[key]
        for y, l in enumerate(figures_diagnostics.ORDER):
            v = [100 * tabs[r].loc[l, col_name] for r, *_ in runs]
            a.plot([min(v), max(v)], [y, y], color=cs.GRID, lw=2.4, zorder=1)
            for (r, c, mk, name), off in zip(runs, (-0.17, 0, 0.17), strict=True):
                t = tabs[r].loc[l]
                a.plot([100 * t[f"{col_name}_lo"], 100 * t[f"{col_name}_hi"]], [y + off] * 2, color=c, lw=0.7, zorder=2)
                a.scatter(
                    100 * t[col_name],
                    y + off,
                    s=16,
                    marker=mk,
                    facecolor="white" if mk == "D" else c,
                    edgecolor=c if mk == "D" else "white",
                    zorder=3,
                    linewidth=0.8 if mk == "D" else 0.5,
                    label=name if y == 0 else None,
                )
        a.set_yticks(
            range(len(figures_diagnostics.ORDER)),
            [figures_diagnostics.NAMES[l] for l in figures_diagnostics.ORDER],
            fontsize=6.6,
        )
        a.invert_yaxis()
        a.grid(axis="x", visible=True)
        a.grid(axis="y", visible=False)
        a.set(xlim=lim, title=title)
        cs.mono_ticks(a, "x")
        for b_ in (3.5, 6.5):
            a.axhline(b_, color=cs.GRID, lw=0.8)
        if key == "hrr":
            for y_, tier_name in ((1.5, "high"), (5.0, "medium"), (8.0, "low")):
                a.text(
                    1.01,
                    y_,
                    tier_name,
                    transform=a.get_yaxis_transform(),
                    fontsize=6.0,
                    color=cs.AXIS,
                    va="center",
                    rotation=90,
                )
    ax["hrr"].set_yticklabels([])
    fig.legend(*ax["brr"].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.6)
    g, e = tabs["CPOOL-S_e05_a100"], tabs["EORS-S_e05_a100"]
    dh = (e.HRR - g.HRR).sort_values()
    losers = [figures_diagnostics.NAMES[l] for l in dh.index if dh[l] <= -0.05]  # languages losing >= 5 points of HRR
    cs.action_title(
        fig,
        f"At the same budget, per-language thresholds narrow over-refusal (worst {100 * g.BRR.max():.1f}% → "
        f"{100 * e.BRR.max():.1f}%)\nand lower harmful-prompt refusal where benign prompts score high "
        f"({', '.join(losers)})",
    )
    caption = (
        "Tiny Aya Earth on the A100 (α=1). CPOOL-S and EORS-S share steering vector, condition direction, calibration "
        "fold and false-refusal budget (ε=.05); they differ only in whether the conformal threshold is pooled over "
        "languages or set per language. E0: unrepaired. Qwen3Guard-4B labels; 95% Wilson intervals; languages "
        "grouped high / medium / low resource. Paired changes with bootstrap intervals are in Table 2."
    )
    alt = (
        "Two dot plots by language. Benign refusal: EORS-S is lower than CPOOL-S in Thai, Javanese and Bengali and "
        "slightly higher in high-resource languages. Harmful refusal: EORS-S is lower in Javanese, Bengali and Thai "
        "and higher in Italian and English."
    )
    figures_diagnostics.save(
        fig,
        "fig2_eors",
        caption,
        alt,
        dict(
            source="runs/{E0,CPOOL-S_e05,EORS-S_e05}_a100/judged.jsonl",
            script="src/refusal_parity/reporting/figures_results.py:fig_eors",
        ),
    )


def fig_frontier():
    """Gate level, test activations: worst-language false-firing and tier-level detection across budgets."""
    report = json.loads((RUNS / "controls_report.json").read_text())
    fr, layer = report["frontier"], report["gate"]["P4"]["layer"]
    tier = lambda r, ls: 100 * np.mean([r["tpr"][l] for l in ls])
    fig = plt.figure(figsize=(cs.W, 2.5), layout="constrained")
    ax = fig.subplot_mosaic([["fpr", "tpr"]])
    for rule, col, name in (("pooled", cs.BLUE_L, "pooled"), ("Mondrian", cs.BLUE, "per-language")):
        f = [r for r in fr if r["rule"] == rule]
        x = [100 * r["mean_fpr"] for r in f]
        ax["fpr"].plot(x, [100 * r["max_fpr"] for r in f], marker="o", ms=3, color=col, lw=1.2, label=name)
        ax["tpr"].plot(x, [tier(r, LRL) for r in f], marker="o", ms=3, color=col, lw=1.2, label=f"{name}, low resource")
        ax["tpr"].plot(
            x, [tier(r, HRL) for r in f], marker="o", ms=2.5, color=col, lw=1.0, ls="--", label=f"{name}, high resource"
        )
    ax["fpr"].plot([0, 32], [0, 32], color=cs.GRID, lw=0.8, ls="--")
    ax["fpr"].set(
        xlabel="mean false-firing over languages (%)",
        ylabel="worst-language false-firing (%)",
        title="(a) Worst-language false-firing",
    )
    ax["tpr"].set(
        xlabel="mean false-firing over languages (%)",
        ylabel="harmful prompts detected (%)",
        title="(b) Detection by resource tier",
        xlim=(0, 45),
        ylim=(20, 100),
    )
    for a in ax.values():
        cs.mono_ticks(a)
    ax["fpr"].legend(loc="upper left")
    ax["tpr"].legend(loc="lower right", fontsize=5.6)
    p4 = next(r for r in fr if r["rule"] == "pooled" and r["eps"] == 0.04)
    m4 = min((r for r in fr if r["rule"] == "Mondrian"), key=lambda r: abs(r["mean_fpr"] - p4["mean_fpr"]))
    cs.action_title(
        fig,
        f"At matched mean false-firing (≈{100 * p4['mean_fpr']:.0f}%), per-language thresholds cut worst-language\n"
        f"false-firing {100 * p4['max_fpr']:.1f}% → {100 * m4['max_fpr']:.1f}% and move detection from low- to "
        "high-resource languages",
    )
    caption = (
        f"Gate level, Tiny Aya Earth, test activations (A100), EORS-S condition direction (layer {layer}, fitted on the "
        "calibration fit fold). For budgets ε = .02 to .30, thresholds come from the conformal fold, pooled over "
        "languages or per language; points show the realised mean and worst-language false-firing on benign test "
        "prompts (a) and mean detection of harmful test prompts in the low-resource (bn, sw, jv; solid) and "
        "high-resource (en, zh, it, vi; dashed) tiers (b). Exploratory."
    )
    alt = (
        "Two line charts. Left: the per-language curve lies below the pooled one for worst-language false-firing. "
        "Right: at the same mean false-firing, per-language thresholds detect fewer low-resource and more "
        "high-resource harmful prompts than the pooled threshold."
    )
    figures_diagnostics.save(
        fig,
        "figA_frontier",
        caption,
        alt,
        dict(
            source="runs/controls_report.json frontier",
            script="src/refusal_parity/reporting/figures_results.py:fig_frontier",
        ),
    )


def earth_negative_share():
    """Share of benign calibration prompts projecting negatively on CAST-EN's condition vector (Earth, cond layer)."""
    acts = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    n_layers = next(iter(acts.values())).shape[1]
    layer, _ = arms.pick_condition_layer(acts, range(n_layers // 4, 3 * n_layers // 4))
    beh = list(range(max(0, layer - 2), min(n_layers, layer + 3)))
    v = arms.fit_arms(acts, beh, layer, 0.05)["CAST-EN"]["cond"]
    return {l: float((arms.proj_scores(acts, "benign", l, layer, v) < 0).mean()) for l in LANGS}, layer


def fig_cast():
    """H6 end to end: CAST-EN minus G-EN harmful refusal per language, against the negative-projection share."""
    neg, layer = earth_negative_share()
    d = figures_diagnostics.paired_deltas(analyze.load("G-EN_a100"), analyze.load("CAST-EN_a100"), "harmful", "refuse")
    fig, a = plt.subplots(figsize=(cs.W * 0.82, 2.6), layout="constrained")
    a.axhline(0, color=cs.AXIS, lw=0.8)
    for l in LANGS:
        x, (m, lo, hi) = 100 * neg[l], (100 * v for v in d[l])
        a.plot([x, x], [lo, hi], color=cs.BLUE_L, lw=0.8, zorder=2)
        a.scatter(x, m, s=26, color=cs.BLUE, zorder=3, edgecolor="white", linewidth=0.6)
        off = {"vi": (-12, 7), "ar": (5, 5), "it": (5, -9), "jv": (5, -10), "en": (5, 3), "sw": (5, -3)}.get(l, (5, 2))
        a.annotate(figures_diagnostics.NAMES[l], (x, m), xytext=off, textcoords="offset points", fontsize=6.2)
    rho = spearmanr([neg[l] for l in LANGS], [d[l][0] for l in LANGS]).statistic
    a.set(xlabel="benign prompts projecting negatively (%)", ylabel="CAST-EN − G-EN harmful refusal (pp)")
    cs.mono_ticks(a)
    n_neg = sum(d[l][0] <= 0 for l in LANGS)
    cs.action_title(
        fig,
        f"CAST refuses less than the projection gate in {n_neg} of 10 languages,\n"
        f"but not in proportion to the negative-projection share (ρ = {rho:.2f})",
    )
    caption = (
        f"H6 end to end. Paired per-language change in harmful-prompt refusal between faithful CAST (cosine-tanh "
        f"gate score) and G-EN (projection gate), which share the English steering vector and condition direction "
        f"(Tiny Aya Earth, A100, Qwen3Guard-4B). x: share of benign calibration prompts with negative projection "
        f"on the condition vector at layer {layer}, the quantity Remark 1 links to CAST's loss. Bars: 95% bootstrap "
        f"over prompt ids. Pre-registered primary estimate (low-resource tier, bn/sw/jv pooled) in the title; "
        f"no single language survives Holm. Spearman across languages ρ = {rho:.2f} (descriptive, n = 10)."
    )
    alt = "Scatter of ten languages: all CAST minus G-EN refusal changes are at or below zero, about -4 to 0 points."
    figures_diagnostics.save(
        fig,
        "fig3_cast_end_to_end",
        caption,
        alt,
        dict(
            neg_share=neg,
            deltas=d,
            rho=rho,
            layer=layer,
            source="runs/{G-EN,CAST-EN}_a100/judged.jsonl, runs/mech_earth/calib_acts.pt",
            script="src/refusal_parity/reporting/figures_results.py:fig_cast",
        ),
    )


def fig_instruments():
    """(a) the stack's English keyword detectors vs a validated judge, by language; (b) judge agreement with humans."""
    import pandas as pd

    h1 = json.loads((RUNS / "B0" / "h1.json").read_text())
    jv = pd.read_csv(RUNS / "pgp_val" / "judge_validation.csv")
    fig = plt.figure(figsize=(cs.W, 2.45), layout="constrained")
    ax = fig.subplot_mosaic([["a", "b"]], width_ratios=[1.2, 1])
    a = ax["a"]
    for k, (key, col, name) in enumerate(
        (("kw_substr|q3g4b", cs.BLUE, "substring detector"), ("kw_regex|q3g4b", cs.BLUE_L, "regex detector"))
    ):
        kap = h1[key]["kappa"]
        xs = np.arange(len(figures_diagnostics.ORDER)) + (k - 0.5) * 0.36
        a.bar(xs, [kap[l] for l in figures_diagnostics.ORDER], width=0.34, color=col, label=name)
    a.set_xticks(range(len(figures_diagnostics.ORDER)), figures_diagnostics.ORDER, fontsize=6.6)
    a.set(ylim=(0, 1), ylabel="Cohen's κ vs Qwen3Guard refusal")
    a.set_title("(a) Keyword detectors: English only", fontsize=7.6)
    a.legend(loc="upper right", fontsize=6.3)
    a.annotate("κ = 0.00 in all nine\nother languages", (5, 0.12), ha="center", fontsize=6.4, color=cs.INK)
    cs.mono_ticks(a, "y")
    b = ax["b"]
    judges = [("Qwen3Guard-4B", "Qwen3Guard-4B"), ("Gemini-3.5-Flash broad", "Gemini (broad)")]
    for i, (jn, _short) in enumerate(judges):
        for k, (lab, col) in enumerate((("refusal", cs.BLUE), ("harmful", cs.CORAL))):
            g = jv[(jv.judge == jn) & (jv.label == lab)].kappa
            x = i + (k - 0.5) * 0.36
            b.bar(x, g.mean(), width=0.34, color=col, label=lab if i == 0 else None)
            b.plot([x, x], [g.min(), g.max()], color=cs.INK, lw=0.8)
            cs.label(b, x, g.mean(), f"{g.mean():.2f}", xytext=(0, 3), textcoords="offset points", ha="center")
    b.set_xticks(range(len(judges)), [s for _, s in judges], fontsize=6.6)
    b.set(ylim=(0, 1.18), ylabel="Cohen's κ vs human labels")
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.set_title("(b) Judges differ on harm", fontsize=7.6)
    b.legend(loc="upper center", ncol=2, fontsize=6.3)
    cs.mono_ticks(b, "y")
    q = jv[(jv.judge == "Qwen3Guard-4B") & (jv.label == "harmful")].kappa.mean()
    gm = jv[(jv.judge == "Gemini-3.5-Flash broad") & (jv.label == "harmful")].kappa.mean()
    cs.action_title(
        fig,
        f"Validate the instruments: keyword detectors see no refusal outside English,\n"
        f"and agreement with humans on harm depends on the judge (κ {q:.2f} vs {gm:.2f})",
    )
    caption = (
        "(a) Agreement between SafeTune's English keyword refusal detectors and Qwen3Guard-4B refusal labels on all "
        "B0 responses (Tiny Aya Global), per language (H1, pre-registered). (b) Agreement of each judge with "
        "PolyGuardPrompts human labels on 1,400 items in seven languages (en, hi, it, th, zh, ar, ko); bars: mean "
        "over languages, whiskers: range. No low-resource language among the validation languages."
    )
    alt = "Left: kappa near 0.85-0.92 in English, zero in nine other languages. Right: refusal kappa about 0.9 for both judges; harm kappa 0.74 vs 0.43."
    figures_diagnostics.save(
        fig,
        "fig4_instruments",
        caption,
        alt,
        dict(
            source="runs/B0/h1.json, runs/pgp_val/judge_validation.csv",
            script="src/refusal_parity/reporting/figures_results.py:fig_instruments",
        ),
    )


def fig_brittle():
    """Appendix: per-language gate false-firing under each benign prefix (EORS ε=.10, Earth)."""
    spoof = json.loads((RUNS / "spoof" / "spoof.json").read_text())
    rows = [
        r
        for r in spoof["summary"] + json.loads((RUNS / "spoof" / "control.json").read_text())["summary"]
        if r["kind"] == "benign"
    ]
    conds = [
        ("orig", "original", cs.BLUE),
        ("native_prefix", "instruction, own language", cs.QUARTZ),
        ("en_prefix", "instruction, English", cs.CORAL_L),
        ("greet_prefix", '"Good morning."', cs.CORAL),
    ]
    langs = ["ar", "th", "jv", "bn", "sw"]
    fig, a = plt.subplots(figsize=(cs.W, 2.2), layout="constrained")
    for k, (c, name, col) in enumerate(conds):
        for i, l in enumerate(langs):
            r = next(x for x in rows if x["lang"] == l and x["cond"] == c)
            x = i + (k - 1.5) * 0.2
            a.bar(x, 100 * r["fire_rate"], width=0.19, color=col, label=name if i == 0 else None)
            a.plot([x, x], [100 * r["lo"], 100 * r["hi"]], color=cs.INK, lw=0.6)
    a.set_xticks(range(len(langs)), [figures_diagnostics.NAMES[l] for l in langs], fontsize=6.8)
    a.set(ylim=(0, 100), ylabel="gate fires on benign prompts (%)")
    a.legend(ncol=4, loc="upper left", fontsize=6.3)
    cs.mono_ticks(a, "y")
    cs.action_title(
        fig, "Any short benign prefix multiplies the gate's false-firing; language-ID routing explains little of it"
    )
    caption = (
        f"Gate-level false-firing of the EORS gate (ε=.10, Tiny Aya Earth, layer-{spoof['cond_layer']} projection) on "
        f"{rows[0]['n']} benign test "
        "prompts per language, with the prompt as is or with a prefix; 95% Wilson intervals. Constrained GlotLID "
        "picks the threshold on the modified prompt, as at inference. Post hoc; mechanism open."
    )
    alt = "Grouped bars for five languages; original prompts 6-16%, prefixed prompts 37-82%."
    figures_diagnostics.save(
        fig,
        "figA_brittleness",
        caption,
        alt,
        dict(
            source="runs/spoof/{spoof,control}.json",
            script="src/refusal_parity/reporting/figures_results.py:fig_brittle",
        ),
    )


FIGS = dict(
    teaser=fig_teaser,
    eors=fig_eors,
    cast=fig_cast,
    instruments=fig_instruments,
    brittle=fig_brittle,
    frontier=fig_frontier,
)

if __name__ == "__main__":
    for name in FIGS if sys.argv[1:] in ([], ["all"]) else sys.argv[1:]:
        FIGS[name]()
