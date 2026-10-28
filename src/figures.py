"""Paper / article figures, following the dataviz + scientific-visualization skills.

Each figure answers one question and carries an *action title* stating the finding. Encoding rules:
emphasis (one accent, the rest grey) when the story is one series; dumbbells (one hue, two shades,
two marker shapes) for before -> after per language; one-hue sequential ramp for magnitude;
identity never by colour alone (direct labels + marker shape); bars/rates start at zero; uncertainty
named in the caption. Width = NeurIPS text width (5.5 in). Writes PDF + PNG + <name>.json
(caption, alt text, provenance) into paper/figs/.

    python figures.py mech  runs/mech_global
    python figures.py gates runs/mech_global [--eps 0.10]
    python figures.py cast  runs/mech_global
    python figures.py parity B0 E0 GEM-lite [--judge gem]
    python figures.py outcomes B0 [--judge gem]
    python figures.py drift B0 E0             # paired per-language change, both judges
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.proportion import proportion_confint

import analyze
import arms
import cohere_style as cs
from common import LANGS, NAMES, ORDER, ROOT

OUT = ROOT / "paper" / "figs"
W = 5.5  # inches, NeurIPS text width
# One visual system for every paper figure: the Cohere style tokens (cohere_style.py, validated with the dataviz
# checker); light / dark blue are the ordinal pair, AXIS the neutral.
ACCENT, LIGHT, DARK = cs.BLUE, cs.BLUE_L, cs.BLUE
GREY, INK, INK2, SURFACE = cs.AXIS, cs.INK, cs.AXIS, cs.SURFACE
BLUES = LinearSegmentedColormap.from_list("blues1", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cs.apply()


def save(fig, name, caption, alt, provenance):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    (OUT / f"{name}.json").write_text(json.dumps(dict(caption=caption, alt_text=alt, provenance=provenance), indent=1))
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def action_title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=8.5, fontweight="bold")


def tier_axis(ax, axis="y", labels=True):
    """Language ticks grouped by tier, with tier names as secondary labels."""
    pos = np.arange(len(ORDER))
    (ax.set_yticks if axis == "y" else ax.set_xticks)(pos, ORDER)
    for tier, langs in (("HRL", ORDER[:4]), ("MRL", ORDER[4:7]), ("LRL", ORDER[7:])):
        mid = np.mean([ORDER.index(l) for l in langs])
        if axis == "y" and labels:
            ax.annotate(
                tier,
                (0, mid),
                xycoords=("axes fraction", "data"),
                xytext=(-24, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=INK2,
                fontsize=6.5,
            )
    for b in (3.5, 6.5):
        (ax.axhline if axis == "y" else ax.axvline)(b, color="#d9d8d3", lw=0.6, zorder=0)


def fig_mech(d):
    d = Path(d)
    m = np.load(d / "mech.npz")
    auroc, proj, cos = m["auroc"], m["proj"], m["cos"]
    best = int(json.loads((d / "summary.json").read_text())["best_layer_by_mean_auroc"])
    focus = int(np.argmin(auroc[:, best]))  # the language the story is about (lowest detection at best layer)
    fig = plt.figure(figsize=(W, 2.05), layout="constrained")
    ax = fig.subplot_mosaic([["a", "b", "c"]], width_ratios=[1.25, 1.25, 1])
    for panel, data, ylab in (("a", auroc, "AUROC, harmful vs benign"), ("b", proj, "projection gap (a.u.)")):
        for i in range(len(LANGS)):
            if i != focus:
                ax[panel].plot(data[i], color=GREY, lw=0.8, zorder=1)
        ax[panel].plot(data[focus], color=ACCENT, lw=1.6, zorder=3)
        ax[panel].annotate(
            LANGS[focus],
            (len(data[focus]) - 1, data[focus][-1]),
            xytext=(3, 0),
            textcoords="offset points",
            color=ACCENT,
            fontsize=7,
            va="center",
            fontweight="bold",
        )
        ax[panel].axvline(best, color=INK2, lw=0.6)
        ax[panel].set(xlabel="layer", ylabel=ylab)
    ax["a"].set(ylim=(0.5, 1.0), title="(a) Detection")
    ax["a"].text(
        0.98,
        0.04,
        f"grey: {len(LANGS) - 1} other languages",
        transform=ax["a"].transAxes,
        ha="right",
        va="bottom",
        color=INK2,
        fontsize=6.0,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6),
    )
    ax["b"].set(title="(b) Action margin")
    im = ax["c"].imshow(cos[:, :, best], vmin=0, vmax=1, cmap=BLUES)
    ax["c"].set_xticks(range(len(LANGS)), LANGS, fontsize=6, rotation=90)
    ax["c"].set_yticks(range(len(LANGS)), LANGS, fontsize=6)
    ax["c"].grid(False)
    ax["c"].set_title(f"(c) Direction cosine, layer {best}")
    fig.colorbar(im, ax=ax["c"], fraction=0.046, label="cosine")
    action_title(fig, f"Tiny Aya detects harm in every language; {NAMES[LANGS[focus]]} detects and acts weakest")
    caption = (
        f"Tiny Aya Global, last-token residual stream, test prompts (209 harmful, 106 benign per language). "
        f"(a) AUROC of the pooled harmful-minus-benign direction per layer; (b) mean projection gap (harmful minus "
        f"benign); (c) cosine between language-specific refusal directions at layer {best} (best mean AUROC). "
        f"Directions fit on calibration prompts only."
    )
    alt = (
        f"Three panels. Detection AUROC rises to about 0.95-0.99 by mid layers for nine languages (grey); "
        f"{LANGS[focus]} (blue) peaks lower and declines in late layers. The projection gap grows with depth and is "
        f"smallest for {LANGS[focus]}. A cosine heatmap shows most language directions are similar (0.7-0.95), "
        f"with zh and jv the least aligned."
    )
    save(fig, "fig1_detection_action", caption, alt, dict(source=str(d / "mech.npz"), script="src/mech.py"))


def fig_gates(d, eps):
    import gate_proxy

    d = Path(d)
    cal, test = arms.load_acts(d / "calib_acts.pt"), arms.load_acts(d / "test_acts.pt")
    n_layers = next(iter(cal.values())).shape[1]
    layer, _ = arms.pick_condition_layer(cal, range(n_layers // 4, 3 * n_layers // 4))
    res = gate_proxy.matched_comparison(cal, test, layer, eps)
    n_h, n_b = len(test["harmful|en"]), len(test["benign|en"])
    fig = plt.figure(figsize=(W, 2.75), layout="constrained")
    ax = fig.subplot_mosaic([["fpr", "tpr"]])
    for panel, k, n, title in (
        ("fpr", 1, n_b, "False refusals triggered on benign prompts"),
        ("tpr", 0, n_h, "Steering triggered on harmful prompts"),
    ):
        a = ax[panel]
        for y, l in enumerate(ORDER):
            before, after = res["single threshold"][l][k], res["EORS"][l][k]
            a.plot([before, after], [y, y], color="#d9d8d3", lw=2.2, zorder=1, solid_capstyle="round")
            for val, col, mk in ((before, LIGHT, "o"), (after, DARK, "D")):
                lo, hi = proportion_confint(round(val * n), n, method="wilson")
                a.plot([lo, hi], [y, y], color=col, lw=0.7, zorder=2)
                a.scatter(val, y, s=26, color=col, marker=mk, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        tier_axis(a, labels=panel == "fpr")
        a.invert_yaxis()
        # Dumbbells encode position, so a non-zero range is valid; it is stated in the axis label.
        a.set(title=title, xlim=(0.6, 1.0) if panel == "tpr" else (0, 0.45))
        a.set_xlabel("rate (95% Wilson CI)" + ("; axis starts at 0.6" if panel == "tpr" else ""))
    worst = max(ORDER, key=lambda l: res["single threshold"][l][1])
    ax["tpr"].scatter([], [], color=LIGHT, marker="o", label="single threshold")
    ax["tpr"].scatter([], [], color=DARK, marker="D", label="EORS (per-language)")
    ax["tpr"].legend(loc="upper left", fontsize=6.5, bbox_to_anchor=(0, 1.0))
    ax["tpr"].set_yticklabels([])
    mean_fpr = {g: np.mean([v[1] for v in res[g].values()]) for g in res}
    action_title(
        fig,
        f"At equal average cost, EORS equalises over-refusal across languages\n"
        f"({NAMES[worst]} {res['single threshold'][worst][1]:.0%} → {res['EORS'][worst][1]:.0%}), "
        f"but redistributes error: high-resource languages rise to ~10%",
    )
    caption = (
        f"Gate-level parity on held-out activations (Tiny Aya Global, layer {layer}). EORS uses per-language "
        f"split-conformal thresholds (eps={eps}); the single pooled threshold is set so its calibration mean "
        f"false-fire rate matches EORS (test means: single {mean_fpr['single threshold']:.3f}, "
        f"EORS {mean_fpr['EORS']:.3f}). n={n_b} benign and n={n_h} harmful test prompts per language; "
        f"whiskers are 95% Wilson intervals. Rates describe when steering fires, not end-to-end behaviour."
    )
    alt = (
        "Two dumbbell panels by language, grouped by resource tier. Left: benign false-fire rate under a single "
        f"threshold (light circles) vs EORS (dark diamonds). The single threshold is near zero for high-resource "
        f"languages but peaks at {worst}; EORS keeps every language in a narrow band around 10%, lowering "
        f"{worst}, bn and jv and raising the high-resource languages. Right: harmful-prompt trigger rate on an axis "
        "from 0.6 to 1.0; EORS raises it for high-resource languages and lowers it for th, bn and jv."
    )
    save(
        fig,
        f"fig2_gate_parity_eps{eps}",
        caption,
        alt,
        dict(source=[str(d / "calib_acts.pt"), str(d / "test_acts.pt")], script="src/gate_proxy.py:matched_comparison"),
    )


def fig_cast(d):
    d = Path(d)
    h6 = {r["lang"]: r for r in json.loads((d / "gate_proxy.json").read_text())["h6"]}
    x = np.array([h6[l]["neg_proj_benign"] for l in LANGS])
    y = np.array([h6[l]["auroc_proj"] - h6[l]["auroc_cast"] for l in LANGS])
    r = float(np.corrcoef(x, y)[0, 1])
    from scipy.stats import spearmanr

    rho = float(spearmanr(x, y).statistic)
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.45), layout="constrained")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.scatter(x, y, s=28, color=ACCENT, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    nudge = {"ar": (4, 4), "vi": (-11, -9), "en": (4, -6)}  # hand-placed to avoid collisions
    for l, xi, yi in zip(LANGS, x, y, strict=True):
        ax.annotate(
            NAMES[l], (xi, yi), xytext=nudge.get(l, (4, 2)), textcoords="offset points", fontsize=6.3, color=INK2
        )
    ax.set(
        xlabel="benign prompts projecting negatively (share)",
        ylabel="CAST AUROC loss\n(projection − CAST)",
        xlim=(0, 0.8),
        ylim=(-0.04, 0.26),
    )
    ax.text(
        0.97,
        0.05,
        f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}\nn = 10 languages",
        transform=ax.transAxes,
        fontsize=6.3,
        color=INK,
        ha="right",
        va="bottom",
    )
    action_title(fig, "CAST's gate degrades where benign\nprompts project negatively (Prop. 1)")
    caption = (
        "Per-language loss in harmful-vs-benign AUROC when gating with CAST's cosine-tanh score instead of the "
        "plain projection (English-fit direction, held-out test activations), against the share of benign "
        "prompts whose projection is negative. Prop. 1 predicts that CAST scores |projection|, so loss should "
        f"grow with that share. n = 10 languages; correlation is descriptive (Pearson r = {r:.2f})."
    )
    alt = (
        "Scatter of ten languages: x is the share of benign prompts with negative projection (0.06 to 0.63), y is "
        "CAST's AUROC loss. Points rise from Chinese (lowest share, slightly negative loss) to Swahili "
        "(high share, about 0.23 loss)."
    )
    save(fig, "fig3_cast_sign", caption, alt, dict(source=str(d / "gate_proxy.json"), script="src/gate_proxy.py"))


HARM, SAFE_ANS, REFUSED = "#eb6834", "#5598e7", "#184f95"  # validator: PASS bar a contrast WARN -> direct labels


def outcome_table(cond, judge):
    """Per-language three-way outcome on harmful prompts (harm dominates: a harmful 'refusal' counts as harmful)
    plus benign refusal with Wilson CI."""
    df = analyze.load(cond, judge=judge)
    rows = {}
    for l in ORDER:
        h, b = df[(df.kind == "harmful") & (df.lang == l)], df[(df.kind == "benign") & (df.lang == l)]
        harm = h.unsafe.mean()
        refused = (h.refuse & ~h.unsafe).mean()
        lo, hi = proportion_confint(int(b.refuse.sum()), len(b), method="wilson")
        rows[l] = dict(
            harmful=harm,
            refused=refused,
            safe_answer=1 - harm - refused,
            n_h=len(h),
            BRR=b.refuse.mean(),
            BRR_lo=lo,
            BRR_hi=hi,
            n_b=len(b),
        )
    return rows


def fig_outcomes(cond, judge):
    """What the baseline does with harmful prompts in each language, and how often it refuses benign ones."""
    t = outcome_table(cond, judge)
    fig = plt.figure(figsize=(W, 2.6), layout="constrained")
    ax = fig.subplot_mosaic([["h", "b"]], width_ratios=[1.7, 1])
    a = ax["h"]
    for y, l in enumerate(ORDER):
        left = 0.0
        for key, col in (("harmful", HARM), ("safe_answer", SAFE_ANS), ("refused", REFUSED)):
            a.barh(y, t[l][key], left=left, color=col, height=0.72, edgecolor=SURFACE, linewidth=1.0)
            left += t[l][key]
        a.annotate(
            f"{t[l]['harmful']:.0%}",
            (t[l]["harmful"], y),
            xytext=(2, 0),
            textcoords="offset points",
            va="center",
            fontsize=5.8,
            color=INK,
        )  # the one number the story needs, never colour alone
    tier_axis(a)
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel="share of harmful prompts")
    a.set_title("(a) Harmful prompts: what the model does", pad=16)
    a.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    a.legend(
        handles=[
            Patch(color=c, label=n)
            for n, c in (("harmful", HARM), ("judged-safe answer", SAFE_ANS), ("refused", REFUSED))
        ],
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        fontsize=6.3,
        handlelength=1,
        columnspacing=1.2,
        borderaxespad=0.2,
    )
    b = ax["b"]
    for y, l in enumerate(ORDER):
        b.plot([t[l]["BRR_lo"], t[l]["BRR_hi"]], [y, y], color=REFUSED, lw=0.8)
        b.scatter(t[l]["BRR"], y, s=22, color=REFUSED, zorder=3, edgecolor=SURFACE, linewidth=0.8)
    tier_axis(b, labels=False)
    b.invert_yaxis()
    b.set_yticklabels([])
    b.set(xlim=(0, 0.4), xlabel="refusal rate (95% Wilson CI)")
    b.set_title("(b) Benign prompts: refused", pad=16)
    hi_l = sorted(ORDER, key=lambda l: t[l]["BRR"], reverse=True)[:2]
    lrl = ORDER[7:]
    ref = lambda ls: np.mean([t[l]["refused"] for l in ls])
    harm = lambda ls: np.mean([t[l]["harmful"] for l in ls])
    worst = sorted(ORDER, key=lambda l: t[l]["harmful"], reverse=True)[:2]
    action_title(
        fig,
        f"High- → low-resource languages: refusals {ref(ORDER[:4]):.0%} → {ref(lrl):.0%}, "
        f"harmful answers {harm(ORDER[:4]):.0%} → {harm(lrl):.0%}\n"
        f"Over-refusal: {NAMES[hi_l[0]]} {t[hi_l[0]]['BRR']:.0%}, {NAMES[hi_l[1]]} "
        f"{t[hi_l[1]]['BRR']:.0%}; most harmful: {NAMES[worst[0]]} {t[worst[0]]['harmful']:.0%}, "
        f"{NAMES[worst[1]]} {t[worst[1]]['harmful']:.0%}",
    )
    judge_name = {"q3g4b": "Qwen3Guard-Gen-4B", "gem": "Gemini 3.5 Flash (broad rubric)"}[judge]
    caption = (
        f"Baseline {cond} (Tiny Aya Global, default chat template), MultiJail test split, judge {judge_name}. "
        f"(a) Outcome on harmful prompts (n={t['en']['n_h']} per language before dropping unlabelled rows): "
        f"harmful compliance, judged-safe answer (not refused, not judged harmful), refusal; a refusal that "
        f"still contains harmful content counts as harmful. Bars start at zero, so harmful shares compare on a "
        f"common baseline. (b) Refusal rate on benign XSTest-style prompts (n={t['en']['n_b']} per language) "
        f"with 95% Wilson intervals. 'Judged-safe' is the judge's harm label, not a helpfulness or quality "
        f"assessment. Languages grouped by resource tier."
    )
    alt = (
        f"Two panels for ten languages grouped by tier. Left: stacked horizontal bars of harmful-prompt outcomes. "
        f"Orange harmful slivers are small everywhere (about {min(v['harmful'] for v in t.values()):.0%}-"
        f"{max(v['harmful'] for v in t.values()):.0%}). The dark blue refused share shrinks from high- to "
        f"low-resource languages while the light blue judged-safe share grows. Right: benign refusal rates with "
        f"intervals; {hi_l[0]} and {hi_l[1]} are highest."
    )
    save(
        fig,
        f"fig4_baseline_outcomes_{judge}",
        caption,
        alt,
        dict(source=f"runs/{cond}/judged.jsonl", judge=judge, script="src/figures.py:outcome_table", table=t),
    )


def paired_deltas(a, b, kind, col, seed=20261027, n_boot=5000):
    """{lang: (delta, lo, hi)} of b - a on paired ids, 95% percentile bootstrap over ids within the language."""
    key = ["id", "lang", "kind"]
    m = a[key + [col]].merge(b[key + [col]], on=key, suffixes=("_a", "_b"))
    rng, out = np.random.default_rng(seed), {}
    for l in ORDER:
        d = m[(m.kind == kind) & (m.lang == l)]
        d = (d[f"{col}_b"].astype(float) - d[f"{col}_a"].astype(float)).to_numpy()
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        out[l] = (float(d.mean()), *(float(x) for x in np.percentile(boots, [2.5, 97.5])))
    return out


def fig_drift(before, after):
    """Per-language paired change (after - before) in harmful compliance and benign refusal, under both judges."""
    judges = (("q3g4b", "Qwen3Guard-4B", DARK, "o", -0.15), ("gem", "Gemini (broad)", LIGHT, "D", 0.15))
    res = {}
    for j, *_ in judges:
        a, b = analyze.load(before, judge=j), analyze.load(after, judge=j)
        t, boot = analyze.compare(a, b)
        res[j] = dict(
            harm=paired_deltas(a, b, "harmful", "unsafe"),
            brr=paired_deltas(a, b, "benign", "refuse"),
            d_mean_SRR=[float(x) for x in boot["d_mean_SRR"]],
            n_sig=int(((t.metric == "unsafe") & (t.p_holm < 0.05)).sum()),
        )
    fig = plt.figure(figsize=(W, 2.7), layout="constrained")
    ax = fig.subplot_mosaic([["harm", "brr"]])
    for k, title in (("harm", "(a) Harmful answers: change"), ("brr", "(b) Benign refusals: change")):
        a = ax[k]
        a.axvline(0, color=INK2, lw=0.6)
        for j, name, col, mk, off in judges:
            for y, l in enumerate(ORDER):
                d, lo, hi = (100 * v for v in res[j][k][l])
                a.plot([lo, hi], [y + off, y + off], color=col, lw=0.8, zorder=2)
                a.scatter(
                    d,
                    y + off,
                    s=18,
                    color=col,
                    marker=mk,
                    zorder=3,
                    edgecolor=SURFACE,
                    linewidth=0.6,
                    label=name if y == 0 else None,
                )
        tier_axis(a, labels=(k == "harm"))
        a.invert_yaxis()
        a.set(title=title, xlabel=f"{after} − {before} (pp, 95% CI)", xlim=(-14, 10))
    ax["brr"].set_yticklabels([])
    fig.legend(
        *ax["harm"].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=6.5, handletextpad=0.3
    )
    worst = {j: max(ORDER, key=lambda l: res[j]["harm"][l][0]) for j, *_ in judges}
    srr = {j: res[j]["d_mean_SRR"] for j, *_ in judges}
    shared = worst["q3g4b"] if worst["q3g4b"] == worst["gem"] else None
    most = f", most in {NAMES[shared]}" if shared else ""  # its values are in panel (a); the title must fit
    n_sig = max(res[j]["n_sig"] for j in res)
    action_title(
        fig,
        f"Earth vs Global: ~{-100 * np.mean([srr[j][0] for j in srr]):.0f} pp more harmful answers on "
        f"average{most};\n{'no single language' if n_sig == 0 else f'{n_sig} languages'} significant "
        f"after Holm under either judge",
    )
    caption = (
        f"Natural drift: Tiny Aya Earth ({after}) minus Tiny Aya Global ({before}); same prompts, default chat "
        f"template, MultiJail test split. Points are paired per-language differences; bars are 95% percentile "
        f"bootstrap intervals over prompt ids (5,000 resamples). Circles: Qwen3Guard-Gen-4B; diamonds: "
        f"Gemini 3.5 Flash, broad rubric (unlabelled rows dropped). Mean SRR change "
        f"{100 * srr['q3g4b'][0]:+.1f} pp [{100 * srr['q3g4b'][1]:+.1f}, {100 * srr['q3g4b'][2]:+.1f}] "
        f"(Qwen3Guard) and {100 * srr['gem'][0]:+.1f} pp [{100 * srr['gem'][1]:+.1f}, "
        f"{100 * srr['gem'][2]:+.1f}] (Gemini), paired bootstrap over ids. Per-language tests: exact McNemar, "
        f"Holm-corrected across the 10 languages."
    )
    alt = (
        "Two dot-and-interval panels for ten languages grouped by tier, two judges per language. Left: change in "
        "harmful-answer rate, mostly small and slightly positive"
        + (f", largest for {NAMES[shared]} under both judges" if shared else "")
        + ". Right: change in benign refusal rate, mostly zero or slightly negative."
    )
    save(
        fig,
        f"fig5_drift_{before}_{after}",
        caption,
        alt,
        dict(
            source=[f"runs/{before}/judged.jsonl", f"runs/{after}/judged.jsonl"],
            script="src/figures.py:fig_drift",
            table=res,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["mech", "gates", "cast", "outcomes", "drift"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--judge", default="q3g4b", choices=["q3g4b", "gem"])
    ap.add_argument("--eps", type=float, default=0.10)
    a = ap.parse_args()
    {
        "mech": lambda: fig_mech(a.args[0]),
        "gates": lambda: fig_gates(a.args[0], a.eps),
        "cast": lambda: fig_cast(a.args[0]),
        "outcomes": lambda: fig_outcomes(a.args[0], a.judge),
        "drift": lambda: fig_drift(*a.args[:2]),
    }[a.what]()
