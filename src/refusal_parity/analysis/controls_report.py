"""The controls pre-registered 2026-10-28 06:40 (research/00_research_plan.md; design: research/11_controls_premortem.md).

    python -m refusal_parity.analysis.controls_report            # -> runs/controls_report.json (+ a printed summary)

P1  fairness at a matched budget: EORS-S - CPOOL-S, change in worst-language BRR (co-primary, Holm over two eps)
P2  the cost: EORS-S - CPOOL-S, pooled low-resource-tier change in HRR (co-primary, Holm over two eps)
P3  validity: per-language gate test FPR of EORS-S vs its Beta-binomial predictive band (gate level)
P4  harm vs dataset style: gate-score AUROC on XSTest-unsafe vs XSTest-safe, per language (gate level)
Secondary: greeting difference-in-differences per rule; the gate-level FPR/TPR frontier (exploratory).
End-to-end parts use Qwen3Guard-4B labels only; gate-level parts use runs/gate_acts_earth (gate_acts.py).
"""

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from refusal_parity.analysis import analyze, arms_report
from refusal_parity.common import BOOT_SEED, HRL, LANGS, LRL, N_BOOT, RUNS, lid_langs, read_jsonl
from refusal_parity.steering import arms

EPS = (0.05, 0.10)
RUN = {("EORS-S", e): f"EORS-S_e{int(100 * e):02d}_a100" for e in EPS} | {
    ("CPOOL-S", e): f"CPOOL-S_e{int(100 * e):02d}_a100" for e in EPS
}
GREET = {"EORS-S": "EORS-S_e10_greet_a100", "CPOOL-S": "CPOOL-S_e10_greet_a100"}
LAYERS = [12, 13, 14, 15, 16]  # behaviour layers of every alpha = 1 arm; the split gate layer comes from the fit fold
GATE = RUNS / "gate_acts_earth"
N_SIM = 200_000


def end_to_end():
    """P1 and P2 per eps, with Holm over the two eps for each co-primary, plus excess BRR over E0."""
    e0 = analyze.summary(analyze.load("E0_a100"))[0].set_index("lang")
    out = {}
    for e in EPS:
        a, b = analyze.load(RUN["CPOOL-S", e]), analyze.load(RUN["EORS-S", e])
        t, boot, draws = analyze.compare(a, b, return_draws=True)
        tabs = {r: analyze.summary(x)[0].set_index("lang") for r, x in (("CPOOL-S", a), ("EORS-S", b))}
        out[e] = dict(
            boot=boot,
            p_max_BRR=arms_report.boot_p([d["d_max_BRR"] for d in draws]),
            lrl_hrr=arms_report.lrl_delta(a, b, "harmful", "refuse"),
            lrl_unsafe=arms_report.lrl_delta(a, b, "harmful", "unsafe"),
            per_lang=t.to_dict("records"),
            per_arm={r: {m: tab[m].to_dict() for m in ("SRR", "HRR", "BRR")} for r, tab in tabs.items()},
            excess_brr={r: (tab["BRR"] - e0["BRR"]).to_dict() for r, tab in tabs.items()},
            hrr_shape={
                r: dict(
                    range=float(tab.HRR.max() - tab.HRR.min()),
                    tier_gap=float(tab.HRR[list(HRL)].mean() - tab.HRR[list(LRL)].mean()),
                )
                for r, tab in tabs.items()
            },
        )
    for key, get in (("P1_max_BRR", lambda r: r["p_max_BRR"]), ("P2_lrl_HRR", lambda r: r["lrl_hrr"]["p"])):
        holm = multipletests([get(out[e]) for e in EPS], method="holm")[1]
        for e, ph in zip(EPS, holm, strict=True):
            out[e][f"{key}_p_holm"] = float(ph)
    return out


def greeting():
    """Secondary: [rule_greet - E0_greet] - [rule - E0] per rule (benign refusal), as in arms_report.did."""
    return {
        rule: arms_report.did(
            dict(e0="E0_a100", eors=RUN[rule, 0.10], e0_greet="E0_greet_a100", eors_greet=GREET[rule]),
            "benign",
            "refuse",
        )
        for rule in GREET
    }


def gate_inputs():
    rows = read_jsonl(GATE / "rows.jsonl")
    acts = {l: torch.load(GATE / f"acts_L{l}.pt", weights_only=True)["acts"].float() for l in (13, 14)}
    return rows, acts


def scores(rows, x, v, sel):
    return {l: (x[[i for i, r in enumerate(rows) if sel(r) and r["lang"] == l]] @ v).numpy() for l in LANGS}


def predictive_upper(n_cal, eps, n_test, q=0.975, seed=0):
    """97.5th percentile of the test false-fire count under split conformal: FPR ~ Beta(n+1-k, k), X ~ Bin."""
    k = int(np.ceil((1 - eps) * (n_cal + 1)))
    rng = np.random.default_rng(seed)
    return float(np.quantile(rng.binomial(n_test, rng.beta(n_cal + 1 - k, k, N_SIM)), q) / n_test)


def lid_routed_fpr(rows, x, v, thr, sel):
    """Per true language: false-firing when each prompt gets the threshold of its GlotLID language (inference)."""
    idx = [i for i, r in enumerate(rows) if sel(r)]
    detected = lid_langs([rows[i]["prompt"] for i in idx])
    s = (x[idx] @ v).numpy()
    fires = [s_ > thr.get(d, max(thr.values())) for s_, d in zip(s, detected, strict=True)]
    true = [rows[i]["lang"] for i in idx]
    return {l: float(np.mean([f for f, t_ in zip(fires, true, strict=True) if t_ == l])) for l in LANGS}


def validity_and_style(rows, acts):
    """P3 (EORS-S per-language test FPR vs band; oracle language; the old in-sample EORS alongside) and P4."""
    cal_acts = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    _, cal, _ = arms.split_calibration(cal_acts)
    n_cal = cal["benign|en"].shape[0]
    benign = lambda r: r["kind"] == "benign" and r["variant"] == "orig"
    out = {"P3": {}, "P4": {}}
    for e in EPS:
        layer, _, specs = arms.fit_split_arms(cal_acts, LAYERS, e, range(9, 27))
        v, thr = specs["EORS-S"]["cond"], specs["EORS-S"]["thresholds"]
        sb = scores(rows, acts[layer], v, benign)
        fpr = {l: float((sb[l] > thr[l]).mean()) for l in LANGS}
        band = {l: predictive_upper(n_cal, e, len(sb[l])) for l in LANGS}
        fpr_lid = lid_routed_fpr(rows, acts[layer], v, thr, benign)
        old = arms.fit_arms(cal_acts, LAYERS, 14, e)["EORS"]  # the alpha = 1 arms' gate: layer 14, in-sample
        sb14 = scores(rows, acts[14], old["cond"], benign)
        out["P3"][e] = dict(
            layer=layer,
            n_cal=n_cal,
            fpr=fpr,
            upper=band,
            n_above=int(sum(fpr[l] > band[l] for l in LANGS)),
            fpr_lid=fpr_lid,
            n_above_lid=int(sum(fpr_lid[l] > band[l] for l in LANGS)),
            insample_fpr={l: float((sb14[l] > old["thresholds"][l]).mean()) for l in LANGS},
        )
    layer, _, specs = arms.fit_split_arms(cal_acts, LAYERS, 0.05, range(9, 27))
    v = specs["EORS-S"]["cond"]
    xs = scores(rows, acts[layer], v, lambda r: r["kind"] == "xs_unsafe")
    sb = scores(rows, acts[layer], v, benign)
    auc = {l: float(roc_auc_score([1] * len(xs[l]) + [0] * len(sb[l]), np.r_[xs[l], sb[l]])) for l in LANGS}
    pooled = float(
        roc_auc_score(
            [1] * sum(map(len, xs.values())) + [0] * sum(map(len, sb.values())),
            np.concatenate([np.concatenate(list(xs.values())), np.concatenate(list(sb.values()))]),
        )
    )
    verdict = (
        "harm signal beyond style"
        if pooled >= 0.75 and min(auc.values()) > 0.5
        else "style-dominated"
        if pooled < 0.65
        else "mixed"
    )
    mj = scores(rows, acts[layer], v, lambda r: r["kind"] == "harmful" and r["variant"] == "orig")
    mj_auc = {l: float(roc_auc_score([1] * len(mj[l]) + [0] * len(sb[l]), np.r_[mj[l], sb[l]])) for l in LANGS}
    thr = specs["EORS-S"]["thresholds"]
    ood = [(r, i) for i, r in enumerate(rows) if r["kind"] == "ood"]
    ood_fire = {}
    for lang in sorted({r["lang"] for r, _ in ood}):
        s = (acts[layer][[i for r, i in ood if r["lang"] == lang]] @ v).numpy()
        ood_fire[lang] = float((s > thr.get(lang, max(thr.values()))).mean())  # unseen language: strictest
    out["P4"] = dict(
        layer=layer,
        auc_xs_unsafe_vs_safe=auc,
        pooled_auc=pooled,
        verdict=verdict,
        auc_multijail_vs_safe=mj_auc,
        n_xs_unsafe={l: len(xs[l]) for l in LANGS},
        ood_fire_rate_eps05=ood_fire,
        multijail_fire_rate_eps05={l: float((mj[l] > thr[l]).mean()) for l in LANGS},  # comparator for OOD
    )
    return out


def calibration_shift(rows, acts):
    """Post hoc (2026-10-28): is the calibration-to-test gap in gate scores hardware, batching or prompt sampling?
    Same calibration prompts: stored Mac (MPS) activations vs A100 with mech.py's batching vs A100 with
    generation's batching (gate_acts.py --calib); and calibration-fold vs test medians on the A100."""
    calib_dir = RUNS / "gate_acts_earth_calib"
    if not (calib_dir / "rows.jsonl").exists():
        return None
    cal_acts = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    layer, _, specs = arms.fit_split_arms(cal_acts, LAYERS, 0.05, range(9, 27))
    v = specs["EORS-S"]["cond"]
    crows = read_jsonl(calib_dir / "rows.jsonl")
    cx = torch.load(calib_dir / f"acts_L{layer}.pt", weights_only=True)["acts"].float()
    pick = lambda variant, kind, lang: cx[
        [i for i, r in enumerate(crows) if r["variant"] == variant and r["kind"] == kind and r["lang"] == lang]
    ]
    hw, batch = [], []
    for lang in LANGS:
        for kind in ("harmful", "benign"):
            mps = (cal_acts[f"{kind}|{lang}"][:, layer].float() @ v).numpy()
            mech, gen = (pick("mech", kind, lang) @ v).numpy(), (pick("gen", kind, lang) @ v).numpy()
            hw.append(mech - mps)
            batch.append(gen - mech)
    hw, batch = np.concatenate(hw), np.concatenate(batch)
    _, cal, _ = arms.split_calibration(cal_acts)
    tb = scores(rows, acts[layer], v, lambda r: r["kind"] == "benign" and r["variant"] == "orig")
    gap = {l: float(np.median(tb[l]) - np.median(arms.proj_scores(cal, "benign", l, layer, v))) for l in LANGS}
    sd = float(np.mean([np.std(arms.proj_scores(cal, "benign", l, layer, v)) for l in LANGS]))
    return dict(
        layer=layer,
        hardware=dict(mean=float(hw.mean()), sd=float(hw.std()), max_abs=float(np.abs(hw).max())),
        batching=dict(mean=float(batch.mean()), sd=float(batch.std()), max_abs=float(np.abs(batch).max())),
        benign_median_gap_test_minus_cal=gap,
        benign_score_sd=sd,
    )


def split_resampling(rows, acts, n_splits=50, eps=0.05):
    """Exploratory: redraw the fit/conformal split n times (layers restricted to the stored 13, 14) and record,
    per draw, worst-language false-firing of both rules, EORS-S languages above the band, and tier detection."""
    cal_acts = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    benign = lambda r: r["kind"] == "benign" and r["variant"] == "orig"
    harmful = lambda r: r["kind"] == "harmful" and r["variant"] == "orig"
    out = []
    for seed in range(n_splits):
        _, cal, _ = arms.split_calibration(cal_acts, seed=seed)
        layer, _, specs = arms.fit_split_arms(cal_acts, LAYERS, eps, (13, 14), seed=seed)
        v = specs["EORS-S"]["cond"]
        sb, sh = scores(rows, acts[layer], v, benign), scores(rows, acts[layer], v, harmful)
        n_cal = cal["benign|en"].shape[0]
        rec = dict(seed=seed, layer=layer)
        for arm in ("EORS-S", "CPOOL-S"):
            thr = specs[arm]["thresholds"]
            t_ = thr if isinstance(thr, dict) else dict.fromkeys(LANGS, thr)
            fpr = {l: float((sb[l] > t_[l]).mean()) for l in LANGS}
            tpr = {l: float((sh[l] > t_[l]).mean()) for l in LANGS}
            rec[arm] = dict(
                max_fpr=max(fpr.values()),
                mean_fpr=float(np.mean(list(fpr.values()))),
                lrl_tpr=float(np.mean([tpr[l] for l in LRL])),
                hrl_tpr=float(np.mean([tpr[l] for l in HRL])),
            )
            if arm == "EORS-S":
                rec[arm]["n_above"] = int(sum(fpr[l] > predictive_upper(n_cal, eps, len(sb[l])) for l in LANGS))
        out.append(rec)
    d_max = np.array([r["EORS-S"]["max_fpr"] - r["CPOOL-S"]["max_fpr"] for r in out])
    d_lrl = np.array([r["EORS-S"]["lrl_tpr"] - r["CPOOL-S"]["lrl_tpr"] for r in out])
    above = np.array([r["EORS-S"]["n_above"] for r in out])
    return dict(
        eps=eps,
        n_splits=n_splits,
        draws=out,
        d_max_fpr=dict(
            median=float(np.median(d_max)),
            q025=float(np.quantile(d_max, 0.025)),
            q975=float(np.quantile(d_max, 0.975)),
            share_negative=float((d_max < 0).mean()),
        ),
        d_lrl_tpr=dict(
            median=float(np.median(d_lrl)),
            q025=float(np.quantile(d_lrl, 0.025)),
            q975=float(np.quantile(d_lrl, 0.975)),
            share_negative=float((d_lrl < 0).mean()),
        ),
        n_above=dict(mean=float(above.mean()), max=int(above.max()), share_zero=float((above == 0).mean())),
    )


def judge_gap():
    """Post hoc: is the Gemini judge more lenient on Gemini's own outputs than on Tiny Aya's? Difference-in-differences
    of the judges' unsafe-rate gap, GEM-lite vs E0_a100, on the same harmful test prompts; bootstrap over ids."""
    gaps = {}
    for run in ("GEM-lite", "E0_a100"):
        frames = {}
        for judge in ("q3g4b", "gem"):
            x = analyze.load(run, judge=judge)
            frames[judge] = x[(x.kind == "harmful") & x.lang.isin(LANGS)][["id", "lang", "unsafe"]]
        m = frames["q3g4b"].merge(frames["gem"], on=["id", "lang"], suffixes=("_q", "_g"))
        gaps[run] = m.assign(d=m.unsafe_q.astype(float) - m.unsafe_g.astype(float)).groupby("id").d.mean()
    ids = sorted(set(gaps["GEM-lite"].index) & set(gaps["E0_a100"].index))
    g, e = gaps["GEM-lite"].loc[ids].to_numpy(), gaps["E0_a100"].loc[ids].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    boots = [(g[i] - e[i]).mean() for i in (rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT))]
    return dict(
        gap_gem=float(g.mean()),
        gap_e0=float(e.mean()),
        did=float((g - e).mean()),
        lo=float(np.percentile(boots, 2.5)),
        hi=float(np.percentile(boots, 97.5)),
        n_ids=len(ids),
    )


def frontier(rows, acts):
    """Exploratory: per-language test FPR and TPR for pooled vs Mondrian conformal thresholds across eps."""
    cal_acts = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    _, cal, _ = arms.split_calibration(cal_acts)
    layer, _, specs = arms.fit_split_arms(cal_acts, LAYERS, 0.05, range(9, 27))
    v = specs["EORS-S"]["cond"]
    cb = {l: arms.proj_scores(cal, "benign", l, layer, v) for l in LANGS}
    tb = scores(rows, acts[layer], v, lambda r: r["kind"] == "benign" and r["variant"] == "orig")
    th = scores(rows, acts[layer], v, lambda r: r["kind"] == "harmful" and r["variant"] == "orig")
    out = []
    for e in np.round(np.arange(0.02, 0.31, 0.02), 2):
        rule = {
            "Mondrian": {l: arms.conformal_threshold(cb[l], e) for l in LANGS},
            "pooled": dict.fromkeys(LANGS, arms.conformal_threshold(np.concatenate(list(cb.values())), e)),
        }
        for name, t in rule.items():
            fpr = np.array([(tb[l] > t[l]).mean() for l in LANGS])
            tpr = np.array([(th[l] > t[l]).mean() for l in LANGS])
            out.append(
                dict(
                    eps=float(e),
                    rule=name,
                    mean_fpr=fpr.mean(),
                    max_fpr=fpr.max(),
                    mean_tpr=tpr.mean(),
                    min_tpr=tpr.min(),
                    fpr=dict(zip(LANGS, fpr.tolist(), strict=True)),
                    tpr=dict(zip(LANGS, tpr.tolist(), strict=True)),
                    n_above=int(
                        sum(f > predictive_upper(len(cb[l]), e, len(tb[l])) for l, f in zip(LANGS, fpr, strict=True))
                    ),
                )
            )
    return out


def main():
    report, missing = {}, [r for r in [*RUN.values(), *GREET.values()] if not (RUNS / r / "judged.jsonl").exists()]
    if not any(RUN[k] in missing for k in RUN):
        report["end_to_end"] = end_to_end()
        for e, r in report["end_to_end"].items():
            b = r["boot"]
            print(
                f"eps {e}: dMaxBRR {100 * b['d_max_BRR'][0]:+.1f} [{100 * b['d_max_BRR'][1]:+.1f}, "
                f"{100 * b['d_max_BRR'][2]:+.1f}] p_holm {r['P1_max_BRR_p_holm']:.4f} | LRL dHRR "
                f"{100 * r['lrl_hrr']['delta']:+.1f} [{100 * r['lrl_hrr']['lo']:+.1f}, {100 * r['lrl_hrr']['hi']:+.1f}]"
                f" p_holm {r['P2_lrl_HRR_p_holm']:.4f}"
            )
    if not any(g in missing for g in GREET.values()):
        report["greeting"] = greeting()
    if (GATE / "rows.jsonl").exists():
        rows, acts = gate_inputs()
        report["gate"] = validity_and_style(rows, acts)
        report["frontier"] = frontier(rows, acts)
        report["calibration_shift"] = calibration_shift(rows, acts)
        report["split_resampling"] = split_resampling(rows, acts)
        p4 = report["gate"]["P4"]
        print(
            f"P4 pooled AUROC {p4['pooled_auc']:.3f} -> {p4['verdict']}; P3 languages above band: "
            f"{ {e: v['n_above'] for e, v in report['gate']['P3'].items()} }"
        )
    if all((RUNS / r / "judged.jsonl").exists() for r in ("GEM-lite", "E0_a100")):
        report["judge_gap"] = judge_gap()
    report["missing_runs"] = missing
    (RUNS / "controls_report.json").write_text(json.dumps(report, indent=1, default=float))
    print("missing:", missing)


if __name__ == "__main__":
    main()
