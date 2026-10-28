"""End-to-end steering results: the comparisons pre-registered in the deviations log (2026-10-27 18:50).

    python -m refusal_parity.analysis.arms_report            # -> runs/arms_report.json (+ a printed summary)

Every comparison is paired by (id, lang, kind) through analyze.compare (per-language McNemar with Holm, paired
bootstrap over ids), under each judge whose labels both runs carry. Runs without finalized judged.jsonl are
skipped and listed, so a partial report never silently looks complete.
"""

import json

import numpy as np

from refusal_parity.analysis import analyze
from refusal_parity.common import BOOT_SEED, LRL, N_BOOT, RUNS

EARTH_ARMS = (
    "G-POOL_a100",
    "G-EN_a100",
    "CAST-EN_a100",
    "G-CAL_a100",
    "EORS_a100",
    "EORS_e10_a100",
    "RESTA_a100",
    "NEG_a100",
)
PAIRS = [("E0_a100", arm, "vs E0") for arm in EARTH_ARMS] + [
    ("B0_a100", "B1_a100", "preamble"),
    ("G-EN_a100", "CAST-EN_a100", "H6 end to end"),  # same vector and direction; only the gate score differs
    ("G-POOL_a100", "EORS_a100", "EORS redistribution"),  # same vector and direction; conformal vs single threshold
    ("G-POOL_a100", "EORS_e10_a100", "EORS redistribution"),
    ("G-POOL_a100", "CAST-EN_a100", "confounded descriptive"),  # only shown if G-EN is missing
    # follow-ups, pre-registered 2026-10-27 23:45 (Qwen3Guard only)
    ("E0_a100", "G-POOL_a05", "vs E0 (alpha .5)"),
    ("E0_a100", "EORS_a05", "vs E0 (alpha .5)"),
    ("E0_a100", "G-EN_a05", "vs E0 (alpha .5)"),
    ("E0_a100", "CAST-EN_a05", "vs E0 (alpha .5)"),
    ("G-POOL_a05", "EORS_a05", "EORS redistribution (alpha .5)"),
    ("G-EN_a05", "CAST-EN_a05", "H6 end to end (alpha .5)"),
    # controls, pre-registered 2026-10-28 06:40 (Qwen3Guard only; primary analysis in controls_report.py)
    *[("E0_a100", f"{arm}_e{e}_a100", "vs E0 (controls)") for e in ("05", "10") for arm in ("CPOOL-S", "EORS-S")],
    *[(f"CPOOL-S_e{e}_a100", f"EORS-S_e{e}_a100", "matched-budget control") for e in ("05", "10")],
]
GREET = dict(e0="E0_a100", eors="EORS_e10_a100", e0_greet="E0_greet_a100", eors_greet="EORS_e10_greet_a100")


def did(runs, kind, col, judge="q3g4b"):
    """[eors_greet - e0_greet] - [eors - e0] per prompt id, paired over all four runs; pooled and per language."""
    key = ["id", "lang", "kind"]
    m = None
    for tag, run in runs.items():
        x = analyze.load(run, judge=judge)
        x = x[x.kind == kind][key + [col]].rename(columns={col: tag})
        m = x if m is None else m.merge(x, on=key)
    d = (m.eors_greet.astype(float) - m.e0_greet.astype(float)) - (m.eors.astype(float) - m.e0.astype(float))
    m = m.assign(d=d)
    rng = np.random.default_rng(BOOT_SEED)
    per_id = m.groupby("id").d.mean()  # ids are parallel across languages: resample ids jointly
    boots = [per_id.sample(len(per_id), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(N_BOOT)]
    out = dict(
        pooled=dict(
            delta=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), n=len(m)
        ),
        per_lang={},
    )
    for lang, g in m.groupby("lang"):
        bl = [g.d.sample(len(g), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(N_BOOT)]
        out["per_lang"][lang] = dict(
            delta=float(g.d.mean()), lo=float(np.percentile(bl, 2.5)), hi=float(np.percentile(bl, 97.5))
        )
    # components, for the reader: how much the prefix moves the unsteered model vs the steered one
    out["components"] = {k: float(m[k].astype(float).mean()) for k in ("e0", "eors", "e0_greet", "eors_greet")}
    return out


def lrl_delta(a, b, kind, col):
    """Pooled low-resource-tier paired change b - a, bootstrap over ids (ids are parallel across languages)."""
    key = ["id", "lang", "kind"]
    m = a[key + [col]].merge(b[key + [col]], on=key, suffixes=("_a", "_b"))
    m = m[(m.kind == kind) & m.lang.isin(LRL)]
    d = (m[col + "_b"].astype(float) - m[col + "_a"].astype(float)).groupby(m.id).mean()  # one value per id
    rng = np.random.default_rng(BOOT_SEED)
    boots = [d.sample(len(d), replace=True, random_state=rng.integers(1 << 31)).mean() for _ in range(N_BOOT)]
    return dict(
        delta=float(d.mean()),
        lo=float(np.percentile(boots, 2.5)),
        hi=float(np.percentile(boots, 97.5)),
        n_ids=len(d),
        p=boot_p(boots),
    )


def boot_p(draws):
    """Two-sided percentile-bootstrap p-value for 'the change is zero', in [1/N_BOOT, 1] (draws at exactly 0,
    common with tied binary outcomes, count on both sides)."""
    d = np.asarray(draws)
    return float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))))


def power(a, b):
    """Observed discordance and McNemar MDE (80% power, alpha 0.05 and Holm's 0.005) per outcome, median language."""
    key = ["id", "lang", "kind"]
    m = a[key + ["unsafe", "refuse"]].merge(b[key + ["unsafe", "refuse"]], on=key, suffixes=("_a", "_b"))
    out = {}
    for kind, col in (("harmful", "unsafe"), ("harmful", "refuse"), ("benign", "refuse")):
        g = m[m.kind == kind].groupby("lang")
        psi = float(np.median([(x[col + "_a"] != x[col + "_b"]).mean() for _, x in g]))
        n = int(g.size().median())
        out[f"{kind}_{col}"] = dict(
            psi=psi, n=n, mde=analyze.mde_mcnemar(n, psi, 0.05), mde_holm=analyze.mde_mcnemar(n, psi, 0.005)
        )
    return out


def main():
    report, missing = [], set()
    have = {r for r in {x for p in PAIRS for x in p[:2]} if (RUNS / r / "judged.jsonl").exists()}
    missing = {x for p in PAIRS for x in p[:2]} - have
    for a_name, b_name, purpose in PAIRS:
        if a_name not in have or b_name not in have:
            continue
        if purpose == "confounded descriptive" and "G-EN_a100" in have:
            continue
        for judge in ("q3g4b", "gem"):
            try:
                a, b = analyze.load(a_name, judge=judge), analyze.load(b_name, judge=judge)
            except LookupError:  # a run without this judge's labels
                continue
            n_pairs = len(a.merge(b, on=["id", "lang", "kind"]))
            if judge == "q3g4b":  # the primary judge labels every row: a silent drop would bias a pre-registered pair
                assert n_pairs == len(a) == len(b), f"{a_name} vs {b_name}: {n_pairs} pairs from {len(a)}/{len(b)} rows"
            t, boot = analyze.compare(a, b)
            row = dict(
                a=a_name,
                b=b_name,
                purpose=purpose,
                judge=judge,
                n_pairs=n_pairs,
                boot={k: [float(x) for x in v] for k, v in boot.items()},
                per_lang=t.to_dict("records"),
                power=power(a, b),
                lrl_hrr=lrl_delta(a, b, "harmful", "refuse"),
                lrl_unsafe=lrl_delta(a, b, "harmful", "unsafe"),
                lrl_brr=lrl_delta(a, b, "benign", "refuse"),
            )
            report.append(row)
            print(
                f"{purpose:24s} {b_name:14s} - {a_name:12s} [{judge}] "
                + " ".join(
                    f"{k}={v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]"
                    for k, v in row["boot"].items()
                    if k in ("d_mean_SRR", "d_mean_HRR", "d_max_BRR")
                )
            )
    greet = {}
    if all((RUNS / r / "judged.jsonl").exists() for r in GREET.values()):
        greet = {
            f"{kind}_{col}": did(GREET, kind, col)
            for kind, col in (("benign", "refuse"), ("harmful", "refuse"), ("harmful", "unsafe"))
        }
        g = greet["benign_refuse"]["pooled"]
        print(
            f"greeting DiD on benign refusal [q3g4b]: {100 * g['delta']:+.1f} pp [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}]"
        )
    else:
        missing |= {r for r in GREET.values() if not (RUNS / r / "judged.jsonl").exists()}
    (RUNS / "arms_report.json").write_text(
        json.dumps(dict(pairs=report, greeting_did=greet, missing_runs=sorted(missing)), indent=1, default=float)
    )
    print("missing (not finalized):", sorted(missing))


if __name__ == "__main__":
    main()
