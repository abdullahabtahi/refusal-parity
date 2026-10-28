"""Does the gate fire where the baseline actually fails? Gate trigger rates on B0's own harmful compliances.

    python -m refusal_parity.analysis.failure_gate ../runs/mech_global ../runs/B0 [--eps 0.10]     # -> runs/B0/gate_on_failures.json

Aggregate TPR averages over all harmful prompts, most of which the baseline already refuses. Steering only matters
on the prompts it answered harmfully, so this scores the matched-cost gates (gate_proxy.matched_thresholds) on
exactly those, per judge. Test activations follow eval.jsonl order within (kind, split, lang) (mech.py), which
is how rows are joined to the judged responses. Per-language failure counts are small (single digits to ~17),
so the pooled rate is the headline and per-language rates are descriptive.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from refusal_parity.analysis import gate_proxy
from refusal_parity.common import DATA, LANGS, read_jsonl
from refusal_parity.steering import arms


def outcome_sets(judged):
    """{(id, lang): labels} for harmful test prompts; a label is None when that judge left the row unlabelled."""
    out = {}
    for r in judged:
        if r["kind"] != "harmful":
            continue
        q = None if r.get("q3g4b_safety") is None else r["q3g4b_safety"] == "Unsafe"
        g = None if r.get("gem_harmful") is None else bool(r["gem_harmful"])
        out[r["id"], r["lang"]] = dict(q3g4b=q, gem=g, refused=bool(r.get("q3g4b_refusal")))
    return out


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return dict(k=int(k), n=int(n), rate=k / n if n else None, lo=lo, hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acts_dir")
    ap.add_argument("run_dir")
    ap.add_argument("--eps", type=float, default=0.10)
    a = ap.parse_args()
    d, run = Path(a.acts_dir), Path(a.run_dir)
    cal, test = arms.load_acts(d / "calib_acts.pt"), arms.load_acts(d / "test_acts.pt")
    layer = json.loads((d / "gate_proxy.json").read_text())["layer"]
    v, tau, t = gate_proxy.matched_thresholds(cal, layer, a.eps)
    labels = outcome_sets(read_jsonl(run / "judged.jsonl"))
    rows = read_jsonl(DATA / "eval.jsonl")

    fires = {g: {} for g in ("single", "EORS")}  # {gate: {(id, lang): bool}}
    for l in LANGS:
        ids = [r["id"] for r in rows if r["kind"] == "harmful" and r["split"] == "test" and r["lang"] == l]
        s = arms.proj_scores(test, "harmful", l, layer, v)
        assert len(ids) == len(s), (l, len(ids), len(s))  # the join below relies on mech.py's row order
        for i, sc in zip(ids, s, strict=True):
            fires["single"][i, l], fires["EORS"][i, l] = bool(sc > t), bool(sc > tau[l])

    res = {}
    for subset, pick in (
        ("harmful_q3g4b", lambda x: x["q3g4b"] is True),
        ("harmful_gem", lambda x: x["gem"] is True),
        ("harmful_either", lambda x: x["q3g4b"] is True or x["gem"] is True),
        ("refused", lambda x: x["refused"]),
    ):
        keys = [k for k, x in labels.items() if pick(x)]
        res[subset] = {
            g: dict(
                pooled=wilson(sum(fires[g][k] for k in keys), len(keys)),
                per_lang={
                    l: wilson(sum(fires[g][k] for k in keys if k[1] == l), sum(k[1] == l for k in keys)) for l in LANGS
                },
            )
            for g in fires
        }
    out = dict(layer=layer, eps=a.eps, acts=str(d), run=str(run), results=res)
    (run / "gate_on_failures.json").write_text(json.dumps(out, indent=1, default=float))
    for subset, r in res.items():
        p = {g: r[g]["pooled"] for g in r}
        print(
            f"{subset:15s} n={p['single']['n']:4d}  single {p['single']['rate']:.2f} "
            f"[{p['single']['lo']:.2f},{p['single']['hi']:.2f}]   EORS {p['EORS']['rate']:.2f} "
            f"[{p['EORS']['lo']:.2f},{p['EORS']['hi']:.2f}]"
        )
    print(
        "per-language EORS on harmful_either:",
        {l: f"{x['k']}/{x['n']}" for l, x in res["harmful_either"]["EORS"]["per_lang"].items()},
    )
    print(
        "per-language single on harmful_either:",
        {l: f"{x['k']}/{x['n']}" for l, x in res["harmful_either"]["single"]["per_lang"].items()},
    )


if __name__ == "__main__":
    main()
