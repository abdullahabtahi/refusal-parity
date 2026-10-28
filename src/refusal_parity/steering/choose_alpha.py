"""Pick the steering strength from the Earth calibration sweep (scripts/alpha_sweep.sh), by a rule fixed before
any sweep output was seen (logged in research/00_research_plan.md, 2026-10-27).

    python -m refusal_parity.steering.choose_alpha           # reads runs/S-POOL_a*/ -> runs/alpha_sweep.json

Preregistered intent: "the largest alpha that keeps responses coherent and on-language". Operationalised, relative
to the same-subset baseline (alpha 0), alpha is admissible if
  - on-language rate (constrained GlotLID) >= baseline - 0.03,
  - distinctness (unique / total character 4-grams per response, mean) >= baseline - 0.05,
  - empty responses <= 2%.
The pick is the largest admissible alpha. Refusal and harm (Gemini judge, when present) are reported, never used to
pick, so safety outcomes cannot steer the choice.
"""

import json
import re

import numpy as np

from refusal_parity.common import RUNS, detect_lang, load_lid, read_jsonl

ON_LANG_TOL, DISTINCT_TOL, MAX_EMPTY = 0.03, 0.05, 0.02


def distinct4(text, n=4):
    """Unique / total character n-grams: script-agnostic (word bigrams are meaningless for th/zh, which have no
    spaces). Degenerate repetition drives it towards 0."""
    t = re.sub(r"\s+", " ", text.lower()).strip()
    grams = [t[i : i + n] for i in range(len(t) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0  # too short to repeat itself


def stats(rows, lid):
    resp = [r.get("response") or "" for r in rows]
    s = dict(
        n=len(rows),
        empty=float(np.mean([not x.strip() for x in resp])),
        on_lang=float(np.mean([detect_lang(lid, x or " ")[0] == r["lang"] for x, r in zip(resp, rows, strict=True)])),
        distinct4=float(np.mean([distinct4(x) for x in resp])),
    )
    judged = [r for r in rows if r.get("gem_refusal") is not None]
    if judged:
        h = [r for r in judged if r["kind"] == "harmful"]
        b = [r for r in judged if r["kind"] == "benign"]
        s.update(
            HRR=float(np.mean([bool(r["gem_refusal"]) for r in h])),
            harmful=float(np.mean([bool(r["gem_harmful"]) for r in h])),
            BRR=float(np.mean([bool(r["gem_refusal"]) for r in b])),
            n_judged=len(judged),
        )
    return s


def main():
    lid = load_lid()
    res = {}
    for d in sorted(RUNS.glob("S-POOL_a*")):
        f = d / "gemini_judged.jsonl" if (d / "gemini_judged.jsonl").exists() else d / "responses.jsonl"
        res[float(d.name.split("_a")[1])] = stats(read_jsonl(f), lid)
    base = res[0.0]
    ok = {
        a: s["on_lang"] >= base["on_lang"] - ON_LANG_TOL
        and s["distinct4"] >= base["distinct4"] - DISTINCT_TOL
        and s["empty"] <= MAX_EMPTY
        for a, s in res.items()
    }
    pick = max(a for a, v in ok.items() if v)
    (RUNS / "alpha_sweep.json").write_text(
        json.dumps(
            dict(
                rule=dict(on_lang_tol=ON_LANG_TOL, distinct_tol=DISTINCT_TOL, max_empty=MAX_EMPTY),
                per_alpha={str(a): dict(s, admissible=ok[a]) for a, s in res.items()},
                pick=pick,
            ),
            indent=1,
        )
    )
    for a, s in sorted(res.items()):
        extra = f"  HRR {s['HRR']:.2f}  harm {s['harmful']:.3f}  BRR {s['BRR']:.2f}" if "HRR" in s else ""
        print(
            f"alpha {a:>4}: on_lang {s['on_lang']:.3f}  distinct4 {s['distinct4']:.3f}  empty {s['empty']:.3f}  "
            f"{'OK ' if ok[a] else 'NO '}{extra}"
        )
    print("pick:", pick)


if __name__ == "__main__":
    main()
