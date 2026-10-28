"""Compare CircuitKIT refusal circuits across languages and hand them to SafeTune.

    python -m refusal_parity.circuits.compare ../runs/circuitkit/kaggle_all      # -> <dir>/circuits.json

Reads parity.json (the gate) and refusal_<lang>_scores.json (CircuitKIT node scores). For each language, SafeTune's
own adapter (load_circuit_info_from_file) turns the top 20% of nodes into layer suggestions; run_arm.py
--behavior_layers can take them. Across languages it reports top-k node overlap (Jaccard) and the rank
correlation of node scores (Spearman, over all nodes), the circuit-level analogue of the activation cosine.
"""

import itertools
import json
import sys
from pathlib import Path

from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
from scipy.stats import spearmanr

TOP_K = 50
TOP_FRACTION = 0.2  # SafeTune's default


def node_scores(path):
    return json.loads(path.read_text())["node_scores"]


def main(d):
    d = Path(d)
    gate = json.loads((d / "parity.json").read_text())
    files = {
        p.name.split("_")[1]: p for p in sorted(d.glob("refusal_*_scores.json")) if p.name.count("_") == 2
    }  # skip discover()'s refusal_<lang>_discovery_scores.json side-car
    scores = {lang: node_scores(p) for lang, p in files.items()}
    per_lang = {}
    for lang, p in files.items():
        info = load_circuit_info_from_file(str(p), top_fraction=TOP_FRACTION)
        ls = info.layer_suggestions
        top = sorted(scores[lang], key=scores[lang].get, reverse=True)[:TOP_K]
        per_lang[lang] = dict(
            n_nodes=len(scores[lang]),
            top10=top[:10],
            layer_subset=ls.layer_subset,
            target_modules=ls.target_modules,
            behavior_layers_arg=",".join(map(str, ls.layer_subset)),
        )
    pairs = {}
    for a, b in itertools.combinations(sorted(scores), 2):
        ta = set(sorted(scores[a], key=scores[a].get, reverse=True)[:TOP_K])
        tb = set(sorted(scores[b], key=scores[b].get, reverse=True)[:TOP_K])
        common = sorted(set(scores[a]) & set(scores[b]))
        rho = spearmanr([scores[a][n] for n in common], [scores[b][n] for n in common]).statistic
        pairs[f"{a}-{b}"] = dict(jaccard_top_k=len(ta & tb) / len(ta | tb), spearman=float(rho), n_common=len(common))
    out = dict(
        parity=dict(gate_pass=gate["gate_pass"], max_kl=gate["max_kl"], top1_agree=gate["top1_agree"]),
        trusted=bool(gate["gate_pass"]),
        top_k=TOP_K,
        top_fraction=TOP_FRACTION,
        per_language=per_lang,
        pairs=pairs,
    )
    (d / "circuits.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(
        json.dumps(
            dict(parity=out["parity"], pairs=pairs, layers={k: v["layer_subset"] for k, v in per_lang.items()}),
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
