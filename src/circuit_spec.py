"""Build the CircuitKIT job spec: parity prompts + per-language refusal-discovery tasks, rendered locally.

    python circuit_spec.py ../scratch_colab/parity_prompts.json ../scratch_colab/circuit_spec.json [--preamble none]

Pairs are token-length matched (deviations log, 13:25). CircuitKIT left-pads clean/corrupt pairs, and the safety
preamble precedes the user turn, so any length difference shifts every preamble position and EAP-IG / patching
compare different tokens (unmatched pairs gave F <= 0.065 at 20% of nodes). Within each rendered-length bucket,
harmful and benign calibration prompts are paired in id order; pairs sorted by harmful id give N_DISCOVER discovery
pairs then N_EVAL held-out evaluation pairs per language (ids are no longer parallel across languages).
Answer tokens are B0's most frequent single-token refusal / compliance openers (runs/B0/first_tokens.json);
'K' and 'আ' are skipped as too generic (they open both refusals and compliances).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from common import DATA, REVISIONS, read_jsonl
from generate import render

N_DISCOVER, N_EVAL = 32, 26  # 58 = the smallest exact-length pool across en/sw/bn (en)
TOKENS = {"en": ("I'm", "To"), "sw": ("Sam", "Katika"), "bn": ("আমি", "প")}

ap = argparse.ArgumentParser()
ap.add_argument("parity")
ap.add_argument("out")
ap.add_argument("--preamble", default="default", choices=["default", "none", "strict"])
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", revision=REVISIONS["CohereLabs/tiny-aya-global"])
for r, c in TOKENS.values():  # the logit-difference metric needs single tokens
    assert all(len(tok(t, add_special_tokens=False).input_ids) == 1 for t in (r, c)), (r, c)

rows = [r for r in read_jsonl(DATA / "eval.jsonl") if r["split"] == "calib"]
n_tok = lambda text: len(tok(text, add_special_tokens=False).input_ids)


def matched_pairs(lang):
    """[(harmful_text, benign_text)] with identical rendered token length, deterministic (id order per bucket)."""
    buckets = {k: defaultdict(list) for k in ("harmful", "benign")}
    for r in sorted((r for r in rows if r["lang"] == lang and r["kind"] in buckets), key=lambda r: r["id"]):
        text = render(tok, r["prompt"], a.preamble)
        buckets[r["kind"]][n_tok(text)].append((r["id"], text))
    pairs = [
        (h, b) for n, hs in buckets["harmful"].items() for h, b in zip(hs, buckets["benign"].get(n, []), strict=True)
    ]
    return [(h[1], b[1]) for h, b in sorted(pairs)]


def task(pairs, lang):
    r, c = TOKENS[lang]
    return dict(harmful=[h for h, _ in pairs], benign=[b for _, b in pairs], refusal_token=r, compliance_token=c)


spec = dict(
    parity=json.loads(Path(a.parity).read_text()),
    preamble=a.preamble,
    pairing="token-length matched",
    discover={},
    eval={},
)
for lang in TOKENS:
    pairs = matched_pairs(lang)
    assert len(pairs) >= N_DISCOVER + N_EVAL, f"{lang}: only {len(pairs)} length-matched pairs"
    assert all(n_tok(h) == n_tok(b) for h, b in pairs)
    spec["discover"][lang] = task(pairs[:N_DISCOVER], lang)
    spec["eval"][lang] = task(pairs[N_DISCOVER : N_DISCOVER + N_EVAL], lang)
Path(a.out).write_text(json.dumps(spec, ensure_ascii=False, indent=1))
print(
    {lang: (len(spec["discover"][lang]["harmful"]), len(spec["eval"][lang]["harmful"])) for lang in TOKENS},
    a.preamble,
    "->",
    a.out,
)
