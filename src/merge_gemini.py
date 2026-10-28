"""Merge Gemini labels (runs/<arm>/gemini_judged.jsonl) into runs/<arm>/judged.jsonl by (id, lang, kind).

python merge_gemini.py runs/B0
"""

import sys
from pathlib import Path

from common import read_jsonl, write_jsonl

d = Path(sys.argv[1])
gem = {
    (r["id"], r["lang"], r["kind"], r.get("response")): r for r in read_jsonl(d / "gemini_judged.jsonl")
}  # response in key: no stale labels
rows = read_jsonl(d / "judged.jsonl")
for r in rows:
    g = gem.get((r["id"], r["lang"], r["kind"], r.get("response")), {})
    r.update({k: v for k, v in g.items() if k.startswith(("gem_", "tok_"))})
write_jsonl(d / "judged.jsonl", rows)
print(f"merged Gemini labels into {sum('gem_refusal' in r for r in rows)}/{len(rows)} rows of {d}/judged.jsonl")
