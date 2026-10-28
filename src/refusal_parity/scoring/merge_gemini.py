"""Merge Gemini labels (runs/<arm>/gemini_judged.jsonl) into runs/<arm>/judged.jsonl by (id, lang, kind).

python -m refusal_parity.scoring.merge_gemini runs/B0
"""

import sys
from pathlib import Path

from refusal_parity.common import read_jsonl, write_jsonl


def main(d):
    key = lambda r: (r["id"], r["lang"], r["kind"], r.get("response"))  # response in key: no stale labels
    gem = {key(r): r for r in read_jsonl(d / "gemini_judged.jsonl")}
    rows = read_jsonl(d / "judged.jsonl")
    for r in rows:
        r.update({k: v for k, v in gem.get(key(r), {}).items() if k.startswith(("gem_", "tok_"))})
    write_jsonl(d / "judged.jsonl", rows)
    print(f"merged Gemini labels into {sum('gem_refusal' in r for r in rows)}/{len(rows)} rows of {d}/judged.jsonl")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
