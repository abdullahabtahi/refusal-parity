"""Add resp_lang (constrained GlotLID) to a judged.jsonl scored remotely with audit.py --no-lid.

python -m refusal_parity.scoring.fill_lang runs/B0/judged.jsonl
"""

import sys

from refusal_parity.common import detect_lang, load_lid, read_jsonl, write_jsonl


def main(path):
    rows, lid = read_jsonl(path), load_lid()
    for r in rows:
        r["resp_lang"] = detect_lang(lid, r.get("response") or " ")[0]
    write_jsonl(path, rows)
    print(f"filled resp_lang for {len(rows)} rows of {path}")


if __name__ == "__main__":
    main(sys.argv[1])
