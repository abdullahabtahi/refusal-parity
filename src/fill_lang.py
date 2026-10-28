"""Add resp_lang (constrained GlotLID) to a judged.jsonl scored remotely with audit.py --no-lid.

python fill_lang.py runs/B0/judged.jsonl
"""

import sys

from common import detect_lang, load_lid, read_jsonl, write_jsonl

rows, lid = read_jsonl(sys.argv[1]), load_lid()
for r in rows:
    r["resp_lang"] = detect_lang(lid, r.get("response") or " ")[0]
write_jsonl(sys.argv[1], rows)
print(f"filled resp_lang for {len(rows)} rows of {sys.argv[1]}")
