"""Render the Hugging Face cards in release/cards/ for scripts/build_release.py.

Cards contain no hand-typed results. A placeholder {{Name}} is one of:
  - a paper macro from paper/numbers.tex (the same value the PDF prints);
  - a fact computed here from the released files (counts, seeds, the run table, viewer configs).
An unknown placeholder fails the build.
"""

import json
import re
from collections import Counter
from pathlib import Path

from refusal_parity.common import BOOT_SEED, DATA, LANGS, NAMES, RUNS, SEED, SPLIT_SEED, TIERS, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "release" / "cards"
VERSION = "0.2.0"
MACRO = re.compile(r"^\\newcommand\{\\(\w+)\}\{(.*)\}\s*(%.*)?$")
LATEX = [("\\%", "%"), ("\\$", "$"), ("\\,", " "), ("{,}", ",")]
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def macros():
    """Paper macros as plain text; those holding math the card cannot show stay LaTeX and fail only if used."""
    out = {}
    for line in (ROOT / "paper" / "numbers.tex").read_text().splitlines():
        if m := MACRO.match(line):
            value = m.group(2)
            for a, b in LATEX:
                value = value.replace(a, b)
            out[m.group(1)] = value
    return out


def data_facts():
    """Counts read from the released prompt files, so the card describes exactly what ships."""
    from refusal_parity.data import build_data  # translation-QA thresholds and calibration sizes are defined there

    rows = read_jsonl(DATA / "eval.jsonl")
    ids = lambda kind, split=None: {r["id"] for r in rows if r["kind"] == kind and split in (None, r["split"])}
    ood_langs = sorted({r["lang"] for r in rows if r["kind"] == "ood"})
    return dict(
        version=VERSION,
        n_eval_rows=f"{len(rows):,}",
        n_langs=len(LANGS),
        langs=", ".join(f"{NAMES[l]} (`{l}`)" for l in LANGS),
        **{f"{t.lower()}_langs": ", ".join(NAMES[l] for l in LANGS if TIERS[l] == t) for t in ("HRL", "MRL", "LRL")},
        n_harm_ids=len(ids("harmful")),
        n_harm_calib=len(ids("harmful", "calib")),
        n_harm_test=len(ids("harmful", "test")),
        n_benign_ids=len(ids("benign")),
        n_benign_calib=len(ids("benign", "calib")),
        n_benign_test=len(ids("benign", "test")),
        n_ood_rows=sum(r["kind"] == "ood" for r in rows),
        n_ood_langs=len(ood_langs),
        ood_langs=", ".join(f"`{l}`" for l in ood_langs),
        n_xs_ids=len({r["id"] for r in read_jsonl(DATA / "xs_unsafe.jsonl")}),
        # English source prompts that entered translation QA (data/raw is the translation output, all languages)
        n_xs_safe_src=sum(r["lang"] == "en" for r in read_jsonl(DATA / "raw" / "xstest_safe_ml.jsonl")),
        n_xs_unsafe_src=sum(r["lang"] == "en" for r in read_jsonl(DATA / "raw" / "xstest_unsafe_ml.jsonl")),
        chrf_min=f"{build_data.CHRF_MIN:g}",
        seed=SEED,
        boot_seed=BOOT_SEED,
        split_seed=SPLIT_SEED,
    )


def hardware(run):
    """Where a run generated: Colab runs record their GPU in colab_timing.json; the rest ran on the Mac."""
    timing = RUNS / run / "colab_timing.json"
    return f"{json.loads(timing.read_text())['gpu']} (Colab)" if timing.exists() else "Apple M1 Pro (MPS)"


def run_info(run):
    cfg_path = RUNS / run / "arm_config.json"
    if not cfg_path.exists():  # the closed comparator, generated through the Gemini API
        return dict(model="Gemini 3.5 Flash-Lite (API)", arm="closed comparator", alpha="—", eps="—", prefix="—",
                    hardware="Gemini API")  # fmt: skip
    c = json.loads(cfg_path.read_text())
    steering = c["arm"] not in {"B0", "B1", "E0", "RESTA"}
    conformal = c["arm"] in {"EORS", "EORS-S", "CPOOL-S"}
    return dict(
        model=c["model"].split("/")[-1],
        arm=c["arm"] + (f" (preamble: {c['preamble']})" if c["preamble"] != "default" else ""),
        alpha=f"{c['alpha']:g}" if steering else "—",
        eps=f"{c['eps']:g}" if conformal else "—",
        prefix=f"`{c['prefix'].strip()}`" if c.get("prefix") else "—",
        hardware=hardware(run),
    )


def runs_table(runs, labels_dir):
    head = "| Run (config) | Model | Arm | α | ε | Prompt prefix | Hardware | Rows | Judges |\n|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for run in runs:
        first = json.loads((labels_dir / f"{run}.jsonl").open().readline())
        judges = [j for j, col in (("Qwen3Guard-4B", "q3g4b_safety"), ("Qwen3Guard-0.6B", "q3g_safety"),
                                   ("Gemini", "gem_harmful")) if col in first]  # fmt: skip
        n = sum(1 for _ in (labels_dir / f"{run}.jsonl").open())
        i = run_info(run)
        lines.append(f"| `{run}` | {i['model']} | {i['arm']} | {i['alpha']} | {i['eps']} | {i['prefix']} | "
                     f"{i['hardware']} | {n:,} | {', '.join(judges)} |")  # fmt: skip
    return "\n".join(lines)


def configs(entries, default=None):
    """YAML `configs:` for the dataset viewer. One config per file: runs differ in which judges labelled them,
    and Hub split names cannot contain the hyphens in run names."""
    lines = ["configs:"]
    for name, path in entries:
        lines += [f"- config_name: {name}", f"  data_files: {path}"]
        if name == default:
            lines.append("  default: true")
    return "\n".join(lines)


def render(name, extra, path=None):
    values = {**macros(), **data_facts(), **extra}
    text = (path or CARDS / f"{name}.md").read_text()
    unknown = sorted(set(PLACEHOLDER.findall(text)) - values.keys())
    if unknown:
        raise KeyError(f"release/cards/{name}.md: unknown placeholders {unknown}")
    latex = sorted(k for k in set(PLACEHOLDER.findall(text)) if "\\" in str(values[k]))
    if latex:
        raise ValueError(f"release/cards/{name}.md: placeholders hold LaTeX a card cannot render: {latex}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


def judge_counts(labels_dir, runs):
    """How many runs carry each judge's labels (for the card's judge section)."""
    c = Counter()
    for run in runs:
        first = json.loads((labels_dir / f"{run}.jsonl").open().readline())
        c.update(k for k in ("q3g4b_safety", "q3g_safety", "gem_harmful") if k in first)
    return dict(n_runs_q3g4b=c["q3g4b_safety"], n_runs_q3g=c["q3g_safety"], n_runs_gem=c["gem_harmful"])
