"""Assemble the public release into hf/release/ (nothing is uploaded or made public here).

    .venv/bin/python scripts/build_release.py

Three Hugging Face repos and one GitHub snapshot (research/12_release_plan.md):
  refusal-parity             OPEN    prompts, per-response LABELS (no response text), reports, judge validation (kappas),
                                     paper PDF and figures, CuratorKIT provenance
  refusal-parity-responses   GATED   full judged.jsonl per run (prompts + model responses, some harmful), judge-validation
                                     items with responses
  refusal-parity-steering    GATED   steering vectors, condition directions, thresholds and split ids per arm
                                     (safetensors + json), and the activations that reproduce the gate-level analyses
  github/                    code, tests, paper sources, research record; no run logs, agent notes or gated files

The open repo is built from a column allowlist and checked: no file in it may carry a `response` field.
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from huggingface_hub import DatasetCard, ModelCard  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import arms  # noqa: E402
import release_cards as cards  # noqa: E402
from common import LANGS, RUNS, read_jsonl, write_jsonl  # noqa: E402

OUT = ROOT / "hf" / "release"
EXCLUDE = {"B0-gem", "pgp_val", "G-POOL"}  # superseded provisional labels, judge-validation workspace, early Mac run
LABEL_COLUMNS = [
    "id", "lang", "tier", "kind", "split", "arm", "model", "preamble",
    "q3g4b_safety", "q3g4b_refusal", "q3g_safety", "q3g_refusal",
    "gem_harmful", "gem_refusal", "gem_blocked", "kw_regex", "kw_substr", "resp_lang", "api_blocked", "api_finish",
    "tok_in", "tok_out", "tok_think",
]  # fmt: skip
USAGE = ("tok_in", "tok_out", "tok_think")
CONFORMAL = {"EORS", "EORS-S", "CPOOL-S"}  # thresholds calibrated at a false-firing budget eps
THRESHOLD_RULE = {
    "CAST-EN": "SafeTune CAST cosine-tanh score; comparator and threshold by max F1 on English",
    "G-EN": "projection; one max-F1 threshold on English",
    "G-POOL": "projection; one max-F1 threshold on pooled languages",
    "G-CAL": "projection; max-F1 threshold per language",
    "EORS": "projection; conformal (1-eps) benign quantile per language, direction fitted on the same calibration "
    "data (in-sample, so the split-conformal guarantee does not apply)",
    "EORS-S": "projection; split-conformal per language, direction from a disjoint fit fold",
    "CPOOL-S": "projection; one split-conformal threshold on pooled scores, direction from the fit fold",
}
# Never public: the agent handoff note and internal hackathon/release planning (they name private repos and secrets'
# locations); everything else in research/ is the study record
RESEARCH_PRIVATE = {"STATUS.md", "06_submission_plan.md", "12_release_plan.md"}
# Superseded or failed-attempt outputs the paper reads none of (the reproduction test would fail if it did)
SUPERSEDED = {"pgp_val/judge_validation_before_qwen.csv", "E0/h3_provisional_gem.json"}
CIRCUITKIT_KEEP = (
    "circuitkit/colab_a100_matched/",
    "circuitkit/kaggle_all/",
    "circuitkit/parity/parity_kaggle_v7_patched",
)
VERSION_FILES = {"colab_versions.txt", "mac_versions.txt"}
PERSONAL = re.compile(r"/Users/[^/\s\"']+|/private/tmp/[^\s\"']+|/var/folders/[^\s\"']+")


def runs():
    return sorted(d.name for d in RUNS.iterdir() if (d / "judged.jsonl").exists() and d.name not in EXCLUDE)


def scrub(text):
    """Absolute local paths -> repo-relative (provenance stays readable, no machine or user names)."""
    text = text.replace(str(ROOT) + "/", "")
    return PERSONAL.sub("<local>", text)


def copy_scrubbed(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(scrub(src.read_text()))


def judged_rows(run):
    """judged.jsonl with tok_* holding generation usage, present only where generation reported it (the closed
    comparator); Tiny Aya rows carry no tok_*.

    The Gemini judge writes its own usage into tok_*, so the released tok_* always mean generation usage, read from
    responses.jsonl; the judge's usage is not released."""
    gen = RUNS / run / "responses.jsonl"
    usage = {(g["id"], g["lang"], g["kind"]): g for g in read_jsonl(gen)} if gen.exists() else {}
    out = []
    for r in read_jsonl(RUNS / run / "judged.jsonl"):
        g = usage.get((r["id"], r["lang"], r["kind"]), {})
        out.append({**{k: v for k, v in r.items() if k not in USAGE}, **{k: g[k] for k in USAGE if k in g}})
    return out


def mirrored(rel, p):
    """Whether runs/<rel> belongs in the open repo's runs/ mirror."""
    text = rel.as_posix()
    if p.suffix not in {".json", ".csv", ".npz"} and p.name not in VERSION_FILES:
        return False
    if rel.parts[0] in EXCLUDE - {"pgp_val"} or {"cache", "mlartifacts"} & set(rel.parts):  # MLflow copies duplicate
        return False
    if text in SUPERSEDED:
        return False
    return rel.parts[0] != "circuitkit" or text.startswith(CIRCUITKIT_KEEP)


def common_files(base):
    """Licence, attribution and the third-party licence texts every HF repo carries."""
    for f in ("LICENSE-DATA", "NOTICE"):
        shutil.copy(ROOT / f, _mk(base / f))
    shutil.copy(ROOT / "LICENSE", _mk(base / "licenses" / "Apache-2.0.txt"))  # Aya red-teaming, Qwen3Guard, GlotLID
    for p in (ROOT / "release" / "licenses").glob("*.txt"):
        shutil.copy(p, _mk(base / "licenses" / p.name))


def dataset_citation():
    """CITATION.cff for the data: the code's file says software / Apache-2.0, which does not describe a dataset."""
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    cff.update(type="dataset", license="CC-BY-NC-4.0")
    cff["abstract"] = cff["abstract"].replace("Code is Apache-2.0; data", "Data")
    return yaml.safe_dump(cff, sort_keys=False, allow_unicode=True, width=120)


def open_repo(base):
    for f in ("eval.jsonl", "xs_unsafe.jsonl", "lid_cache.json"):  # lid_cache: constrained GlotLID per prompt
        shutil.copy(ROOT / "data" / f, _mk(base / "data" / f))
    for d in ("curatorkit", "curatorkit_xs_unsafe"):
        for p in (ROOT / "data" / d).glob("*"):
            if p.is_file() and p.suffix in {".json", ".txt", ".md"}:
                copy_scrubbed(p, base / "data" / d / p.name)
    for run in runs():
        # only the columns a run has: an all-null judge column would read as "labelled by that judge"
        write_jsonl(
            base / "labels" / f"{run}.jsonl", [{k: r[k] for k in LABEL_COLUMNS if k in r} for r in judged_rows(run)]
        )
    # every small summary the paper is generated from (reports, configs, gate diagnostics, CircuitKIT appendix) and
    # the library versions: json / csv / npz / version files only, never jsonl (row-level text), activations or logs
    for p in sorted(RUNS.rglob("*")):
        rel = p.relative_to(RUNS)
        if p.is_file() and mirrored(rel, p):
            dst = base / "runs" / rel
            copy_scrubbed(p, dst) if p.suffix in {".json", ".txt"} else shutil.copy(p, _mk(dst))
    shutil.copy(ROOT / "paper" / "main.pdf", _mk(base / "paper" / "refusal_parity.pdf"))
    for p in (ROOT / "paper" / "figs").glob("*.png"):
        shutil.copy(p, _mk(base / "paper" / "figs" / p.name))
    for p in (ROOT / "paper" / "figs").glob("*.json"):  # provenance: caption, alt text, sources
        copy_scrubbed(p, base / "paper" / "figs" / p.name)
    common_files(base)
    (base / "CITATION.cff").write_text(dataset_citation())


def responses_repo(base):
    for run in runs():
        write_jsonl(base / "responses" / f"{run}.jsonl", judged_rows(run))
    for p in (RUNS / "pgp_val").glob("*.jsonl"):  # the Gemini judge's own usage (tok_*) is not released
        write_jsonl(
            base / "judge_validation" / p.name, [{k: v for k, v in r.items() if k not in USAGE} for r in read_jsonl(p)]
        )
    common_files(base)


def steering_repo(base):
    """Vectors, directions and thresholds exactly as run_arm.py fits them for the alpha = 1 Earth arms (layers read
    from the runs' arm_config.json)."""
    acts = arms.load_acts(RUNS / "mech_earth" / "calib_acts.pt")
    cfg = json.loads((RUNS / "EORS_a100" / "arm_config.json").read_text())
    beh, cond = cfg["behavior_layers"], cfg["cond_layer"]
    index = {}
    for eps in (0.05, 0.10):
        specs = arms.fit_arms(acts, beh, cond, eps)
        layer_s, _, split_specs = arms.fit_split_arms(acts, beh, eps, range(9, 27))
        for name, spec, cond_layer in [
            *((k, v, cond) for k, v in specs.items()),
            *((k, v, layer_s) for k, v in split_specs.items()),
        ]:
            tag = f"{name}_eps{eps:.2f}" if name in CONFORMAL else name
            if tag in index:
                continue
            tensors = {f"steer.layer{l}": v.contiguous().float() for l, v in spec["vecs"].items()}
            if spec["cond"] is not None:
                tensors["condition"] = spec["cond"].contiguous().float()
            save_file(tensors, _mk(base / "arms" / f"{tag}.safetensors"))
            thr = spec["thresholds"]
            index[tag] = dict(
                arm=name,
                eps=eps if name in CONFORMAL else None,  # max-F1 thresholds (G-*, CAST-EN) do not depend on eps
                gate=spec["gate"],
                threshold_rule=THRESHOLD_RULE.get(name),
                condition_layer=cond_layer if spec["cond"] is not None else None,
                behaviour_layers=beh,
                thresholds=thr if not hasattr(thr, "item") else float(thr),
                comparator=spec.get("comparator"),
                alpha=1.0,
                model="CohereLabs/tiny-aya-earth",
            )
    _, _, ids = arms.split_calibration(acts)
    index["_split"] = {f: {k: v.tolist() for k, v in d.items()} for f, d in ids.items()}
    _mk(base / "arms" / "index.json").write_text(json.dumps(index, indent=1, default=float))
    for src, dst in (("mech_earth/calib_acts.pt", "activations/mech_earth/calib_acts.pt"),
                     ("mech_global/calib_acts.pt", "activations/mech_global/calib_acts.pt"),
                     ("mech_global/test_acts.pt", "activations/mech_global/test_acts.pt")):  # fmt: skip
        shutil.copy(RUNS / src, _mk(base / dst))
    for d in ("gate_acts_earth", "gate_acts_earth_calib"):
        for p in (RUNS / d).glob("*"):
            if p.suffix in {".pt", ".jsonl"}:
                shutil.copy(p, _mk(base / "activations" / d / p.name))
    common_files(base)


GITHUB_INCLUDE = [
    "src", "scripts", "tests", "README.md", "LICENSE", "LICENSE-DATA", "NOTICE", "CITATION.cff", "pyproject.toml",
    "Makefile", ".pre-commit-config.yaml", "release", ".github", "SECURITY.md", "CONTRIBUTING.md",
]  # fmt: skip
PAPER_INCLUDE = ["main.tex", "numbers.tex", "checklist.tex", "refs.bib", "neurips_2026.sty", "main.pdf"]


def github_snapshot(base):
    for item in GITHUB_INCLUDE:
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, base / item, ignore=shutil.ignore_patterns("__pycache__", "cache", "logs", "*.pyc"))
        else:
            shutil.copy(src, _mk(base / item))
    for f in PAPER_INCLUDE + [p.name for p in (ROOT / "paper").glob("table_*.tex")]:
        shutil.copy(ROOT / "paper" / f, _mk(base / "paper" / f))
    shutil.copy(ROOT / "release" / "github.gitignore", base / ".gitignore")  # public ignores, not the workspace's
    # workspace-only tooling: sync.sh mirrors results to the author's private HF dataset
    (base / "scripts" / "sync.sh").unlink()
    make = (base / "Makefile").read_text()
    make = make.replace(" sync\n", "\n", 1)
    (base / "Makefile").write_text(make[: make.index("\nsync:")] + "\n")
    for sub in ("figs", "fonts"):
        shutil.copytree(ROOT / "paper" / sub, base / "paper" / sub)
    for p in (ROOT / "research").glob("*.md"):
        if p.name not in RESEARCH_PRIVATE:
            copy_scrubbed(p, base / "research" / p.name)


def _mk(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".tex", ".sh", ".toml", ".cff", ".yaml", ".bib", ""}
SECRET = re.compile(r"hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35}|ghp_[A-Za-z0-9]{36}|-----BEGIN [A-Z ]*PRIVATE KEY")
RESPONSE_KEY = re.compile(r'"response"\s*:')


def check(out):
    """Release gates; collects every failure before raising, so one run lists all of them.

    Open repo: no row may carry a response, and only data/ rows may carry a prompt (every row of every JSONL is
    read; JSON files are searched for a response key). All four parts: no secret-like string, no personal path."""
    fails, base_open = [], out / "refusal-parity"
    for p in base_open.rglob("*.jsonl"):
        in_data = p.relative_to(base_open).parts[0] == "data"
        bad = [i for i, r in enumerate(read_jsonl(p)) if "response" in r or ("prompt" in r and not in_data)]
        if bad:
            fails.append(f"row-level text in {p} (first at row {bad[0]})")
    for p in base_open.rglob("*.json"):
        if RESPONSE_KEY.search(p.read_text(errors="ignore")):
            fails.append(f"response field: {p}")
    for base in sorted(d for d in out.iterdir() if d.is_dir()):
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in TEXT_SUFFIXES or p.name == Path(__file__).name:
                continue  # this script defines the patterns it searches for
            text = p.read_text(errors="ignore")
            if SECRET.search(text):
                fails.append(f"secret-like string: {p}")
            if m := PERSONAL.search(text):
                fails.append(f"personal path: {p}: {m.group(0)}")
    if fails:
        raise SystemExit("release checks failed:\n  " + "\n  ".join(fails))


def write_cards():
    """README.md (the Hub card) for each HF repo, rendered from release/cards/ with the paper's macros."""
    labels = OUT / "refusal-parity" / "labels"
    beh = json.loads((OUT / "refusal-parity-steering" / "arms" / "index.json").read_text())["EORS-S_eps0.05"]
    extra = {
        "refusal-parity": dict(
            configs=cards.configs(
                [("prompts", "data/eval.jsonl"), ("xs_unsafe", "data/xs_unsafe.jsonl")]
                + [(r, f"labels/{r}.jsonl") for r in runs()],
                default="prompts",
            ),
            runs_table=cards.runs_table(runs(), labels),
        ),
        "refusal-parity-responses": dict(
            configs=cards.configs([(r, f"responses/{r}.jsonl") for r in runs()]), n_resp_runs=len(runs())
        ),
        "refusal-parity-steering": dict(behaviour_layers=", ".join(map(str, beh["behaviour_layers"]))),
    }
    for name, values in extra.items():
        text = cards.render(name, values)
        card = ModelCard(text) if name.endswith("steering") else DatasetCard(text)
        card.validate()  # the Hub's own metadata validation (network); raises on a card the Hub would reject
        (OUT / name / "README.md").write_text(text)


def write_checksums(base):
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(base)}"
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    ]
    (base / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    # the repository README is rendered from its template too, before the GitHub snapshot copies it
    (ROOT / "README.md").write_text(cards.render("github-readme", {}))
    parts = {
        "refusal-parity": open_repo,
        "refusal-parity-responses": responses_repo,
        "refusal-parity-steering": steering_repo,
        "github": github_snapshot,
    }
    for name, build in parts.items():
        build(OUT / name)
        n = sum(1 for p in (OUT / name).rglob("*") if p.is_file())
        mb = sum(p.stat().st_size for p in (OUT / name).rglob("*") if p.is_file()) / 1e6
        print(f"{name:26s} {n:4d} files {mb:8.1f} MB")
    write_cards()
    for name in ("refusal-parity", "refusal-parity-responses", "refusal-parity-steering"):
        write_checksums(OUT / name)
    check(OUT)
    print(f"release checks passed; runs: {len(runs())}; languages: {len(LANGS)}")


if __name__ == "__main__":
    main()
