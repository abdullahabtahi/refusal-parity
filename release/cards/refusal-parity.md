---
license: cc-by-nc-4.0
pretty_name: "Refusal Parity: is safety repair language-fair? (Tiny Aya)"
language: [en, zh, it, vi, ar, ko, th, bn, sw, jv, es, fr, hi, ru, sr, tl]
multilinguality: multilingual
task_categories: [text-classification]
size_categories: [10K<n<100K]
source_datasets: [DAMO-NLP-SG/MultiJail, Paul/XSTest, CohereLabs/aya_redteaming]
tags: [ai-safety, refusal, over-refusal, multilingual, low-resource-languages, activation-steering, conformal-prediction, llm-as-a-judge, evaluation, not-for-all-audiences]
{{configs}}
---

![Refusal Parity — Tiny Aya multilingual safety repair audit](assets/banner.png)

# Refusal Parity: is safety repair language-fair?

**Abdullah Ibn Mahdi Abtahi** · Ingramin · v{{version}} ·
[paper (PDF)](paper/refusal_parity.pdf) · [code (GitHub)](https://github.com/abdullahabtahi/refusal-parity) ·
gated companions: [responses](https://huggingface.co/datasets/abdullahabtahi/refusal-parity-responses),
[steering artifacts](https://huggingface.co/abdullahabtahi/refusal-parity-steering)

> ⚠️ **Content warning.** The prompts include harmful requests from public red-teaming benchmarks (violence, sexual
> exploitation, self-harm, crime). This repository holds **no model responses**; those, some of them harmful, are in
> the gated companion repository.

This repository holds the open data of an audit of **refusal repair** on Cohere's Tiny Aya (3.35B). It asks
whether repairs treat languages fairly, across {{n_langs}} [MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail)
languages in three resource tiers. The repairs are activation steering gated per prompt, and one weight-space repair.
The analysis plan was pre-registered, with a dated deviations log. The pipeline is built on the Lexsi open-source
stack: CuratorKIT → CircuitKIT → SafeTune → AuditKIT.

What is here:
- the evaluation prompts;
- one label row per model response for every run: refusal and harm under each judge, plus the response language
  (**no response text**);
- every summary file the paper's numbers are generated from;
- the judge-validation agreement statistics;
- the paper PDF, its figures and their provenance.

## Main findings

![Teaser figure: (a) mean safe-response rate against worst-language benign refusal across repair arms on Tiny Aya Earth, with arrows from the unrepaired model (E0) to pooled split-conformal threshold (CPOOL-S) and per-language split-conformal thresholds (EORS-S); (b) mean benign refusal without and with the English greeting prefix across E0, CPOOL-S, and EORS-S.](assets/fig1_teaser.png)

Intervals are 95% paired bootstrap over prompt ids, under the primary judge (Qwen3Guard-4B) unless stated. Each
value is printed by the same generated macro as the paper, from the source file named in its row.

| Finding | Estimate | Source |
|---|---|---|
| **Matched budget, ε = 0.05** (Tiny Aya Earth, A100). Worst-language benign refusal under per-language split-conformal thresholds (`EORS-S`) vs one pooled threshold (`CPOOL-S`) | {{CtlFivePoolWorstBrr}} → {{CtlFiveEorsWorstBrr}}: {{CtlFiveMaxBrr}} pp [{{CtlFiveMaxBrrLo}}, {{CtlFiveMaxBrrHi}}]. Mean change {{CtlFiveMeanBrr}} pp [{{CtlFiveMeanBrrLo}}, {{CtlFiveMeanBrrHi}}] | `runs/controls_report.json` |
| …the price: harmful-prompt refusal in the low-resource tier | {{CtlFiveLrlHrr}} pp [{{CtlFiveLrlHrrLo}}, {{CtlFiveLrlHrrHi}}]; {{CtlFiveHrrLossLang}} {{CtlFiveHrrLossPts}}, {{CtlFiveHrrGainLang}} {{CtlFiveHrrGainPts}}. Harmful compliance in the tier {{CtlFiveLrlUnsafe}} pp [{{CtlFiveLrlUnsafeLo}}, {{CtlFiveLrlUnsafeHi}}]. Tier gap in harmful-prompt refusal {{CtlFivePoolTierGap}} → {{CtlFiveEorsTierGap}} points | `runs/controls_report.json` |
| The same contrast at ε = 0.10 | Worst language {{CtlTenMaxBrr}} pp [{{CtlTenMaxBrrLo}}, {{CtlTenMaxBrrHi}}]. Low-resource harmful-prompt refusal {{CtlTenLrlHrr}} pp [{{CtlTenLrlHrrLo}}, {{CtlTenLrlHrrHi}}] | `runs/controls_report.json` |
| Per-language false-firing bound on held-out benign prompts: languages above the predictive band | {{CtlFiveValidAbove}} of {{n_langs}} (ε = 0.05) and {{CtlTenValidAbove}} of {{n_langs}} (ε = 0.10). With detected-language routing: {{CtlFiveValidAboveLid}} and {{CtlTenValidAboveLid}} | `runs/controls_report.json` (gate level) |
| The gate separates XSTest's unsafe prompts from its matched safe ones (AUROC) | {{CtlStyleAuc}}; lowest language {{CtlStyleAucMin}} ({{CtlStyleAucMinLang}}) | `runs/controls_report.json` (gate level) |
| A benign `Good morning.` prefix, at ε = 0.10: mean benign refusal of the repaired model (secondary test) | {{CtlGreetBrrEors}} → {{CtlGreetBrrEorsGreet}} (`EORS-S`) and {{CtlGreetBrrPool}} → {{CtlGreetBrrPoolGreet}} (`CPOOL-S`). Unrepaired: {{GreetBrrEZero}} → {{GreetBrrEZeroGreet}} | `runs/controls_report.json`, `runs/arms_report.json` |
| SafeTune's English keyword refusal detectors (Tiny Aya Global, against the human-validated Gemini refusal labels): highest recall in any language other than English | {{HOneRecallOtherMax}} | `runs/B0/h1.json` |
| Closed comparator (Gemini 3.5 Flash-Lite), safe-response rate: same-family judge vs Qwen3Guard (post hoc) | {{GemGSrr}} vs {{GemQSrr}}. The leniency is {{CtlJudgeDid}} points [{{CtlJudgeDidLo}}, {{CtlJudgeDidHi}}] larger than on Tiny Aya's outputs | `labels/GEM-lite.jsonl`, `runs/controls_report.json` |

Further results, each labelled in the paper:
- **Split redraws (exploratory).** Across {{CtlSplitN}} redraws of the fit/conformal split, the worst-language
  false-firing rate is lower under per-language thresholds in {{CtlSplitDMaxShare}} of draws.
- **Judge agreement with humans.** Refusal: κ = {{RefKappaMean}} (Gemini) and {{QRefKappaMean}} (Qwen3Guard-4B).
  Harm: κ = {{HarmKappaMean}} and {{QHarmKappaMean}}.

**How to read these.** Equal false-refusal rates across languages whose gate-score distributions differ force
unequal protection. Which error to equalise is a deployment decision; the paper reports both at matched budgets.

![Per-language refusal dot plots with Wilson intervals grouped by resource tier: (a) benign prompts refused and (b) harmful prompts refused for unrepaired E0, pooled threshold CPOOL-S, and per-language threshold EORS-S.](assets/fig2_eors.png)

## Contents

| Path | What it is |
|---|---|
| `data/eval.jsonl` | {{n_eval_rows}} evaluation prompts (see *Composition*) |
| `data/xs_unsafe.jsonl` | XSTest's unsafe contrast prompts that pass translation QA ({{n_xs_ids}} of {{n_xs_unsafe_src}}), ×{{n_langs}} languages; a gate diagnostic only, never used for fitting |
| `data/lid_cache.json` | Constrained GlotLID language label for every prompt (reproduces language routing without fastText) |
| `data/curatorkit*/` | CuratorKIT manifests, provenance and checksums for both prompt sets |
| `labels/<run>.jsonl` | One row per (prompt, language) and run: judge labels, response language, generation usage. No prompt or response text |
| `runs/` | Every summary file the paper is generated from: pre-registered reports, per-run configs and per-language tables, gate diagnostics, judge validation, CircuitKIT outputs, timing and library versions. `runs/<run>/colab/` holds that run's scoring job (AuditKIT summary, scoring time); the top level holds generation. Mirrors the code's `runs/` layout |
| `paper/` | The paper PDF, figure PNGs and one provenance JSON per figure (caption, alt text, sources) |
| `SHA256SUMS` | Checksums of every file |
| `LICENSE-DATA`, `NOTICE`, `licenses/`, `CITATION.cff` | Licence, third-party attributions and licence texts, citation |

## Loading

```python
from datasets import load_dataset

prompts = load_dataset("abdullahabtahi/refusal-parity", "prompts", split="train")
labels = load_dataset("abdullahabtahi/refusal-parity", "EORS-S_e05_a100", split="train")  # one config per run
```

To rebuild the code's `data/` and `runs/` layout, run `make data` from the code repository. Open data alone reproduces
`runs/arms_report.json` exactly; `make data GATED=1` reproduces every number in the paper exactly.

## Runs

One config per run. Arms:
- **References:** `B0`/`B1` is Tiny Aya Global with/without its default safety preamble; `E0` is Tiny Aya Earth,
  unrepaired.
- **Activation steering** on Earth:
  - `NEG`: random vectors (control);
  - `CAST-EN`: SafeTune's CAST;
  - `G-EN`, `G-POOL`, `G-CAL`: projection gate, max-F1 thresholds;
  - `EORS`: per-language conformal thresholds;
  - `EORS-S` / `CPOOL-S`: split-conformal per-language / pooled thresholds at a matched budget ε.
- **Weight-space repair:** `RESTA`, SafeTune's ReSta.
- **Closed comparator:** `GEM-lite`, Gemini 3.5 Flash-Lite.

α is the steering strength.

{{runs_table}}

Comparisons never mix hardware: steering arms are paired with `E0_a100`. The Apple-MPS runs (`B0`, `E0`) are the
references for the hardware check in `runs/E0_a100/hardware_check.json`.

## Data fields

**`data/eval.jsonl`**:
- `id`: prompt id, parallel across languages, so one id is one prompt and its translations;
- `lang`: ISO 639-1 code;
- `tier`: `HRL` / `MRL` / `LRL`;
- `kind`: `harmful`, `benign` or `ood`;
- `split`: `calib` or `test`;
- `prompt`;
- `xs_type`: XSTest's prompt category (benign rows);
- `harm`, `scope`: Aya red-teaming's harm categories and global/local scope (`ood` rows).

**`labels/<run>.jsonl`.** Present only where the run carries them:
- `id`, `lang`, `tier`, `kind`, `split`;
- `arm`, `model`, `preamble`.

Judge labels:
- `q3g4b_safety`, `q3g4b_refusal`: Qwen3Guard-Gen-4B, the primary judge. Safety is `Safe` / `Controversial` /
  `Unsafe`; refusal is boolean.
- `q3g_safety`, `q3g_refusal`: Qwen3Guard-Gen-0.6B.
- `gem_harmful`, `gem_refusal`, `gem_blocked`: Gemini 3.5 Flash, broad rubric (the gated responses also carry
  `gem_finish`, its finish reason).
- `kw_regex`, `kw_substr`: SafeTune's English keyword refusal detectors.

Other columns:
- `resp_lang`: constrained-GlotLID language of the response.
- `api_blocked`, `api_finish`: closed comparator only.
- `tok_in`, `tok_out`, `tok_think`: closed comparator only. Generation token usage; the judges' usage is not released.

A missing label means the judge failed or was blocked. The analysis drops such rows; it never counts them as safe.

## Composition and collection

![Study pipeline diagram showing the seven subpackages in src/refusal_parity/ connected to the Lexsi open-source stack (CuratorKIT, CircuitKIT, SafeTune, AuditKIT).](assets/pipeline.png)

**Languages ({{n_langs}}):** {{langs}}. Tiers:
- high-resource: {{hrl_langs}};
- mid-resource: {{mrl_langs}};
- low-resource: {{lrl_langs}}.

**Prompt sets:**
- **Harmful.** MultiJail (MIT): the {{n_harm_ids}} prompt ids present in all {{n_langs}} languages (ids missing a
  language are dropped), {{n_harm_calib}} calibration and {{n_harm_test}} test.
- **Benign.** XSTest's safe prompts (CC-BY-4.0), translated from English with NLLB-200 (distilled 600M). A
  translation id is kept only if it passes QA in every language: constrained GlotLID must detect the target
  language, and the back-translation must reach chrF ≥ {{chrf_min}}. {{n_benign_ids}} of {{n_xs_safe_src}} ids pass:
  {{n_benign_calib}} calibration, {{n_benign_test}} test.
- **Out-of-distribution harmful.** Aya red-teaming (Apache-2.0): {{n_ood_rows}} prompts in {{n_ood_langs}}
  languages ({{ood_langs}}). They measure gate firing out of distribution and are answered by the closed
  comparator; they are not part of the paired end-to-end comparisons.

**Splits and curation.** Calibration and test ids are disjoint and drawn with seed {{seed}}. Ids missing any
language are dropped, so every paired test is exact. CuratorKIT ran schema checks, cleaning, exact deduplication
and provenance logging; its manifests and checksums are in `data/curatorkit*/`.

**Generation.** Tiny Aya runs use greedy decoding with at most 96 new tokens, through AuditKIT's `generate`. The
closed comparator runs on Vertex AI with temperature 0, at most 256 output tokens and the default safety settings
(what a deployer of the closed model gets). Steering arms are fitted
offline from calibration activations only.

**Judges.** Refusal and harm are judged separately, because refusal is not safety.
- Each judge was checked against {{NValidation}} human-labelled PolyGuardPrompts items. Item-level labels are in
  the gated repository.
- The aggregate agreement statistics are in `runs/pgp_val/judge_validation.csv`.

## Uses

**Intended uses:**
- research on multilingual safety evaluation, over-refusal, language fairness of guardrails, and judge validity;
- reproducing and auditing the paper.

**Out of scope:**
- training models to comply with harmful requests;
- certifying any model as safe;
- treating these labels as ground truth in languages where the judges were not validated.

## Limitations and biases

- **Judges.** They are not validated against humans in Bengali, Swahili, Javanese or Vietnamese, and their harm
  agreement is moderate. Harm metrics are therefore reported under two judges.
- **Benign prompts are translated.** Translation artefacts can shift both the gate scores and the judges.
- **Scope.** One model family (Tiny Aya), one decoding setting, and short responses (≤ 96 tokens).
- **Precision.** Per-language effects below a few points are beyond this design's power.
- **Benchmark-prompt calibration.** Calibration on clean benchmark prompts does not survive ordinary framing (the
  greeting result above).

## Reproducibility

- **Seeds:** {{seed}} (data splits, per-language bootstrap), {{boot_seed}} (tier and difference-in-differences
  bootstraps), {{split_seed}} (calibration fit/conformal split).
- **Pinned inputs:** model and dataset revisions are pinned in the code (`src/refusal_parity/common.py`, `REVISIONS`).
- **Library versions:** per run in `runs/**/colab_versions.txt` and `runs/mac_versions.txt`.
- **Hardware:** A100 for all paired comparisons; the A100-vs-MPS difference is reported, never mixed.
- **Checks:** reproduction from these files was verified byte-for-byte before release.

## Licence and attribution

**CC BY-NC 4.0** covers this project's own contributions: the translations, labels,
aggregates, figures and paper. It is inherited from the Tiny Aya outputs and the NLLB translations; model outputs
are also subject to the Cohere Labs Acceptable Use Policy.

Third-party text redistributed verbatim keeps its original licence:
- MultiJail prompts: MIT (text in `licenses/`);
- English XSTest prompts: CC BY 4.0;
- Aya red-teaming prompts: Apache-2.0 (text in `licenses/`).

Judge labels come from Qwen3Guard (Apache-2.0) and from Gemini on Vertex AI (Google Cloud terms). Sources and
attributions are listed in `NOTICE`. The code is Apache-2.0.

## AI use

Implementation and drafting were a collaboration between the author and an AI agent (Claude, Anthropic, via
Claude Code). Independent AI reviewer agents critiqued drafts and code, and every number is generated from the
result files by script.

## Citation

```bibtex
@misc{abtahi2026refusalparity,
  title        = {A Repair Is Only as Fair as Its Gate: Auditing Multilingual Refusal Steering in a Small Multilingual Model},
  author       = {Abtahi, Abdullah Ibn Mahdi},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/datasets/abdullahabtahi/refusal-parity},
  note         = {Version {{version}}}
}
```

## Contact

Abdullah Ibn Mahdi Abtahi, [abdullah.abtahi@ingramin.com](mailto:abdullah.abtahi@ingramin.com). Questions are welcome in this repository's Community tab; please report harmful
content or misuse concerns privately by email.
