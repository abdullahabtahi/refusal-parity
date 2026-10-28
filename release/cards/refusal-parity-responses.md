---
license: cc-by-nc-4.0
pretty_name: "Refusal Parity: model responses (gated)"
language: [en, zh, it, vi, ar, ko, th, bn, sw, jv, es, fr, hi, ru, sr, tl]
multilinguality: multilingual
task_categories: [text-generation, text-classification]
size_categories: [10K<n<100K]
tags: [ai-safety, refusal, over-refusal, multilingual, red-teaming, llm-as-a-judge, evaluation, not-for-all-audiences]
extra_gated_heading: "Access to model responses for safety research"
extra_gated_prompt: >-
  This dataset contains responses of language models to harmful requests, and some responses are harmful. Access
  is granted for safety research only, after manual review. By requesting access you agree to use the data under
  CC BY-NC 4.0 and the Cohere Labs Acceptable Use Policy, and not to use it to train or tune models to comply with
  harmful requests or to redistribute harmful responses outside a research setting.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use:
    type: select
    options:
      - Safety evaluation research
      - Judge or classifier validation
      - Reproducing the paper
      - Teaching
      - label: Other
        value: other
  Briefly describe your intended use: text
  I agree to use this data for safety research only and not to train models to comply with harmful requests: checkbox
  I agree to the CC BY-NC licence and the Cohere Labs Acceptable Use Policy: checkbox
extra_gated_button_content: "Request access"
{{configs}}
---

# Refusal Parity: model responses (gated)

**Abdullah Ibn Mahdi Abtahi** · Ingramin · v{{version}} · open data, findings and paper:
[`abdullahabtahi/refusal-parity`](https://huggingface.co/datasets/abdullahabtahi/refusal-parity)

> ⚠️ **Content warning.** This repository holds model responses to harmful requests, and some responses are
> harmful. It is gated for safety research and access is reviewed manually.

This repository holds the full judged records behind the open labels in `refusal-parity`: the prompt, the model's
response and every judge label, one row per (prompt, language) and run. Use it to:
- re-judge responses with another judge;
- audit individual refusal or harm decisions;
- reproduce the paper exactly, with `make data GATED=1` in the code repository.

## Contents

| Path | What it is |
|---|---|
| `responses/<run>.jsonl` | {{n_resp_runs}} runs; one config each |
| `judge_validation/*.jsonl` | The {{NValidation}} PolyGuardPrompts items (CC BY 4.0) behind the judge-validation statistics. `responses.jsonl`: items with their human labels. Judge labels on the same items: `judged.jsonl` (Qwen3Guard-4B and 0.6B, SafeTune detectors, and a merged copy of the Gemini labels), `gemini_broad_judged.jsonl` (Gemini 3.5 Flash, broad rubric: the study's Gemini judge), `gemini_judged.jsonl` (Gemini 3.5 Flash, strict rubric), `gemini38_judged.jsonl` (Gemini 3.8 Flash, strict rubric; a partial run, reported in the validation table only) |
| `SHA256SUMS` | Checksums of every file |
| `LICENSE-DATA`, `NOTICE` | Licence and third-party attributions |

**Fields.** These are the label fields documented in the open repository's card, plus `prompt` and `response`.
- `tok_*` is the closed comparator's generation usage; the judges' own usage is not released.
- A missing label means the judge failed or was blocked; it is never counted as safe.

**Runs.** The arms and their settings are listed in the open repository's card. Responses are greedy, with at
most 96 new tokens for Tiny Aya.

## Uses

**Intended uses:**
- safety-evaluation research;
- validating judges and classifiers, especially in low-resource languages;
- reproducing and auditing the paper.

**Prohibited uses:**
- training or tuning models to comply with harmful requests;
- extracting harmful content for use outside research;
- redistributing harmful responses without equivalent gating.

## Limitations

The judge labels share the open repository's limitations:
- they are not validated against humans in Bengali, Swahili, Javanese or Vietnamese;
- harm agreement is moderate (κ = {{QHarmKappaMean}} for the primary judge);
- benign prompts are machine-translated;
- the study covers one model family.

## Licence and attribution

The data are released under **CC BY-NC 4.0**, inherited from the Tiny Aya outputs and the NLLB translations, and
are subject to the Cohere Labs Acceptable Use Policy. The closed comparator's responses come from Gemini 3.5
Flash-Lite on Vertex AI, under the Google Cloud Service Terms and the Generative AI Prohibited Use Policy. Third-party
text keeps its original licence: MultiJail prompts (MIT), XSTest prompts and PolyGuardPrompts items (CC BY 4.0) and
Aya red-teaming prompts (Apache-2.0); licence texts are in `licenses/` and sources in `NOTICE`.

## AI use and citation

Implementation and drafting were a collaboration between the author and an AI agent (Claude, Anthropic, via
Claude Code). Independent AI reviewer agents critiqued drafts and code, and every number is generated from the
result files by script.

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
