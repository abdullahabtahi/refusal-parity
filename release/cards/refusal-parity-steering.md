---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-earth
language: [en, zh, it, vi, ar, ko, th, bn, sw, jv]
tags: [ai-safety, activation-steering, refusal, conformal-prediction, multilingual, interpretability, safetensors, not-for-all-audiences]
extra_gated_heading: "Access to steering artifacts for safety research"
extra_gated_prompt: >-
  These steering vectors and condition directions are the refusal directions of Tiny Aya. Inverted, they can
  suppress refusal. Access is granted for safety and interpretability research only, after manual review. By
  requesting access you agree to use them under CC BY-NC 4.0 and the Cohere Labs Acceptable Use Policy, and never
  to use them to remove or weaken a model's safety behaviour outside a research setting.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use:
    type: select
    options:
      - Safety or steering research
      - Interpretability research
      - Reproducing the paper
      - label: Other
        value: other
  Briefly describe your intended use: text
  I agree not to use these artifacts to remove or weaken safety behaviour outside a research setting: checkbox
  I agree to the CC BY-NC licence and the Cohere Labs Acceptable Use Policy: checkbox
extra_gated_button_content: "Request access"
---

# Refusal Parity: steering artifacts (gated)

**Abdullah Ibn Mahdi Abtahi** · Ingramin · v{{version}} · open data, findings and paper:
[`abdullahabtahi/refusal-parity`](https://huggingface.co/datasets/abdullahabtahi/refusal-parity)

> ⚠️ **Dual use.** The steering vectors and condition directions encode Tiny Aya's refusal behaviour. Added, they
> increase refusal; subtracted, they can suppress it. The repository is gated for safety research and access is
> reviewed manually.

This repository holds every steering arm exactly as the paper's runs applied it to
[Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth). For each arm it holds:
- the steering vectors per behaviour layer;
- the gate's condition direction and thresholds;
- the calibration split ids.

It also holds the activations that the arms were fitted from and that the gate-level analyses read. The exported
thresholds and layers were checked against each run's `arm_config.json` before release.

## Model details

- **What they are:** activation-steering interventions, not model weights. They are applied at inference on top of
  `CohereLabs/tiny-aya-earth`, built on SafeTune's `CASTModel`.
- **Steering vectors:** contrastive activation addition (difference of means between harmful and benign
  calibration prompts). They are added at layers {{behaviour_layers}}, scaled by SafeTune's strength α: α = 1 in
  the main runs; the α = 0.5 replication uses the same vectors at half strength.
- **Gate:** fires when a prompt's last-token activation, projected on the condition direction at the condition
  layer, exceeds the threshold. SafeTune's CAST cosine-tanh score is used for `CAST-EN` only.
- **Thresholds:**
  - max-F1 for `G-*` and `CAST-EN`;
  - per-language conformal (1 − ε) benign quantile for `EORS` (direction fitted on the same calibration data, so the
    split-conformal guarantee does not apply) and per-language split-conformal for `EORS-S`;
  - one pooled split-conformal threshold for `CPOOL-S`.
- **Split-conformal arms:** `EORS-S` and `CPOOL-S` take their condition direction and layer from a fit fold (1/3 of
  calibration ids, seed {{split_seed}}) and their thresholds from the disjoint conformal fold.

## Files

| Path | What it is |
|---|---|
| `arms/<arm>.safetensors` | `steer.layer<l>`: steering vector per behaviour layer, float32, shape (d,). `condition`: unit condition direction, for gated arms |
| `arms/index.json` | Per arm: gate type, threshold rule, condition layer, behaviour layers, thresholds (a number, or one per language), comparator, ε, α, base model. `_split`: fit/conformal ids per prompt kind |
| `activations/mech_earth/calib_acts.pt` | Last-token residual stream of Tiny Aya Earth on the calibration prompts, every layer: `{"harmful|en": (N, layers, d) fp16, ...}`. The arms are fitted from this |
| `activations/mech_global/*.pt` | The same for Tiny Aya **Global** (not Earth), calibration and test prompts; used by the mechanistic analyses, not by the arms |
| `activations/gate_acts_earth*/` | Condition-layer activations of the test prompts, plus `rows.jsonl` with the prompts, under generation-style batching (gate-level validity, style and calibration-shift tests) |
| `SHA256SUMS` | Checksums of every file |

Load the tensors safely, with no pickle execution:

```python
import json, torch
from safetensors.torch import load_file

arm = load_file("arms/EORS-S_eps0.05.safetensors")  # {"steer.layer12": ..., "condition": ...}
meta = json.load(open("arms/index.json"))["EORS-S_eps0.05"]
acts = torch.load("activations/mech_earth/calib_acts.pt", weights_only=True)
```

The code repository does not read these files: `src/refusal_parity/generation/run_arm.py` refits each arm from
`activations/mech_earth/calib_acts.pt` with `src/refusal_parity/steering/arms.py` and applies it with `src/refusal_parity/steering/steer.py`. The safetensors are
exported from that same fit, so they equal what the runs applied, for use outside the code.

## Intended use

**Intended uses:**
- research on conditional steering, language fairness of gated interventions, and conformal calibration of gates;
- interpretability of refusal directions across languages;
- reproducing the paper's gate-level results.

**Out of scope and prohibited:**
- deploying these arms as a safety mechanism. The paper shows calibration on benchmark prompts does not survive
  ordinary framing: a "Good morning." prefix raises the mean benign refusal of `EORS-S` at ε = 0.10 from
  {{CtlGreetBrrEors}} to {{CtlGreetBrrEorsGreet}}.
- inverting or subtracting the vectors to weaken refusal outside a controlled research setting.

## Evaluation

The evaluation is in the paper and the open repository.

**At a matched false-refusal budget ε = 0.05**, `EORS-S` lowers worst-language benign refusal by {{CtlFiveMaxBrr}}
pp against `CPOOL-S`. It costs {{CtlFiveLrlHrr}} pp of harmful-prompt refusal in the low-resource tier.

**Gate-level validity.** The per-language false-firing bound is not contradicted on held-out prompts:
{{CtlFiveValidAbove}} of {{n_langs}} languages fall above the predictive band.

## Caveats

- **Scope:** fitted on one model (Tiny Aya Earth), at 3.35B parameters.
- **Thresholds are prompt-distribution-specific.** They depend on the calibration prompt distribution (MultiJail
  and translated XSTest) and on the batching used to compute activations. The calibration-shift diagnostic is in
  `runs/controls_report.json`.
- **Per-language thresholds route on language identity.** At inference they use constrained GlotLID; an unseen
  language falls back to the highest threshold, at which the gate fires least.

## Licence

CC BY-NC 4.0, as derived from Tiny Aya (CC BY-NC 4.0, Cohere Labs Acceptable Use Policy). See `NOTICE`.

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
