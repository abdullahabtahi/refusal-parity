# Premortem and forecast — Refusal Parity

*Written 2026-10-26, after the smoke test and before any evaluation data. The exercise: it is Nov 22, the paper was rejected and the hackathon entry didn't place. Why?*

## Evidence in hand

- Smoke test: Tiny Aya Global refuses clear-cut harmful prompts in English, Swahili and Bengali under **all three** preamble conditions, and answers the benign homonym prompt. Strong safety, consistent with Cohere's 91% MultiJail number.
- Throughput: **about 0.6 responses/s** (52 tok/s at batch 32). The default preamble adds 373 prompt tokens to every request.
- **Novelty check (critical):** Aziz, Hanif and Koto (Findings EMNLP 2026, arXiv:2606.01196) already train a gate on high-resource activations and **calibrate its threshold per target language** from a few examples. Their selectivity is 54.5 vs 33.6 for CAST, on 23 languages and 3 models (7–9B).

## Failure modes, ranked by likelihood × impact

| # | Failure mode | Likelihood | Impact | Mitigation (now in the plan) |
|---|---|---|---|---|
| 1 | **Novelty collapse.** "EORS is Aziz et al. with a different threshold rule." | High (already true for the bare idea) | Fatal for a method paper | Reframe: the paper is an **audit of language-fair repair under drift**. The method contribution narrows to *FPR-constrained calibration with a finite-sample guarantee*: the per-language quantile threshold is split-conformal, so P(benign false fire) ≤ ε + 1/(n_ℓ+1) per language. Add an **Aziz-style per-language max-F1 threshold (G-CAL)** as the baseline EORS must beat on max-BRR parity. |
| 2 | **Ceiling / no headroom.** Global is ~91% safe, so the steering arms differ by less than the CIs (±4–5 pp at n=209). | High on Global, medium on Earth | Null results everywhere | (a) Make **Earth** the primary repair testbed (worst language ~78%). (b) Add MultiJail's **"intentional" scenario** (English jailbreak template plus a target-language request, as in Deng et al. 2024) to create headroom. (c) Pool harmful sets; bootstrap the worst-language statistic. |
| 3 | **Judge invalidity in LRLs.** Qwen3Guard-4B mislabels sw/jv/bn, so every per-language number is suspect. | Medium | High: undermines all claims | Two judges plus translate-then-judge; per-language agreement; a stratified ~200-item audit (labelled as Claude annotation, not native speakers); headline claims only where judges agree. |
| 4 | **Compute.** 0.6 resp/s × ~3,150 prompts ≈ 90 min per condition × ~14 conditions, plus judging, comes to well over 24 h. | Certain without changes | Late or partial results | 96-token generations (refusals are decided in the first tokens per the shallow-alignment literature; the judge reads the first sentences). **Activation-level proxies first:** gate TPR/FPR per language from forward passes alone predicts each arm's parity before any generation. Generation is reserved for the arms that matter. Optional: the user's free Colab/HF credits. |
| 5 | **Steering breaks a 3B model.** Incoherent text, or refusals switching to English. | Medium | Medium; reportable | α capped by the benign-perplexity criterion; language-fidelity metric; this is itself a finding about small models. |
| 6 | **LID dependency.** EORS needs the prompt's language; code-mixed or romanized prompts get the wrong threshold. | Medium | Medium | Strictest-threshold fallback; evaluate on Aya red-teaming OOD; report the LID error rate and its effect. |
| 7 | **Benign-set artefacts.** NLLB can drop the "scary" word (e.g. *terminate* → *stop*), underestimating LRL over-refusal. | Medium | Low–medium | Report results by XSTest type; flag homonym items; limitations section. |
| 8 | **H1 judged trivial.** "Of course English regexes miss Swahili refusals." | High | Low | Make H1 consequential: show the English detector **changes SafeTune's chosen steering layer** and AuditKIT's verdicts, not just that recall is low. |
| 9 | **CircuitKIT infeasible on 16 GB.** TransformerLens fp32 would need about 13.4 GB plus activations. | Medium–high | Hackathon: lose one repo | Try bf16 on MPS with small n; otherwise run on Colab T4. The pipeline still chains 3 repos without it. |

## Forecast (my calibrated guesses)

| Outcome | P |
|---|---|
| H1: multilingual-vs-English detector disagreement large in LRLs (κ gap ≥ 0.2) | 0.90 |
| H2: preamble changes harmful compliance significantly in ≥ 1 tier | 0.45 |
| H3: Earth's deficit is mainly an *action* failure (AUROC preserved) | 0.60 |
| H4: single-threshold steering leaves a worst-language gap and loads over-refusal onto LRLs | 0.65 |
| H5: EORS beats a single threshold on worst-language SRR at bounded BRR | 0.50 (Global) / 0.60 (Earth) |
| EORS beats Aziz-style G-CAL on max-BRR parity | 0.40 |
| No capability loss on Belebele (< 1 pp) for gated arms | 0.85 |

## Honest venue assessment

- **As scoped (one model family, 10 languages, machine judges, M1 compute):** a strong **NeurIPS/ICLR workshop paper** (SoLaR, Safe GenAI, MRL). P(accept) ≈ 0.6–0.7. **NeurIPS main track ≈ 0.10–0.15**: reviewers would cite Aziz et al. and BabelSteering and ask for more models and human labels.
- **What would lift it to main-track range (≈ 0.3):** ≥ 3 model families or sizes, native-speaker labels for 3 languages, induced-drift fine-tunes with 3 seeds, the conformal guarantee proved and validated empirically, and the AIM jailbreak setting.
- **Hackathon:** strong position. All 4 Lexsi repos chained, real before/after metrics, an evidence-based critique of the tools (the organisers asked for feedback), and rigour above typical entries. The main risk is presentation, not substance.

## Decisions taken from this premortem

1. Reframe the title and contributions around the **audit** (measurement, drift, repair parity); EORS becomes a principled variant, not the headline.
2. Add arms **G-CAL** (Aziz-style per-language max-F1 threshold) and the **MultiJail intentional (AIM) scenario**.
3. Run order: activations → gate-parity proxies → baselines → only the decisive generation arms, with Earth as the primary repair testbed.
4. max_new_tokens = 96.
