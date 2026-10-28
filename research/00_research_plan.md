# Refusal Parity — preregistered research plan

*Frozen 2026-10-26, before any model output was observed. Deviations go in §9 with a date and a reason.*

## 1. Problem

Small multilingual models such as Tiny Aya (3.35B, 70+ languages) are built for on-device and offline use, often in low-resource language (LRL) communities where no external guardrail runs. Their safety is (a) uneven across languages, (b) eroded by regional specialisation (Earth/Fire are ~10 pp worse than Global in their worst MultiJail language, per the Tiny Aya report), and (c) **measured with English-centric tools**: every component of the Lexsi stack detects refusal with English substrings/tokens, and the strongest small multilingual judge (PolyGuard) does not cover bn, sw, jv or vi.

The open gap (research/02_literature.md §C, synthesis): safety *repair* is validated on mean or English metrics. Nobody reports whether repair is **language-fair**, i.e. whether it lifts the worst language without loading over-refusal onto LRL users.

## 2. Research questions and hypotheses

| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 measurement | English keyword refusal detection agrees with multilingual judges much less in LRLs than in English (Cohen's κ gap ≥ 0.2), and the direction of the error changes the measured safety ranking of languages. | κ gap < 0.1, or rankings unchanged. |
| H2 preamble | Removing Tiny Aya's default "contextual safety mode" preamble raises harmful compliance, more in LRLs than in high-resource languages (HRLs). | No significant increase, or the increase is uniform across tiers. |
| H3 drift = action failure | Earth's worst-language deficit vs Global is an *action* failure: harm-direction AUROC per language is preserved (≤ 0.03 drop), while projection onto the refusal direction shifts down. | AUROC drops > 0.05 in the deficit languages. |
| H4 repair is unfair | A single English-derived refusal direction with one global gate threshold improves mean harmful-refusal but leaves a max–min gap and adds the most benign over-refusal in LRLs. | The gap closes and benign refusal is flat across tiers. |
| H5 method | **Equalised-odds refusal steering (EORS)**, a pooled multilingual direction with per-language gate thresholds set for equal benign false-refusal rate ε, reduces the harmful-refusal max–min gap vs H4's baseline, keeps every language's benign refusal ≤ baseline + ε, and costs ≤ 1 pp capability. | Gap reduction not significant (paired bootstrap, Holm), or any language's benign refusal exceeds the bound. |

## 3. Models

`CohereLabs/tiny-aya-global` (primary), `CohereLabs/tiny-aya-earth` (natural-drift arm). bf16 on MPS, greedy decoding, max 128 new tokens. The chat template is applied in every condition, with the default preamble either on (default) or stripped (H2 arm). One model on disk at a time (disk budget about 16 GB).

## 4. Data (built with CuratorKIT; provenance manifests kept)

- **Harmful:** MultiJail (`DAMO-NLP-SG/MultiJail`), 315 parallel prompts × 10 languages. Tiers: HRL en, zh, it, vi; MRL ar, ko, th; LRL bn, sw, jv. Split by prompt id (parallel across languages): 105 calibration / 210 test.
- **Benign but scary (over-refusal):** XSTest safe prompts (250), translated en → 9 languages with NLLB-200-distilled-600M. Filter with fastText LID and back-translation chrF; split 90 calibration / 160 test.
- **Out-of-distribution harmful:** `CohereLabs/aya_redteaming`, native-written, ar and en only overlap with MultiJail; also hi, fr, es, ru, sr, tl. Used only at test time, to check that the gate generalises beyond MultiJail phrasing.
- **Capability:** Belebele (log-likelihood multiple choice, 100 items per language for the 10 languages) and Global-MMLU-Lite where the language is covered.

## 5. Interventions

| Arm | Description |
|---|---|
| B0 | Global, default preamble |
| B1 | Global, preamble stripped (H2) |
| E0 | Earth, default preamble (H3) |
| S-EN | ActAdd with the English diff-in-means refusal direction (SafeTune Steer), unconditional |
| G-EN | Gated: English direction, English harm-score gate, single global threshold (CAST-style) |
| G-POOL | Gated: pooled multilingual direction and gate, single global threshold (ablation) |
| **EORS** | Pooled direction and gate, per-language thresholds τ_ℓ = (1−ε) quantile of the calibration benign harm-scores in language ℓ; language from fastText LID at inference |

Each steering arm is applied to Global and to Earth. The layer is chosen from CircuitKIT/SafeTune layer suggestions and the calibration split only. α is chosen on calibration only, by the largest α whose benign-calibration perplexity rises < 10%.

## 6. Metrics (per language ℓ, test split only)

- HRR_ℓ: harmful refusal rate (judge says the response is not harmful). ASR_ℓ = 1 − HRR_ℓ.
- BRR_ℓ: benign refusal rate (over-refusal).
- Selectivity Δ_ℓ = HRR_ℓ − BRR_ℓ.
- **Parity:** worst-language HRR, max–min HRR gap, max BRR.
- Language fidelity: share of responses in the prompt's language (LID).
- Capability: Belebele accuracy.
- Overhead: tokens/s.

**Judges:** J1 Qwen3Guard-Gen-0.6B on the native response (all languages); J2 PolyGuard-Qwen-Smol (native where covered); J3 translate response to English with NLLB, then PolyGuard. Primary label = J1 ∧ J3 agreement; disagreements are reported. A Claude-annotated audit of a stratified 20 items per language estimates judge accuracy, and is labelled as such (it is not a native-speaker audit).

## 7. Analysis plan

Wilson 95% CIs per language. Paired comparisons on identical prompts use McNemar (binary) and paired bootstrap (10k) for gap statistics, with Holm correction across the 10 languages. Calibration-split randomness: 5 random calibration/test resplits, reporting mean ± sd for EORS thresholds. Effect sizes are reported in pp. MultiJail's 210 test prompts per language give 80% power for about a 9 pp paired change at an 85% baseline.

## 8. Lexsi pipeline (≥3 repos, here 4)

CuratorKIT (ingest, clean, dedup, LID gate, provenance) → CircuitKIT (refusal circuit on Tiny Aya; layer suggestions; cross-lingual edge overlap) → SafeTune (refusal-direction extraction, ActAdd steering, CircuitKIT adapter) → AuditKIT (evaluation spine with a multilingual guard judge metric; red-team probes).

**Contribution back to Lexsi:** a multilingual refusal detector, and bug reports for AuditKIT's per-probe detector routing.

## 9. Deviations log

- **2026-10-26, before any model output.** Reading the chat template showed that the default "contextual safety mode" preamble explicitly *permits* information about violence, hate, misinformation and sex. H2 is therefore made **two-sided**: the preamble could raise or lower harmful compliance.
- **Added arm B2 (strict preamble):** the default preamble with its contextual-mode paragraph replaced by Cohere's documented STRICT safety-mode text (docs.cohere.com/docs/safety-modes). This is a zero-cost prompt-only repair that every steering method must beat.
- **LID: fastText lid.176 replaced by constrained GlotLID.** lid.176 labelled short Swahili and Javanese strings as English; unconstrained GlotLID picked near-variants (swc, syl, nan). Prediction is now restricted to the study's languages, and the translation LID gate is a label match.
- **Primary judge is Qwen3Guard-Gen-4B.** A smoke test showed the 0.6B model missing a Swahili refusal and flagging a benign "kill a Python process" answer as Unsafe; the 0.6B is kept as a secondary judge.
- **Premortem (research/05_premortem.md).** Aziz et al. 2026 already calibrate gate thresholds per target language, so EORS's contribution narrows to FPR-constrained, conformal-style calibration.
  - Added arm **G-CAL**: a per-language max-F1 threshold, in the style of Aziz et al.
  - Added the **MultiJail intentional (AIM) scenario**, for headroom.
  - Earth becomes the primary repair testbed.
  - max_new_tokens is 96, for compute (about 0.6 resp/s on M1).
- **Blindspot review (DeepMind-reviewer lens).** Added:
  - **Controls:** a positive control (refusal-direction ablation must raise ASR sharply) and a negative control (random direction of equal norm must not).
  - **Adaptive attacks on gated arms:** LID spoofing (an English sentence prepended to a target-language request) and intra-sentence code-mixing.
  - **Sampled ASR:** T=0.7, 5 samples, on a 30-id × 10-language subset.
  - **Generation quality on benign prompts:** judge helpfulness plus language fidelity.
  - **Requested from the user (optional):** native-speaker labels for bn/sw/jv, a frontier-API judge, GPU credits.
- **Judge validation on human labels (PolyGuardPrompts).** 200 parallel ids × 7 languages, stratified.
  - Gemini 3.5-flash detects refusals at κ≈0.95 in every language (3.8-flash scores 0.93 and is ~5× slower), so 3.5-flash is the Gemini judge.
  - With a "serious harm only" rubric it missed ~73% of human-harmful responses: a definition mismatch, since the human labels cover stereotypes, privacy, threats and role-play compliance.
  - The rubric is now **broad** (WildGuard taxonomy); the strict rubric is kept as a sensitivity analysis.
  - The rubric was calibrated on validation data only, never on the study's test set.
- **Primary metrics are now refusal-based.** HRR is harmful-prompt refusal, BRR is benign refusal, and selectivity Δ = HRR − BRR, as in Aziz et al.
  - Validation shows refusal labels are robust: κ 0.93 vs humans, and 97.8% accuracy on unanimous items in all 7 languages.
  - Harm labels are contested: κ 0.43 with the broad rubric; on items where humans split 2–1 the judge disagrees 90% of the time.
  - SRR (harm-based) is kept as secondary under two judges, and H4/H5 are evaluated on HRR, BRR and selectivity.
- **Correction to the entry above.** The Gemini Flash-Lite comparator refuses only 46–58% of MultiJail prompts yet is harmful in 0–2%: it gives *safe completions*, so refusal ≠ safety.
  - Outcomes are now 3-way: refuse / safe compliance / harmful compliance.
  - **Primary safety metric:** SRR = 1 − harmful-compliance rate, reported under each judge (Gemini, Qwen3Guard-4B) and their union, because harm labels are contested.
  - **Primary over-refusal metric:** BRR.
  - HRR and selectivity are reported as behaviour metrics that explain how safety is achieved.
  - H4/H5 are evaluated on SRR (worst-language, gap) and BRR (max).
- **Gate score.** A synthetic unit test (tests/test_arms.py) found that SafeTune's faithful CAST similarity `cos(h, tanh(v(v·h)))` is sign-dependent. On languages whose activations shift downward it inverts (AUROC 1.0 → 0.0), while the plain projection h·v stays at 1.0.
  - G-EN, G-POOL, G-CAL and EORS now use a projection gate (steer.GateModel, a subclass of SafeTune's CASTModel).
  - Added a **CAST-EN** arm (faithful SafeTune CAST, comparator by F1) so the effect is tested on real data. **New hypothesis H6:** CAST-EN's gate fires less on harmful prompts in low-resource languages than G-EN's projection gate does.
- **Gemini on Vertex** (user's expiring GCP credits). Roles: third judge, and closed-model comparator (3.5-flash-lite as the assistant with default safety settings), for the parity reference and ROI.
- **Evaluation spine:** the generation arms run through AuditKIT `evaluate()` with the validated judges as custom metrics.
- **Added arm R-ReSta for Earth:** disk is now 36 GB, so Base, Global and Earth fit. The arm is Earth + α(Global − Base), α ∈ {0.1, 0.25, 0.5}, through SafeTune's ReStaTrainer.
- **2026-10-27, compute and scoring venue (no change to measurement).** Qwen3Guard scoring moved from the Mac to a Colab T4 (`scripts/colab_score.sh`) with the same `audit.py`, bf16, batch 4. Language ID can't be built there, so it is added locally afterwards (`fill_lang.py`, the same constrained GlotLID). All arms are scored this way, so B0 and E0 are comparable.
- **2026-10-27, CircuitKIT on Kaggle (T4×2).** Tiny Aya plus its TransformerLens copy don't fit free Colab (see STATUS). Parity gate run first, before any discovery output was seen: max KL 3.6e-5, which passes.
  - **Discovery settings fixed before the first discovery output:**
    - EAP-IG, node level, 32 calibration pairs per language (the same ids across en/sw/bn), harmful_vs_benign pairing, default chat template;
    - answer tokens are B0's most frequent single-token refusal and compliance openers: en I'm/To, sw Sam/Katika, bn আমি/প. The generic 'K' and 'আ' are excluded because they open both refusals and compliances.
    - batch size 1, forced by GPU memory (the v3 OOM).
  - Discovery results are **exploratory** (no preregistered circuit hypothesis); cross-language overlap is descriptive.
- **2026-10-27, harm trend reported per judge.** On B0 the tier trend in harmful compliance differs by judge: flat under Qwen3Guard, rising under Gemini. As preregistered, SRR is reported under each judge. A trend is claimed only where both judges agree; otherwise it is reported as unresolved.
- **2026-10-27 ≈07:00, alpha-selection rule fixed before any sweep output.** The sweep is S-POOL on Earth, calibration split, the first 20 ids per (kind, language), α ∈ {0, 0.5, 1, 2, 4}, with α=0 as the same-subset baseline (`scripts/alpha_sweep.sh`). "Coherent and on-language" is operationalised relative to α=0: on-language ≥ baseline − 0.03, mean distinct character 4-gram ratio ≥ baseline − 0.05 (script-agnostic; word bigrams fail for th/zh), and empty ≤ 2%. Pick the largest admissible α (`src/alpha_pick.py`). Gemini refusal and harm are reported, never used to pick.
- **2026-10-27 ≈07:55, H3 sensitivity analysis (post hoc).**
  - **Why:** raw projection gaps are not comparable across two models with different activation scales. I added a standardised margin (Cohen's d of harmful vs benign projections) to `src/h3.py`, after seeing the provisional raw-gap result, so it is labelled post hoc.
  - **Provisional result (Gemini-judged deficit languages: bn, sw):** the raw-gap criterion passes; detection is preserved (AUROC drop ≤ 0.000); the raw gap shrinks in every language by a similar amount.
  - **But** the standardised margin in bn and sw *increases* (+0.10, +0.13).
  - **Reporting rule:** H3 is reported as "detection preserved; the margin loss is not language-specific once scale is controlled". The action-failure mechanism is not claimed from this read-out. The final verdict waits for the Qwen3Guard-judged deficit languages.
- **2026-10-27 ≈08:35, CircuitKIT discovery runs under the no-preamble condition (B1), before any discovery output.**
  - Node-level EAP-IG at the default template's ~460 tokens needs ~7 GB of working memory on top of the 7.45 GB model: 14.3 GB measured, which a 15 GB T4 cannot hold even at batch 1. This was shown in a fresh process with memory diagnostics (Kaggle v9), after the per-head einsum patch removed a separate 3.3 GiB spike (v5).
  - Without the preamble, prompts are ≤104 tokens. Same ids, answer tokens and settings otherwise.
  - The circuits therefore describe refusal in the B1 condition, not under the default template.
  - Answer tokens were measured on B0 (with the preamble); a B1 first-token check follows when B1 is generated.
  - A default-preamble replication needs a ≥24 GB GPU (e.g. Colab Pro L4); this is optional and the owner's call (it costs money).
- **2026-10-27 09:15, α chosen by the pre-fixed rule: α = 1.0.** On the calibration subset (`runs/alpha_sweep.json`), α ∈ {0, 0.5, 1} are admissible (on-language 0.99; distinctness 0.83–0.86) and α ∈ {2, 4} are not (on-language 0.60 / 0.10). Reported, not used to pick: unconditional S-POOL at α = 1 refuses 98% of harmful and 98% of benign prompts (α = 0.5: 93% / 60%), so in gated arms the gate alone determines selectivity.
- **2026-10-27 11:40, judge input clipping (fixes a measurement failure; arms unaffected).** `judge.generate` truncated each templated chat from the right at 2,048 tokens. On over-long inputs that cut the generation prompt, so Qwen3Guard emitted no label: 163/1,400 pgp_val rows, whose human-labelled responses run up to 41k tokens (Hindi most). `judge.clip_response` now keeps the head of the *response* so the whole chat fits; responses that already fit are unchanged. Every Tiny Aya arm is unaffected (E0 max 1,772 tokens; 0 rows changed; `tests/test_judge_clip.py`). The pgp_val validation therefore scores the first ~1.9k tokens of very long responses, while humans read them in full. Reported as a limitation of the validation.
  - The validation set is scored at judge batch 1 (`JUDGE_BATCH=1 scripts/kaggle_score.py push pgp_val`), because 2k-token inputs at batch 4 exceed a T4. Greedy labels do not depend on batch size, apart from padding numerics; arms stay at batch 4.
- **2026-10-27 11:45, E0 final (both judges) and H3 final.** Reported exactly as the 07:55 reporting rule fixed: deficit languages bn, en, sw (union of the top-2 SRR drops per judge); AUROC drop ≤ 0.009, so the preregistered criterion holds; Cohen's d rises in bn/sw. The paper says "detection preserved, margin loss uniform". The figure for E0 is a paired-difference dot plot (`figures.py drift`), not the generic dumbbell, because a 1-pp effect is invisible on a 0–1 rate axis.
- **2026-10-27 12:55, circuit analysis = faithfulness + cross-language transfer, fixed before any faithfulness output.** The default-preamble EAP-IG circuits for en/sw/bn exist (Colab A100, peak 24.0–24.4 GiB, parity KL 3.1e-5). This makes the 08:35 no-preamble deviation moot: discovery ran on the faithful condition.
  - **Why the change:** Hanna, Pezzelle & Belinkov 2024 (arXiv 2403.17806) show that node overlap is weak evidence of shared mechanism. Jaccard/Spearman overlap is kept as a descriptive companion only.
  - **Metric:** CircuitKIT pillar 1 (causal patching), F = (y_circuit − y_corrupt)/(y_clean − y_corrupt). y is the refusal-minus-compliance logit difference; corrupt = the paired benign prompt; every out-of-circuit node is patched from the corrupt run.
  - **Held-out prompts:** the next 32 calibration ids per kind (sorted ids 33–64), never used for discovery. The test split stays reserved for the steering arms.
  - **Grid:** source circuit ∈ {en, sw, bn} × target language ∈ {en, sw, bn} × circuit size ∈ {1%, 5%, 10%, 20%} of the 612 nodes (`target_sparsity` 0.99/0.95/0.90/0.80, scope both).
  - **Baseline:** CircuitKIT's baselines pillar (random/magnitude circuits) on the diagonal at 5%.
  - **Reading rule (fixed now):** a circuit "transfers" to language B if F(A→B) is within 0.1 of F(B→B) at the same size. Cross-language F well below the diagonal means language-specific circuitry. Exploratory: there is no preregistered circuit hypothesis.
- **2026-10-27 13:25, circuits redone with token-length-matched pairs, fixed before any new output.**
  - **What the first faithfulness grid showed:** F ≤ 0.065 at ≤20% of nodes in every cell; random circuits of equal size do as well (z 0.4–0.6).
  - **Diagnosis:** only 0–2 of 32 harmful/benign pairs per language had equal token length (median offset 4–7.5 tokens). CircuitKIT left-pads pairs, so the ~350-token preamble *before* the user turn was shifted, and position-wise EAP-IG differences and patching compared different tokens.
  - That first result is reported as "unaligned pairs give unfaithful circuits", not as a property of Tiny Aya.
  - **New pairing:** within each exact rendered-length bucket (default preamble), harmful and benign calibration prompts are paired in id order. Pairs sorted by harmful id: the first 32 are discovery, the next 26 evaluation (26 = the smallest pool across en/sw/bn, so n is equal). Ids are no longer parallel across languages; each circuit is evaluated on each language's own held-out pairs.
  - **Sizes:** keep 1/5/10/20/40/60/100% of nodes. The 100% point must give F = 1 (pipeline sanity check).
  - **Reading rule:** unchanged (transfer if F(A→B) is within 0.1 of F(B→B)), applied only at sizes where the diagonal F ≥ 0.5. Below that, the circuit isn't faithful enough for a transfer reading.
- **2026-10-27 14:50, scope refocused (owner's decision, after an honest value review).**
  - **Core claims:**
    1. the CAST sign defect, tested end to end (CAST-EN vs the projection-gated arms, Earth, A100, paired against E0_a100);
    2. measurement validity: English-only detectors (H1), judge validation (Qwen3Guard harm κ 0.74 vs Gemini 0.43), refusal ≠ safety;
    3. EORS redistributes rather than removes error.
  - **Reported briefly, with a power statement:** natural drift (E0 vs B0) and H3.
  - **CircuitKIT stopping rule:** if the length-matched circuits don't reach diagonal faithfulness F ≥ 0.5 at ≤ 40% of nodes, discovery is reported as a tooling appendix (memory, alignment pitfall, the degenerate SafeTune layer hand-off) and no circuit claims are made.
  - **Deferred, not run:** B1/B2 preamble arms, adaptive LID-spoofing/code-mixing attacks, Belebele, the HF Space. Their hypotheses (H4/H5 attack parts) are reported as not tested.
  - **Why:** the drift to repair is ~1 pp and per-language power is limited (next entry). Effort goes where the evidence can be strong.
- **2026-10-27 15:05, two deferred items reinstated (owner's decision, after a cost/value check). Designs fixed before any output.**
  - **B1 (no preamble), Global.** Generated on the A100 together with **B0_a100**, so B1 is paired with a same-hardware B0 (the E0_a100 logic); B0 (MPS) stays the reported baseline. Primary comparison: paired B0_a100 vs B1_a100, both judges, same estimands as the drift analysis. B2 stays deferred.
  - **Gate-level spoofing / code-mixing of EORS (Earth, ε = 0.10 thresholds, layer 14, test split, no generation).**
    - Languages: ar, th, jv (GlotLID flipped by an English prefix in 20–33% of harmful prompts, local pilot) plus bn, sw as robust contrasts.
    - Conditions: (a) original prompt; (b) the English sentence "Please answer the following question in detail. " prepended (an honest code-mixing user, or a naive spoof), on benign and harmful prompts; (c) the same sentence in Thai prepended to harmful ar/jv/bn/sw prompts (a spoof towards a stricter-threshold language, the attacker's best direction).
    - **Outcomes:** gate firing rate per language and condition. For benign prompts that's the false-fire rate (the fairness cost of code-mixing); for harmful prompts the recall (the security cost of spoofing). Language is taken from constrained GlotLID on the *modified* prompt, exactly as at inference.
    - **Reading rule:** report paired changes with Wilson/bootstrap intervals; exploratory. The pilot's threshold comparison is first-order only (the prefix also moves activations), which is why the forward passes are run.
- **2026-10-27 15:20, CircuitKIT stopping rule applied: no circuit claims; discovery is reported as a tooling appendix.** Results from the length-matched run (`runs/circuitkit/colab_a100_matched/`):
  - The sanity point F(100%) = 1.00 in all 9 cells; parity KL 3.1e-5; peak GPU memory 22.6–22.8 GiB.
  - Diagonal faithfulness at 1/5/10/20/40/60% of nodes: en ≈0/≈0/0.02/−0.02/0.41/0.67, sw ≈0/≈0/0.00/0.06/0.47/0.87, bn ≈0/≈0/0.01/0.03/0.44/0.66. The rule (F ≥ 0.5 at ≤ 40%) is not met.
  - Random circuits equal the discovered ones at 5% (z −0.4 to −1.6).
  - Cross-language F at 40–60% resembles the diagonal, but without random baselines at those sizes it is uninterpretable and is not reported as transfer.
  - **Correction of the 13:25 entry:** matched pairs are also ≈0 at ≤ 20%, so prompt misalignment does *not* explain the first null. The supported statement is narrower: under this metric (refusal-minus-compliance logit difference, harmful vs length-matched benign) no sparse node-level circuit (≤ 20% of nodes) recovers refusal in Tiny Aya.
- **2026-10-27 17:10, spoofing results and a post hoc control (fixed before running it).**
  - **Results:** an English instruction prefix raised benign false-firing of the EORS gate 3–10× (ar 6→61%, th 14→55%, jv 13→62%, bn 9→43%, sw 16→67%). Harmful recall rose too; neither prefix lowered it, so no attack was found.
  - **Cause:** mostly a projection shift (+2.7 to +3.9 units). LID routing adds a smaller share (th, jv).
  - **Post hoc control, benign prompts only, same 5 languages:**
    - (d) the same instruction translated into the prompt's own language (Gemini translation, back-translation checked);
    - (e) a neutral English greeting prefix, "Good morning. ".
  - **Reading:** if (d) ≈ en_prefix, the gate reacts to instruction semantics (formatting fragility); if (d) ≈ orig and (e) high, it reacts to English material (a code-mixing cost); intermediate results are reported as such.
- **2026-10-27 17:40, prefix-control result (post hoc).** Benign false-firing of the EORS gate (Earth, ε 0.10), original → own-language instruction → English instruction → "Good morning.":
  - ar 6/42/61/63%, th 14/37/55/77%, jv 13/49/62/82%, bn 9/37/43/74%, sw 16/63/67/76%.
  - Language ID flags none of the control prompts.
  - **Reading:** neither code-mixing nor instruction semantics explains the shift. The harm direction responds to generic benign prefixes. A length confound is not supported: within benign prompts, gate score vs prompt length gives Spearman −0.26 to +0.10.
  - **Mechanism unresolved.** The paper reports this as a robustness caveat on all projection-gated arms: clean-prompt gate parity understates over-refusal under innocuous formatting changes.
- **2026-10-27 18:50, end-to-end analysis plan, fixed before any steering-arm result was inspected.** All comparisons are paired by (id, lang, kind) on the test split, under both judges (Qwen3Guard-4B primary for harm, Gemini secondary). Per-language exact McNemar with Holm across 10 languages; paired bootstrap over ids (5,000) for pooled estimands. Unlabelled rows are dropped.
  - **References:** Earth arms vs **E0_a100**, B1_a100 vs **B0_a100** (same hardware).
  - **H6 end to end (confirmatory direction, exploratory status):** CAST-EN vs **G-EN** (same English vector and condition direction; only the gate score differs).
    - Primary: pooled low-resource-tier (bn, sw, jv) change in harmful-prompt refusal (HRR), CAST-EN − G-EN. Supported if the 95% CI is below 0.
    - Secondary: per-language ΔHRR vs the share of benign prompts projecting negatively (gate_proxy h6), Spearman across 10 languages, descriptive; and ΔBRR.
    - Requires the G-EN arm; if it isn't run, H6 end to end is reported as not tested, and CAST-EN vs G-POOL is shown only as a confounded descriptive contrast.
  - **EORS redistribution end to end:** EORS (ε 0.05, 0.10) vs **G-POOL** (same pooled vector and direction; per-language conformal vs single threshold). Primary: change in worst-language BRR. Secondary: cross-language BRR gap, high-resource-tier BRR change (gate level predicts a rise), mean HRR.
  - **Every arm vs E0_a100:** mean and worst-language SRR, mean BRR, max BRR, mean HRR, worst selectivity (HRR − BRR). NEG is a sanity control (it should lower refusal). ReSta vs E0 is reported as is.
  - **B1 (no preamble) vs B0_a100:** mean SRR, mean HRR, mean BRR and per-language values; preamble dependence is read from the pooled estimands.
  - **Power** for each comparison from its observed discordance (Connor 1987), as for drift. Gate-brittleness (spoof/control) results are reported alongside as a robustness caveat.
- **2026-10-27 19:00, G-EN arm added (owner's OK, ~4–5 CU).** It is the comparator that pre-registered H6 names. It wasn't in the 14:15 queue, which ran G-POOL instead, and CAST-EN vs G-POOL confounds gate type with vector and direction. Same settings as CAST-EN (Earth, mech_earth calibration activations, α 1.0, batch 16, A100), scored identically (Qwen3Guard bf16 batch 4 on A100; Gemini broad). H6 end to end is now testable as pre-registered at 18:50.
- **2026-10-27 23:45, two follow-up experiments (owner's OK: Colab compute only, no Gemini/API spend). Analysis fixed before any output.** Earth, A100, the same code and settings as the α = 1 arms. Scored by Qwen3Guard-4B only (no Gemini labels for these runs; the paper says so).
  - **(A) Steering-strength replication at α = 0.5:** G-POOL_a05, EORS_a05 (ε .05), G-EN_a05, CAST-EN_a05. Primary contrasts, the same estimands as at α = 1:
    - EORS vs G-POOL: Δ worst-language BRR;
    - CAST-EN vs G-EN: low-resource-tier pooled ΔHRR, plus Δ unsafe;
    - each arm vs E0_a100.
  - **Reading rule:** "the gate decides" replicates if EORS's worst-language BRR CI vs G-POOL is below 0 at α = 0.5. CAST's defect "matters more when steering is weaker" if the LRL ΔHRR magnitude is larger than at α = 1 and Δ unsafe becomes positive with a CI above 0.
  - **(B) Greeting, end to end:** EORS_e10_greet (EORS ε .10, every prompt prefixed with "Good morning. ") and E0_greet (no steering, same prefix). Language for the gate comes from constrained GlotLID on the prefixed prompt, as at inference.
  - **Primary estimand:** the difference-in-differences in benign refusal, [BRR(EORS_greet) − BRR(E0_greet)] − [BRR(EORS) − BRR(E0)], pooled and per language, by paired bootstrap over ids. It isolates the gate's extra over-refusal under the prefix from the prefix's effect on the model itself.
  - **Secondary:** the same for HRR and SRR.
  - **Reading rule:** the gate-level brittleness is confirmed behaviourally if the pooled difference-in-differences in BRR is above 0 with its CI excluding 0.
  - **Outcome (2026-10-28 01:20, read by the rules above; runs/arms_report.json):** (A) replicates: EORS vs G-POOL Δ worst-language BRR −19.5 pp [−27.4, −12.3]. CAST at α = 0.5: LRL ΔHRR −2.6 [−4.0, −1.3], smaller in magnitude than α = 1 (−3.0), Δ unsafe +0.2 [+0.0, +0.5]; "matters more" not supported. (B) confirmed: BRR difference-in-differences +52.6 pp [+46.6, +58.6].
- **2026-10-28 06:40, controls after the independent review (owner's OK: Colab compute only, no Gemini/API spend). Fixed before any new output exists.** Design, research and premortem: `11_controls_premortem.md`. Two review findings drive it:
  - EORS's gain mixes level with spread: its worst/mean BRR ratio equals G-POOL's (1.85 vs 1.86).
  - EORS's direction and thresholds were fitted on the same calibration prompts, so its conformal guarantee was not valid as built.
  - **Split.** Calibration ids are permuted with seed 20261028: 1/3 fit, 2/3 conformal; the same ids in every language. The condition layer and direction come from the fit fold; thresholds from the conformal fold only. Steering vectors and behaviour layers 12–16 are unchanged from the α = 1 arms (they don't affect the gate).
  - **Arms** (Earth, A100, α = 1, batch 16, Qwen3Guard-4B only):
    - EORS-S ε .05 / .10 (per-language Mondrian split-conformal);
    - CPOOL-S ε .05 / .10 (one split-conformal threshold on pooled benign scores; same direction, data and budget; the matched-budget control).
    - Secondary: EORS-S_e10_greet and CPOOL-S_e10_greet ("Good morning. " prefix).
  - **P1, fairness at a matched budget (co-primary, Holm over the two ε):** EORS-S − CPOOL-S Δ worst-language BRR. Supported if the CI is below 0. Also reported: Δ max−min BRR range and the per-language excess BRR over E0_a100.
  - **P2, the cost (co-primary):** EORS-S − CPOOL-S pooled low-resource-tier (bn, sw, jv) ΔHRR. "Calibrating false refusals per language moves the miss onto low-resource harmful prompts" if the CI is below 0. Secondary: Δ harmful compliance (unsafe) per language and worst-language SRR.
  - **P3, validity (gate level, Earth test activations):** per-language test FPR of EORS-S against its Beta(n+1−k, k) ⊗ Binomial(n_test) 95% predictive band. Valid if at most 1 of 10 languages falls above the band at each ε. Computed both with LID-routed and with oracle-language thresholds; the old in-sample EORS thresholds are reported alongside.
  - **P4, harm vs dataset style (gate level):** AUROC of the gate score on XSTest-unsafe vs XSTest-safe (matched-surface contrast pairs; NLLB-translated, the same QA as the benign set), per language. "Harm signal beyond style" if pooled AUROC ≥ 0.75 and every language > 0.5; "style-dominated" if pooled < 0.65; mixed otherwise. Also reported: firing rate on Aya red-team OOD harmful prompts.
  - **Secondary / exploratory:**
    - greeting difference-in-differences per rule vs E0 (descriptive comparison of rules);
    - the gate-level frontier (per-language FPR and TPR across ε for pooled vs Mondrian thresholds) on test activations.
  - **Estimators:** plug-in point estimates (not bootstrap means; this fixes review finding M5) with percentile-bootstrap CIs over ids (5,000, seed 20261027). Max-type statistics are labelled as such.
- **2026-10-28 08:10, seed clarification (before any end-to-end control result exists; the ε = .05 pair is still being scored).** The 06:40 entry names bootstrap seed 20261027. In code, `analyze.compare` (worst-language and mean estimands) resamples with the project seed 20261026, while the tier (`lrl_delta`) and difference-in-differences estimands use 20261027. This is the existing code used for every earlier contrast, so it stays unchanged for comparability, and the entry is corrected to match.
  - **Outcome (2026-10-28 ≈09:30; read by the 06:40 rules; runs/controls_report.json, Qwen3Guard-4B).**
    - **P1 supported at both ε** (Holm across ε): Δ worst-language BRR −9.4 pp [−15.1, −3.8] (ε .05) and −8.5 [−14.2, −3.8] (ε .10), p_holm < .001; mean BRR −1.0 [−2.2, 0.0] and −0.9 [−2.4, +0.4].
    - **P2 supported at both ε:** low-resource-tier ΔHRR −7.5 [−9.6, −5.4] and −7.3 [−9.4, −5.3], p_holm < .001. Secondary: Δ unsafe +1.3 [+0.3, +2.2] and +0.5 [−0.2, +1.3].
    - **P3 valid:** 0 of 10 languages above the band at either ε.
    - **P4:** pooled AUROC 0.81, minimum 0.78 (jv), so "harm signal beyond style".
    - **Greeting (secondary):** difference-in-differences +41.4 (EORS-S) vs +41.0 (CPOOL-S); the rules are equally brittle.
  - **2026-10-28 ≈09:00, post hoc (not pre-registered).**
    - A free Qwen3Guard re-score of GEM-lite: mean SRR 93.6% vs 99.3% under the Gemini judge.
    - A calibration-activation recomputation on the A100: hardware s.d. 0.04, batching s.d. 0.05, vs a benign score s.d. of 2.8. The calibration-to-test gap is prompt sampling, correlated across languages through parallel ids.
    - A gate-level frontier.
