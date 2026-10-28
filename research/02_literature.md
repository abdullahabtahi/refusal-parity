# 02 — Literature Survey: Safety of Small Multilingual Models (Tiny Aya / Aya Expanse)

*Survey date: 2026-10-26. Scope: 2024–2026. Target: a 2-week hackathon paper that improves the safety of Cohere Labs' Tiny Aya (3.35B) using Lexsi Labs tools (SafeTune, CircuitKIT, CuratorKIT, AuditKIT) on an M1 Pro (16 GB), possibly with a Colab T4.*

**Citation policy.** I located every paper below during this session unless it is tagged **[ID from memory, not re-verified]**. Those are canonical papers whose arXiv IDs I did not re-open here. Check them before submission.

---

## 0. The target model: what we know about Tiny Aya's safety

- **Tiny Aya** (Salamanca et al., arXiv:2603.11510, Mar 2026; COLM 2026). Dense decoder, 36 layers, d_model 2048, 16 heads / 4 KV heads (GQA), 262k vocabulary (0.5B embedding parameters), 3:1 sliding-window:global attention, parallel blocks. Released as Base, Global, and three regional variants: **Earth** (Africa, West Asia), **Fire** (South Asia), **Water** (Europe, Asia-Pacific). Safety was distilled through "Fusion" from Command A, and the regional models were *merged with Global* to reduce cross-language safety gaps.
- **Reported safety (Table 7 of the report):** MultiJail (10 languages) min/mean safe rate is Global 87.0/91.1, Earth **77.5**/87.8, Fire **78.1**/90.0, Water 82.9/89.7. English XSTest over-refusal is 10.4% (Global), the *highest* among peers (Gemma3-4B 3.6%, Qwen3-4B 4.0%). Under-refusal is 15.5%. The judge was Command A in "contextual safety mode", applied without translation.
- **Implications.** (i) Out of the box, Tiny Aya is already the safest model in its size class on MultiJail. A "make the base model safer" paper will hit a ceiling. (ii) **Regional specialization costs up to ~10 pp of worst-language safety**, which amounts to a naturally occurring safety-drift experiment. (iii) Over-refusal was measured **only in English**. (iv) The report itself notes that safety in a language requires SFT data in that language. So a downstream fine-tune for a new language or domain is exactly where safety should erode.
- **Existing Tiny Aya safety work (novelty threats):** the Cohere "Tiny Aya Expedition" (blog, Jul 2026) included *Cross-Lingual Emergent Misalignment* (an anonymous OpenReview submission fine-tunes Tiny Aya on English insecure or risky data and finds a shared EM direction across 8 languages; code at Cohere-Labs-Community/crosslingual-emergent-misalignment), *Mix, Fine-Tune, Break* (code-mixed safety evaluation), and an SAE study showing that most features are shared across languages. An unpublished HF dataset (Turbs/xprmt-tiny-aya-global-multijail-judge-eval) suggests someone has run English, local, and global steering on Tiny Aya with MultiJail; there is no paper. **SafeTune's docs explicitly support Tiny Aya.** For ReSta they recommend α≈0.25, because α=1 "breaks the model". This is useful practical knowledge.
- **Aya Expanse 8B** is feasible on 16 GB only in 4-bit, which complicates activation and weight surgery. **Recommendation: use Tiny Aya Global plus its regional siblings as the primary testbed.**

---

## A. Cross-lingual and low-resource-language jailbreaks

**Key findings.**
- Translation to low-resource languages (LRLs) bypasses safety. Yong et al., *Low-Resource Languages Jailbreak GPT-4* **[arXiv:2310.02446, ID from memory]**. Deng et al., *Multilingual Jailbreak Challenges in LLMs* (MultiJail, ICLR 2024) **[arXiv:2310.06474, ID from memory]**; the MultiJail repo was located. Wang et al., *All Languages Matter* (XSafety; Findings ACL 2024, arXiv:2310.00905). Shen et al., *The Language Barrier* (Findings ACL 2024) find that RLHF/SFT in LRLs helps little and place the bottleneck in pretraining.
- Human-written multilingual red-teaming separates **global from local harms**. Aakanksha et al., *The Multilingual Alignment Prism* (EMNLP 2024, arXiv:2406.18682), produced the Aya Red-teaming dataset (8 languages). DPO cut harmful generations by 54.7%, and training on local harms transferred to global harms.
- **Code-switching and perturbations.** CSRT (Yoo et al., ACL 2025) gets +46.7% attack success over English, and intra-sentence mixing works best. CMP-RT (arXiv:2505.14226) combines code-mixing with phonetic misspellings and attacks the tokenizer. *Attributional Safety Failures* (OpenReview 2026) reports ASR rising from ~9% to ~69% under code-mixing. **Minionese** (Ifebi, Kong, Mehrotra; arXiv:2607.10112) covers 18 languages, 4 tiers, and 4 perturbations: transliteration collapses the harm representation, code-switching disrupts language routing, and safety drops sharply between tiers 2 and 3. *Register Shifts Break LLM Safety* (BanglaSafe, arXiv:2608.22335) finds that writing style *within* Bengali (formal vs casual) matters more than the switch from English to Bengali. IndicSafeEval (Findings EMNLP 2026) studies persuasion attacks in Hindi, Bengali, Marathi, and Punjabi.
- **Guards.** PolyGuard (Kumar et al., COLM 2025, arXiv:2504.04377) covers 17 languages, is trained on 1.91M samples, and beats prior guards by 5.5%. Even PolyGuard does not cover most of Tiny Aya's 70 languages.
- Surveys. Yong et al., *The State of Multilingual LLM Safety Research* (EMNLP 2025, arXiv:2505.24119), find English safety work outnumbers work on any other language about 10:1. *LLM Safety Alignment in Low-Resource Languages* (SLR, arXiv:2608.14626) notes that African languages are the least covered.

**SOTA.** Attacks: code-switching, transliteration, and register attacks. Defenses: multilingual DPO with local-harm data, multilingual guards (PolyGuard), and inference-time steering (Section B).

**Open problems.** Coverage of the long tail of the 70+ languages, romanized and code-mixed inputs, culturally local harms, and attacks on *small, on-device* models where no external guard runs.

**Not done for small multilingual models.** Systematic tier-wise vulnerability of Tiny Aya's regional variants, and whether defenses (not just attacks) generalize to transliteration and code-mixing. The code-mixed *evaluation* angle on Tiny Aya is partly taken by the Expedition project.

---

## B. Refusal direction, representation engineering, and steering

**Key findings.**
- Refusal is mediated by a single residual-stream direction. Arditi et al., NeurIPS 2024, arXiv:2406.11717. Refinements: refusal is **affine**, so ACE combines projection with a bias and stays coherent where ablation fails (Marshall, Scherlis, Belrose, arXiv:2411.09003). Refusal also lives in **multi-dimensional concept cones** with representationally independent directions (Wollschläger et al., ICML 2025, arXiv:2502.17420).
- **Multilingual universality.** Wang et al., *Refusal Direction is Universal Across Safety-Aligned Languages* (NeurIPS 2025, arXiv:2505.17306), use PolyRefuse in 14 languages. A refusal vector from *any* safety-aligned language ablates refusal in all of them, and the directions are nearly parallel. The *failure* in non-English languages is that **harmful and harmless prompts are poorly separated**. A small-model replication (*One Direction, Many Languages*, an unreviewed PDF, Gemma-2-2B / Llama-3.2-1B / Qwen2.5-1.5B) finds cosines of 0.85–0.97 and causal transfer that is weaker for Hindi. It also reports that **keyword refusal detectors agree with a judge only at chance (κ≈0)**.
- **The detection/action split is the 2026 consensus.** Three papers converge on it. *Low-Resource Safety Failures Are Action Failures, Not Representation Failures* (Aziz, Hanif, Koto; Findings EMNLP 2026, arXiv:2606.01196) studies 23 languages and 3 models (7–9B): harmful refusal falls from 87.9% in high-resource languages to 43.9% in LRLs, yet the harmfulness direction still separates LRL prompts. Scores are only *shifted down*. A recalibrated low-rank gate that routes between refusal steering and harm-direction ablation reaches selectivity Δ 54.5, against 33.6 for CAST and 6.8 for AdaSteer. *Decided Upstream, Written Late* (arXiv:2608.08032, an Indic MoE) finds the harm direction ≈ language-invariant (cos≈0.9) but orthogonal to a late refusal-"writer" circuit. Minionese finds sub-threshold projection onto the refusal direction.
- **Steering for safety.** CAST (Lee et al., ICLR 2025 spotlight, arXiv:2409.05907) applies refusal steering only when a condition vector fires. **BabelSteering** (Stein et al., arXiv:2608.16577) uses English refusal vectors plus an orthogonalized false-refusal ablation across 8 languages on Gemma-7B. Harmful refusal rises 11 pp on average and 17 pp for Bengali, but **pseudo-harmful refusal rises 13 pp on average and 23 pp for Bengali**: over-refusal grows most where safety was weakest. Further cautions: *Analysing the Safety Pitfalls of Steering Vectors* (arXiv:2603.24543) shows non-safety steering vectors shift ASR by up to ±50–57% because they overlap the refusal direction. *Steering Vectors are an Adversarial Attack Surface* (arXiv:2606.05958) shows poisoned vectors.

**SOTA.** Calibrated, gated steering (Aziz et al.) for LRL repair; ACE for precise control; concept-cone directions for robustness.

**Open problems.** (i) Does the language-universal direction survive **fine-tuning drift**, or does drift break the gate's calibration? (ii) Over-refusal measured per language and per resource tier. (iii) Behaviour at ~3B, where representations are less separable. (iv) Interaction with romanized and code-mixed inputs (Minionese suggests a different failure route).

**Thoroughly done (avoid as a headline):** "is the refusal direction shared across languages?" (Wang et al. 2025) and "English steering vectors improve multilingual safety" (BabelSteering, Aziz et al.).

---

## C. Fine-tuning safety drift and repair

**Key findings (general).** A few harmful examples, or even benign data, break alignment (Qi et al., *Fine-tuning Aligned LMs Compromises Safety*, ICLR 2024 **[arXiv:2310.03693, ID from memory]**). Alignment is **shallow**, concentrated in the first few output tokens; token-constrained fine-tuning is the fix. Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025 Outstanding Paper **[arXiv:2406.05946, ID from memory]**; the ICLR version was located. The refusal direction **drifts** during instruction fine-tuning, mostly early in training. ProCon anchors it with a projection loss (Du et al., arXiv:2509.06795). RefusalGuard (arXiv:2605.01913) preserves refusal-cone geometry.

**Multilingual drift.**
- *Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks* (Poppi, Yong et al., Findings NAACL 2025, arXiv:2410.18210): a harmful fine-tune in one language breaks all languages, and changing ~20% of parameters suffices (SIL localization). The **alternative-pathways hypothesis**, that freezing safety parameters does not stop attacks, matters for any "freeze the circuit" idea.
- *Tongue-Tied* (Upadhayay & Behzadan, CALCS 2025): benign LoRA on Newari or invented languages jailbreaks the model, apparently through late-layer pivots past English-centric safeguards.
- *The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning* (Hawkins et al., PMLR v306, 2026): Llama-3.2, Qwen3, and Gemma-3 tuned on benign data in 9 languages. Adversarial compliance rises **up to 4×**, depends on both the fine-tuning and the evaluation language, and is decoupled from capability. Non-English fine-tuning causes *smaller* representational drift that still tips the model toward exaggerated compliance or refusal. They released Multilingual-Benign-Tune and SORRY-Bench-Multilingual. **They measure drift but do not repair it.**
- *AI Safety Lost in Translation* (Wu & Brandao, LREC 2026) and RefusEU (Findings ACL 2026): bilingual SFT can *reduce* safety in small models, and English-only alignment does not transfer.

**Repair and hardening methods.** Post-hoc: **ReSta** (Bhardwaj, Anh, Poria, ACL 2024, arXiv:2402.11746) adds the safety vector with DARE; it covered Hindi and Chinese SFT and evaluated in English, Chinese, and Vietnamese. **Safe LoRA** (Hsu et al., NeurIPS 2024, arXiv:2405.16833) projects onto the alignment matrix. **Antidote** (Huang et al., ICML 2025, arXiv:2408.09600) uses one-shot pruning. Pre- and post-tuning **merging** (Farn et al., Findings EMNLP 2025). SOMF (arXiv:2405.09055). Safety Arithmetic (Hazra et al., EMNLP 2024, arXiv:2406.11801). Alignment-stage: **Vaccine** **[arXiv:2402.01109, ID from memory]**, **Booster** (ICLR 2025, arXiv:2409.01586), **LoX** (low-rank extrapolation, arXiv:2506.15606; −11 to −54 pp ASR). During fine-tuning: **SafeGrad** (gradient surgery with KL alignment, ICASSP 2026, arXiv:2508.07172), SPF (arXiv:2601.10141), **SaLoRA** (ICLR 2025, arXiv:2501.01765), SPPFT (freezing safety layers, see D).

**Open problems / not done.**
- **Does repair work equally across languages?** I found no paper that benchmarks ReSta, LoX, SafeGrad, Safe LoRA, or steering repairs with **per-language parity** after fine-tuning for language adaptation. ReSta's multilingual evidence is three high-resource languages. Hawkins et al. stop at measurement.
- Repairs for **3B-scale models**. SafeTune's note that Tiny Aya breaks at ReSta α≥0.5 suggests small models have a narrow repair window, a worthwhile empirical point.
- **Regional specialization as drift.** Tiny Aya's Earth and Fire variants are real deployed models with a ~10 pp worst-language deficit relative to Global.

---

## D. Safety circuits, neurons, and layers, and whether they are language-agnostic

**Key findings.**
- **Safety layers**: a contiguous middle block distinguishes malicious queries, and freezing it (SPPFT) preserves safety (Li et al., ICLR 2025, arXiv:2408.17003). About 5% "safety neurons", when patched, restore more than 90% of safety and overlap heavily with helpfulness neurons, which explains the alignment tax (*A Mechanistic Perspective from Safety Neurons*, NeurIPS 2025 poster). Safety-critical attention heads: Zhou et al., *On the Role of Attention Heads in LLM Safety*, ICLR 2025 **[arXiv:2410.13708, ID from memory]**.
- **Language-shared components.** **SS-Neurons** (Zhang et al., ICML 2026, arXiv:2602.01283) are a small set of cross-lingually shared safety neurons; tuning only them lifts safety in non-high-resource languages. *Who Bridges Safety?* (Miao et al., arXiv:2608.09095) finds shared cross-layer safety *pathways*. **Sparse Weight Editing** (Liang et al., ICML 2026, arXiv:2602.22554) maps LRL harmful representations onto the high-resource safety subspace in closed form, under a null-space constraint. Language-specific neurons sit in the top and bottom layers (LAPE, Tang et al., arXiv:2402.16438). For multilingual circuit claims, *Triangulation* (arXiv:2512.24842) proposes necessity, sufficiency, and cross-language invariance as an acceptance rule.
- The emerging picture is a **language-specific periphery** (early layers read the language, late layers write the language) around a **largely shared semantic and safety core**. The cross-lingual gap sits at the *interface* (routing and calibration into the core), not in the core itself.

**Open problems / not done.** (i) Edge-level circuits (EAP-IG) for refusal in a 70-language model, with faithfulness tested *per language* (most work uses neurons or directions). (ii) Using a discovered shared circuit to **constrain language-adaptation fine-tuning**, with circuit-restricted LoRA on the complement, and testing it against Poppi et al.'s alternative-pathways objection. (iii) No circuit work exists on Tiny Aya's parallel-block, GQA, NoPE/RoPE-interleaved architecture.

---

## E. Over-refusal and the safety–utility frontier

**Key findings.** XSTest (Röttger et al., NAACL 2024) uses 250 safe and 200 unsafe prompts, but many depend on English homonyms, so **it does not translate well** (BabelSteering makes the same point). OR-Bench (Cui et al., ICML 2025 / PMLR v267) is a large automatically built set. **MORBench** (Pan et al., EMNLP 2025, arXiv:2505.18325) has 8.4k prompts in 7 high-resource languages, built by steering-vector-guided boundary sampling, and finds over-refusal varies by language. LinguaSafe (arXiv:2508.12733) has 45k entries in 12 languages and includes an oversensitivity rate. Native contrast sets exist for single languages, such as Safety-SeGa (Basque/Spanish/English) and a Turkish over-refusal set (an injection guard flags benign Turkish 71× more often than English).

**Frontier.** Every safety intervention moves along a curve of harmful-refusal against benign-refusal. In multilingual settings that curve is **language-dependent** and steepest in LRLs (BabelSteering's Bengali result).

**Open problems / not done.** Over-refusal in **low-resource African and South Asian languages** (MORBench covers 7 high-resource languages only). Tiny Aya has no multilingual over-refusal number at all. Nobody reports a **per-language Pareto frontier** for repair methods. Metric recommendation: report harmful-refusal, benign-refusal, *selectivity Δ = harmful − harmless refusal* (as in Aziz et al.), and worst-language values.

---

## F. Evaluation methodology best practice

- **Harm scoring.** StrongREJECT scores refusal × specificity × convincingness **[arXiv:2402.10260, ID from memory]**. The HarmBench classifier **[arXiv:2402.04249, ID from memory]**. WildGuard **[arXiv:2406.18495, ID from memory]**. Llama Guard 3 is officially multilingual for 8 languages **[not re-verified]**. PolyGuard covers 17 languages (verified), and its Qwen2.5-0.5B "PG Smol" variant runs on an M1.
- **Judge reliability across languages is poor and biased.** Fu & Liu, *How Reliable is Multilingual LLM-as-a-Judge?* (Findings EMNLP 2025): Fleiss κ ≈ 0.3 across 25 languages, worst in LRLs. MM-Eval (arXiv:2410.17578): safety-subset accuracy drops 18.4% in LRLs. *LLM Evaluators are Biased across Languages* (arXiv:2607.14480): evaluators score LRLs **more leniently**, with acceptance gaps up to 43% under a global threshold, *invisible to pairwise accuracy*. Harmful LRL content therefore passes filters more often, and **judge leniency can masquerade as model safety**. The multilingual joint testing exercise (Vij et al., arXiv:2601.15706 / AISI report) found LLM-vs-human discrepancies above 5% in 6 of 10 languages, with the most common miss being a disclaimer followed by harmful content. *Challenges and Recommendations for LLM-as-a-Judge in Multilingual Settings* (Findings EMNLP 2026) offers a checklist.
- **Additional failure modes to log:** off-language or invalid responses (Qwen3-4B gave 91% invalid responses in Swahili per the Tiny Aya report), language consistency (RefusEU), and the "disclaimer then comply" pattern.
- **Statistical rigour (recommended protocol):** at least 2 judges from different families (for example PolyGuard plus an API LLM judge), with per-language Cohen's κ, and a native-speaker or bilingual audit of ~50 items per language for 2–3 focal languages. Report both translate-to-English-then-judge and direct judging. Use Wilson 95% CIs per language; paired McNemar or paired bootstrap for before/after intervention comparisons on the same prompts; Holm correction across languages; and at least 3 seeds for any fine-tune. Report the **worst-language** metric and the **max–min gap** alongside the mean. Power check: MultiJail has ~315 prompts per language. A paired test detects a ~7–8 pp change at 80% power when the baseline is 85–90%, but a 3 pp change needs more data, so pool across benchmarks (MultiJail, Aya Red-teaming, PolyRefuse, SORRY-Bench-Multilingual).

---

## Synthesis: where the gap is

1. *Measuring* the multilingual safety gap is mature, and *explaining* it (detection/action split, shared direction, shared neurons) has converged in 2026.
2. *Repairing* it with steering or editing is now active for **static** 7–9B models (Aziz et al., BabelSteering, SS-Neurons, sparse editing).
3. **Under-studied: the dynamic case, where safety drift is introduced by language adaptation or regional specialization in small models, and whether existing repairs restore safety equally across languages without inflating over-refusal in LRLs.** Tiny Aya's base, global, and regional family, together with SafeTune's drift→repair→evaluate API, is close to purpose-built for this question.

---

## Candidate research questions (ranked by impact × novelty × feasibility × tool fit)

Scores are 1–5 for impact (I), novelty (N), feasibility on M1/T4 (F), and Lexsi tool fit (T).

### RQ1 (rank 1): Is safety repair language-fair? Repair parity after language-adaptation drift in Tiny Aya. **I5 N4 F4 T5**
- **Hypothesis.** Benign LoRA adaptation of Tiny Aya Global to one LRL (for example Hausa, Yoruba, or Swahili), with English and one high-resource language as controls, causes heterogeneous cross-language safety drift. Standard repairs (ReSta, LoX, SafeGrad, refusal-direction steering or ACE) restore high-resource languages almost fully but leave a **residual worst-language gap**, and they **raise over-refusal most in LRLs**. A *language-balanced* repair (a safety vector or steering direction estimated from multilingual anchor sets built with CuratorKIT, or a per-language-calibrated gate in the style of Aziz et al.) closes the gap at equal utility.
- **Why it matters.** Practitioners adapt small open models to their own language. That is Tiny Aya's stated purpose, for phones and offline use in Africa and South Asia. If repairs are validated only in English, billions of LRL users receive a model that is certified safe but is not.
- **Minimal convincing experiment.** 2 adaptation languages (1 LRL, 1 high-resource) × LoRA with 3 seeds (QLoRA on T4, or MLX-LoRA on M1, ~2–5k benign instructions, for example Aya Dataset or Multilingual-Benign-Tune). Evaluate on 8–10 languages (MultiJail plus Aya Red-teaming) for harm, multilingual benign contrast prompts for over-refusal, and Global-MMLU or Belebele subsets for utility. Compare 4 repairs: ReSta (Global−Base delta, α sweep 0.1–0.5), LoX (hardening before the fine-tune), SafeGrad (during the fine-tune), and steering or ACE. Primary metrics: worst-language safe rate, max–min gap, selectivity Δ per language, and utility.
- **Expected headline.** "English-validated safety repairs leave an X-pp residual safety gap and a Y-pp over-refusal surplus in low-resource languages. Language-balanced repair halves it at no utility cost."
- **Risk.** Moderate. Drift on Tiny Aya may be small with benign data, given strong distilled safety. Mitigation: add a 1–5% harmful-mix condition, as in the SafeGrad threat model. Judge leniency in LRLs (Section F) could hide drift, so use 2 judges plus an audit.

### RQ2 (rank 2): Repairing regional-specialization drift without training (Earth, Fire, Water vs Global). **I4 N5 F5 T5**
- **Hypothesis.** The deficit of Tiny Aya Earth and Fire in worst-language safety (77–78% vs 87%) comes from regional SFT and is recoverable training-free: a sparse or low-rank **safety-delta transplant** from Global (ReSta or LoX top-k on Global−Base, restricted to the refusal subspace or a CircuitKIT-localized circuit), or activation steering with the Global refusal direction, restores min-language safety with no loss in regional quality (chrF on the focus languages).
- **Why it matters.** These regional variants are the models aimed at African and South Asian users. The work is cheap, fully reproducible, and needs no harmful training data.
- **Minimal experiment.** Inference and weight arithmetic only, all feasible on the M1: 3 regional models × 3 repairs × α sweep, scored on MultiJail and Aya Red-teaming (10+ languages), multilingual over-refusal, and FLORES/WMT chrF for the region's languages.
- **Expected headline.** "A 0.x%-parameter transplant from Global recovers Z of Earth's 10-pp worst-language safety deficit while keeping >99% of its regional translation gain."
- **Risk.** The gap may sit within judge noise for some languages, so CIs are essential. Cohere already merged with Global, so the remaining headroom may be small. **Recommendation: run RQ2 as the zero-training arm of RQ1. Together they form one paper on "natural vs induced drift".**

### RQ3 (rank 3): Detection or action? Diagnosing which part of the safety pipeline drift breaks, per language, and choosing the repair accordingly. **I4 N3 F5 T4**
- **Hypothesis.** In Tiny Aya, LRL safety failures and post-fine-tune drift mainly *shift harmfulness scores* (calibration), not refusal-direction integrity. The per-language projection shift onto the harm and refusal directions therefore **predicts which repair works**: calibrated gating for shift failures, and weight-space repair for broken directions.
- **Minimal experiment.** Extract harm and refusal directions per language before and after the fine-tune (SafeTune refusal-direction utilities). Measure AUROC, mean shift, and cosine drift (as in ProCon). Correlate them with per-language repair success from RQ1.
- **Novelty caveat.** The detection/action split is established for 7–9B models (Aziz et al. 2026, Minionese, Decided Upstream). The new contribution is the *drift* setting at 3B with 70 languages. Best used as RQ1's mechanistic section, not as a standalone paper.

### RQ4 (rank 4): Circuit-restricted language adaptation, freezing a cross-lingually shared refusal circuit. **I4 N4 F2–3 T5**
- **Hypothesis.** EAP-IG (CircuitKIT) finds a refusal circuit whose core edges are shared across languages (accepted only if necessity and sufficiency hold in at least 4 languages, per triangulation). Adapting to an LRL with LoRA restricted to the *complement* of that circuit preserves cross-lingual safety better than SPPFT or Safe LoRA at equal target-language gain.
- **Risk: high.** Poppi et al. show that alternative pathways defeat freezing. EAP-IG on 3.35B with long multilingual prompts is memory-heavy (needs a T4). Faithfulness may fail in LRLs. Prior overlap: SS-Neurons and shared pathways (2026). Worth doing only if RQ1 is already running.

### RQ5 (rank 5): A low-resource multilingual over-refusal and code-mixed audit of Tiny Aya, with false-refusal ablation. **I3 N2–3 F5 T4**
- **Hypothesis.** Tiny Aya's English over-refusal (10.4%) is larger in LRLs and in romanized or code-mixed inputs. Ablating a false-refusal direction cuts it without raising MultiJail ASR.
- **Novelty caveat.** MORBench, BabelSteering's OR-Bench translation, CSRT, Minionese, and Tiny Aya's own code-mixing expedition project cover much of this. **Recommendation: build the contrast set with CuratorKIT and use it as the over-refusal axis of RQ1/RQ2, not as the main contribution.**

### Already thoroughly done (do not headline)
Universality of refusal directions across languages; English-derived steering for multilingual safety; conditional and calibrated steering for LRLs; cross-lingual fine-tuning attacks; benign multilingual drift *measurement*; shared safety neurons; code-switching jailbreaks; cross-lingual emergent misalignment on Tiny Aya.

---

## References

**Target models and tools**
1. Salamanca, A. R., Abagyan, D., D'souza, D., ... Kreutzer, J., Fadaee, M. *Tiny Aya: Bridging Scale and Multilingual Depth.* arXiv:2603.11510, 2026 (COLM 2026).
2. Cohere Labs. *Tiny Aya Expedition Drives Multilingual Innovation* (blog), Jul 2026; Cohere-Labs-Community/crosslingual-emergent-misalignment (GitHub).
3. Anonymous. *Cross-Lingual Emergent Misalignment: Shared Multilingual Circuits Can Propagate Safety Failures.* OpenReview (id JjtZl9jTSW), 2026, under review.
4. Lexsi Labs. SafeTune docs (ReStaTrainer; Tiny Aya α≈0.25 note), safetune.lexsi.ai; Sadhu, S. et al. *SafeTune* (arXiv:2609.22153; EMNLP 2026 Demos per Lexsi); CircuitKIT, Seth, P. et al. (arXiv:2607.19317); AuditKIT README.

**A. Cross-lingual jailbreaks**
5. Yong, Z.-X., Menghini, C., Bach, S. *Low-Resource Languages Jailbreak GPT-4.* arXiv:2310.02446, 2023 **[ID from memory]**.
6. Deng, Y. et al. *Multilingual Jailbreak Challenges in LLMs* (MultiJail). ICLR 2024, arXiv:2310.06474 **[ID from memory; repo located]**.
7. Wang, W. et al. *All Languages Matter: On the Multilingual Safety of LLMs.* Findings ACL 2024, arXiv:2310.00905.
8. Shen, L. et al. *The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts.* Findings ACL 2024.
9. Aakanksha, Ahmadian, A., Ermis, B., Goldfarb-Tarrant, S., Kreutzer, J., Fadaee, M., Hooker, S. *The Multilingual Alignment Prism.* EMNLP 2024, arXiv:2406.18682.
10. Yoo, H. et al. *Code-Switching Red-Teaming (CSRT).* ACL 2025 (long, 657).
11. *Code-Mixed Phonetic Perturbations for Red-Teaming LLMs (CMP-RT).* arXiv:2505.14226, 2025/26.
12. Ifebi, C., Kong, B., Mehrotra, A. *Minionese.* arXiv:2607.10112, 2026.
13. *Register Shifts Break LLM Safety: A Bengali Benchmark (BanglaSafe).* arXiv:2608.22335, 2026.
14. Kumar, P. et al. *PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages.* COLM 2025, arXiv:2504.04377.
15. Yong, Z.-X., Ermis, B., Fadaee, M., Bach, S., Kreutzer, J. *The State of Multilingual LLM Safety Research.* EMNLP 2025, arXiv:2505.24119.
16. *LLM Safety Alignment in Low-Resource Languages* (systematic review). arXiv:2608.14626, 2026.

**B. Refusal direction and steering**
17. Arditi, A. et al. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024, arXiv:2406.11717.
18. Marshall, T., Scherlis, A., Belrose, N. *Refusal in LLMs is an Affine Function.* arXiv:2411.09003, 2024.
19. Wollschläger, T. et al. *The Geometry of Refusal in LLMs: Concept Cones and Representational Independence.* ICML 2025, arXiv:2502.17420.
20. Wang, X., Wang, M., Liu, Y., Schütze, H., Plank, B. *Refusal Direction is Universal Across Safety-Aligned Languages.* NeurIPS 2025, arXiv:2505.17306.
21. Lee, B. W. et al. *Programming Refusal with Conditional Activation Steering.* ICLR 2025, arXiv:2409.05907.
22. Stein, E. V., Meier, D., Ruas, T., Wahle, J. P., Gipp, B. *BabelSteering.* arXiv:2608.16577, 2026.
23. Aziz, R., Hanif, I. A., Koto, F. *Low-Resource Safety Failures Are Action Failures, Not Representation Failures.* Findings EMNLP 2026, arXiv:2606.01196.
24. Kompella, R. P., Mahajan, A. *Decided Upstream, Written Late.* arXiv:2608.08032, 2026.
25. *Analysing the Safety Pitfalls of Steering Vectors.* arXiv:2603.24543, 2026.
26. *Steering Vectors are an Adversarial Attack Surface.* arXiv:2606.05958, 2026.
27. *One Direction, Many Languages: Causal Cross-Lingual Refusal Transfer Across Small Open Models.* Unreviewed PDF, 2026 **[venue unverified]**.

**C. Fine-tuning drift and repair**
28. Qi, X. et al. *Fine-tuning Aligned Language Models Compromises Safety.* ICLR 2024, arXiv:2310.03693 **[ID from memory]**.
29. Qi, X. et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025 (Outstanding Paper) **[arXiv:2406.05946 from memory]**.
30. Poppi, S., Yong, Z.-X. et al. *Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks.* Findings NAACL 2025, arXiv:2410.18210.
31. Upadhayay, B., Behzadan, V. *Tongue-Tied: Breaking LLMs Safety Through New Language Learning.* CALCS 2025.
32. Hawkins, W., Rawal, K., Rystrøm, J., Tsirtsis, S., Fu, Z. *The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning.* PMLR v306, 2026.
33. Wu, A., Brandao, M. *AI Safety Lost in Translation.* LREC 2026.
34. *Multilingual Refusal Alignment for Safer LLMs (RefusEU).* Findings ACL 2026.
35. Bhardwaj, R., Anh, D. D., Poria, S. *Language Models are Homer Simpson! (RESTA).* ACL 2024, arXiv:2402.11746.
36. Hsu, C.-Y. et al. *Safe LoRA.* NeurIPS 2024, arXiv:2405.16833.
37. Hazra, R. et al. *Safety Arithmetic.* EMNLP 2024, arXiv:2406.11801.
38. Huang, T. et al. *Antidote.* ICML 2025, arXiv:2408.09600.
39. Huang, T. et al. *Booster.* ICLR 2025, arXiv:2409.01586.
40. Huang, T. et al. *Vaccine.* NeurIPS 2024, arXiv:2402.01109 **[ID from memory]**.
41. Perin, G. J. et al. *LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning.* arXiv:2506.15606, 2025.
42. Yi, B. et al. *SafeGrad: Gradient Surgery for Safe LLM Fine-Tuning.* ICASSP 2026, arXiv:2508.07172.
43. Li, M. et al. *SaLoRA.* ICLR 2025, arXiv:2501.01765.
44. *Understanding and Preserving Safety in Fine-Tuned LLMs (SPF).* arXiv:2601.10141, 2026.
45. Du, Y. et al. *Anchoring Refusal Direction (ProCon).* arXiv:2509.06795, 2025.
46. *RefusalGuard.* arXiv:2605.01913, 2026.
47. Farn, H. et al. *Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging.* Findings EMNLP 2025.
48. *SOMF: safety realignment via subspace-oriented model fusion.* arXiv:2405.09055, 2024.

**D. Safety circuits, neurons, and layers**
49. Li, S., Yao, L., Zhang, L., Li, Y. *Safety Layers in Aligned LLMs.* ICLR 2025, arXiv:2408.17003.
50. *A Mechanistic Perspective from Safety Neurons.* NeurIPS 2025 poster.
51. Zhou, Z. et al. *On the Role of Attention Heads in LLM Safety.* ICLR 2025 **[arXiv:2410.13708 from memory]**.
52. Zhang, X. et al. *Who Transfers Safety? Cross-Lingual Shared Safety Neurons.* ICML 2026, arXiv:2602.01283.
53. Miao, S. et al. *Who Bridges Safety? Cross-Lingual Shared Safety Pathways.* arXiv:2608.09095, 2026.
54. Liang, J., Wang, Z., Wang, H. *Multilingual Safety Alignment via Sparse Weight Editing.* ICML 2026, arXiv:2602.22554.
55. Tang, T. et al. *Language-Specific Neurons (LAPE).* ACL 2024, arXiv:2402.16438.
56. *Triangulation as an Acceptance Rule for Multilingual Mechanistic Interpretability.* arXiv:2512.24842.
57. Syed, A., Rager, C., Conmy, A. *Attribution Patching Outperforms Automated Circuit Discovery (EAP).* BlackboxNLP 2024.

**E. Over-refusal**
58. Röttger, P. et al. *XSTest.* NAACL 2024.
59. Cui, J., Chiang, W.-L., Stoica, I., Hsieh, C.-J. *OR-Bench.* ICML 2025 (PMLR v267) **[arXiv:2405.20947 from memory]**.
60. Pan, L. et al. *Understanding and Mitigating Overrefusal... (RASS, MORBench).* EMNLP 2025, arXiv:2505.18325.
61. Ning, Z. et al. *LinguaSafe.* arXiv:2508.12733, 2025.

**F. Evaluation**
62. Souly, A. et al. *A StrongREJECT for Empty Jailbreaks.* NeurIPS 2024 D&B **[arXiv:2402.10260 from memory]**.
63. Mazeika, M. et al. *HarmBench.* ICML 2024 **[arXiv:2402.04249 from memory]**.
64. Han, S. et al. *WildGuard.* NeurIPS 2024 D&B **[arXiv:2406.18495 from memory]**.
65. Fu, X., Liu, W. *How Reliable is Multilingual LLM-as-a-Judge?* Findings EMNLP 2025.
66. Son, G. et al. *MM-Eval.* arXiv:2410.17578.
67. *LLM Evaluators are Biased across Languages.* arXiv:2607.14480, 2026.
68. Vij, A. et al. *Improving Methodologies for LLM Evaluations Across Global Languages.* arXiv:2601.15706, 2026.
69. *Challenges and Recommendations for LLM-as-a-Judge in Multilingual Settings and for Low-Resource Languages.* Findings EMNLP 2026.
