# 04 — Cohere Labs Tiny Aya: Technical & Safety Fact Sheet

Compiled 2026-10-26. Sources: HF Hub API, rendered model cards, the arXiv tech report, Cohere docs, and community repos. The official weights, `config.json` and README are **gated** (`"gated":"auto"`; raw fetches return 401). Architecture values below come from the tech report plus **ungated byte-identical-size mirrors** (`mlx-community/tiny-aya-fire`, `1-800-LLMs/tiny-aya-global`). Notation: **[V]** = I checked it against the source. **[U]** = uncertain or inferred.

---

## 1. Model family

| Model | HF repo | Specialization (official wording) | Post-training |
|---|---|---|---|
| Base | `CohereLabs/tiny-aya-base` | Pretrained, 70+ languages | none (pretrained only) |
| Global | `CohereLabs/tiny-aya-global` | "Best balance across languages and regions" | SFT (all-region mix) + minimal preference tuning |
| Earth | `CohereLabs/tiny-aya-earth` | "Best for West Asian and African languages" | Cluster SFT (Europe + West Asia + Africa), then merged with Global |
| Fire | `CohereLabs/tiny-aya-fire` | "Best for South Asian languages" | Cluster SFT (South Asia only: hi, mr, bn, gu, pa, ta, te, ne, ur), then merged with Global |
| Water | `CohereLabs/tiny-aya-water` | "Best for European and Asia Pacific languages" | Cluster SFT (Europe + West Asia + Asia-Pacific), then merged with Global |

- Sources: model cards (https://huggingface.co/CohereLabs/tiny-aya-global, …-fire, …-water, …-earth, …-base), docs (https://docs.cohere.com/docs/tiny-aya), and tech report §2.3.3 and §3.3 (https://arxiv.org/html/2603.11510) [V].
- Cluster composition is from the report: "first cluster groups European, West Asian and Asia-Pacific; the second European, West Asian and African; the third solely South Asian." The cluster-to-model mapping (Water / Earth / Fire) is inferred from the per-model descriptions [U, high confidence]. The report states that **Fire and Water "do not have support for" African languages**, and African-language performance drops in those two models [V].
- TechCrunch describes Water as "Asia Pacific, West Asia, and Europe". That matches the training-data cluster, not the model card's tagline (https://techcrunch.com/2026/02/17/cohere-launches-a-family-of-open-multilingual-models/).
- **Region merging**: SimMerge picks among Linear, SLERP and TIES merges of each regional SFT checkpoint with the Global checkpoint. The final pick is checked so that "multilingual safety metrics do not regress" (report §3.3) [V].
- **Later additions (Sep 2026)**: `CohereLabs/tiny-aya-l2-thinker` and `tiny-aya-en-thinker` are reasoning variants (paper arXiv:2609.10445, "Building Multilingual Bridges…", 2026-09-09: L2 reasoning rate >93% across 60 languages). `CohereLabs/tiny-aya-base-32K` is a long-context base (gated, little public detail). Their parameter count is 3,350,243,328, slightly more than the originals, and their tokenizer.json is larger (30.2 MB vs 21.4 MB). This suggests a modified tokenizer [U].

### Core specs (all five original models are identical in shape)

| Field | Value | Source |
|---|---|---|
| Params | **3,349,227,520** (BF16), reported as 3.35B. Embedding 0.5B, non-embedding 2.8B | HF API `safetensors`; report Table 2 [V] |
| `model_type` / class | `cohere2` / `Cohere2ForCausalLM` (same arch class as Command R7B) | HF API config [V] |
| Layers / hidden / FFN | 36 / 2048 / 11008 (SwiGLU, no biases) | mirror config + report Table 2 [V] |
| Heads | 16 Q heads, **4 KV heads (GQA 4:1)**, head_dim 128 | [V] |
| Attention pattern | 3 sliding-window (4096, RoPE) layers : 1 full-attention layer (**NoPE**), repeated 9 times. Full-attention layers are at indices 3, 7, …, 35 (`layer_types`, `sliding_window_pattern: 4`) | config + report §3.1 [V] |
| Block | **Parallel** attention+MLP blocks (one LayerNorm per block), tied embeddings (`use_embedding_sharing: true`), `logit_scale: 1.0`, `rope_theta: 50000` | config [V] |
| Vocab / tokenizer | **262,144** (stated as "262k"). Single shared BPE tokenizer trained on 50 GB of FineWeb-2 with language-bucket weighting, GPT-4o pre-tokenization regex, no normalization. BOS `<BOS_TOKEN>`=2, EOS `<EOS_TOKEN>`=3, PAD=0 | report §2.1; config [V] |
| Context | **8K input + 8K output** (card, docs, report Table 2). Note: `max_position_embeddings: 500000` in config. Do not trust it beyond 8K | [V]; the discrepancy is real |
| Pretraining | 6T tokens, WSD schedule, 70 languages + code, FP8 mixed precision, 256× H100 (JAX "Fax"). Post-training: 16× H100, <24h per cluster, 3 epochs, LR 2.5e-5 → 1.2e-6, global batch size 32 | report §3.1–3.2 [V]. Press reports "64 H100s" (https://betakit.com/from-amharic-to-zulu-coheres-new-multilingual-ai-model-supports-more-than-70-languages/), which conflicts with the paper [U] |
| Knowledge cutoff | "June 2024" (stated in the chat template) | chat template [V] |
| Release | HF repos created 2026-02-13. Public launch 2026-02-17 (India AI Summit). Blog: https://cohere.com/blog/cohere-labs-tiny-aya | HF API; TechCrunch [V] |
| Tech report | **arXiv:2603.11510**, "Tiny Aya: Bridging Scale and Multilingual Depth", Salamanca et al., v1 2026-03-12 | https://arxiv.org/abs/2603.11510 [V] |
| License | **CC-BY-NC-4.0** + Cohere Labs Acceptable Use Policy, gated with a non-commercial checkbox | HF tags; GGUF README [V]. The card's "Terms of Use" paragraph mentions "a 111 billion parameter model". That is a copy-paste error from Command A |
| Signatures | Each repo includes `signatures/*.sig` + verification instructions (model signing) | HF file list [V] |

### Chat template (tokenizer_config, readable via HF API)
`<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble …<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{assistant}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`

**Safety-critical detail [V]**: the template *always* injects a hard-coded system preamble. It does this whether or not you pass a system message. The preamble includes, verbatim:
> "You are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful outcomes."

The preamble then gives the trained-language list and a "Default Preamble" ("Your name is Aya…"). Any user `system` message goes into a "# Developer Preamble" section that "take[s] precedence over … default preamble and user prompt". The strings come from https://huggingface.co/api/models/CohereLabs/tiny-aya-global (`config.tokenizer_config.chat_template`).

Implications:
- **Contextual safety mode is the default.** There is no STRICT mode in the template. Cohere's API STRICT text ("You will reject requests to generate content related to violence, hate, misinformation or sex to any amount…", https://docs.cohere.com/docs/safety-modes) could be swapped in by hand. That is untested for Tiny Aya [U].
- Raw-prompt experiments that skip `apply_chat_template` remove the safety preamble. This is a confound for refusal measurements.

---

## 2. Safety: what Cohere reports

From tech report §4.3.2, §5.3, Table 7, Table 26, App. C (https://arxiv.org/html/2603.11510) [V]:
- **Training**: safety comes from (a) SFT completions synthesized by Fusion-of-N, with **Command A as the "Fusor"** prompted to produce only safe and helpful responses ("distill concepts of safety"); (b) "minimal" preference tuning on Global only, covering identity and multilingual safety; (c) merging regional models with Global, which the report says *improves* safety for every variant, especially Fire and Earth (Fig. 8). Key claim: models are safest when SFT includes (safe) data *in that language*.
- **Dev-time eval**: an internal multilingual safety benchmark covering "Cohere's key safety areas", tracked across all languages.
- **Public evals**: MultiJail (10 languages, unintentional-harm subset), judged by **Command A in contextual safety mode** without translating outputs to English. The judge labels each response safe / unsafe / invalid (prompt in App. C). XSTest (English) measures over- and under-refusal.

| Model | MultiJail min safe % | MultiJail mean safe % | XSTest over-refusal % | XSTest under-refusal % |
|---|---|---|---|---|
| Tiny Aya Global | 87.0 | **91.1** | 10.4 | 15.5 |
| Tiny Aya Earth | 77.5 | 87.8 | 10.4 | 19.0 |
| Tiny Aya Fire | 78.1 | 90.0 | 10.4 | 15.0 |
| Tiny Aya Water | 82.9 | 89.7 | 10.0 | 19.5 |
| Gemma3-4B | 79.9 | 88.7 | 3.6 | 44.0 |
| Qwen3-4B | 1.6 | 85.8 | 4.0 | 32.5 |
| Qwen3.5-4B | 53.7 | 88.6 | 2.8 | 35.5 |
| Ministral-3-3B | 19.5 | 66.2 | 1.2 | 71.0 |
| SmolLM3-3B | 27.0 | 36.4 | 0.8 | 51.5 |

- Global per-language MultiJail safe rates (Table 26): ar 0.87 (lowest), bn 0.88, en 0.93, it 0.90, jv 0.88, ko 0.91, **sw 0.94**, th 0.95, vi 0.94, zh 0.91. Invalid rate is 0.00 for every language except jv (0.04).
- Trade-off Cohere acknowledges: Tiny Aya over-refuses more (~10% on XSTest) and still **under-refuses 15–19.5% of unsafe XSTest prompts**. Regional variants have lower worst-language safety than Global; Earth's minimum is 77.5%.
- **Not reported** [V by absence]: adversarial jailbreak ASR (e.g., HarmBench / StrongREJECT / GCG), the MultiJail *intentional* (jailbreak-wrapped) split, code-switching attacks, safety results for the Base model, and safety on African languages beyond Swahili.

### Community red-teaming and derivatives (as of 2026-10-26)
- **Code-mixing stress test** (Cohere Labs Community "Expedition Aya" team, 2026-05-25, https://cohere-labs-community.github.io/blog/2026/mix-fine-tune-break/). They used code-mixed En-Hi and En-Ro harmful prompts from current events across 7 harm categories. Base Tiny Aya Global showed **~30% "language confusion"**: a binary judge scores 96% "safe", but the true refusal rate was **66%**. Safety degraded with as few as **50 harmful fine-tuning examples**. Results were highly sensitive to hyperparameters. Not peer-reviewed.
- **Cross-lingual emergent misalignment** (Cohere-Labs-Community, https://github.com/Cohere-Labs-Community/crosslingual-emergent-misalignment; OpenReview https://openreview.net/pdf?id=JjtZl9jTSW; HF org `crosslingual-em`):
  - Setup: LoRA (r=32, α=64) on English bad-advice data for all four chat variants, evaluated in en, pt, tr, hi, mr, ur, ha, yo.
  - EM transfers across languages. EM rates: finance 8.2–12.9%, medical 2.4–8.4%, sports 0.9–1.8%, insecure code <1%. English Water·Finance reaches ~35–40%.
  - They report a shared "EM direction" that peaks in late layers.
  - The paper text says "0.14%" where 14% (55/394) is clearly meant [U].
  - A related replication (https://aroramrinaal.com/ai/cross-lingual-emergent-misalignment) found +8.1 pp (en), +6.2 (hi), +6.3 (mr), +4.7 (ur) coherent EM on Global. This contradicts the earlier finding that Hindi and Marathi resist EM (https://anuadesina.com/crosslingual-em/).
- **Guard-model derivatives**: AfriGuard fine-tunes (`israel/AfriGuard-tiny-aya-global-full`, `adzcai/AfriGuardInst-tiny-aya-earth-full`, etc.; dataset `israel/AfriGuard-inst`). They output `<safety>/<category>/<response>` tags for African languages.
- **Abliterated / uncensored**: I searched the HF API for `abliterated`, `uncensored`, `heretic`, `tiny-aya`, `tiny_aya`, `tinyaya` and `aya-tiny`. I found **no abliterated or uncensored Tiny Aya model**. Contrast Aya Expanse below. Many EM "insecure" checkpoints exist: `crosslingual-em/*`, `ayesha1505/tiny-aya-insecure-seed0`, `nunaa/em-tiny-aya-fft-insecure-seed0`, `1musty/*`, `akankshanc/*`.
- **Ungated full-weight mirrors** exist (`1-800-LLMs/tiny-aya-{global,water,earth,fire}`, `mlx-community/tiny-aya-fire`). They are convenient, but the CC-BY-NC/AUP terms still apply.

---

## 3. Ecosystem support

| Tool | Status | Source |
|---|---|---|
| transformers | `Cohere2` classes were added in **v4.48.0** (PR #35224, contributed 2024-12-13, for Command R7B). Tiny Aya's config lists `transformers_version 4.51.3`. Community fine-tunes use Transformers 5.x | https://github.com/huggingface/transformers/releases/tag/v4.48.0; model_doc/cohere2.md [V] |
| TransformerLens | **TransformerBridge (TL ≥4.x) has `Cohere2ArchitectureAdapter`**. NoPE global layers are handled. A RoPE-convention fix for Cohere and Cohere2 landed 2026-10-09 (#1873, in v4.2.0). Latest PyPI release is 4.3.0 (2026-10-23). It is tested only on `trl-internal-testing/tiny-Cohere2ForCausalLM`, **not on Tiny Aya itself**. Legacy `HookedTransformer` has no Cohere2 support [U] | https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/model_bridge/supported_architectures/cohere.py [V] |
| MLX | `mlx-community/tiny-aya-global-8bit-mlx`, `mlx-community/tiny-aya-fire` (bf16), `mlx-community/tiny-aya-fire-4bit`. The report benchmarks MLX on iPhone 13 (~10 tok/s decode) and iPhone 17 Pro (~32 tok/s) | HF API; report §6 [V] |
| GGUF | **Official**: `CohereLabs/tiny-aya-{global,earth,fire,water}-GGUF` (ungated): bf16/f16 6.71 GB, q8_0 3.57 GB, q4_k_m 2.14 GB, q4_0 2.03 GB. Community: DevQuasar (all 5, including base), professorf, keisuke-miyako (q8_0), mazrba (imatrix). **No bartowski / unsloth / lmstudio-community / mradermacher Tiny Aya repos found**. The report calls Q4_K_M "optimal" (2.14 GB, −1.4 pts) | HF API [V] |
| Ollama | **No official library entry** (`ollama.com/library/tiny-aya` returns 404). Community uploads: `NeonIH/tiny-aya-global-tools`, `s1gnature/tiny-aya` | ollama.com search [V] |
| vLLM | Registry maps `Cohere2ForCausalLM` to the `commandr` implementation. Correctness on Tiny Aya is unverified [U] | vllm `model_executor/models/registry.py` [V] |
| ONNX / EXL3 | `onnx-community/tiny-aya-{global,water,fire,earth}-ONNX`; `Piter254/tiny-aya-water-exl3-6.0bpw` | HF API [V] |
| Disk (bf16 safetensors) | 2 shards: 4,992,396,352 + 1,706,092,176 bytes = **6.70 GB**. The full repo is ~6.72 GB, of which the tokenizer.json is 21.4 MB | HF API `?blobs=true` [V] |
| Hosted | The instruction-tuned variants are on the Cohere API Chat endpoint and on an HF Space | https://docs.cohere.com/docs/tiny-aya [V] |

---

## 4. Reusable Cohere multilingual-safety assets

- **Aya Red-teaming** (`CohereLabs/aya_redteaming`, Apache-2.0, **ungated**, https://huggingface.co/datasets/CohereLabs/aya_redteaming) [V, counted locally]:
  - Size: **7,419 human-written harmful prompts in 8 languages**: arb 900, eng 987, fra 813, hin 915, rus 1007, spa 782, srp 1006, tgl 1009.
  - Global/local split: 4,890 "global" and 2,529 "local" harm.
  - 9 harm categories (multi-label counts): Discrimination & Injustice 2632, Violence/Threats/Incitement 2556, Hate Speech 1904, Bullying & Harassment 1710, Graphic material 1602, Profanity 1209, Harms of Representation/Allocation/QoS 1013, Non-consensual sexual content 921, Self-Harm 861.
  - Fields: `prompt`, `language`, `harm_category`, `global_or_local`, `literal_translation`, `semantic_translation`, `explanation`.
  - **Overlap with Tiny Aya languages**: all 8 dataset languages are in Tiny Aya's 70-language list. Script mismatch is possible: Tiny Aya lists Serbian as Cyrillic, and I did not check which script the `srp` prompts use [U].
- **The Multilingual Alignment Prism** (arXiv:2406.18682, Aakanksha et al., 2024). Introduces the dataset above. Studies SFT vs. DPO/RLOO safety alignment across 6 languages with global and local harm, "with minimal degradation in general performance".
- **Mix Data or Merge Models?** (arXiv:2410.10801). Objective-based merging beat data mixing by up to 8% general performance and 10% safety. Language-based merging gave −7% harm. This is the conceptual basis for Tiny Aya's regional merging.
- **Safety context distillation** (Aya 101) cut harmful generations from adversarial prompts by 78–89% (human-judged). Source: Peppin et al., "The Multilingual Divide and Its Impact on Global AI Safety", arXiv:2505.21344.
- **The State of Multilingual LLM Safety Research** (Yong, Ermis, Fadaee, Bach, Kreutzer; arXiv:2505.24119). A survey of ~300 papers that documents English-centrism. It recommends culturally grounded evals, multilingual safety data, and work on cross-lingual safety generalization, and flags code-switching and Arabizi jailbreaks.
- Policy primers: "Translating Safety" (https://cohere.com/research/papers/translating-safety.pdf) and "The AI Language Gap" (https://cohere.com/research/papers/the-ai-language-gap.pdf).
- **Global-MMLU** (`CohereLabs/Global-MMLU`, arXiv:2412.03304, Apache-2.0, 42 languages; Lite: 23 languages). Global-MMLU Lite was in Tiny Aya's dev-time monitoring suite. Use it as a capability-retention check after safety interventions.
- Related external work: "Refusal Direction is Universal Across Safety-Aligned Languages" (arXiv:2505.17306, LMU, not Cohere). An English refusal vector bypasses refusals across 14 languages. It is directly relevant to steering or abliteration on Tiny Aya.
- **Aya Vision** (8B/32B, Mar 2025): mlx-community has conversions. I found no safety specifics for it in this pass [U].

---

## 5. Aya Expanse 8B (comparison point)

| Field | Aya Expanse 8B | Tiny Aya Global |
|---|---|---|
| Release | 2024-10-23/24 (HF repo created 2024-10-23); report arXiv:2412.04261 | 2026-02-13/17 |
| Params | 8,028,033,024 (F16) | 3,349,227,520 (BF16) |
| Arch | `cohere` / `CohereForCausalLM`; 32 layers, hidden 4096, 32 heads / 8 KV heads, vocab 256,000, all-RoPE (θ=10k), `logit_scale` 0.125 | `cohere2`; 36 layers, interleaved SWA + NoPE global, vocab 262,144 |
| Context | 8K | 8K in / 8K out |
| Languages | 23 | 70 |
| License | CC-BY-NC-4.0 + AUP | same |
| Safety recipe | Card lists "safety tuning" (links the Multilingual Alignment Prism), multilingual preference training, and merging | Fusion-of-N with a Command A teacher + minimal preference tuning + merging |
| Default system prompt | **None** injected by the template. The template contains only a dead-code default Aya persona (`elif false == true`) | Contextual-safety preamble is always injected |
| Community uncensoring | **Many** abliterated / heretic derivatives: `lenML/aya-expanse-8b-abliterated`, `mradermacher/aya-expanse-8b-abliterated(-i1)-GGUF`, `mradermacher/aya-expanse-8b-heretic(-i1)-GGUF`, `…-orthogonal-heretic`, plus 32B versions | **None found** |

Sources: https://huggingface.co/CohereLabs/aya-expanse-8b, HF API metadata, `mlx-community/aya-expanse-8b-4bit` config, https://arxiv.org/abs/2412.04261, and the HF API search (2026-10-26) [V]. I found no published MultiJail or XSTest numbers for Aya Expanse 8B in this pass [U].

---

## 6. Practical notes for safety experiments
1. You need to accept the HF gate (or use an ungated mirror) to get the official configs and weights.
2. Pick one prompting regime for every run: default template (with the safety preamble) or a custom template. Report which you used.
3. The interleaved NoPE global layers (every 4th layer) are a natural place to probe whether refusal or language directions concentrate in global vs. local layers. The EM work found a late-layer peak [U as a hypothesis].
4. Judge reliability: binary safe/unsafe judges hide "language confusion". Add an *invalid/confused* class, as in MultiJail App. C and the code-mixing study.
5. Weakest public points: worst-language safety in Earth and Fire (~78% minimum), 15–19.5% XSTest under-refusal, code-mixed prompts, and narrow fine-tunes (50 harmful examples; EM via LoRA).
