# 03 — Open-Source Tools Survey: Multilingual Safety for Tiny Aya on Apple Silicon

*Survey date: 2026-10-26. Every repo, dataset, and model below was checked live against the GitHub API, the Hugging Face Hub API, or PyPI on that date. Stars and last-push dates come from the GitHub API, and gating and licence come from HF Hub metadata. Anything not verified is labelled **(unverified)**.*

**Target:** `CohereLabs/tiny-aya-global` (also `-base`, `-earth`, `-fire`). `Cohere2ForCausalLM` / `model_type: cohere2`, 3,349,227,520 params, CC-BY-NC-4.0, **gated=auto** (click-through on HF). The bf16 safetensors are 6.7 GB. The official GGUF repo `CohereLabs/tiny-aya-global-GGUF` is not gated: q4_k_m is 2.14 GB, q8_0 is 3.57 GB, bf16 is 6.71 GB. `mlx-community/tiny-aya-global-8bit-mlx` is 3.56 GB.

**Hardware budget:** M1 Pro, 16 GB unified memory (MPS, no CUDA), about 17 GB free disk. Free Colab T4 is optional.

**Required core:** Lexsi `safetune` 0.1.10 (harden/recover/unlearn/steer), `circuitkit` 0.2.0 (circuit discovery and interventions), `auditkit` 1.1.3 (evaluate any model on any dataset), and `curatorkit` 0.1.22 (curation and synthetic data). All four are on PyPI with repos under `github.com/Lexsi-Labs`. This document covers only the tools that **complement** them: data, judges, cross-checks, Apple-Silicon runtimes, and figures.

> **Finding to act on first:** Tiny Aya's default chat template injects a system preamble: "You are in contextual safety mode… you will reject requests to generate CSAM… accept to provide information… related to violence, hate… but not content that could directly or indirectly lead to harmful outcomes." Every baseline must state whether this default preamble is on. A custom `system` message gets appended as a "Developer Preamble" and does not replace the default. This alone can move refusal rates by many points, so it is a confound to control and report.

---

## 1. Multilingual safety datasets

### 1a. Harmful-prompt sets (measure attack success and refusal rate)

| Dataset (HF id / repo) | Languages | Size | Licence | Gating | Use it for |
|---|---|---|---|---|---|
| `DAMO-NLP-SG/MultiJail` | 10 (en, zh, it, vi, ar, ko, th, bn, sw, jv) | 315 prompts × 10 (parallel) | MIT | open | **Headline cross-lingual jailbreak eval.** Parallel structure gives clean per-language comparisons across high/mid/low-resource tiers. |
| `CohereLabs/aya_redteaming` | 8 (en, hi, fr, es, ar, ru, sr, tl) | ~7K (1K–10K bucket) | Apache-2.0 | open | **Native-speaker (not translated) harmful prompts from Cohere itself**, the most on-distribution red-team set for Aya. Includes harm categories and "global vs local" harm labels. |
| `ToxicityPrompts/XSafety` | 10 (en, zh, ar, es, fr, de, bn, ru, ja, hi) | 27,999 (≈2.8K/lang) | ODC-By | open | Broad 14-category safety sweep. Mirror of the XSafety benchmark (repo `Jarviswang94/Multilingual_safety_benchmark`, Apache-2.0). |
| `ToxicityPrompts/RTP-LX` | 28 language codes (incl. sw, he, th, hi, ar, uk, BCMS) | 30,253 (~1K/lang) | ODC-By | open | **Toxic-completion / culturally specific toxicity** in many languages. The repo `microsoft/RTP-LX` is stale (2025-05). |
| `ToxicityPrompts/PolyGuardPrompts` | 17 (ar, zh, cs, nl, en, fr, de, hi, th, it, ja, ko, pl, pt, ru, es, sv) | 29,325 | CC-BY-4.0 | open | Held-out eval with **prompt-harm, response-harm and refusal labels**. Lets you check your judge as well as the model. |
| `ToxicityPrompts/PolyGuardMix` | same 17 | 1,910,372 | CC-BY-4.0 | open | **Training pool** for SFT/DPO refusal data. Subsample with CuratorKIT. Too big to download whole: stream it. |
| `felfri/M-ALERT` | 5 (en, de, fr, it, es) | 14,763 rows × 5 langs (~75K) | **CC-BY-NC-SA-4.0** | open | Fine-grained 32-category taxonomy (ALERT) for category-level heatmaps. |
| `sorry-bench/sorry-bench-202503` | en + mutations in zh, fr, ml, mr, ta | 440 base × 21 variants | other (custom ToU) | auto | **44 fine-grained refusal categories plus linguistic mutations** (5 languages, 6 writing styles, persuasion, encodings). The repo `SORRY-Bench/sorry-bench` ships an evaluator. |
| `NASK-PIB/RefusEU` | 12 European (pl, en, cs, sk, sl, lt, lv, de, it, fr, es, pt) | ~86K rows; 1,400/lang test | `llama3.1` tag | open | DPO-ready refusal preference pairs plus a balanced test split on the Llama-Guard 14-hazard taxonomy (ACL Findings 2026). |
| PolyRefuse in `github.com/mainlp/Multilingual-Refusal` | 14 (ar, de, en, es, fr, it, ja, ko, nl, pl, ru, th, zh, yo) | 128 train/lang + val/test, harmful **and** harmless | Apache-2.0 | GitHub | **Ready-made contrastive pairs for refusal-direction extraction per language**, from "Refusal Direction is Universal Across Safety-Aligned Languages". Directly reusable for steering. |
| `JailbreakBench/JBB-Behaviors` | en | 100 harmful + 100 matched benign | MIT | open | Small, standard, citable English anchor. |
| `walledai/HarmBench` | en | ~400 behaviours (standard/contextual/copyright) | MIT | auto | Standard English anchor. Translate with NLLB for a parallel multilingual set. |
| `walledai/AdvBench` | en | 520 | MIT | auto | Classic refusal-direction source (Arditi et al.). |
| `walledai/StrongREJECT` | en | 313 | MIT | auto | Pairs with the StrongREJECT rubric/judge for "harmfulness × specificity" scoring. |
| `allenai/wildjailbreak` | en | 262K train + 2.2K eval (adversarial and benign) | ODC-By | auto (AI2 ToU) | Adversarial jailbreak tactics plus **adversarial-benign** contrasts, useful for anti-over-refusal training. |
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | en | ~33K | CC-BY-4.0 | open | Response-level human safety labels. Use it to calibrate judges. |

### 1b. Benign-but-scary prompts (measure over-refusal)

| Dataset | Languages | Size | Licence | Gating | Use it for |
|---|---|---|---|---|---|
| `Paul/XSTest` (also `walledai/XSTest` auto-gated, repo `paul-rottger/xstest` CC-BY-4.0) | en | 250 safe + 200 unsafe contrasts | CC-BY-4.0 | open | **Primary over-refusal probe.** Translate with NLLB plus GlotLID QA to build your own multilingual XSTest. `jkminder/xstest-overrefusal` is a cleaned 214-prompt safe-only subset. |
| `Aaron-Pan/MORBench` | 7 (en, zh-cn, it, ja, fr, de, es) | 16,800 rows (2,400/lang, 12 categories) | MIT | open | **The only sizeable public *multilingual* over-refusal benchmark found** (paper arXiv 2505.18325). Single CSV of 2.6 MB. Note: the "train" split is their test set. |
| `bench-llm/or-bench` | en | 80K / hard-1K / toxic-600 | CC-BY-4.0 | open | Use `or-bench-hard-1k` as a stress test, paired with `or-bench-toxic` so a model can't win by never refusing. |
| `HiTZ/safety-SeGa` | eu, es, en | 120 safe + 120 unsafe | (check card) | open | Low-resource (Basque) XSTest-style contrast set. |

**Not found:** no public HF dataset named **"MultiXSTest"** exists. Searches of the Hub and the web found nothing under that name. Build it yourself (XSTest → NLLB → GlotLID filter → native spot check), or use MORBench.

**Recommended core eval set (fits in <50 MB):** MultiJail + aya_redteaming (harmful) · XSTest translated to 8–10 langs + MORBench (benign) · PolyGuardPrompts (judge validation). Use PolyRefuse or PolyGuardMix subsamples for steering and training pairs.

---

## 2. Safety judges and classifiers (local)

Memory rule of thumb: bf16 ≈ 2 bytes/param. Tiny Aya bf16 takes about 7 GB resident, or about 2–3.5 GB as 4/8-bit MLX/GGUF.

| Model | Params | Languages | Licence | Gated | Fits alongside Tiny Aya on 16 GB? | Use it for |
|---|---|---|---|---|---|---|
| `ToxicityPrompts/PolyGuard-Qwen-Smol` | 0.49B (Qwen2.5-0.5B) | **17** | CC-BY-4.0 | no | **Yes, easily** (file is 1.98 GB) | **Primary judge.** Outputs prompt-harm, response-harm **and response-refusal** labels in 17 languages. GGUF: `mradermacher/PolyGuard-Qwen-Smol-GGUF`. |
| `ToxicityPrompts/PolyGuard-Qwen` | 7.6B | 17 | CC-BY-4.0 | no | Only as GGUF Q4 (~4.7 GB), run sequentially | Higher-accuracy re-judge on a stratified sample. GGUF: `mradermacher/PolyGuard-Qwen-GGUF`. |
| `ToxicityPrompts/PolyGuard-Ministral` | 8.0B | 17 | CC-BY-4.0 | no | GGUF Q4, sequential | Alternative 8B PolyGuard. |
| `Qwen/Qwen3Guard-Gen-0.6B` | 0.75B | **119 languages/dialects** | Apache-2.0 | no | **Yes** (1.5 GB) | **Second, independent judge** with the widest language coverage. 3-way Safe/Controversial/Unsafe plus a refusal flag. `-Gen-4B` (4.4B) and `-Gen-8B` exist, plus `-Stream-*` for token-level moderation. |
| `meta-llama/Llama-Guard-3-1B` | 1.5B | 8 (en, fr, de, hi, it, pt, es, th) | Llama 3.2 | **manual** | Yes (3.0 GB) | Industry-standard MLCommons-hazard taxonomy, so reviewers recognise it. Weak outside its 8 languages. |
| `meta-llama/Llama-Guard-4-12B` | 12B (multimodal) | multilingual (same core set) | Llama 4 | **manual** | No (Q4 GGUF ~7 GB at best, and slow) | Skip locally. Colab only, if needed at all. |
| `google/shieldgemma-2b` (also 9b, 27b) | 2.6B | English-focused | Gemma | **manual** | Yes (~5.2 GB) | Per-policy probability scores. Low value for multilingual work. |
| `allenai/wildguard` | 7.2B | English | Apache-2.0 | auto | GGUF Q4 only | English harm + refusal detection. Superseded by PolyGuard for this project. |
| `ibm-granite/granite-guardian-3.1-2b` / `-3.2-3b-a800m` / `-4.1-8b` | 2.5B / 3.3B MoE / 8.4B | primarily English | Apache-2.0 | no | 2B: yes (~5 GB) | Groundedness and "harm" risk scores. Repo `ibm-granite/granite-guardian` was active 2026-10. Optional. |
| `qylu4156/strongreject-15k-v1` | LoRA on `google/gemma-2b` | English | (no licence tag; base is Gemma, manual-gated) | base gated | Yes | StrongREJECT fine-tuned judge (via `dsbowen/strong_reject`, MIT). Use for English "harmfulness × specificity" scores, comparable with papers. |
| `cais/HarmBench-Mistral-7b-val-cls` / `-Llama-2-13b-cls` | 7.2B / 13B | English | MIT | no | 7B as GGUF only; 13B no | Only if you need HarmBench-comparable ASR numbers. |

**Recommendation:** PolyGuard-Qwen-Smol is the primary judge. Qwen3Guard-Gen-0.6B is the cross-check, so you can report judge agreement (Cohen's κ). Re-judge a stratified ~300-item sample with PolyGuard-Qwen-7B Q4 via llama.cpp after unloading Tiny Aya, and hand-label 50–100 items in 2–3 languages to report judge accuracy. For non-English **response-language** failures (the model answers in English), add a GlotLID check (section 6). Neither judge catches that.

---

## 3. Steering and representation-engineering libraries

| Library | Repo / PyPI | Licence · stars · last push | Works with arbitrary HF causal LM on MPS? | Use it for |
|---|---|---|---|---|
| **nnsight** | `ndif-team/nnsight` · `nnsight` 0.8.1 | MIT · 1.1K · 2026-10-22 | **Yes.** Wraps any `torch.nn.Module`. Run locally with `LanguageModel(id, device_map="mps")`. No architecture adapter needed. | **Default tracing and patching tool** for cohere2: activation capture, diff-in-means, ablation, patching. Cross-checks CircuitKIT results. |
| **TransformerLens v4** | `TransformerLensOrg/TransformerLens` · `transformer-lens` 4.3.0 (2026-10-23) | MIT · 4.0K | **Cohere2 IS supported** via `TransformerBridge`: `Cohere2ArchitectureAdapter` (PR #1485, 2026-07-06), with a RoPE/NoPE-interleaving fix (#1873, 2026-10-09). Both are in 4.3.0. MPS via `device="mps"`. Tiny Aya itself not tested, so **(unverified for Tiny Aya specifically)**. | Logit lens, attention-pattern figures, hook-based ablations with familiar `hook_resid_post` names. **Pin ≥4.3.0.** The legacy `HookedTransformer` path does not cover cohere2. |
| **repeng** | `vgel/repeng` · `repeng` 0.4.0 | MIT · 765 · 2025-09 | Mostly yes: wraps `model.model.layers`, pure torch + sklearn PCA. Cohere2 has that layout. **(unverified on cohere2)** | Fastest "control vector" baseline (PCA/mean-diff from contrastive pairs) in ~20 lines. Can export GGUF control vectors for llama.cpp `--control-vector`. |
| **steering-vectors** | `steering-vectors/steering-vectors` · 0.12.2 | MIT · 165 · 2025-02 | Yes. Layer matching is auto-guessed with a custom `layer_config` override. | Clean API for CAA-style vectors with token-position control. Good for ablating where to steer. |
| **IBM activation-steering (CAST)** | `IBM/activation-steering` (git install, not on PyPI) | Apache-2.0 · 195 · 2026-10-23 | Yes in principle (HF + hooks) **(unverified on MPS/cohere2)** | **Conditional steering:** steer toward refusal *only when* a condition vector detects harm. Directly targets the safety vs over-refusal trade-off. Strong paper-worthy baseline. |
| **dialz** | `cardiffnlp/dialz` · `dialz` 1.1.4 | MIT · 28 · 2026-03 | Yes (repeng-derived) | Steering with built-in datasets and visualisation. Optional. |
| **pyvene** | `stanfordnlp/pyvene` · 0.1.8 | Apache-2.0 · 914 · 2026-03 | Yes (generic torch) | Interchange interventions / DAS / ReFT-style trained interventions. Use only if you need learned rather than mean-diff directions. |
| **refusal_direction** | `andyrdt/refusal_direction` | Apache-2.0 · 468 · 2025-06 | Research code with per-family model wrappers (no cohere2), CUDA assumed | **Reference implementation** of diff-in-means refusal direction, direction selection, and weight orthogonalisation. Port its selection logic; don't run it as-is. |
| **mainlp/Multilingual-Refusal** | GitHub | Apache-2.0 · 19 · 2025-11 | Research code | Multilingual refusal-direction extraction plus PolyRefuse data. **Closest prior work.** Cite it, and note the contrary 2026 MRL-workshop finding that the English direction did *not* transfer causally in Qwen2.5-7B / Llama-3.2-3B. |
| **heretic** | `p-e-w/heretic` · `heretic-llm` 1.4.0 | **AGPL-3.0** · 36.5K · 2026-10-24 | Dense HF models; Optuna search is compute-heavy on M1 | **Attack/diagnostic only:** automated abliteration shows how removable Tiny Aya's refusal is per language. AGPL, so keep it out of anything you redistribute. Lighter alternatives: `Tsadoq/ErisForge` (no licence file), `FailSpy/abliterator` (MIT, stale 2024). |
| **EasySteer** | `ZJU-REAL/EasySteer` | Apache-2.0 · 312 · 2026-10-25 | **No.** Built on vLLM, so it needs CUDA. | Colab T4 only, for high-throughput steering sweeps. Not for the laptop. |

**Recommendation:** use **nnsight** to capture activations and hook interventions, **repeng** for quick control-vector baselines, and **IBM CAST** for the conditional-steering comparison against SafeTune's `steer`. Use TransformerLens 4.3 only for interpretability figures, after a 5-minute logit-parity check on Tiny Aya.

---

## 4. Eval harnesses for capability retention

| Harness | Version / activity | Multilingual tasks (verified in repo) | M1 suitability | Use it for |
|---|---|---|---|---|
| **lm-evaluation-harness** (`EleutherAI/lm-evaluation-harness`, `lm-eval` 0.4.13) | MIT · 14.3K · 2026-09 | `global_mmlu`, `belebele`, `xcopa`, `mgsm`, `okapi/arc_multilingual` (m_arc), `okapi/hellaswag_multilingual`, `xnli`, `xstorycloze`, `xwinograd`, `include`, `openai-mmmlu`, `afrimgsm`, `afrixnli` | **Good.** `--model hf --device mps`, or **`mlx_lm.evaluate`** (ships in mlx-lm, wraps lm-eval with an MLX backend, faster). Use `--limit` to subsample. | **Capability-retention suite.** Log-likelihood tasks (Global-MMLU-Lite, Belebele, XCOPA, m_arc) are cheap. MGSM is generative and slow, so subsample it. |
| **inspect_ai** + **inspect_evals** | `inspect-ai` 0.3.285 (2026-10-25); `UKGovernmentBEIS/inspect_evals` MIT · 718 | Ready tasks: `xstest`, `strong_reject`, `mgsm`, `agentharm`, `wmdp`, `mask`, `b3` | **Good** against a local OpenAI-compatible server (`mlx_lm.server` or llama.cpp `llama-server`). Excellent logs/viewer. | **Safety-eval runner** with model-graded scorers. Swap the grader to a local PolyGuard endpoint. Logs are reproducible and good for appendices. |
| **lighteval** (`huggingface/lighteval` 0.13.0) | MIT · 2.6K · 2026-10 | Large `tasks/multilingual` module | OK but heavier, and its accelerate/vLLM backends are CUDA-oriented | Optional alternative to lm-eval. |
| **HELM** (`stanford-crfm/helm`) | Apache-2.0 · 3.0K | HELM Safety | **Too heavy** for M1 (API-judge-oriented, large deps) | Cite its taxonomy only. |

**Datasets for retention:** `CohereLabs/Global-MMLU-Lite` (23 langs, Apache-2.0, open) is preferred over full Global-MMLU (42 langs, 100K–1M rows). Also `facebook/belebele` (96 langs on HF, CC-BY-SA-4.0), `cambridgeltl/xcopa` (11 langs, CC-BY-4.0), and `juletxara/mgsm` (10 langs, CC-BY-SA-4.0). **Budget estimate (unverified):** Global-MMLU-Lite + Belebele subset (200/lang × 10 langs) + XCOPA comes to roughly 1–2 h on M1 Pro with MLX 8-bit.

---

## 5. Apple-Silicon training and inference

| Tool | Status for cohere2 | Use it for |
|---|---|---|
| **MLX / mlx-lm** (`ml-explore/mlx-lm`, `mlx-lm` 0.32.2, MIT · 7.5K · 2026-10-24) | **`mlx_lm/models/cohere2.py` exists.** `mlx-community/tiny-aya-global-8bit-mlx` and `tiny-aya-fire-4bit` are published, so loading is proven by the community. Includes `lora.py` (LoRA/QLoRA/DoRA, grad checkpointing), `fuse.py`, `convert.py` (`-q --q-bits 4`), `server.py`, `evaluate.py`. | **Main local fine-tuning path.** QLoRA on a 4-bit Tiny Aya fits easily in 16 GB. Fast generation for mass eval. For DPO/ORPO on MLX, the third-party `mlx-lm-lora` (PyPI) adds preference trainers **(not vetted here)**. |
| **llama.cpp** (`ggml-org/llama.cpp`, MIT · 132K · 2026-10-26) | `src/models/cohere2.cpp` present, plus the official Cohere GGUFs. | Run 7–8B judges as Q4 GGUF. `llama-server` gives an OpenAI-compatible endpoint for inspect/promptfoo/garak. `--control-vector` applies repeng vectors at inference. |
| **PyTorch MPS + PEFT/TRL** (`peft` 0.21.2, `trl` 1.16.0) | `Cohere2ForCausalLM` is native in transformers | Needed wherever Lexsi tools, nnsight, or CAST want HF modules. Caveats: (1) no bitsandbytes 4-bit QLoRA on MPS, so LoRA runs in fp16/fp32 (~7 GB weights in fp16, plus activations: tight but feasible with batch 1–2, seq ≤512, and gradient checkpointing). (2) bf16 on MPS needs macOS 14+, and M1-generation GPUs have no native bf16 math, so prefer **fp16 for inference and fp32 LoRA params**. Watch for fp16 overflow/NaN (Cohere models use a logit scale). (3) No FlashAttention: use `attn_implementation="sdpa"` or `"eager"`. (4) Set `PYTORCH_ENABLE_MPS_FALLBACK=1`. (5) Some ops in sliding-window attention may fall back to CPU. |

**Throughput expectations (estimates from M1 Pro's ~200 GB/s memory bandwidth and public MLX benchmarks, unverified on this exact machine):** decode is roughly 15–25 tok/s in bf16/fp16 (PyTorch MPS, slower end), about 35–50 tok/s at 8-bit MLX, and about 50–80 tok/s at 4-bit MLX. Batched prompt processing runs at several hundred tok/s. MLX QLoRA training on a 3B 4-bit model should land in the low hundreds of tokens/s at batch 1–4. Plan about 30–90 min for a 1–2K-example SFT run of 1–2 epochs. **Colab T4** (16 GB, fp16 only, no bf16) suits PyTorch LoRA/DPO with TRL and EasySteer, but its free-tier session limits make it a burst resource.

---

## 6. Translation and language ID

| Tool | Size / licence | Use it for |
|---|---|---|
| `facebook/nllb-200-distilled-600M` | 2.46 GB (pytorch_model.bin) · **CC-BY-NC-4.0** · open | **Default translator** for building multilingual XSTest/HarmBench (200 languages, runs fine on MPS/CPU). Non-commercial licence, which suits a research hackathon. `-1.3B` is better and twice the size. |
| `google/madlad400-3b-mt` | Apache-2.0 · open. **Warning:** the repo holds an 11.76 GB safetensors **and** GGUFs (q4k 1.65 GB), so download with `allow_patterns`. | Apache-licensed alternative with 400+ languages. Use the GGUF via llama.cpp (T5 support) or `Nextcloud-AI/madlad400-3b-mt-ct2-int8` with CTranslate2. |
| Tiny Aya itself | already on disk | Back-translation sanity check, and "self-translation" prompts. Avoid using it as the sole translator of its own eval set, since that circularity biases results. |
| `cis-lmu/glotlid` (repo `cisnlp/GlotLID`) | model.bin 1.69 GB · licence "other" (Apache-2.0 for the model per authors, **verify**) · open | **Language-ID QA**: filter bad translations and detect when Tiny Aya **answers in the wrong language** (a key multilingual failure mode). Covers 2,000+ labels including low-resource languages. |
| fastText `lid.176.ftz` / `lid.176.bin` | 0.9 MB / 126 MB · CC-BY-SA-3.0 (from fasttext.cc; HF mirror `facebook/fasttext-language-identification` is CC-BY-NC-4.0) | Tiny, fast first-pass LID for the 176 major languages. PyPI wrapper `fasttext-langdetect` 1.1.1. |

Pipeline: English source → NLLB-600M → GlotLID confidence ≥0.9 filter → back-translate a 10% sample and score chrF (`sacrebleu`) → native-speaker spot check of 20 items per language where you can.

---

## 7. Red-teaming frameworks

| Framework | Repo · version | Multilingual capability | Use it for |
|---|---|---|---|
| **garak** | `NVIDIA/garak` · 0.17.1 · Apache-2.0 · 9.7K · 2026-10-23 | **`langservice` translation layer:** set `run.target_lang` and translate probe prompts plus detector triggers **locally** with Helsinki-NLP opus-mt or `facebook/m2m100_418M/1.2B`, or remotely with DeepL/Riva/Google. The `LRLBuff` (low-resource languages) requires DeepL. | **Broad automated vulnerability scan** (encoding, DAN, latent injection, etc.) against `llama-server`/`mlx_lm.server`, run in 2–3 languages. Caveat: its HF detectors are mostly English. |
| **PyRIT** | `microsoft/PyRIT` · `pyrit` 1.1.0 · MIT · 4.7K · 2026-10-26 (the old `Azure/PyRIT` is **archived**) | `TranslationConverter` / `RandomTranslationConverter` (LLM-driven, so point them at a local OpenAI-compatible endpoint) | Programmatic multi-turn attacks (Crescendo, PAIR/TAP-style orchestration) with language conversion. Heavier setup. |
| **promptfoo** | `promptfoo/promptfoo` · MIT · 26.4K · 2026-10-26 (Node CLI, `npx promptfoo`) | Top-level `redteam.language: [...]` generates tests per language (the old `multilingual` strategy is deprecated). Default suggestions are bn/sw/jv. | Fast YAML-driven regression red-team between model versions. **Caveat:** test generation defaults to a hosted OpenAI model, so configure a local provider for an offline/no-API setup. |

**Recommendation:** garak (local m2m100 translation) is the one automated scan to report. Static datasets carry the main results, because they are reproducible.

---

## 8. Statistics and paper-grade figures

| Tool | Status | Use it for |
|---|---|---|
| `scipy.stats.bootstrap` (BCa) | SciPy ≥1.7 | 95% CIs on per-language refusal/ASR rates; paired bootstrap for Δ(before → after). |
| `statsmodels` (`proportion_confint` Wilson, `mcnemar`, `multipletests`) | stable | Wilson CIs for small per-language n, **McNemar** for paired refusal flips on the same prompts, Holm/BH correction across ~10 languages. |
| `seaborn` + `matplotlib` | stable | Language × category heatmaps, and safety-vs-over-refusal Pareto scatter plots with CI error bars. |
| `SciencePlots` (`garrettj403/SciencePlots`, 2.2.2, MIT · 9.3K · 2026-06) | active | One-line IEEE/NeurIPS-style rcParams (`plt.style.use(['science','no-latex'])`). |
| `rliable` (`google-research/rliable`, 1.2.0) | Apache-2.0 · **archived 2024** | Stratified-bootstrap IQM and probability-of-improvement plots across languages and seeds. Still pip-installable and works, but frozen. Copying its ~100-line stratified bootstrap is an option. |
| `sacrebleu` (chrF) | stable | Translation-quality QA for the multilingual prompt sets. |
| `krippendorff` / `sklearn.metrics.cohen_kappa_score` | stable | Judge-vs-judge and judge-vs-human agreement. |

---

## 9. Recommended stack (sized for 16 GB RAM / 17 GB disk)

| Component | Tool | Why | Install | Disk / peak RAM |
|---|---|---|---|---|
| Core (required) | Lexsi SafeTune, CircuitKIT, AuditKIT, CuratorKIT | Hackathon core: harden/steer/recover, circuits, eval, data curation | `pip install safetune circuitkit auditkit curatorkit` | <1 GB / shares model RAM |
| Target model (HF) | `CohereLabs/tiny-aya-global` bf16 | Needed for PyTorch tools (Lexsi, nnsight, CAST, TL) | `hf auth login` (accept terms), then `hf download CohereLabs/tiny-aya-global` | **6.7 GB** / ~7.5 GB fp16 on MPS |
| Target model (fast) | MLX 4-bit conversion of the same weights | Fast mass generation and QLoRA | `pip install mlx-lm` then `mlx_lm.convert --hf-path CohereLabs/tiny-aya-global -q --q-bits 4` | ~2 GB / ~3 GB |
| Fine-tuning | mlx-lm LoRA (local); PEFT+TRL (Colab T4 for DPO) | cohere2 is native in mlx-lm; QLoRA fits in 16 GB | `pip install mlx-lm peft trl` | adapters <100 MB / 6–9 GB |
| Primary judge | PolyGuard-Qwen-Smol (0.5B) | 17 languages; harm **and** refusal labels; CC-BY | `hf download ToxicityPrompts/PolyGuard-Qwen-Smol` | 2.0 GB / ~1.5 GB |
| Cross-check judge | Qwen3Guard-Gen-0.6B | 119 languages, Apache-2.0, independent family, gives κ | `hf download Qwen/Qwen3Guard-Gen-0.6B` | 1.5 GB / ~1.8 GB |
| Big-judge audit (optional) | PolyGuard-Qwen 7B Q4 GGUF via llama.cpp | Accuracy check on a 300-item sample; run with Tiny Aya unloaded | `brew install llama.cpp`; `hf download mradermacher/PolyGuard-Qwen-GGUF --include "*Q4_K_M*"` | ~4.7 GB / ~5.5 GB (**delete after use** if disk is tight) |
| Interp / steering | nnsight + repeng + IBM CAST (+ TransformerLens ≥4.3 for figures) | Arch-agnostic hooks; quick control vectors; conditional-steering baseline | `pip install nnsight repeng "transformer-lens>=4.3" && pip install git+https://github.com/IBM/activation-steering` | ~0.5 GB (deps) / uses model RAM |
| Harmful eval data | MultiJail, aya_redteaming, PolyGuardPrompts (+ XSafety subset) | Parallel, native, and labelled multilingual sets | `datasets.load_dataset(...)` | <100 MB |
| Over-refusal data | XSTest (self-translated) + MORBench + OR-Bench-hard-1K | Multilingual over-refusal with toxic contrasts | `load_dataset("Paul/XSTest")`, `load_dataset("Aaron-Pan/MORBench")`, `load_dataset("bench-llm/or-bench","or-bench-hard-1k")` | <50 MB |
| Steering/training pairs | PolyRefuse (`mainlp/Multilingual-Refusal`) + streamed PolyGuardMix subsample | Per-language contrastive pairs; large refusal-SFT pool | `git clone --depth 1 https://github.com/mainlp/Multilingual-Refusal`; `load_dataset(..., streaming=True)` | <200 MB |
| Capability retention | lm-eval via `mlx_lm.evaluate`: Global-MMLU-Lite, Belebele, XCOPA, MGSM (subsampled) | Standard multilingual retention numbers | `pip install lm-eval` | ~0.5 GB data |
| Safety eval runner | inspect_ai + inspect_evals (xstest, strong_reject) vs `mlx_lm.server` | Reproducible logs, model-graded scorers | `pip install inspect-ai inspect-evals` | <0.3 GB |
| Translation | NLLB-200-distilled-600M | 200 languages, light | `hf download facebook/nllb-200-distilled-600M` | 2.5 GB / ~2.5 GB (delete after building sets) |
| Language ID | GlotLID (+ fastText lid.176.ftz) | Translation QA and wrong-language-response detection | `pip install fasttext-langdetect` (bundles fastText) or `pip install fasttext`; `hf download cis-lmu/glotlid model.bin` | 1.7 GB (or 1 MB for lid.176.ftz) |
| Red-team scan | garak with local m2m100 langservice | One automated multilingual scan | `pip install garak` | ~1 GB incl. m2m100_418M |
| Stats / figures | scipy, statsmodels, seaborn, SciencePlots, sacrebleu | CIs, McNemar, Holm, publication styling | `pip install scipy statsmodels seaborn SciencePlots sacrebleu` | <0.3 GB |

**Disk total (steady state):** Tiny Aya HF 6.7 + MLX-4bit 2.0 + two small judges 3.5 + GlotLID 1.7 + data/libs ~2.5 ≈ **16.4 GB**, which leaves only ~0.6 GB headroom. To stay safe: (a) use `lid.176.ftz` instead of GlotLID (−1.7 GB), or (b) download NLLB, the 7B judge GGUF, and garak's m2m100 only temporarily and `hf cache delete` them after use, or (c) skip the MLX copy and use the official `tiny-aya-global-GGUF` q4_k_m (2.14 GB) through llama.cpp. Run `hf cache scan` regularly to see what is filling `~/.cache/huggingface`.

**RAM plan:** never co-load Tiny Aya fp16 (~7.5 GB) with a 7B judge. Generate all responses first (MLX 4-bit, ~3 GB), save JSONL, unload, then judge in batches (PolyGuard-Smol + Qwen3Guard-0.6B together use <4 GB). Run interpretability/steering in fp16 on MPS with only small judges resident.

**Do not use locally:** EasySteer (CUDA/vLLM), HELM (too heavy), Llama-Guard-4-12B and HarmBench-13B classifier (RAM), legacy `HookedTransformer` without the v4 bridge, `Azure/PyRIT` (archived, use `microsoft/PyRIT`). heretic is AGPL, so use it only as an attack baseline and do not redistribute derivatives.
