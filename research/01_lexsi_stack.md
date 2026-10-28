# 01 — Lexsi Track-2 Stack Audit (SafeTune, CircuitKIT, AuditKIT, CuratorKIT)

Audit date: 2026-10-26. Clones under `vendor/` (paths below are relative to `vendor/`).
Versions audited: SafeTune 0.1.8 (`26f59e6`, 2026-10-21), CircuitKIT 0.2.0 (`e71d5e1`, 2026-10-08), AuditKIT 1.1.3 (`71316da`, 2026-10-07), CuratorKIT 0.1.21 (`dd9e835`, 2026-10-12).

## TL;DR for the decision

- **Tiny Aya (cohere2) is supported by all four repos.** SafeTune has Cohere-specific fixes, CircuitKIT ships a TransformerLens 3.8 port for `cohere2`, AuditKIT's `hf:` default is literally `CohereLabs/tiny-aya-global`, and CuratorKIT documents Aya as generator and judge.
- **One venv works:** Python 3.12, `transformers>=5.15,<6`, `transformer-lens==3.8.0`. TL 3.8.0 needs `transformers>=5.9` (PyPI metadata), so nothing conflicts.
- **The integration is thin and mostly file-based.** CircuitKIT `*_scores.json` (`safety_units`/`layer_suggestions`) goes into the SafeTune adapter, which only `CThetaTrainer` consumes. `lexsi_provenance.json` records lineage in every repo. No shipped example imports two of these libraries together.
- **Big shared gap: multilingual safety.** Refusal detection is English substrings everywhere. AuditKIT's red-team module is 26 hard-coded English prompts with keyword detectors, and its detector-selection logic is buggy. CircuitKIT's refusal task uses the English answer tokens " I" / " Sure". For a multilingual model like Tiny Aya, this is the obvious place for a novel contribution.
- **M1 16 GB constraints:** Steer, interpret (SafeTune's native HF EAP) and AuditKIT run locally. Recover (needs 3 models loaded), Unlearn (forced fp32 full-parameter training), the default 7B judges, and TransformerLens EAP on Tiny Aya need Kaggle or Colab, or a code change.

---

## 1. SafeTune

### 1.1 Method catalogue (counts: Harden 27, Recover 26, Steer 19, Unlearn 6, Interpret 6, Evaluate 24)

Sources: `SafeTune/docs/reference/feature-map.md`, `docs/reference/references.md` (paper/arXiv/inputs), `docs/getting-started/taxonomy.md`. Runner classes: `src/safetune/runner/{steer,recover,harden,unlearn}/_*.py`. Call contracts are in taxonomy.md under "Per-class usage contracts":

| Pillar | Call shape | Training? |
|---|---|---|
| Harden | `harden.XTrainer(model_id=… or model, tok).train(ds, safety_dataset=…)`: LoRA, then merge, then save | yes (replaces SFT) |
| Recover | `recover.XTrainer(drifted, base_model=, aligned_model=).apply()` | no (weight arithmetic), with SCRUB and RepNoise-recover as exceptions |
| Unlearn | `unlearn.XTrainer(model).unlearn(forget, retain)` | yes |
| Steer | `steer.XTrainer(model, tok, …).calibrate(harmful, harmless)` returns `(wrapped, dir)`; generate inside `with wrapped:` | no (calibration only) |
| Interpret | `safety_circuit_info(...)`, `identify_safety_neurons(...)`, `eap_safety_circuit(...)` return `CircuitInfo` | no |
| Evaluate | `safetune.evaluate.evaluate(model, tokenizer, benchmarks, judge, max_prompts)` | no |

**Steer** (all Faithful). Activation steering:
- `RefusalDirectionTrainer(layers, alpha=20.0, orthogonalize)`: Arditi 2406.11717.
- `CAATrainer(target_layers, multiplier=20)`: 2312.06681.
- `AlphaSteerTrainer`: 2506.07022.
- `SafeSteerTrainer`: 2506.04250.
- `SafeSwitchTrainer`: 2502.01042.
- `SCANSTrainer`: 2408.11491.
- `STATrainer`: 2505.20322. Needs an SAE.
- `AdaSteerTrainer(alpha=15, layers)`: 2504.09466.
- `CASTTrainer(behavior_layers, condition_layers, alpha=1.0)`: CAST with a condition gate.
- `LinearProbeGuardTrainer(layer, threshold)`.

Decoding steering, each of which needs a second model:
- `SafeDecodingTrainer`: 2402.08983. Needs an expert model.
- `ContrastiveDecodingTrainer`: 2309.09117. Needs an amateur model.
- `ProxyTuningTrainer`: 2401.08565.
- `NudgingTrainer`: 2410.09300.

The following are filed under `steer` but are actually train-time LoRA: `CircuitBreakerTrainer`/`CircuitBreakerRRTrainer` (2406.04313), `RepBendTrainer` (2504.01550), `RRFAEnsembleTrainer`, `TARSteerTrainer`. Their inference hook is a no-op (taxonomy.md, "Filed under steer but actually TRAIN-TIME"). Signatures are at `runner/steer/_activation.py:25-340` and `_decoding.py:5-100`. Steer methods need no training, so they are the easiest to run on CPU/MPS.

**Recover** (training-free unless noted). Whole-model:
- `TaskArithmeticTrainer`: 2212.04089.
- `SOMFTrainer`: 2405.09055. Variant.
- `PrePostMergeTrainer`.
- `WiseFTTrainer`: 2109.01903.

Subspace:
- `LoXTrainer`: 2506.15606.
- `LSSFTrainer`: 2602.00038.
- `SafetyVectorRestoreTrainer`.

Layer:
- `ReStaTrainer(base_model, aligned_model, alpha, dare, dare_drop_rate)`: 2402.11746.
- `SafeMergeTrainer`: 2503.17239.
- `SafeLoRATrainer`: 2405.16833.
- `SafeDeltaTrainer`: 2505.12038. Simplified.
- `QReSafeTrainer`: 2506.20251. Needs preference data.
- `AAQTrainer`: 2511.07842.
- `RepNoiseRecoverTrainer`: 2405.14577. Needs retain data.

Neuron:
- `NLSRTrainer`: 2412.12497.
- `MSCPTrainer`: 2508.09190.
- `AntidoteTrainer`: 2408.09600.
- `AntidoteV2Trainer`.
- `PKETrainer`: 2410.03772.
- `SafeReActTrainer`.
- `SCRUBTrainer`: 2302.09880. Trains.

Saliency:
- `GradSelectiveRecoverTrainer`.
- `OneShotSafetyPatchTrainer`.

Circuit:
- `CThetaTrainer(base_model, aligned_model, strength)`: C-ΔΘ, 2602.04521.

Also `tracin_influence` (2002.08484), which is a diagnostic. Signatures are at `runner/recover/_layer.py`, `_low_rank.py`, `_neuron.py`, `_saliency.py`, `_other.py`, `_whole_model.py` and `_circuit.py`. Most methods need the aligned model and often the base model too. Tiny Aya conveniently ships `tiny-aya-base` (pre-alignment) and `tiny-aya-global` (aligned).

**Harden** (all LoRA training).
- Gradient surgery: `SafeGradTrainer` (2508.07172).
- Regularisation: `AsFTTrainer` (2506.08473; needs an `aligned_model_path`, otherwise it is plain SFT), `BoosterTrainer` (2409.01586), `SaLoRATrainer` (2501.01765).
- Representation perturbation: `VaccineTrainer(rho)` (2402.01109), `TVaccineTrainer`, `SAPTrainer` (2505.16737), `SurgeryTrainer` (2602.05228; needs `attn_implementation="eager"`).
- Data shaping: `LisaTrainer` (2405.18641), `DeRTaTrainer` (2407.09121), `STARDSSTrainer` (2505.17196), `SPPFTTrainer` (2408.17003), `CSTTrainer` (2404.00495), `LookAheadTrainer` (2503.19041).
- Data selection: `SEALTrainer` (2410.07471).
- Distribution constraint: `ConstrainedSFTTrainer` (2406.05946).
- Tamper-resistant: `TARTrainer` (2408.00761), `RepNoiseTrainer`, `SEAMTrainer` (2505.12186), `CTRAPTrainer` (2505.16559), `DOORTrainer` (2503.03710), `MARTTrainer` (2311.07689), `DeepRefusalTrainer` (2509.15202), `AntibodyTrainer` (2603.00498), `LoXHardenTrainer`.
- `PlainSFTTrainer` produces a drifted control model (`runner/harden/_gradient_surgery.py:17`).
- `ASRTCallback` is a Variant (`docs/community/scope.md:81-99`).

**Unlearn** (trains):
- `RMUTrainer(layer_id=7, update_layer_ids=[5,6,7])`: 2403.03218.
- `NPOTrainer`: 2404.05868.
- `GradientAscentTrainer` / `GradDiffTrainer`: 2401.06121.
- `FLATTrainer`: 2410.11143.
- `SimDPOTrainer`: 2405.14734.
- `crisp_unlearn`: Variant.

**Interpret:**
- `identify_safety_neurons(model, refusal_direction_per_layer, config)` (`instrumentation/interpret/safety_neurons.py:470`). Weight mode is Faithful; activation mode is a Variant.
- `safety_circuit_info(model, tok, harmful, harmless, top_k_per_layer=16, target_module="mlp.down_proj", method="weight")` (`:541`).
- `eap_safety_circuit(model_name_or_model, harmful, harmless, config, tokenizer=)` (`interpret/eap.py:814`). This is a native HF implementation of EAP/EAP-IG with no TransformerLens dependency (`eap.py:1-45`), so it runs on any HF decoder, including Tiny Aya on MPS.

**Evaluate:**
- `evaluate(model, benchmarks, judge="wildguard", tokenizer, max_prompts, …)` (`instrumentation/evaluate/suite/evaluate.py:206-261`). It returns `asr`, `refusal_rate`, `harmfulness_score`, `n_evaluated` and `headline_metric`.
- Judges: `wildguard` (default, gated 7B, about 14.5 GB), `harmbench` (13B classifier), `llama_guard_3`. OR-Bench and AILuminate use Llama-3.1-8B (`evaluate/judges/__init__.py:60-75`).
- `StringMatchJudge` (`core/eval/pipeline/scorer.py:86`) is the CPU option. `OpenAIJudge` implements StrongREJECT.
- Benchmarks (`suite/benchmarks.py:55-230`): harmbench, wildjailbreak, sorrybench_v1, advbench, hexphi, ailuminate, orbench (hard/toxic), xstest, jailbreakbench, beavertails, star1, muse, rwku, safedialbench, plus gsm8k, mmlu, humaneval and medmcqa for utility. All are English.
- Red team: `AbliterationAttack(model, tok).fit(...).run(mode="runtime_ablate"|"weight_orthogonalize")` and `BoNAttack(generator, judge, BoNConfig)` (`evaluate/redteam/`).
- `SpectralEntropyMonitor(model, tok, SpectralMonitorConfig)` (`suite/spectral.py:152`) runs on CPU.
- `TamperBenchEvaluator` and Pareto utilities are also included.

### 1.2 Cohere2 / Tiny Aya support

- Layer access is generic, so there is no model-type map to fall through. `_get_decoder_layers` resolves `model.model.layers` for "Llama / Mistral / Qwen / Gemma / Cohere" (`src/safetune/_refusal_helpers.py:13-45`). Loading goes through `AutoModelForCausalLM` (`:69-98`), and transformers 5.x registers `cohere2` natively.
- Explicit Tiny Aya hardening:
  - MPS loads go to CPU first and then move, because loading `tiny-aya-global` straight onto MPS in fp32 segfaults or silently zeros the weights (`runner/utils/model_utils.py:69-76`).
  - `max_len` is sized from the chat template's ~366-token preamble (`runner/utils/data_utils.py:144`; CHANGELOG.md:72-78).
  - The refusal-direction sweep has a layer floor, because the unguarded sweep picked layer 1 of 36 on Tiny Aya (`interventions/steer/refusal_direction.py:80-100`).
  - The ReSta docs recommend alpha≈0.25, since alpha=1 breaks Tiny Aya (`runner/recover/_layer.py:14`).
  - Unlearn upcasts to fp32 because about 80% of bf16 updates were lost on Tiny Aya (`runner/unlearn/_base.py:52-68`).
  - Tests: `tests/runtime/test_hackathon_cohere_fixes.py`.
- README section "Cohere / hackathon notes" (`README.md:251-266`) advises `RefusalDirectionConfig(pick_layer=24)` and `pip uninstall -y torchao` on Colab.
- Device selection is cuda > mps > cpu, with bf16 on MPS from macOS 14 (`config.py:324-354`). Training falls back to fp32 on CPU (`runner/harden/_base.py:55-80`).

### 1.3 How SafeTune consumes CircuitKIT

- `core/circuit_kit/adapter.py:23` `load_circuit_info_from_file(path, top_fraction=0.2)` accepts `safety_units` + `layer_suggestions`, or raw `node_scores`.
- With raw scores, `circuit_info_from_node_scores` (`:66-115`) parses node names `A<L>.<H>` / `MLP <L>` / `L<L>.<n>` into `model.layers.L.self_attn|mlp` and Llama projection names (`:58-63`). Those projection names also hold for Cohere.
- The dataclasses `SafetyRelevantUnits`, `LayerModuleSuggestions` and `CircuitInfo` are in `core/circuit_kit/interface.py:12-40`. `get_circuit_info(source=path)` is at `adapter.py:161`.
- **Only real consumer:** `CThetaTrainer.apply(circuit_info=…)` calls `apply_ctheta`, which uses `layer_suggestions.layer_subset` and `target_modules` (`runner/recover/_circuit.py:25-32`; `interventions/recover/ctheta.py:47-90`).
- The C-ΔΘ mask works at **layer × projection** granularity: every head in a selected layer is edited (`ctheta.py:12-45`). With no circuit it edits every `.weight`, embeddings included (`ctheta.py:20-21`).
- `SafetyLoRAConfig.resolve_target_modules` (`core/safety_lora.py:35-40`) and `core/neuron_safety.py` can do circuit-guided LoRA targeting, **but no Harden trainer or the CLI calls them**. A grep for `circuit_info` / `safety_lora` in `runner/` and `cli.py` finds only `_circuit.py`.
- `eap.py:8` says PKE, NLSR, LSSF and DeepRefusal "can target" a `CircuitInfo`, but none of them accepts one. Wiring it requires manual mapping to `target_layers` or `region_mask`.

## 2. CircuitKIT

- **Algorithms** (`src/circuitkit/backends/__init__.py:46-73`):
  - Stable: `eap`, `eap-ig` (default), `eap-gp`, `acdc`, `ibcircuit`, `cdt`.
  - Research: `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `relp`, `peap`, `eap-ifr`.
  - Pruning selectors: random, magnitude, taylor, wanda, multi_granular, sparsegpt.
  - Quantization: gptq, awq, tacq.
- **Tasks** (`tasks/builtins/`): ioi, greater_than, sva, double_io, capital_country, hypernymy, gender_bias, boolq, glue, gsm8k, ifeval, mmlu, truthfulqa, winogrande, wmdp.
- **Safety tasks:**
  - `tasks/safety_datasets.py:93-139`: `register_advbench/harmbench/sorrybench/jailbreakbench/truthfulqa/stereoset`, built on `SafetyPromptAdapter` (`data/adapters/safety_prompt.py`).
  - Pairing modes: `answer_contrastive` (same prompt, refusal token " I" vs compliance token " Sure", `:56-57`), `harmful_vs_benign` (Arditi-style) and the reverse.
  - Custom data: `GenericTaskSpec`, a YAML loader, or explicit `corrupted_prompt`/`corrupted_answer` CSV columns. See `examples/case-studies/24-safety-refusal-custom-data.py` and `examples/experiments/e4_custom_data_jailbreak.py`, which runs on Qwen2.5-1.5B-Instruct. Case study 24 notes that auto-corruptions are meaningless for safety prompts, so you must author the pairs yourself.
- **API:**
  - `ck.load_model(name, dtype="bfloat16", device=None, fold_ln=False…)` (`quick.py:204`). The default device is cuda, else cpu, so MPS is not auto-selected (`:236`).
  - `ck.discover(model, task, algorithm="eap-ig", level="node"|"neuron", n_examples=128, batch_size=4, sparsity=0.3)` (`quick.py:408`).
  - Also `faithfulness`, `prune`, `quantize`, `export_checkpoint`, `selective_finetune` and `load_scores`, plus a `Pipeline` class.
- **TransformerLens and Cohere:**
  - Pins `transformer-lens==3.8.0` (`pyproject.toml`) and patches it in memory through an import hook. `import circuitkit` must come before `transformer_lens` (`_tl_patches/README.md`). Patch `0006-cohere.patch` adds Cohere and Cohere2 weight conversion.
  - `backends/_tl_compat/cohere.py` registers `CohereLabs/tiny-aya-{base,global,earth,fire,water}`. It maps interleaved RoPE, sliding window plus NoPE on global layers, the parallel attention/MLP block, and GQA (`docs/advanced/tiny-aya.md`). GGUF variants are not supported.
  - Validated on real weights: eap, eap-ig, eap-gp, ibcircuit, cdt. ACDC is too slow at 3–8B (`README.md:118-140`).
  - The parity gate requires TL-vs-HF KL < 1e-4 and is opt-in via `CIRCUITKIT_RUN_TINY_AYA=1` (`tests/backends/tl_compat/test_cohere_parity.py`).
  - Single-token-answer caveat: the 262k vocab can split answer strings, and `ioi` fails fast (`tiny-aya.md` "Task caveat").
- **Memory:**
  - The TL copy keeps a separate W_U (about 1 GB more for Tiny Aya's 262k×2048).
  - "qkv-flag algorithms (eap, eap-ig, eap-gp, acdc) and ibcircuit … exceed a single 47 GB GPU" for 7–8B models, which then fall back to CPU. CD-T needs neither (`docs/advanced/experimental-models.md:453-462`).
  - For Tiny Aya (3.35B, bf16 ≈ 6.7 GB) on M1 16 GB, CPU discovery with batch 1–2 and small `n_examples` is feasible only marginally. The ~366-token chat preamble makes activation memory large. `CIRCUITKIT_TL_MEMORY_PATCH=1` lowers VRAM use.
- **Outputs:** `Circuit.save()` writes `.pt` plus `*_scores.json` with `node_scores`, plus `interop_fields()` → `safety_units`, `layer_suggestions` and `provenance` (`circuit.py:252-266`; `artifacts/scores.py:42-115`). It ranks the top 20% of nodes and uses `arch_registry` projection names. The Cohere family is at `applications/arch_registry.py:123-187`. Note that the fields are named `safety_units` whatever the task is.
- **Interventions** (`applications/`):
  - `steering/steering.py:23` `ActivationSteering(model, circuit_scores, score_threshold)`.
  - `steering/weight_steering.py:135` `CircuitWeightSteering(model, scores, top_k_frac=0.01)`: per-head contrastive weight steering via `fine_tune_positive/negative`, then `apply_steering(k)`. This is head-sliced, unlike SafeTune's C-ΔΘ.
  - `steering_enhanced.py`: `SteeringComposer`, `SafetyDatasetSynthesis`, `SteeringEvaluationGates`.
  - Pruning (`pruning/*`).
  - Circuit-aware quantization (`quantization/*`, needs optimum-quanto or llmcompressor).
  - Editing: ROME, MEMIT, CAKE, MCircKE (`editing/*`).
  - Circuit-restricted LoRA (`finetuning/circuit_tuning.py:132` `CircuitTuner`) and selective fine-tuning with gradient masks (`selective_finetuning/finetune_utils.py:125`).
  - Six-pillar faithfulness evaluation.
- Vision models (Aya Vision, North) are rejected because TL has no vision tower.

## 3. AuditKIT

- **Red team** (`src/auditkit/redteam/`): `RedTeamRunner(model).run(probes, detectors)`.
  - Probes (`probes/builtin.py`): `prompt_injection` (8), `jailbreak` (8), `encoding` (5: base64/rot13/reverse/leet) and `refusal` (5 benign prompts for over-refusal). That is 26 hard-coded English prompts in total.
  - Detectors (`detectors/builtin.py`): `keyword`, `refusal` (English regexes), `system_prompt_leak`, `injection_success`.
- **Bugs and limits:**
  - (a) With the default `detectors=None`, the runner takes the **union** of detectors over all probes and applies it to every probe (`runner.py:200-206`). In a default run, a *refused* jailbreak is therefore scored as a failure by `RefusalDetector`.
  - (b) `KeywordDetector` counts "as an ai" and "system prompt" as attack success (`detectors/builtin.py:20-21`), so ordinary refusals trip it.
  - (c) Jailbreak probes have no harm judge, so harmful compliance passes unless it contains "hacked" and similar strings.
  - (d) One prompt is a literal placeholder, `'[instructions to build a weapon]'` (`probes/builtin.py:42`).
  - (e) `pass_rate` counts (prompt × detector) rows.
  - (f) Not multilingual.
- **The proper safety path** is `ak.evaluate(samples, model, [GuardJudge(...)])`. `metrics/guard.py:187-232` defines these profiles:
  - `llama_guard` (Llama-Guard-3-8B)
  - `wildguard`
  - `harmbench` (13B classifier)
  - `shield_gemma` (google/shieldgemma-2b)
  - `granite_guardian` (granite-guardian-3.1-2b)

  The 2B guards are runnable on M1, and `judge_model=` accepts any `hf:`/`api:` spec. Other metrics: `ToxicityScore` (English `unitary/toxic-bert`, `metrics/toxicity.py:70`), `HateSpeechScore`, `RepresentationSkew`, `LLMJudge`/`GEval`/`BiasJudge` (`metrics/judge.py`), and `ak.compare(base, tuned)` for regression diffs.
- **Model plug-in** (`model/__init__.py:555-642`): `hf:` (HFGenModel takes an id or path and picks device cuda > mps > cpu, `hf_gen.py:107-126`), `vllm:`, `api:` (any OpenAI-compatible server), `litellm:`, `openai:`, `anthropic:`, `groq:`, `openrouter:`, `agent:`, plus **any `list[str] -> list[str]` callable** (`CallableModel`, `:509`). The callable is how to wrap a SafeTune-steered model, which needs `with wrapped:` and your own chat template. `hf:` with no name defaults to `CohereLabs/tiny-aya-global` (`:577`).
- **Datasets:** no built-in safety datasets. The scenarios are arc, gsm8k, hellaswag, humaneval, mmlu and truthfulqa. `loaders.load_hf`/`load_dataset` reads any HF dataset or a CuratorKIT export folder (`loaders.py:63-160`). lm-eval runs via `engine="lmeval"`.
- **Multilingual:** nothing safety-related. Cohere notes cover a 5-language translation probe and chat-template caveats (`README.md:140-189`; `docs/known_issues.md:22`).

## 4. CuratorKIT

- **Tasks** (`README.md`, "What it generates"): `qa`, `preference` (DPO, `single_call` or `two_pass`, with `chosen_/rejected_prompt_template`, `config.py:234-235`), `grpo`, `multiturn`, `evol`, `cot`, `adversarial_preference` and `adversarial_qa`.
- **"Adversarial" here means hallucination injection**: contradicts_source, parametric_drift, domain_mismatch and so on (`generators/adversarial_preference.py:1-25`, `adversarial_qa_generator.py:1-40`). It does **not** generate jailbreaks or refusals. Refusal/compliance DPO pairs have to come from custom templates in instruction mode.
- **Backends:** LiteLLM (any provider) plus a native `OllamaBackend` (`llm/ollama.py:25`, localhost:11434), or a local vLLM / `transformers serve` OpenAI-compatible endpoint via `llm_api_base`. It can run fully local with `ollama/aya-expanse` (8B). Tiny Aya is not in the Ollama library and has to be served via `transformers serve` (`README.md:150-171`). Small judges often fail to emit JSON, and failed rows are rejected unless you set `judge_on_error="pass"`.
- **Gates:** hallucination, reward judge, diversity, schema. Hygiene (`hygiene/`):
  - `ToxicityGate`: Detoxify stage 1, with a `"multilingual"` model option (`toxicity.py:104-105`), then an LLM judge for borderline cases.
  - Secrets: detect-secrets.
  - PII: Presidio, `pii_language="en"` (`config.py:428`).
- **Outputs:** HF-loadable folder (`sft_messages`, `dpo`, `grpo`, …), `manifest.json`, `rejected.jsonl` and `lexsi_provenance.json`.

## 5. Integration points that already exist

| Link | Mechanism | Evidence |
|---|---|---|
| CircuitKIT → SafeTune | `*_scores.json` (`safety_units`/`layer_suggestions`), read by `load_circuit_info_from_file` and then `CThetaTrainer.apply(circuit_info=)` | `CircuitKIT/src/circuitkit/artifacts/scores.py:42`; `SafeTune/src/safetune/core/circuit_kit/adapter.py:23` |
| SafeTune → CircuitKIT | `ck.load_model("<SafeTune checkpoint dir>")` | `CircuitKIT/README.md:58-61` |
| CuratorKIT → SafeTune/AuditKIT | `load_dataset("output/", "sft_messages")` for Harden; `auditkit.loaders.load_hf("output/", name=…)` | `CuratorKIT/README.md:205-225` |
| Any model → AuditKIT | `hf:<dir>` or a callable | §3 |
| Lineage | `lexsi_provenance.json` (`lexsi.provenance/1`), nested `inputs[].provenance` | `SafeTune/src/safetune/provenance.py`; `CircuitKIT/src/circuitkit/provenance.py`; `AuditKIT/src/auditkit/provenance.py`; CuratorKIT `manifest.py` |

**Chained examples:** none cross library boundaries in code. SafeTune's "full six-pillar pipeline" (`examples/notebooks/full_pipeline.ipynb`, `examples/demos/full_pipeline.py`) stays inside SafeTune: PlainSFT drift, then `safety_circuit_info`, then ReSta, then RefusalDirection steer, then evaluate, all on Qwen2.5-0.5B. The only cross-repo snippets are in READMEs (CuratorKIT §Handing off; CircuitKIT §Single environment). A real CuratorKIT → CircuitKIT → SafeTune → AuditKIT pipeline on Tiny Aya would therefore be new work.

## 6. Gaps, stubs, and opportunities for a contribution

1. **No multilingual safety anywhere.**
   - Refusal substrings in SafeTune (`interventions/steer/refusal_direction.py:128-163`) drive layer selection for refusal directions. On a model that answers in Hindi or Swahili, the sweep silently mis-scores.
   - AuditKIT's regexes, CircuitKIT's " I"/" Sure" answer tokens, toxic-bert and Presidio defaults are all English.
   - Tiny Aya's whole selling point is roughly 70 languages, so possible work includes cross-lingual refusal-direction transfer, language-conditioned safety circuits, and multilingual ASR/over-refusal evaluation.
2. **AuditKIT red-team bugs** (detector union, refusals tripping keywords, no harm judge, placeholder prompt). A small fix with high value is per-probe detectors plus `GuardJudge`-backed jailbreak scoring and multilingual probes.
3. **Circuit-guided LoRA is not wired into Harden.** `SafetyLoRAConfig`/`neuron_safety` exist, but no trainer uses them, and the claim at `eap.py:8` about PKE/NLSR/LSSF is unimplemented.
4. **C-ΔΘ is coarse.** It works at layer × projection, while CircuitKIT supplies head-level units. A head-sliced C-ΔΘ (as `CircuitWeightSteering` already does in TL space) is a clear extension.
5. **Stubs:**
   - GuardSpace (`core/patches/guardspace.py:4,58`).
   - `safetune.core.backend_factory`, so `safetune.core.training` orchestration is unimplemented (`core/training/orchestrator.py:11-23`).
   - The vLLM steer backends raise for unsupported models (`steer/backends/vllm_hook.py:280-294`).
   - CoSAlign is "inspired by" only (`core/runtime/cosa/cosalign.py:53`).
   - Five Variant methods (`docs/community/scope.md:81-99`).
6. **CircuitKIT:**
   - Auto-corruptions are useless for safety prompts.
   - The single-token answer assumption breaks on the 262k vocab.
   - `use_qk_norm`/`attention_bias` Command R releases raise NotImplementedError.
   - TL multi-GPU (`n_devices`) is broken, so Aya Expanse 32B cannot run discovery.
   - ENVIRONMENT.md still lists transformers 4.57.6, which contradicts the pyproject's `>=5.15`.
7. **Memory engineering:** Recover loads three full models, about 20 GB in bf16 for Tiny Aya. Streaming the safetensors deltas for the base and aligned models would make ReSta and C-ΔΘ fit on a 16 GB Mac.
8. **Research hook:** the Tiny Aya global, fire, water and earth checkpoints are regional derivatives sharing the same base, which gives natural "drift" checkpoints for cross-lingual safety-drift measurement and recovery.

## 7. Environment and coexistence

| Package | SafeTune | CircuitKIT | AuditKIT | CuratorKIT |
|---|---|---|---|---|
| Python | **>=3.12** | >=3.10 | >=3.10 | >=3.11 |
| torch | >=2.7 | any | >=2.4 (extra) | n/a |
| transformers | >=5.15,<6 | >=5.15,<6 | >=5.15,<6 (extra) | n/a (LiteLLM) |
| transformer-lens | >=2.0 (`interpret` extra) | **==3.8.0** (needs transformers>=5.9) | n/a | n/a |
| other notable | trl<1.15 (Triton DPO bug), peft<1, lm-eval>=0.4.8, bitsandbytes, wandb | hf-hub>=1.5,<2, spacy>=3.8, torch-pruning, bert-score, sentence-transformers | zero core deps; vllm and sglang are mutually exclusive (`[tool.uv] conflicts`) | pydantic>=2.7; extras `generation` (litellm), `hygiene` (detoxify, presidio, spacy) |

**Verdict:** the four libraries co-install in one Python 3.12 venv (`uv` and `python3.12` are present at `~/.local/bin`). Install `circuitkit` and import it before anything that pulls in `transformer_lens`. Skip the `vllm`/`sglang` extras on macOS. `bitsandbytes` and `nvidia-ml-py` are no-ops or skipped on Darwin. Disk: Tiny Aya is about 7 GB per checkpoint in bf16, so base plus global plus one regional variant is about 21 GB, which exceeds the ~16–17 GB free disk. Plan on 1–2 local checkpoints and move the rest of the work to Kaggle (30 GB RAM, 2×T4).

**What runs on the M1 16 GB:**
- Runs: SafeTune Steer (bf16, MPS); `eap_safety_circuit` and `safety_circuit_info`; `SpectralEntropyMonitor`; AuditKIT with `hf:` or a callable plus a 2B guard; CuratorKIT with Ollama aya-expanse.
- Marginal: Harden LoRA on MPS (slow); CircuitKIT TL discovery on CPU.
- Does not run: Recover (three models), Unlearn (forced fp32 full-parameter; `upcast=False` helps only partly), and the default 7–13B judges in `evaluate()`.
