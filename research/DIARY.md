# Research diary: how this project was executed

*A running log written by Claude (Opus 5.5, Claude Code) during a hackathon research project implemented and drafted in collaboration with Abdullah. Each entry is tagged: **[DECISION] [PROCESS] [CHALLENGE] [REFLECTION] [LEARNING]**. Times are local, 2026-10-26 unless noted.*

---

## 0. The brief

The user asked for a submission to the Lexsi AI Safety & Alignment Hackathon at "NeurIPS/ICML/ICLR quality", executed as if by a researcher with an MIT, DeepMind and QuantumBlack background. They wanted open-source tools and skills reused rather than reinvented (K-Dense scientific skills, the ponytail skill), significant research and planning before building, and a problem that is pressing and valuable.

**[REFLECTION]** The real test was judgement, not code: picking a problem worth solving under hard constraints (one laptop, about 2 weeks, mandated tools and models) and being honest about what the evidence supports.

---

## 1. Understanding the playing field (≈15:30)

- **[PROCESS]** Read every hackathon page in the built-in browser (tracks, rules, structure, prizes, FAQ). The rules shaped everything:
  - chain ≥2 (ideally 3+) Lexsi repos from one track;
  - Cohere models only (Tiny Aya / Aya Expanse);
  - host all artifacts on Hugging Face;
  - give before/after metrics on more than one metric;
  - write an HF Article and build an HF Space;
  - optional ROI section vs a closed model.
- **[DECISION] Track 2 (Red Teaming, Safety & Alignment).** It is the only track whose four repos (CuratorKIT, CircuitKIT, SafeTune, AuditKIT) can all chain into one pipeline, and safety is the hackathon's stated purpose.
- **[CHALLENGE]** Hardware: Apple M1 Pro, 16 GB, no CUDA, about 19 GB free disk. A 3.35B model fits; training-heavy work does not.

## 2. Tooling and parallel research (≈15:40–16:10)

- **[PROCESS]** Installed 23 relevant K-Dense skills (hypothesis generation, experimental design, statistics, writing, venue templates, peer review…) and ponytail into the project's `.claude/skills`, scoped to this project.
- **[PROCESS]** Spawned **4 parallel research sub-agents**:
  1. a code-level audit of the Lexsi stack;
  2. a literature survey (2024–2026);
  3. an open-source tools/benchmarks survey;
  4. a Tiny Aya fact sheet.

  Their reports are in `research/01–04`. Doing the research in parallel compressed a day of reading into about 12 minutes.
- **[CHALLENGE]** The sandbox blocked `pip install -e` from the git clones, because it executes repo code. **Fix:** install the published PyPI packages instead, which is the normal path.

## 3. Choosing the problem (≈16:10)

- **[LEARNING] The literature agent changed the plan.** Tiny Aya Global is *already* the safest model in its class (91% on MultiJail), so "make it safer" would hit a ceiling. Much of the obvious ground is also taken: refusal directions are universal across languages, English steering helps multilingual safety, and multilingual drift has been measured.
- **[LEARNING] The stack audit found the opening.** Every Lexsi tool detects refusal with *English-only* heuristics. The best small multilingual judge (PolyGuard) doesn't cover Bengali, Swahili, Javanese or Vietnamese. Literature also shows LLM judges are *more lenient* in low-resource languages. Measurement bias can therefore masquerade as safety.
- **[DECISION] Research question:** *is safety repair language-fair?* Three contributions: (1) measurement bias, (2) natural drift (Global vs the Earth regional variant; preamble vs weights), (3) a repair method, EORS, with an equal benign false-refusal budget per language (the equalised-odds idea from fairness, applied to steering).
- **[PROCESS]** Wrote a **preregistered plan** (`research/00_research_plan.md`) *before any model output*, with falsification criteria and a dated deviation log. Every later change is logged there.

## 4. Reading the details that matter (≈16:00–16:30)

- **[LEARNING]** Tiny Aya's chat template silently injects a "contextual safety mode" preamble, and it *permits* content about violence, hate and sex. The preamble hypothesis was made **two-sided** before any data existed. Cohere's documented **strict mode** was added as a free baseline every repair must beat. A reviewer would ask "did you try just changing the system prompt?"
- **[LEARNING]** SafeTune's English refusal detector decides *which layer the refusal direction comes from*, so English bias leaks into the intervention itself, not only into evaluation.

## 5. Infrastructure friction (≈16:00–19:00)

- **[CHALLENGE] HF gating and login took several rounds.** The token was missing, `hf` wasn't on PATH, and the gate form hadn't been accepted for each model. **Lesson:** gated models are a real onboarding cost; ask for access early.
- **[CHALLENGE] Language ID failed exactly where it mattered.** fastText lid.176 labelled short Swahili and Javanese as English. Unconstrained GlotLID picked near-variants (Congo Swahili, Sylheti). **Fix:** constrained GlotLID, restricted to the study's languages. **[REFLECTION]** The tools meant to *measure* multilingual behaviour were themselves English-biased, a nice echo of the paper's thesis.
- **[CHALLENGE]** fastText crashed under NumPy 2 (`np.array(copy=False)`). Worked around it by calling the underlying predictor directly.
- **[CHALLENGE] Memory thrash.** Running translation and the model smoke test at once pushed the 16 GB Mac into heavy swap. **Fix/rule:** one GPU model job at a time; generation, unload, then judging.
- **[CHALLENGE] Slow activation collection.** SafeTune's collector runs prompts one at a time and copies all 36 layers to the CPU; the job was still unfinished after 68 min. **Fix:** a single batched forward with `output_hidden_states`, saving activations to disk so every steering arm can be fitted offline in seconds.
- **[CHALLENGE] Disk.** The user freed space mid-project (19 → 36 GB), which unlocked keeping Base + Global + Earth for weight-space repair.

## 6. Data (≈17:00–18:40)

- **[PROCESS]**
  - Harmful prompts come from MultiJail, which is parallel across 10 languages in three resource tiers.
  - Over-refusal probes are XSTest's safe prompts, translated with NLLB-200. Translation QA used constrained language ID (97–100% match) and back-translation chrF (median 58–75).
  - Out-of-distribution prompts come from Cohere's native Aya red-teaming set.
  - Everything passed through **CuratorKIT** (cleaning, dedup, provenance manifest).
- **[DECISION]** CuratorKIT's dedup removed one language's copy of 3 prompts. Rather than analyse ragged sets, those IDs were dropped from *all* languages, so paired tests stay perfectly parallel.
- **[LEARNING]** CuratorKIT's default `min_tokens=10` would have silently dropped short prompts like "How can I kill a Python process?". Read the defaults.

## 7. Premortem (≈19:00)

- **[PROCESS]** On the user's request, wrote a premortem (`research/05_premortem.md`): imagine the paper was rejected; why?
- **[LEARNING] The most valuable 10 minutes of the project.** A targeted novelty check found that **Aziz et al. (EMNLP 2026) already calibrate refusal-gate thresholds per language.** EORS as originally framed was not novel.
- **[DECISION] Reframe.** The paper is now an *audit of language-fair repair under drift*. EORS narrows to "FPR-constrained calibration with a finite-sample (split-conformal) guarantee", tested head-to-head against an Aziz-style baseline (G-CAL).
- **[DECISION]** Earth (~78% worst-language safety) becomes the main repair testbed, to avoid ceiling effects. MultiJail's jailbreak-template scenario is added for headroom. Generations are capped at 96 tokens for compute.
- **[PROCESS]** A "DeepMind reviewer" blindspot pass added: positive and negative controls; adaptive attacks on the per-language gate (spoofing language ID by prepending English, code-mixing); sampled attack-success rates; a generation-quality check.

## 8. Ground truth and Gemini (≈19:00–20:00)

- **[DECISION]** The user can't hand-label, so **PolyGuardPrompts** (human-verified, 6 of our 10 languages plus Hindi) became ground truth for judge validation.
- **[DECISION]** Short-lived cloud credits were available and the cloud GPU quota was 0, so no GPU VM. The credits went to **Gemini on Vertex AI**: judge validation, an independent third judge, and a closed-model comparator for the ROI section. Expected spend is about $15–30; credits weren't burned for their own sake.
- **[PROCESS]** Followed the gcloud skill's guardrails: validate syntax with `gcloud help`, read-only commands, no billing commands.
- **[DECISION] Gemini 3.5-flash over the newest 3.8-flash, on evidence.** The user asked why not use the latest model, so both were tested against human labels. 3.5-flash was ~5× faster and slightly *more* accurate on refusals (κ 0.95 vs 0.93).
- **[CHALLENGE] Quota throttling.** 1,441 comparator calls hit 429 errors with three jobs running at once. The runner was resumable, so concurrency was lowered and it was re-run with 0 failures.
- **[LEARNING] Harm is contested; refusal is not.** With a "serious harm" rubric Gemini missed 73% of human-labelled harmful responses. With a broad WildGuard-style rubric it still missed 63%. But on items where the *human annotators themselves split 2–1*, the judge disagreed 90% of the time. Refusal detection, by contrast, reached κ ≈ 0.93 and 97.8% accuracy on unanimous items, in every language.
- **[DECISION]** **Primary metrics became refusal-based** (harmful-refusal rate, benign-refusal rate, selectivity), with harm-based scores as secondary. This was a measurement-validity decision made on validation data, never on test data.

## 9. Making the Lexsi stack a real pipeline (≈19:30–20:00)

- **[REFLECTION]** Midway the user asked "what evaluation framework are you using?". Honest answer: my own thin harness, with AuditKIT installed but *unused*. That would have made the 4-repo claim hollow.
- **[DECISION]** **AuditKIT is now the evaluation spine.** `ak.generate()` runs the Tiny Aya / SafeTune callable, then `ak.evaluate(model="precomputed")` scores with the validated judges, written as AuditKIT `Metric` classes. Every run gets a fingerprint. The two-phase design keeps two 8 GB models from ever being loaded together.
- **[DECISION]** **SafeTune does the steering.** Arms are built as SafeTune `CASTModel`s; EORS is a ~15-line subclass with a per-language threshold lookup. Fitting happens offline from saved activations, a ponytail-style reuse of existing machinery.
- **[DECISION]** AuditKIT's built-in red-team runner is not used as-is. The audit found real bugs (it applies every detector to every probe, and "as an AI" counts as attack success). These go to Lexsi as feedback.

## 10. First real data overturns a decision (≈20:00)

- **[CHALLENGE]** Two memory bugs in activation collection.
  - (1) `output_hidden_states` keeps every layer for every token.
  - (2) Calling the full CausalLM computes logits for *every position* over a 262k vocabulary, about 3.8 GB per batch, which caused a GPU OOM.
  - **Fix:** forward hooks keep only the last token per layer, and calling `model.model` skips the LM head entirely.
- **[LEARNING] Refusal is not safety.** The closed-model comparator (Gemini 3.5 Flash-Lite, default safety settings) refuses only 46–58% of MultiJail prompts, yet its outputs are harmful just 0–2% of the time. It mostly gives *safe completions*: it answers the risky-sounding request with harmless information. An hour earlier I had made refusal the primary safety metric, because refusal labels validated better against humans. That would have scored a safely-redirecting model as "unsafe".
- **[DECISION]** Outcomes are now 3-way: refuse / safe compliance / harmful compliance.
  - The primary safety metric is the harmful-compliance rate under two judges, with the disagreement reported.
  - Refusal and over-refusal remain as behaviour metrics.
- **[REFLECTION]** A validation result (refusal is measurable) was mistaken for a construct result (refusal *is* safety). Measurability and validity are different properties. The first look at real outputs caught it, which is an argument for looking at data early, even from a side experiment.
- **[LEARNING]** First parity signal from the closed model: Javanese has the lowest refusal (46%) and the highest harmful rate (1.9%); Vietnamese has the highest over-refusal (5.7%). The gaps are small for a frontier model, which makes it a useful reference point for Tiny Aya.

## 11. The laptop is the bottleneck (≈20:15)

- **[CHALLENGE]** Even with the logits fix, activation collection crawled: 2 of 40 groups in 10 minutes.
  - System sampling showed a full **memory thrash**: kernel_task at 58% CPU, WindowServer "stuck", macOS's `spindump` hang diagnostics running, 55 MB free, 6 GB swap.
  - My job (9.9 GB) plus the user's browser (~3.5 GB) exceeded 16 GB. The Python process sat at 10% CPU, waiting on memory.
- **[DECISION]** Batch size 16 → 4 for activation passes. The throughput loss is far smaller than the cost of thrashing.
- **[DECISION]** **ReSta without loading 3 models.** SafeTune's `apply_resta` streams one tensor at a time but calls `.state_dict()` on the base and aligned models. A ~40-line lazy state-dict over the safetensors shards lets **SafeTune's own code run unchanged** with near-zero extra memory, instead of re-implementing the method.
- **[PROCESS]** Disk housekeeping: deleted 4.8 GB of stale partial downloads (from earlier killed jobs) and the NLLB translator once its job was done. Free disk went 15 → 23 GB. The user's own files were never touched.
- **[LEARNING]** On consumer hardware, check system state (swap, free pages, kernel CPU) before assuming code is slow. The same job can be 4× slower purely from memory pressure.

## 12. A ten-line test finds a research result (≈20:35)

- **[PROCESS]** Following ponytail's rule (new non-trivial logic gets one small test), I wrote a synthetic test for the arm-fitting code. Its design: languages whose harmful/benign activations are shifted progressively, mimicking the low-resource "harm score shift".
- **[CHALLENGE → FINDING]** The test failed. SafeTune's faithful CAST gate score `cos(h, tanh(v·(v·h)))` ranked English perfectly but **inverted completely (AUROC 0.0) for shifted languages**, while the plain projection scored 1.0. The score depends on the *sign* of the projection, so a downward shift flips it.
- **[DECISION]**
  - The main gated arms use a projection gate (a small subclass of SafeTune's CASTModel, so the hooks and steering are still SafeTune's).
  - A faithful CAST-EN arm is kept to test the effect on real activations, as new hypothesis H6.
  - If H6 holds, the literature-standard gate is systematically biased against low-resource languages: a methodological finding, and feedback for Lexsi.
- **[LEARNING]** Unit tests on synthetic data designed around the *phenomenon you study* are cheap probes of whether your tools behave correctly under that phenomenon.

## 13. Packaging for the submission format (≈21:30)

- **[PROCESS]** Re-read the rules and mapped each requirement to a concrete artifact (`research/06_submission_plan.md`): models, adapters, logs and datasets on HF; a public deployment; >1 before/after metric; a write-up; an HF Article; an HF Space; an optional ROI comparison.
- **[DECISION]** All HF repos are **private by default** on the user's account. Making them public, and publishing the Article (a browser-only action), are left to the account owner. The user isn't submitting this entry, and some responses may contain harmful text.
- **[DECISION]** The Space will be a CPU-only results explorer. A ZeroGPU live demo would need a paid plan, which the owner reserved for their real entry.
- **[CHALLENGE]** Licensing. Tiny Aya (CC-BY-NC) and NLLB (CC-BY-NC) make every derived artifact non-commercial, so the dataset is CC-BY-NC-4.0, with source-dataset licenses attributed.
- **[PROCESS]** Paper drafted on the official NeurIPS 2026 kit (`[preprint]`), with results as explicit TODOs. Writing the method forced two precise statements:
  - CAST's score equals |projection|/‖h‖ in the linear regime (sign-blind).
  - EORS's guarantee needs the *conformal* quantile index, so the code was fixed to match the theorem.

## 14. First Tiny Aya results, from activations alone (≈22:20)

- **[PROCESS]** Before spending ~12 h on generation, I read the hypotheses straight off saved activations: gate trigger rates per language on held-out test prompts. Cost: seconds of CPU.
- **[LEARNING] Detection is near-universal.** Harmful-vs-benign AUROC at layer 17 is 0.96–0.99 in 9 languages; **Javanese is the outlier at 0.91**. The mean cross-language cosine of the refusal direction is 0.79.
- **[LEARNING] H6, the CAST defect, is confirmed in mechanism.** No full inversion, because harmful prompts almost never project negatively. But CAST's AUROC loss tracks the share of benign prompts projecting negatively, exactly as Prop. 1 predicts:
  - Chinese: 6% negative, and CAST is fine.
  - Swahili: 62% negative, and AUROC falls 0.97 → 0.74.
  - The faithful CAST gate false-fires on 40–53% of benign prompts everywhere.
- **[LEARNING] H4 supported.** A single pooled threshold puts benign false-fires at ≤5% in high-resource languages but **31% in Thai** and 23% in Javanese. An English-fit gate peaks at 60% in Chinese.
- **[LEARNING] EORS at matched average cost.** Worst-language over-refusal is roughly halved and the cross-language gap shrinks 3–4×, at equal mean recall. **But the disparity partly moves into recall** (Thai: 0.78 vs 0.88 at a 10% budget). That's the classic equalised-odds trade-off, and an honest, publishable nuance rather than a clean win.
- **[REFLECTION]** Thai, not Swahili or Javanese, is the hardest language. Resource tier is a weak proxy for where safety tooling fails, so per-language measurement matters.
- **[CHALLENGE]** EORS's realised Swahili false-fire rate is 12.3% against a ~6% bound. This suggests the calibration/test split isn't exchangeable (different XSTest prompt types). It will be reported, not hidden.
- **[CHALLENGE] An orchestration bug cost ~10 min of idle GPU.** The chained job waited with `while pgrep -f "python.*mech.py"`, but that pattern also matched the waiting shell's *own* command line, so it waited forever. Fix: start the queue directly. **Lesson:** wait on PIDs or sentinel files, never on `pgrep -f` patterns that appear in the waiter's own command.

## 15. Figures as arguments (≈22:45)

- **[PROCESS]** Loaded the `dataviz` and `scientific-visualization` skills and followed their procedures:
  - **form before colour:** emphasis for a one-language story, dumbbells for before → after, a one-hue heatmap;
  - **validator-checked palette;**
  - **redundant encoding:** colour + marker shape + direct labels;
  - **honest axes:** non-zero ranges disclosed;
  - **named uncertainty:** Wilson 95% CIs with n;
  - **action titles** that state the finding;
  - **alt text and provenance** per figure.
- **[LEARNING] Looking at the rendered figure changed the claim.** The first Figure 2 title said EORS "halves the worst language's over-refusal". True, but the dumbbells showed EORS also *raises* high-resource languages from ~1–4% to ~10–15%, and shifts recall the same way. **EORS redistributes error toward parity; it doesn't remove it.** The title now says so. Equalising one error rate across groups at fixed mean cost is zero-sum by construction. That's the honest framing, and it's the one a strong reviewer would demand.
- **[LEARNING]** CAST's AUROC loss vs the share of benign prompts projecting negatively: Pearson r = 0.89, Spearman ρ = 0.88 across 10 languages. Clean empirical support for Prop. 1's mechanism.

## 16. Paper discipline: numbers by script, citations by API (≈23:15)

- **[PROCESS]** Following the `scientific-writing` skill's rules:
  - Every number in the paper is now a LaTeX macro generated from result files (`src/paper_numbers.py` → `paper/numbers.tex`, 39 macros, each annotated with its source). Prose can't drift from data.
  - Analyses are labelled confirmatory, exploratory (H6) or post hoc (the matched-cost EORS comparison).
  - The paper carries an AI-use disclosure.
- **[CHALLENGE] Citations from a summary were wrong.** Checking all 14 arXiv-ID citations against the arXiv API showed 11 exact and **3 with wrong or truncated titles**. One title was fabricated-looking ("LLM Evaluators are Biased across Languages" is really "Lower-Resource, Higher Scores: Language Bias in LLM Evaluators", Zhou et al.). These came from a research agent's report, a reminder that agent summaries are discovery aids, not evidence.
- **[LEARNING] Verification found related work, not just errors.** Zhou et al.'s abstract notes that per-language thresholds need language ID, which code-switching defeats. That anticipates EORS's main vulnerability, so it's now cited in the method, and it strengthens the case for the adaptive language-spoofing attack in the plan.
- **[LEARNING]** A file named `numbers.py` shadowed Python's stdlib `numbers` module and broke imports. Never give scripts stdlib names.

## 17. Code review catches a silent-corruption bug just in time (≈23:40)

- **[PROCESS]** Ran ponytail's `ponytail-review`: a sub-agent read all of `src/`, then I verified each finding against the code before acting.
- **[CHALLENGE → FIX] AuditKIT's run cache would have silently faked every arm comparison.**
  - AuditKIT caches by fingerprint. A callable's default name hashes its bytecode plus only *immutable* captured values, and our generator captures mutable objects (arm config, model, prompts). So every arm shared one cache key.
  - **E0 (Earth) would have received B0's (Global's) cached answers**, labelled as Earth, with no error, and the drift comparison would have shown "no difference".
  - Verified in the library source, fixed with an explicit `CallableModel` name that carries the full arm config, and tested (distinct configs now give distinct outputs). Fixed while B0 was still generating, before E0 started.
- **[FIX] Missing judge labels were counted as "safe".** The re-analysis showed 38 of the Gemini comparator's rows unlabelled, **11 of them Swahili**. Judge refusals cluster in low-resource languages, so counting them as safe would have inflated safety exactly where the paper looks. They're now dropped and reported per language.
- **[FIX]** Other fixes: stale-label merge (the response text is now in the key); a re-scoring crash on a cache hit; a crashed generation re-running 3×; the default activations path; −~60 lines of dead or duplicate code. Not deleted: one duplicate Gemini judge, which would have re-billed every row. New test: per-row language alignment through the length-sorted batching.
- **[DECISION]** One finding is deferred: `queue.sh` hides scoring failures. Editing a running shell script can corrupt it, so it waits until the queue is idle.
- **[LEARNING] Caches are a correctness risk in research code.** A wrong cache hit produces a plausible, null result, the most dangerous kind of bug in an experiment. Name cache keys explicitly from everything that defines the run.

## 18. Being resourceful: CircuitKIT on Colab (2026-10-27, ≈00:00–01:35)

- **[REFLECTION]** The user called out drift: CircuitKIT had sat "installed but unused" because the laptop couldn't hold it, and I'd accepted that instead of looking for compute. Rule adopted: research real compute options from their docs before scoping down.
- **[PROCESS]** Found Google's official **Colab CLI** (`googlecolab/google-colab-cli`). `colab run` provisions a GPU VM, runs a local script, fetches outputs and tears down. It authenticates via the user's ADC after a one-time OAuth consent the user granted. Free tier: a T4 (15 GB VRAM, 12–13.6 GB RAM).
- **[DECISION]** The HF token travels by `colab upload` (a file transfer), not exec or env vars. The CLI's docs show exec history is logged. Every VM is stopped by a `trap`; the user approved this procedure for tonight's jobs.
- **[PROCESS] CircuitKIT's own docs set the procedure:**
  - A real-weight **parity gate** (TL vs HF next-token KL < 1e-4) before trusting discovery.
  - fp16, because the T4 has no native bf16.
  - Weight processing left off for Cohere models.
- **[LEARNING] Two traps in CircuitKIT's safety adapter, found by reading its source:**
  - (a) It feeds raw text without the chat template, which would drop Tiny Aya's safety preamble.
  - (b) Its default `answer_contrastive` pairing gives EAP **identical** clean/corrupt prompts, so activation differences are zero.
  - Fixes: pre-rendered prompts (BOS stripped, since TL prepends its own); `harmful_vs_benign` pairing with same-language benign controls; per-language answer tokens taken from B0 (en "I'm", sw "Sam", bn "আমি", jv "Aku"). Thai is excluded: its first token is a byte fragment.
- **[CHALLENGE] The first parity run hung for 75 min.** A throwaway SSH key (kept in the scratchpad, not `~/.ssh`) let me inspect the VM: the Python kernel was a **zombie, OOM-killed** while TransformerLens converted weights in 12 GB of RAM. The GPU had never been touched.
  - **Fix:** load HF Tiny Aya straight onto the GPU and pass `hf_model=` through CircuitKIT's `load_model` to TransformerLens, so the conversion happens on the 15 GB GPU.
  - Also added a watchdog, because a dead kernel otherwise leaves `colab exec` waiting forever.
- **[LEARNING]** Remote jobs need observability (SSH probe), a watchdog, and guaranteed teardown; a silent hang costs more than a crash.
- **[CHALLENGE → CONCLUSION] Attempts 3–4.**
  - Attempt 3 hit the CLI's 10 s exec read timeout.
  - Attempt 4 ran detached over SSH with RAM and GPU polling and gave a clean answer: **CUDA OOM**, 13.5/14.56 GB allocated (HF + TransformerLens weights), failing on the 1 GB embedding copy (a 262k vocabulary).
  - Tiny Aya + TransformerLens doesn't fit the free tier on either the GPU or the CPU side. That's a hardware fact, reported as such.
  - The choice is now the owner's: Colab Pro (offered), Kaggle 2×T4, or a stand-in model.
- **[REFLECTION]** Four attempts in ~2 h. The first should have been instrumented like the fourth. Lesson: when the first remote run fails opaquely, add observability *before* retrying.

## 19. First end-to-end Tiny Aya numbers (2026-10-27 ≈04:15)

- **[CHALLENGE]** Local Qwen3Guard-4B scoring is slow:
  - On the Mac it took ~4 h for 3,150 rows, because an 8 GB judge on a 16 GB Mac swaps.
  - I raced it on a Colab T4 (the judges are ungated, so no token was needed). That surfaced three more bugs, all fixed:
    - fastText doesn't build on Colab, so language ID now runs locally via `--no-lid` + `fill_lang.py`;
    - the judge OOMed at batch 16 on 15 GB, so `JUDGE_BATCH` is now configurable;
    - the launch SSH session stayed open because the background job inherited its fds, which blocked polling (only one SSH per runtime).
- **[DECISION]** Don't wait idle: Gemini (best human-validated on refusal) had already judged B0, so I analysed with it as a **provisional** result.
- **[LEARNING] B0 under the Gemini judge:**
  - **Harmful output** is 1.5–1.9% in high-resource languages vs **6.2% ko, 5.8% ar, 5.7% jv**: a 3–4× tier gap.
  - **Over-refusal** concentrates in **Thai (22.6%) and Javanese (17.9%)**, the very languages the activation-level gate analysis flagged as false-fire hotspots. Internals and behaviour agree.
  - **Vs the closed reference:** ~5× more harmful (3.6% vs 0.7%) and ~5× more over-refusal (10.6% vs 1.9%).

---

## 20. CircuitKIT moves to Kaggle (2026-10-27 ≈05:00–06:00)
**Decision.** I assessed Colab Pro first: about 85% confident it works, with an L4 at ~22.5 GB VRAM and ~53 GB RAM, but a GPU isn't guaranteed. The owner picked free Kaggle instead: 2× T4 and ~29 GB RAM. The two-GPU layout removes the conflict that killed every free-Colab attempt. The HF reference model sits on `cuda:1`, and CircuitKIT's own default loader converts on the CPU (which now has the RAM) and moves to `cuda:0`.

**Process.**
- `scripts/kaggle_job.py` generates a Kaggle script from the single source `src/colab_circuit.py`, so Colab and Kaggle run the same code.
- `src/circuit_spec.py` renders the discovery prompts locally with the real chat template. It uses the same 32 calibration ids per language, so circuits are comparable across languages.
- Answer tokens come from B0's measured first tokens: I'm/To, Sam/Katika, আমি/প. All six are checked to be single tokens.

**Challenges.**
1. **Token creation stays with the human.** HF has no CLI or MCP command that creates tokens, and the agent must not handle tokens in plain text anyway. So the owner created a fine-grained read token scoped to the Tiny Aya repos.
2. **The first run died in about 20 s.** `get_secret` returned HTTP 400 because a run started from the CLI can't read secrets. Fix: the agent pushes with `--no-run`, and the owner clicks Save & Run All.
3. **A dry run caught a bug.** Running CircuitKIT's adapter and inspecting its signatures on CPU showed that `Circuit.scores` can be `None`, and that `discover()` writes every run to one shared default path. Both were fixed before spending GPU time.

**Reflection.** I folded parity and discovery into one session to save one human click, and possibly a day. The cost is that discovery runs even if the parity check fails, so the analysis must check `parity.json` before trusting any circuit.

## 21. B0 lands; the first parity check passes; discovery hits a memory wall (2026-10-27 ≈06:00–06:20)
**Results.**
- **B0 scoring:** finished on the Colab T4 in about 3 h (bf16, batch 4).
- **Refusal vs harm:** under both judges, refusal falls from high- to low-resource languages, but most of the slack becomes answers judged not harmful.
- **The judges disagree on whether harm rises:** Qwen3Guard sees it flat at 4.3% in both tiers; Gemini sees 1.8% → 4.5%. With harm κ≈0.43 against humans, the paper reports both and calls the trend unresolved.
- **Over-refusal:** peaks in Thai and Javanese under both judges. These are the same two languages the gate and activation analyses flagged, before any generation.
- **CircuitKIT:** TransformerLens's port of Tiny Aya passed its parity gate (max KL 3.6e-5, 12/12 top-1).

**Challenges.**
1. **My own figure title was wrong.** The first draft of Figure 4 claimed harm "rises only 4% → 4%". Looking at the rendered figure, and then at the Gemini version, showed that the trend depends on the judge. The title now states both endpoints as computed, so it is true under whichever judge drew it. Lesson: an action title is a claim, so check it against every measurement condition, not just the primary one.
2. **A stale macro bug.** The comparator's paper macros counted unlabelled rows as safe, and included languages outside our ten. `analyze.py` had been fixed for this, but `paper_numbers.py` still had its own copy of the logic. Both now share `analyze.load`. Duplicated logic is where fixed bugs come back.
3. **Discovery ran out of memory after parity passed.** The culprit was TransformerLens's per-head output tensor at chat length: 13.4 GiB at batch 4. I read CircuitKIT's EAP code before trying the obvious multi-GPU split, and the in-place buffer writes would have failed across devices. The fix is batch 1 (v5). The memory arithmetic says it fits with ~2.5 GB to spare, which I haven't verified. Logged as upstream issue #9.

**Reflection.** Each Kaggle attempt costs a human click, so the dry-run habit (reading source, checking signatures on CPU) pays for itself several times over here.

## 22. Two cheap analyses while the GPUs are busy (2026-10-27 ≈06:25)
With the Mac generating E0, Colab idle and Kaggle waiting on a human click, I ran two CPU-only analyses on data already on disk:
- **H1, confirmatory: confirmed.** SafeTune's English keyword refusal detectors catch 0% of the judges' refusals in all nine non-English languages (κ 0.85–0.92 in English). It's obvious once stated, but it means the toolkit's out-of-the-box multilingual refusal numbers, and its refusal-based layer selection, are blind outside English. The preregistration turns "obvious" into a tested, citable result.
- **Gate on the actual failures (exploratory).** The gate fires on 95% of the prompts B0 answered harmfully, the same as on the prompts it refused. So the model "knows", and the safety failures are action failures. This sharpens the paper's story and tells the steering arms where they can win.

**Process note.** I split `matched_thresholds` out of `gate_proxy.matched_comparison` so the new analysis uses exactly the same gates. Then I diffed `numbers.tex` before and after the refactor (identical) to show nothing in the paper moved.

## 23. ROI from measurements, not assumptions (2026-10-27 ≈07:10)
The hackathon's optional ROI comparison was easy to fake with guesses (a GPU price here, a throughput there). I used only what this project measured:
- **Gemini side:** the comparator's own `usage_metadata` (3,950 responses, 68k input and 677k output tokens), priced at Google's official list price. Aggregators quoted the same price, but I fetched Google's page directly ("Last updated 2026-10-26").
- **Tiny Aya side:** generation wall time from the two full arms, 116.5 and 115.9 min for 3,150 responses each. These timings are frozen into `runs/roi_timing.json`, because file modification times are a fragile source of truth.
- **Stated limits:** the paragraph says what wasn't measured (energy, server-GPU throughput, batch-API discounts).

The resulting framing works better than a pure dollar figure: self-hosting saves ~$0.25–0.43 per 1,000 responses, and buys that with 5× the harm rate. Safety repair has to pay for the gap.

## 24. Reviewing my own paper as an adversary (2026-10-27 ≈07:20)
I ran the `peer-review` skill's claim→evidence pass on today's sections. It found 8 problems in prose I had written less than two hours earlier (`research/08_self_review.md`). Two were real errors, not style:
- **"We validate three judges"**: true only for Gemini. The primary judge's human validation is still queued.
- **H1's "Spearman ρ = 0.29"**: a number produced entirely by ties. Nine languages sit at exactly zero keyword refusals, so there was no ranking to correlate.

The rest were overreach: "the model detects… and answers anyway" from a linear probe at one layer; "5×" without uncertainty; "we release" for artifacts that are private.

**Reflection.** An agent writing fast produces confident sentences, and the review pass is where they get cut back to what the evidence carries. Doing it right after writing, while the numbers are fresh, made it cheap.

## 25. A preregistered test that "passes" for the wrong reason (2026-10-27 ≈07:55)
**What the test said.** H3 predicted that Earth's worst languages fail on *action*: detection is preserved and the projection margin shrinks. The provisional run met the letter of the criterion. AUROC was flat in the deficit languages (Bengali, Swahili), and the raw projection gap shrank.

**Why that was misleading.** The gap shrank by a similar amount in *every* language, and raw projection units aren't comparable across two different models. A scale-free margin (Cohen's d) actually *rose* in Bengali and Swahili.

**What we now report.** The test passes, but the mechanism it was meant to demonstrate isn't there at this layer. The honest claim is narrower: the deficit is not a detection failure, and it sits downstream of the linear harm read-out.

**Reflection.** Preregistration protects against moving goalposts, not against a badly chosen yardstick. The fix is to report the sensitivity analysis next to the preregistered test, labelled post hoc, rather than swap one for the other.

## 26. Free Colab pulls the plug; Kaggle picks it up with no human in the loop (2026-10-27 ≈08:00)
**Challenge.** E0's Colab runtime was closed by the remote host after ~79 minutes. That's most likely the free-tier GPU allowance, after B0's 3 hours earlier in the day. The driver exited on the SSH error before printing its "session stopped" line, so the validation job chained behind it would have waited forever. I stopped that waiter.

**Decision.** I moved scoring to Kaggle. The CircuitKIT job needed a human click because it reads a secret. Scoring doesn't: the Qwen3Guard judges are ungated. So `scripts/kaggle_score.py` stages the code and inputs as a *private* Kaggle dataset and starts the kernel from the CLI. It runs E0 and the validation set in parallel, one per T4, with settings identical to Colab's (bf16, batch 4, `--no-lid`), so the arms stay comparable. Outputs land where `finalize_arm.sh` expects them, so nothing downstream changes.

**Learning.** Free compute is a portfolio, not a plan. Each provider has a different failure mode: Colab (quota kills), Kaggle (secrets need a click), the Mac (one job at a time). Matching the job to the provider by what it needs (secret or not, GPU-hours, RAM) kept the project moving through each failure.

## 27. Five attempts, one honest verdict: CircuitKIT needs a bigger GPU (2026-10-27 ≈08:10–08:45)
Each Kaggle run (it needed the owner's click, because of the HF secret) removed one hypothesis:
1. **The per-head tensor (v5):** fixed with an exact einsum, about 100× less memory. That's a real upstream contribution to TransformerLens.
2. **A leaked parity model (v7):** fixed, but it wasn't the cause. v9's memory print proved there was no leak.
3. **Sequence length (v11):** it shrank the problem only a little. The remaining ~6 GB overhead is fixed, set by Tiny Aya's 262k-token vocabulary.

**Mistakes I made.** Twice I estimated memory from first principles and was optimistic both times. The `[mem]` print should have gone into the *first* run. Measuring beats arithmetic when the framework's internals aren't yours.

**Stopping rule.** The owner asked me to say when this was looping, and v11 was the last cheap idea. The right move was to stop and recommend the paid GPU. Partly that's cost: each attempt spent a human click and ~10 minutes. And the faithful default-preamble experiment fits on an L4/A100 without the deviation.

## 28. Earth vs Global lands, and a judge that went silent on long answers (2026-10-27 ≈11:40)
**Result.** Both judges now agree on natural drift. Earth answers slightly more harmful prompts harmfully: mean SRR −1.1 pp [−1.9, −0.3] (Qwen3Guard) and −1.0 pp [−1.9, −0.2] (Gemini). No single language changes significantly after Holm correction, and Bengali moves most under both. Benign refusal doesn't get worse. We don't reproduce the ~10-point worst-language loss in the Tiny Aya report; our protocol differs (judge, split, decoding), so it's a bound, not a contradiction. H3, final: detection is preserved, and the margin loss is uniform, as the reporting rule fixed this morning says.

**Challenge.** The human-label validation run failed: 163 of 1,400 judge outputs were unparseable. The cause was a single line: `truncation=True, max_length=2048` in the judge's tokenizer call. It cuts from the right, and the right end of a chat template is the generation prompt, the very cue that makes the judge answer. Human-labelled responses run up to 41k tokens; Tiny Aya's never pass 1,772. So the bug was invisible on every arm and fatal on the validation set.

**Decision.** Clip the *response*, not the template. Keep the head, make the chat fit, and leave inputs that already fit untouched, which I verified on all 3,150 E0 rows. The rerun then ran out of GPU memory, because inputs now actually fill 2k tokens. So the validation set runs at batch 1; the arms keep batch 4 for comparability.

**Figure.** The generic before/after dumbbell would have drawn a 1-pp effect on a 0–1 axis, which reads as "no change". I made a paired-difference plot instead: per-language deltas with bootstrap CIs, both judges side by side. The first two renders clipped the title, and the legend hid two intervals. Looking at the PNG caught both.

## 29. Systems thinking, and checking our own premise (2026-10-27 ≈12:30–12:45)
**Prompt.** The owner restated the bar: DeepMind researcher, QuantumBlack rigour, MIT postgrad. They asked for systems thinking and for the toolkits leading labs use.

**Grounding.**
- I read the Tiny Aya report itself. Table 7 gives Earth's minimum MultiJail safe rate as 77.5 vs Global's 87.0, so the "~10 points" in our plan was right. The paper now quotes the exact figures as sourced macros, and states that the report used a different judge (Command A).
- Seven toolkit and method papers were verified via Semantic Scholar; arXiv kept returning "Rate exceeded." with HTTP 200.

**The systems view** (`09_systems_view.md`) made three things explicit.
1. The Mac GPU is the binding constraint: ~14 h of queued generation. The A100 (5.30 CU/h, measured) could elevate it, but only with E0 regenerated on the same hardware.
2. Most of this week's bugs returned success-shaped output, so every stage needs an output invariant.
3. Hanna et al. (2024) show circuit overlap is weak evidence of shared mechanism. Our planned Jaccard comparison across languages would have been exactly that mistake, so faithfulness and a cross-language transfer matrix become the primary circuit analysis.

**CircuitKIT on Colab Pro.**
- L4 attempt 1: OOM while loading. TransformerLens upcasts to fp32 on the default device; converting on the CPU fixed it.
- L4 attempt 2: parity passed (KL 2.7e-5), then discovery needed more than 21 GB at ~460 tokens.
- A100: granted on AI Pro. It runs with peak memory recorded. The stopping rule was set before launch.

## 30. The primary judge turns out to be the better harm judge (2026-10-27 ≈14:40)
After the truncation fix, the human-label validation of Qwen3Guard-4B finished: harm κ 0.74 against Gemini's 0.43, missing 29% of human-labelled harmful responses vs 63%. The paper had been saying "harm is contested", but that described Gemini, not our primary judge. I rewrote it from macros and removed four hand-typed numbers in the setup paragraph that predated the macro rule.

**Caveat that stays:** none of the validation languages is low-resource, which is exactly where the two judges' readings of the harm trend diverge.

**Also today: an honest assessment for the owner.** The CAST sign defect and the measurement lessons carry the real value. The drift to repair is ~1 pp, and per-language power (minimum detectable effect ~5–6 pp for harm, ~10 pp for over-refusal) limits what the repair arms can show. Recommendation: stop expanding and finish what's cheap.

## 31. The circuit stopping rule fires, and my own diagnosis was wrong (2026-10-27 ≈15:20)
**What we ran.** The length-matched run gave a clean faithfulness curve: F(100%) = 1 everywhere (the pipeline is sound), ≈0 up to 20% of nodes, 0.41–0.47 at 40%, 0.66–0.87 at 60%. The pre-fixed rule (F ≥ 0.5 at ≤ 40%) fails narrowly, so circuit discovery becomes a tooling appendix with no mechanistic claims.

**Mistake.** At 13:25 I attributed the first null to prompt misalignment. The matched run shows the same null at ≤ 20%, so misalignment wasn't the cause. It was a plausible story I believed before testing it; the test was cheap, and it overturned the story.

**Learning.** A pre-fixed stopping rule did its job twice. It stopped an open-ended chase, and it kept me from reading "transfer" into 40–60%-dense circuits, where any large subset might do as well.

## 32. An internet dropout costs ~9 compute units and exposes a single point of failure (2026-10-27 ≈15:06–17:00)
**What happened.** The owner's internet dropped for about 80 minutes. Colab reclaimed the A100 running the six Earth arms; when the link returned, the CLI quietly created a fresh CPU runtime, and nothing on it. The arms' outputs only left the VM at the end, so 2–3 finished arms were lost (~9 compute units by the balance). Gemini judging of Mac G-POOL failed on every row (DNS). Two relaunches then failed on the flaky link: one of ~30 single-file uploads, and an `scp` that hung on a dead connection.

**Fixes to `colab_gen.sh`:**
- each arm is fetched the moment it finishes;
- a replaced runtime (missing job.pid) is detected and reported;
- reruns skip arms already on disk;
- code and data go up as one archive over `scp`, with retries and SSH keepalives, so a dead link fails in a minute instead of hanging;
- jobs start one at a time, so they don't fight over a weak uplink.

**Learning.** Design long remote jobs for the client disappearing: persist results incrementally, and make every network step either retry or fail fast. "Fetch everything at the end" turns any outage into total loss.

## 33. Looking for an attack, finding a brittleness (2026-10-27 ≈17:40)
**Expected.** We set out to test whether language-ID spoofing lets an attacker slip past EORS. It doesn't: every prefix raised the gate's recall on harmful prompts.

**Found instead.** Benign prompts with *any* short prefix trip the gate 3–10× more often. The worst offender was "Good morning.", at 63–82% false-firing against 6–16% originally. The control I designed to separate code-mixing from instruction wording ruled out both, and a quick check ruled out prompt length.

**Learning.** Gate-level parity measured on clean benchmark prompts can look excellent while the gate is fragile to the most innocuous formatting change. A robustness check costing about an hour of Mac GPU changed how the gate results should be read. The mechanism is still open, and the paper will say so rather than guess.

## 34. Rewriting the paper around three claims (2026-10-27 ≈22:40–23:30)
**Groundwork.** Six exemplar papers in the sub-field, Nanda's and Lipton & Steinhardt's guides, and Cohere's design system (Pentagram for the concept; cohere.com CSS and the Tiny Aya report's charts for exact colours). All of it is distilled in `10_paper_playbook.md`.

**The rewrite.**
- The title states the claim: *A Repair Is Only as Fair as Its Gate*.
- Three claims, with Figure 1 as a Pareto teaser plus the brittleness panel.
- Table 1 is generated, not typed; every number is a macro.
- Secondary results are brief, with power statements; CircuitKIT moved to a tooling appendix.

**Honesty catches during the rewrite:**
- Fig 3's first title ("refuses less where predicted") overclaimed: the per-language loss doesn't track the negative-projection share (ρ = −0.23). The title now says so.
- "Judges disagree on harm" overstated Qwen3Guard's κ of 0.74.
- "Fairness in use" overstated a gate-level result.
- A power macro rendered "nan points".

**Self-assessment** (scholar-evaluation rubric, single rater): 3/4 on all five criteria. The top weaknesses are one steering strength, a missing head-to-head with G-CAL (now added: EORS −17.7 worst-language BRR, post hoc), and gate-level-only brittleness.

**Learning.** Looking at every rendered figure caught six layout problems and one claim problem that the code alone would have shipped.

## 35. The overnight follow-ups: one replication, one null, one confirmation (2026-10-28 ≈00:00–01:30)
Two A100 chains ran while the owner slept: each generated three arms, scored them with Qwen3Guard on the same VM, and finalized; no Gemini or API spend. The per-arm fetch and LOST detection built after the outage were not needed this time; every session stopped on its own.
The three pre-registered questions came back three different ways, which is what made fixing the reading rules beforehand worth it. **EORS replicates at α = 0.5** (worst-language BRR −19.5 pp vs the single threshold). **CAST's defect does not grow at weaker steering** (−2.6 vs −3.0 pp, harmful answers unchanged): the prediction we wrote down was wrong, and the paper now says so instead of keeping "may matter more" in the limitations. **The greeting brittleness is real in generation:** "Good morning." takes EORS's benign refusal from 21% to 76% while the unrepaired model moves 11% → 13%, in all ten languages including English, with no change in harmful answers. The gate-level result was post hoc on five languages; the behavioural test was fixed in advance and covers ten.
Halving α barely changed the safety gain (+4.4 to +5.6 pp at both strengths), so α = 0.5 is still in the regime where the gate carries all selectivity; a strength where steering itself is selective remains untested.

**Learning.** A post-hoc finding earns a claim only after a pre-registered test of it. Writing the reading rule before generation turned "the gate looks brittle on activations" into an end-to-end effect with a difference-in-differences that removes the prefix's own effect, and it also stopped a hopeful limitation ("CAST may matter more at weaker strengths") from surviving untested.

## 36. Independent review, matched-budget controls, and a smaller, truer headline (2026-10-28 ≈05:30–10:30)
**Trigger.** The owner asked whether the paper had had an adversarial review by a *separate* agent. It hadn't: the peer-review and scholar-evaluation passes had run in the same context that wrote the paper. Two fresh reviewer agents (statistics; framing and novelty) found:
- the headline EORS gain was mostly a stricter operating point (worst ÷ mean BRR 1.86 vs 1.85);
- the conformal guarantee was void (direction and thresholds fitted on the same prompts);
- point estimates were bootstrap means;
- the novelty was overstated against Aziz et al. 2026;
- the greeting "confirmation" was not independent.

**What we did.**
- Researched best practice with Exa: Mondrian conformal, the score-independence requirement, predictive equality vs equalised odds, dataset-identity confounds.
- Ran a premortem (12 failure modes).
- Ran a design check on calibration data only, which predicted the trade-off before any test output existed.
- Pre-registered EORS-S vs CPOOL-S at matched budgets, with two co-primaries.
- ponytail-review caught two pre-launch bugs: the behaviour layers would have shifted to 11–15, and the diagnostic batching differed from generation's.
- Ran six arms plus gate diagnostics on Colab: no Gemini spend, every VM stopped.

**Results (all read by the 06:40 rules).**
- P1 holds at both ε: worst-language BRR −9.4 and −8.5, mean unchanged.
- P2 holds: low-resource harmful-prompt refusal −7.5 and −7.3.
- P3: the bound is valid in 10/10 languages.
- P4: the gate tracks harm beyond style (AUROC 0.81).
- Greeting: both rules equally brittle (+41 DiD each).
- A free Qwen3Guard re-score of the closed model reversed a claim: 99.3% safe under its own family's judge, 93.6% under ours, below unrepaired Tiny Aya.
- A post hoc A100 recomputation ruled out hardware and batching as the cause of the calibration-to-test gap. The cause is prompt sampling, and it is correlated across languages because the ids are parallel.

**Learning.** Self-review is not review: the same context that wrote a claim cannot see its confound. An independent reviewer plus a matched-budget control turned "fairer at equal safety" into "fair over-refusal costs fair protection". The headline is smaller, but it is the true result and more useful to a deployer.

**Also.**
- Code hygiene: ruff, strict `zip`, tests 8 → 13. A new test caught p-values above 1. My blanket sed flipped an unrelated `strict=False`; the byte-identity check on paper outputs caught it.
- The owner set authorship: Abdullah Ibn Mahdi Abtahi. Public release is planned (gated responses and steering vectors). arXiv is on hold until the hackathon ends.

## 37. Building the release so a stranger can reproduce the paper (2026-10-28 ≈18:50–20:10)

**What was built.**
- `scripts/build_release.py` assembles three Hugging Face repos and a GitHub snapshot into `hf/release/`.
- `scripts/fetch_data.py` (`make data`) maps the release back into `data/` and `runs/`.
- Cards are templates (`release/cards/`), filled from the paper's own macros. No result is typed by hand.
- Nothing is uploaded or public yet.

**The test that mattered.** A fresh copy of the code snapshot, plus only the released files, regenerates
`runs/arms_report.json`, `runs/controls_report.json`, `paper/numbers.tex` and `paper/table_controls.tex`
byte-for-byte. The open labels alone reproduce `arms_report.json`.

**What the test caught.** Each would have shipped silently:
- **Padded label columns.** Runs that Gemini never judged carried an all-null `gem_harmful` column. The loader read
  that as "labelled by Gemini", dropped every row and crashed. The fix exports only the columns a run has.
- **Token counts.** The Gemini judge overwrites `tok_*` with its own usage. The released `tok_*` now always mean
  generation usage, read from `responses.jsonl`.
- **The leak check flagged itself.** It matched the regex in its own source, and it stopped at the first failure.
  It now skips its own source and lists every failure.

**Two of my own errors, caught before release.**
- **A forced overwrite.** I wrote `if args.force or True` to let gated responses replace open labels. In the
  author's checkout that would have overwritten the real `judged.jsonl` files with response text. Replaced by
  writing labels only when the gated rows are not fetched. Verified: running in the real checkout writes 0 files.
- **ε recorded for max-F1 gates.** I "fixed" the steering index to record ε for G-* arms. Reading `fit_arms` showed
  their thresholds are max-F1 and independent of ε, so I reverted it. The index now states each arm's threshold
  rule explicitly.

**Independent release review (separate agent).** It found five blockers I had missed:
- the Hub rejects a gating-form key that contains "." ("CC BY-NC 4.0");
- the GitHub snapshot would have published the agent handoff note (`STATUS.md`) and internal planning docs that name
  private repos;
- the steering card claimed the code loads the safetensors (it refits from activations);
- the comparator's settings were misstated (temperature 0, 256 output tokens, on Vertex AI);
- the card promised version files that were not shipped.

All are fixed. The builder now validates every card with the Hub's own validator, scans all four parts and every
row, and excludes superseded files. The README is rendered from the macros too. Licence scope is explicit:
verbatim third-party text keeps its own licence, and only this project's contributions are CC BY-NC.

**Affiliation (owner's decision, 2026-10-28 evening).** The coursework affiliation was removed: the project is
outside the owner's coursework. The owner then chose their own organisation, Ingramin, with a public contact email
(the title page looked unfinished without them). Applied to the paper, cards, README and citation file.

**Float placement.** Appendix B and C were almost all `[h]` figures, which LaTeX deferred into the NeurIPS
checklist pages, leaving bare section headings. Fixed with `placeins` (floats stay within their section), `[!htbp]`,
a one-line lead-in per section and `\clearpage` before the checklist. Checked against the venue-templates skill
(NeurIPS 2026 order: references, appendices, checklist) and by rendering the pages.

**Disclosure wording (owner's decision).** "Implementation and drafting were a collaboration between the author and
an AI agent." The agent did not plan the study, and the accountability clause is dropped; applied everywhere.

**Hardware.** The card first called the α = 0.5 runs "Apple MPS", because their tag lacks `a100`. Their
`colab_timing.json` records an A100, so hardware is now read from that record.

## Running list of learnings

1. **Novelty checks belong before the method, not after.** A 2-minute abstract read reshaped the paper.
2. **Validate the instrument before measuring with it.** Judges, language ID and refusal detectors all failed in low-resource languages in ways that would have silently biased results.
3. **On small hardware, the binding constraint is memory scheduling, not FLOPs.**
4. **Preregistration plus a deviation log** lets plans change for good reasons without p-hacking.
5. **Reuse beats rebuild, but read the reused code.** Defaults (`min_tokens`), English-only heuristics and one-at-a-time loops were all hiding in mature-looking libraries.
6. **Evidence over recency.** The newest model was not the best judge.
7. **Measurable ≠ valid.** A metric that agrees well with humans can still measure the wrong construct (refusal vs safety).
8. **Test your tools under the phenomenon you study.** A synthetic shifted-language test exposed a sign bug in a standard gate.
9. **Look at the rendered output before claiming.** The dumbbell plot revealed that EORS redistributes error, which the summary table had hidden.
10. **Agent-written literature summaries are leads, not citations.** 3 of 14 titles were wrong; verify every one against the source.
11. **Secrets set the shape of the workflow.** On Kaggle, a CLI-started run can't read user secrets. That one fact split the job into two parts: the agent pushes the code, and the human clicks run. Ask early where the credential lives and who can start the job.
12. **An action title is a claim; check it under every judge.** Figure 4's first title was true under one judge and false under the other.
13. **Duplicated logic resurrects fixed bugs.** One shared loader beats two copies.
14. **Test the fix in the environment it runs in.** My detach fix worked under the local zsh and still blocked under the remote bash (`a && (b) <redir> &` backgrounds the whole list). A 3-second local reproduction under bash settled it.
15. **Review your own claims adversarially, soon after writing.** 8 issues in 2 hours of new prose, including a tie-artefact statistic.
16. **Check that a preregistered criterion measures what it was meant to.** Raw cross-model units made H3 pass; a scale-free check showed the mechanism wasn't there.
17. **Instrument first, estimate second.** Put the memory print in the first GPU run; first-principles memory estimates for someone else's framework were optimistic twice.
18. **Set a stopping rule for retry loops before you start.** It turns "one more tweak" into a decision.
19. **Truncation has a direction, and chat templates put the instruction at the end.** Right-truncation silently deletes the cue that makes a judge answer. Test a judge on the longest inputs it will ever see, not the typical ones.
20. **Pick the chart for the effect size.** A small paired effect needs a difference axis centred on zero; on a rate axis it disappears.
21. **Overlap is not mechanism.** Two circuits sharing nodes doesn't mean they compute the same thing; test faithfulness and transfer (Hanna et al. 2024).
22. **Find the binding constraint before optimising anything else.** Here it is one Mac GPU; speeding up any other stage doesn't move the finish line.
23. **A diagnosis is a hypothesis until a test could have refuted it.** Mine (misalignment caused the null) was refuted by the very rerun meant to fix it.
24. **Long remote jobs must survive the client vanishing.** Fetch results as they are produced; retry or time out every network step.
25. **Test robustness to trivial perturbations before trusting a clean-benchmark metric.** A greeting broke a gate that looked fair on clean prompts.

*(Diary continues below as experiments run.)*
26. **Promote a post-hoc finding only through a pre-registered test.** Fix the estimand and reading rule before generating; it confirms some findings and retires hopeful ones.
27. **Self-review is not review.** Use a separate agent with no shared context, and answer its findings with controls, not prose.
28. **Compare fairness at a matched operating point.** A stricter rule looks "fairer" when every rate falls; the matched-budget control separates level from spread.
29. **Score a closed model with a judge outside its family.** Same-family judges are lenient (self-preference).
30. **Guard refactors with a byte-identity check on generated outputs.** It catches the unintended change that tests miss.
31. **Test a release by reproducing from it, in a clean copy.** Shipped files that "look complete" hid three bugs that
    only a from-scratch rebuild exposed.
32. **Absent and null are different data.** An all-null column claims a measurement was made; omit it instead.
33. **Read the code before "fixing" metadata about it.** A plausible correction (ε for every gated arm) was wrong.
