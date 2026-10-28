# Systems view of the study (2026-10-27)

*Why this exists:* the owner asked for systems thinking: treat the study as a system with stocks, flows, constraints and feedback loops, not as a list of tasks. It sits beside `STATUS.md` (state) and `DIARY.md` (history). Measured numbers carry their source; anything not yet measured says so.

## 1. Purpose and output
**Purpose.** Valid, language-fair conclusions about safety repair on Tiny Aya.
**What a result has to survive.** It counts only after passing every gate below: a validated instrument, paired comparisons, two judges, the preregistered or labelled status, and a rendered-figure check.

## 2. The pipeline as a flow
| Stage | Where it runs | Throughput (measured) | Gate that stops bad output |
|---|---|---|---|
| Data (CuratorKIT, NLLB) | Mac CPU | one-off | GlotLID + back-translation chrF; parallel ids |
| Activations (`mech.py`) | Mac MPS | ~2 h per model | held-out AUROC; calibration-only fitting |
| Generation (`run_arm.py`) | **Mac MPS, one job at a time** | **27 responses/min, ~2 h per arm** (`runs/roi_timing.json`) | row counts; language fidelity |
| Judging (Qwen3Guard) | Kaggle T4×2 / Colab | ~3 h per arm on a T4 | ≤2% unparseable assert (it caught the truncation bug) |
| Judging (Gemini) | Vertex API | ~minutes per arm | unlabelled rows dropped, never counted safe |
| Circuits (CircuitKIT) | Colab A100 | **5.30 CU/h** (`colab usage`, 12:40) | parity KL < 1e-4 (passed on Kaggle T4 and Colab L4) |
| Analysis → paper/HF | Mac CPU | minutes | macros only; claim→evidence comments; peer review |

## 3. The binding constraint
**The constraint.** Generation on the Mac GPU is the critical path. Seven Earth arms at ~2 h each is ~14 h, and nothing downstream can start before an arm exists. Every other stage already runs in parallel off the Mac (theory of constraints: *subordinate* everything to the bottleneck).

**Elevating it.** Moving generation to a Colab A100 could shorten the critical path several-fold. The 200 CU/month at 5.30 CU/h buys ~37 A100-hours. The A100 throughput for Tiny Aya generation is **not measured yet**; one timed arm would settle it.

**The catch: a hardware confound.**
- B0 and E0 were generated on MPS (bf16). Greedy decoding on CUDA bf16 is not bit-identical to MPS.
- Steering arms generated on the A100 would differ from E0 by hardware as well as by intervention.
- The fix is to regenerate E0 on the A100 as the paired reference for every A100 arm, and report the MPS-vs-CUDA E0 agreement as a reproducibility check.
- This costs compute units, so it is the **owner's decision**.

## 4. Feedback loops
**Balancing loops (they keep the system honest):**
1. Instrument validation → metric choice: judge κ against humans set refusal as the measurable construct.
2. Assertions on silent failures: the unparseable ≤2% assert, the parity gate, row-count checks.
3. Adversarial review → claims: the self-review found 8 issues. Today it found the drift claim needed exact figures, which the source check gave: 87.0 → 77.5 (Table 7).
4. The deviations log keeps plan changes visible, so changing the plan can't quietly become p-hacking.

**Reinforcing loop (risk).** Retry loops on infrastructure: each failed GPU attempt invites "one more tweak". The counter is the stopping rule, written down before the next attempt (DIARY §27; today: A100 once, then the no-preamble L4 fallback, then stop).

## 5. Failure modes are a property of the system
Most bugs this week returned *success-shaped* output:
- Kaggle secrets returning 400 from the CLI;
- a detach that blocked;
- right-truncation that deleted the judge's answer cue;
- arXiv returning HTTP 200 with "Rate exceeded."

**Design rule.** Every stage asserts an invariant on its output (count, parse rate, KL, rendered PNG), rather than trusting exit codes.

## 6. Leverage points, ranked by impact per unit of effort
1. **Circuit analysis by faithfulness, not overlap.** Hanna, Pezzelle & Belinkov 2024 (arXiv 2403.17806, verified via Semantic Scholar) show that circuit overlap is weak evidence of a shared mechanism; faithfulness is the test. Plan:
   - CircuitKIT's own `evaluate_circuit` pillars (patching, ablation, baselines) per language;
   - a cross-language transfer matrix: discover on language A, evaluate faithfulness on B, using CircuitKIT's `TransferMatrix` pattern;
   - Jaccard/Spearman overlap stays only as a descriptive companion.
2. **Elevate the generation constraint** (§3), subject to the owner's OK on compute units.
3. **Instrument first.** Peak GPU memory is now recorded per discovery run (`discover_summary.json`), so memory claims in the paper and the upstream issue are measured, not estimated.

## 7. Research-toolkit map (what leading labs use; what we use and why)
All papers verified against Semantic Scholar on 2026-10-27.

| Need | Field standard | In this study | Decision |
|---|---|---|---|
| Circuit discovery | EAP / EAP-IG (Syed et al. 2023, arXiv 2310.10348; Hanna et al. 2024), ACDC (Conmy et al., NeurIPS 2023, arXiv 2304.14997) | CircuitKIT's EAP-IG on TransformerLens (mandated stack) | Keep. It is the Hanna et al. method; add their faithfulness test (§6.1) |
| Model internals / interventions | TransformerLens; NNsight + NDIF (Fiotto-Kaufman et al., ICLR, arXiv 2407.14561); pyvene (Wu et al., NAACL, arXiv 2403.07809) | TransformerLens via CircuitKIT; raw PyTorch hooks in `mech.py` | Keep. An independent pyvene/NNsight patching check of the top nodes is an option if the faithfulness results look surprising |
| Feature-level (SAEs) | Gemma Scope (Lieberum et al., BlackboxNLP, arXiv 2408.05147) | none | Not applicable: no SAEs exist for Tiny Aya, and training one is out of scope |
| Steering | Refusal direction (Arditi et al. 2024), CAST (Lee et al. 2025) | SafeTune `CASTModel` subclasses | Keep. We found and test CAST's sign defect (H6) |
| Evaluation | judge validation against humans | AuditKIT + validated judges + paired McNemar/bootstrap | Keep |
