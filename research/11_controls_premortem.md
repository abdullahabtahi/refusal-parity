# Control experiments after the independent review: research, premortem, design (2026-10-28)

The two independent reviews (statistics; framing) found that the headline EORS result mixes two effects. EORS lowers the over-refusal *level* and also changes its *spread* across languages. On top of that, its "guarantee" was fitted in-sample. This file records:
- the prior work these experiments are benchmarked against;
- what could go wrong (premortem);
- the design that answers both review points.

The binding pre-registration is the deviations-log entry of the same date in `00_research_plan.md`.

## 1. What best practice says (sources checked 2026-10-28 via Exa; arXiv IDs to be verified before citing)

| Question | Best practice | Source |
|---|---|---|
| Per-language thresholds with a finite-sample guarantee | This is **Mondrian (group-conditional) split-conformal**: a separate conformal quantile per group gives P(error \| group) ≤ ε. | Vovk et al. 2003; Gibbs, Cherian & Candès, JRSS-B 2025 (arXiv 2305.12616); MAPIE docs |
| What the guarantee requires | The score function must be fitted on data **independent** of the calibration fold. Validity is marginal over calibration draws, and the realised FPR given calibration is Beta(n+1−k, k), k = ⌈(1−ε)(n+1)⌉. | Angelopoulos & Bates, gentle intro (2107.07511); Vovk 2012, conditional validity |
| A high-probability (PAC) version | Learn-then-Test / Clopper–Pearson on a fixed threshold family gives P(FPR > ε) ≤ δ. With n ≈ 60 per language this is near-degenerate at ε = .05, so it is reported as a limitation and not run. | Angelopoulos et al., LTT (2110.01052); 2610.19811 (LTT for allow/block gates) |
| Matched-budget comparison | A fairness claim about a decision rule must hold at a **matched operating point**. Otherwise a stricter threshold looks "fairer" simply because every rate is lower. The natural control for Mondrian conformal is **marginal (pooled) split-conformal at the same ε**: same score, same data, same budget, only the per-group conditioning differs. | Threshold-invariant fairness (Chen & Wu 2020); fairness at all thresholds (2203.07490) |
| Naming | Equal FPR across groups is **predictive equality** (Corbett-Davies et al. 2017). Equalised odds needs equal TPR as well (Hardt et al. 2016). When base score distributions differ, FPR and TPR can't both be equal (Chouldechova 2017; Kleinberg et al. 2016). | as named |
| Dataset-identity confound | Harmful and benign contrast sets drawn from different sources let a probe learn provenance. Best practice is to test on **matched-surface contrast pairs** and on out-of-source harmful prompts. | "When Benchmarks Lie" (2602.14161); "False Sense of Security" probing (2509.03888); XSTest's contrast design (Röttger et al. 2024) |
| Closest prior method | Aziz et al. 2026 (2606.01196; EMNLP Findings) already run per-language calibrated conditional steering **end to end** with CAST and AdaSteer as baselines, scored by refusal selectivity Δ = HRR − BRR. Our contribution has to be stated *relative to it*: a budgeted FPR objective with a guarantee, the trade-off it implies, three-way outcomes (harmful compliance), and robustness. | arXiv/ACL Anthology record |

## 2. Design check (calibration data only; no test output inspected)

The Earth calibration activations were split by a seeded permutation of ids (seed 20261028; ids are parallel, so a prompt never straddles folds):
- 1/3 fit: 35 harmful and 30 benign ids per language;
- 2/3 conformal: 70 harmful and 60 benign ids per language.

The layer was chosen on the fit fold (layer 13, AUROC 0.968). Within the conformal fold, 200 random halvings were used to predict the gate-level behaviour of the two rules:

| ε | Rule | mean FPR | worst-language FPR | lowest TPR |
|---|---|---|---|---|
| .05 | Mondrian (EORS-S) | 3.2% | 3.6% | th 0.33 |
| .05 | pooled conformal (CPOOL-S) | 5.1% | 12.0% (th) | vi 0.48 |
| .10 | Mondrian | 9.9% | 10.6% | jv 0.70 |
| .10 | pooled conformal | 10.0% | 22.4% (th) | ar 0.72 |

**Prediction.** At a matched budget, Mondrian calibration does equalise false-firing. In the languages whose benign scores sit high (th, bn, jv), it pays by firing less on harmful prompts: at ε = .05, Thai TPR falls from 0.80 to 0.33.

This is the predictive-equality trade-off. It is the result the paper should report, in place of "the same safety at half the over-refusal".

**Power.** About 106 benign test prompts per language, so the SE of a worst-language BRR difference is ≈ 3.5–4 pp. The predicted gaps are ≈ 8 pp (ε .05) and ≈ 11 pp (ε .10), detectable at roughly 2× and 3× the SE.

## 3. Premortem: "the controls ran and the paper still failed review". Why?

| # | Failure | Likelihood | Mitigation (built into the design) |
|---|---|---|---|
| 1 | The leak was not the cause: FPR still exceeds ε because LID routes prompts to the wrong language's threshold | low (LID accuracy logged per run) | Log `lid_accuracy_on_eval`. Gate-level validity is computed both with LID-routed thresholds and with oracle language |
| 2 | Halving calibration makes thresholds so noisy that per-language FPR scatters, and EORS-S looks "unfair" by chance | medium | Pre-register the Beta–binomial predictive band per language; judge validity by exceedances, not by point FPR |
| 3 | The control is not matched end to end: equal gate FPR still gives unequal BRR because base refusal (E0) differs by language | medium | Report BRR *and* the excess BRR over E0 (BRR − BRR_E0) per language. Gate-level FPR is reported from activations |
| 4 | Forking paths: many ε, rules, judges and metrics; we pick the flattering one | high if unmanaged | Two primary estimands, fixed now, Holm over two ε; everything else labelled secondary or exploratory; Qwen3Guard only, named |
| 5 | The max-over-languages statistic is biased and its bootstrap unreliable | certain | Plug-in point estimates; say "percentile bootstrap of a max". Also report the range (max−min) and the full per-language table |
| 6 | The style confound is real: the gate fires on XSTest-unsafe no more than on XSTest-safe | medium | The XSTest unsafe/safe contrast in 10 languages (matched surface) plus Aya red-team OOD harmful prompts. The reading rule is fixed now; a negative result is reported as a limitation of every projection-gated arm |
| 7 | Machine translation of XSTest-unsafe is poor in jv/sw, so the diagnostic is noise | medium | Same NLLB pipeline and QA columns (GlotLID, back-translation chrF) as the benign set; report the AUROC with and without chrF < 40 rows |
| 8 | Same test prompts reused (no fresh test set) | certain | Disclose. The new estimands are new; the earlier test-set analyses were descriptive of other arms. The OOD Aya and XSTest-unsafe sets are genuinely unseen |
| 9 | Colab dropouts or OOM | medium | The proven driver (per-arm fetch, LOST detection, resume), at most two sessions, VM always stopped |
| 10 | Hardware mixing | low | All arms on A100, paired with E0_a100. Steering vectors and behaviour layers 12–16 are identical to the α = 1 arms, so only the gate differs |
| 11 | Harm labels in jv/bn/sw are unvalidated, so the "cost" (P2) rests on unvalidated labels | certain | P2's primary is HRR (refusal; κ 0.89 validated across languages). Harmful compliance is secondary, with the caveat |
| 12 | The reviewer's "it's just a classifier + forced refusal" point | certain | Embrace it: frame the gate as the decision rule and the study as threshold-policy evaluation, which is what the evidence supports |

## 4. What is *not* run, and why
- **PAC/LTT thresholds:** degenerate at n ≈ 60 per language and ε = .05 (it needs zero calibration errors). Discussed as a limitation.
- **Equal-opportunity (per-language TPR) gates end to end:** this is the other pole of the trade-off. It is shown at gate level on test activations (exploratory frontier), not generated.
- **A second model family:** out of scope for this budget; stated as the main external-validity limit.
