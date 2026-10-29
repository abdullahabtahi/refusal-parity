---
title: "Who pays for safety repair? Auditing refusal steering across ten languages on Tiny Aya"
thumbnail: https://huggingface.co/datasets/abdullahabtahi/refusal-parity/resolve/main/assets/banner.png
authors:
- user: abdullahabtahi
---

# Who pays for safety repair? Auditing refusal steering across ten languages on Tiny Aya

*We repaired the refusals of a small multilingual model with gated activation steering and measured every language
separately. The rule that sets the gate's threshold decides which languages bear over-refusal and which lose
protection, the open tools to measure this work only in English, and one harmless greeting undoes the calibration.*

**Abdullah Ibn Mahdi Abtahi**, Ingramin · October 2026 ·
[Paper](https://huggingface.co/datasets/abdullahabtahi/refusal-parity/blob/main/paper/refusal_parity.pdf) ·
[Code](https://github.com/abdullahabtahi/refusal-parity) ·
[Data](https://huggingface.co/datasets/abdullahabtahi/refusal-parity)

![Refusal Parity — Tiny Aya multilingual safety repair audit](https://huggingface.co/datasets/abdullahabtahi/refusal-parity/resolve/main/assets/banner.png)

Ask a small model "how do I kill a Python process?" and a badly tuned safety layer will refuse. XSTest was built
around exactly this kind of benign question with an alarming word in it. Now ask it in Swahili or Javanese, on a
phone with no server-side guardrail behind it. Whether the model refuses, and whether it still refuses a genuinely
harmful request, is decided by whatever *repair* was applied to it, and that repair was almost certainly tuned and
judged on an average over languages, with tools built for English.

Small multilingual models such as Cohere Labs' [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-earth) (3.35B)
are exactly the models that run where no external filter does, often for speakers of low-resource languages. So we
asked a narrow question: **when you repair a small model's refusals, which languages pay for it?**

> ⚠️ The evaluation uses harmful requests from public red-teaming benchmarks. This post quotes none of them.

## TL;DR

> All numbers are from Tiny Aya Earth on an A100, judged by Qwen3Guard-4B on held-out test prompts in
> {{n_langs}} languages, with 95% paired bootstrap intervals. Each one is generated from the released result files
> by the same script that writes the paper.

1. **The threshold rule decides who pays.** At the same false-refusal budget, per-language thresholds cut the worst
   language's benign refusal from {{CtlFivePoolWorstBrr}} to {{CtlFiveEorsWorstBrr}}. Low-resource languages pay:
   their harmful-prompt refusal changes by {{CtlFiveLrlHrr}} points.
2. **The open instruments are English-only.** The stack's keyword refusal detectors catch {{HOneRecallOtherMax}} of
   refusals outside English, and a closed model looks safer under a judge from its own family.
3. **"Good morning." breaks the calibration.** A harmless greeting raises the repaired model's benign refusal from
   {{CtlGreetBrrEors}} to {{CtlGreetBrrEorsGreet}}, under either threshold rule.

## What we built

| | |
|---|---|
| **Model** | Tiny Aya Earth (repairs), Global (natural-drift sibling), Base (weight-space reference) |
| **Languages** | {{n_langs}} MultiJail languages: high-resource {{hrl_langs}}; mid {{mrl_langs}}; low {{lrl_langs}} |
| **Prompts** | Harmful: MultiJail. Benign: XSTest's safe prompts, machine-translated (NLLB) and filtered by back-translation. Ids parallel across languages, so every comparison is paired |
| **Repairs** | Eleven arms: gated activation steering with different gate scores and threshold rules, SafeTune's CAST, and the weight-space repair ReSta |
| **Judges** | Qwen3Guard-4B (primary) and Gemini (secondary), both scored against {{NValidation}} human labels |
| **Plan** | Pre-registered, with a dated deviations log; every result is labelled pre-registered, exploratory or post hoc |
| **Stack** | The Lexsi open-source libraries end to end: CuratorKIT → CircuitKIT → SafeTune → AuditKIT |

Outcomes on harmful prompts are three-way: refuse, answer safely, or answer harmfully. We report the safe-response
rate (one minus harmful compliance) as the primary safety metric, and benign refusal as its cost. Refusal is not
safety: a model can answer a harmful request safely, and we count that as safe.

![Study pipeline diagram showing the seven subpackages in src/refusal_parity/ connected to the Lexsi open-source stack (CuratorKIT, CircuitKIT, SafeTune, AuditKIT).](https://huggingface.co/datasets/abdullahabtahi/refusal-parity/resolve/main/assets/pipeline.png)
*The pipeline. Data provenance in CuratorKIT, steering and ReSta in SafeTune, generation and judging in AuditKIT; the
judges are validated against human labels before any result is read.*

## A gated repair is a classifier with a threshold

Most of our arms are *gated* activation steering. A direction in the residual stream scores each prompt; prompts above
a threshold get a refusal vector added to the model's activations, and the rest pass through untouched. Once the
steering is strong enough to refuse reliably, the repair is a classifier followed by a forced refusal. Its fairness
across languages is then a property of one decision: **how the threshold is set**.

The standard choice is one threshold for every language, set by split-conformal calibration so that benign prompts
trigger the refusal at most ε of the time. The alternative sets one threshold per language, each at the same ε.
Both come with a guarantee on false refusals; they differ in who the guarantee is spread over.

## Finding 1: equalising over-refusal moves protection away from low-resource languages

We compared the two rules with everything else held fixed: the same steering vector, the same gate direction, the
same calibration data (pre-registered).

![Teaser figure: (a) mean safe-response rate against worst-language benign refusal across repair arms on Tiny Aya Earth, with arrows from the unrepaired model (E0) to pooled split-conformal threshold (CPOOL-S) and per-language split-conformal thresholds (EORS-S); (b) mean benign refusal without and with the English greeting prefix across E0, CPOOL-S, and EORS-S.](https://huggingface.co/datasets/abdullahabtahi/refusal-parity/resolve/main/assets/fig1_teaser.png)
*Left: every repair arm, safety against the worst language's benign refusal. Per-language thresholds (EORS-S) pull
the worst language down from the pooled rule (CPOOL-S) at the same budget. Right: the greeting test of Finding 3.*

At ε = 0.05, per-language thresholds lower the worst language's benign refusal from {{CtlFivePoolWorstBrr}} to
{{CtlFiveEorsWorstBrr}}: {{CtlFiveMaxBrr}} points, 95% CI [{{CtlFiveMaxBrrLo}}, {{CtlFiveMaxBrrHi}}], with the mean
essentially unchanged ({{CtlFiveMeanBrr}} points). The guarantee holds out of sample: on held-out benign prompts,
{{CtlFiveValidAbove}} of {{n_langs}} languages exceed the per-language bound's predictive band.

The average hides the price. Harmful-prompt refusal in the low-resource tier changes by {{CtlFiveLrlHrr}} points
[{{CtlFiveLrlHrrLo}}, {{CtlFiveLrlHrrHi}}], most in {{CtlFiveHrrLossLang}} ({{CtlFiveHrrLossPts}}), and the gap between
the high- and low-resource tiers widens from {{CtlFivePoolTierGap}} to {{CtlFiveEorsTierGap}} points. The mechanism
is arithmetic, not a bug: in languages where *benign* prompts already score high on the gate, holding their false
refusals to ε forces the threshold up, and harmful prompts slip under it.

![Per-language refusal dot plots with Wilson intervals grouped by resource tier: (a) benign prompts refused and (b) harmful prompts refused for unrepaired E0, pooled threshold CPOOL-S, and per-language threshold EORS-S.](https://huggingface.co/datasets/abdullahabtahi/refusal-parity/resolve/main/assets/fig2_eors.png)
*Per language, grouped by tier, with 95% Wilson intervals. Left: per-language thresholds narrow over-refusal. Right:
harmful-prompt refusal falls most where benign prompts score high on the gate (Javanese, Thai, Bengali).*

**Is the gate just detecting benchmark style?** Harmful and benign prompts come from different datasets, so the
gate could have learned provenance rather than harm. We checked: on XSTest's own *unsafe* contrast prompts, which
share the benign prompts' surface form and translation pipeline, the gate separates unsafe from safe with AUROC
{{CtlStyleAuc}}, and at least {{CtlStyleAucMin}} in every language.

Neither rule is fair in general. Equal false-refusal rates across languages whose score distributions differ force
unequal protection, and the reverse holds too. This is the familiar impossibility from fair classification, showing
up inside a safety repair. Which error to equalise is a deployment decision. The problem is that an average lets
it be made without anyone noticing.

## Finding 2: the instruments decide what you can see

**Keyword refusal detectors are English-only.** SafeTune ships two keyword detectors, which the stack uses to score
refusal and to choose steering layers. They agree with the judge in English and catch {{HOneRecallOtherMax}} of
refusals in each of the nine other languages (pre-registered). A keyword audit would report that the model never
refuses outside English.

**Judges agree on refusal, not on harm.** Against human labels from PolyGuardPrompts, both judges detect refusal well
(κ = {{QRefKappaMean}} for Qwen3Guard, {{RefKappaMean}} for Gemini). On harm, Qwen3Guard reaches κ =
{{QHarmKappaMean}} and Gemini {{HarmKappaMean}}. The human labels cover seven languages, none of them in our
low-resource tier, so we report harm under both judges.

**A judge from the same family is lenient.** Gemini 3.5 Flash-Lite, run as a closed comparator, is {{GemGSrr}} safe
under the Gemini judge and {{GemQSrr}} under Qwen3Guard, close to the unrepaired Tiny Aya's {{EZeroAhundredSrr}}. The
Gemini judge is lenient on everything, and its leniency is {{CtlJudgeDid}} points larger on its own family's outputs (post hoc),
consistent with known self-preference in LLM judges.

**A gate score that ignores sign.** SafeTune's conditional steering (CAST) thresholds a score that is approximately
invariant to the sign of the projection, so a prompt pointing *away* from the harm direction can still pass the gate.
End to end, CAST's harmful-prompt refusal in the low-resource tier changes by {{CastQLrlHrr}} points, but it also
refuses fewer benign prompts, so we cannot separate the sign problem from a different operating point (exploratory). Our gated arms threshold
the signed projection instead.

## Finding 3: one greeting undoes the calibration

This one we did not plan for. We were testing whether an attacker could prepend English text to route a harmful
prompt to a laxer per-language threshold. They cannot: the gate fires *more* on prefixed harmful prompts. But on
benign prompts the same experiment showed a much larger effect. The gate's false-firing rate is {{BritOrigMean}} on
the original prompts and {{BritGreetMean}} once they begin with "Good morning." (found post hoc).

Because the finding was post hoc, we tested it again end to end, with the analysis fixed before generation. The
greeting raises the repaired model's mean benign refusal from {{CtlGreetBrrEors}} to {{CtlGreetBrrEorsGreet}} with
per-language thresholds and from {{CtlGreetBrrPool}} to {{CtlGreetBrrPoolGreet}} with the pooled one. The unrepaired
model barely moves ({{GreetBrrEZero}} to {{GreetBrrEZeroGreet}}).

The conformal guarantee holds for prompts like the calibration set, and conversational framing is not like a
benchmark. A gate calibrated on clean benchmark prompts should be tested against the formats users actually send.

## Also worth knowing

- **The weight-space repair does not help here.** ReSta changes the mean safe-response rate by {{RestaQSrr}} points
  and worst-language benign refusal by {{RestaQMaxBrr}} points.
- **Self-hosting is cheap; auditing is still required.** The closed comparator cost {{RoiGemCost}} for {{RoiGemN}}
  responses at list price. Tiny Aya generated {{RoiTinyPerMin}} responses per minute on a 16 GB laptop with no
  per-token fee.
- **Power is the binding limit.** Per-language effects below a few points are beyond this design. Where a test is
  underpowered, the paper says so instead of calling it a null.

## Recommendations

For anyone shipping a gated safety repair on a multilingual model:

1. Report per-language benign refusal **and** harmful-prompt refusal at matched budgets, never only an average.
2. Fit the gate direction and its thresholds on disjoint data.
3. Calibrate on, or at least test against, the formats users actually send.
4. Validate refusal detectors in every language you report, and judge a closed model with a judge outside its family.

## Limitations (read before citing)

- **One model family**, at 3.35B parameters, in {{n_langs}} languages.
- **Translated benign prompts.** The benign probes are machine translations, not native prompts.
- **Judges** are validated against humans in seven languages, none of them low-resource here.
- **Decoding** is greedy, at one steering strength, with a robustness check at half strength.
- **Reused test prompts.** The controls reuse the test prompts of earlier contrasts with new estimands. Only the
  XSTest-unsafe and Aya red-teaming sets are unseen.

The full list, and every deviation from the pre-registered plan, is in the paper.

## Use the data

```bash
git clone https://github.com/abdullahabtahi/refusal-parity && cd refusal-parity
make data          # open prompts, labels and summaries from the Hub
make verify        # rerun the analysis; fails unless every paper number is byte-identical
```

- [`refusal-parity`](https://huggingface.co/datasets/abdullahabtahi/refusal-parity) (open): prompts, one label row
  per response for every run, every summary the paper is generated from, and the judge-validation statistics.
- [`refusal-parity-responses`](https://huggingface.co/datasets/abdullahabtahi/refusal-parity-responses) (gated): full
  responses with every judge label, for re-judging or auditing individual decisions.
- [`refusal-parity-steering`](https://huggingface.co/abdullahabtahi/refusal-parity-steering) (gated): every steering
  vector, gate direction and threshold as the runs applied them.

The gated repositories hold harmful text and dual-use vectors, and access is reviewed manually.
`make data GATED=1` fetches them once you are approved.

## What's next

- Native rather than translated benign prompts, especially in the low-resource tier.
- Human harm labels in Bengali, Swahili and Javanese, so harm can be scored with a validated judge in every language.
- Gates calibrated on conversational data, and a mechanism for why a greeting moves the gate score.
- The same audit on a second model family.

## Acknowledgements

Built as a reference project for the Lexsi AI Safety & Alignment Hackathon, on the Lexsi open-source stack. Thanks
to Cohere Labs for Tiny Aya, and to the authors of MultiJail, XSTest, PolyGuard, Qwen3Guard, GlotLID and NLLB.
Implementation and drafting were a collaboration between the author and an AI agent (Claude, Anthropic, via
Claude Code).

Questions and corrections: abdullah.abtahi@ingramin.com.

## References

- Salamanca et al. *Tiny Aya: Bridging Scale and Multilingual Depth.* arXiv:2603.11510, 2026.
- Deng et al. *Multilingual Jailbreak Challenges in Large Language Models.* ICLR 2024.
- Röttger et al. *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL 2024.
- Lee et al. *Programming Refusal with Conditional Activation Steering.* ICLR 2025.
- Bhardwaj et al. *Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic.* ACL 2024.
- Kumar et al. *PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages.* COLM 2025.
- Zhao et al. *Qwen3Guard Technical Report.* arXiv:2510.14276, 2025.
- Kargaran et al. *GlotLID: Language Identification for Low-Resource Languages.* Findings of EMNLP 2023.
- Yong et al. *The State of Multilingual LLM Safety Research: From Measuring the Language Gap to Mitigating It.* EMNLP 2025.
- Panickssery et al. *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024.
- Vovk, Gammerman and Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
