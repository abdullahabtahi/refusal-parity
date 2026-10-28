# Paper playbook: what the rewrite must do (2026-10-27)

Distilled from three inputs: six exemplar papers in this sub-field, two widely cited writing guides, and Cohere's own design system and chart style. Every rule below cites where it came from. The rewrite of `paper/main.tex` follows this file.

## 1. Benchmark: six papers with clear community acceptance, and what they do well
| Paper | Craft pattern worth copying |
|---|---|
| Arditi et al. 2024, *Refusal in LMs Is Mediated by a Single Direction* (NeurIPS) | **The title is the claim.** The abstract runs context, gap, scoped claim ("13 models up to 72B"), two-way evidence (ablate removes, add induces), application, implication. Figure 1 captions **lead with the finding**. Hedges are specific ("more of an existence proof"). |
| Lee et al. 2025, *Programming Refusal with Conditional Activation Steering* (ICLR) | Run-in labels in the intro ("Problem:", "Contribution:"). Three crisp bullet contributions. **Tables mirror the headline figure**, so readers can check the numbers. Arrow notation for change vs base (90.7←45.8). |
| Yong et al. 2023, *Low-Resource Languages Jailbreak GPT-4* | **Reframes a fairness issue as a risk to everyone.** One memorable number benchmarked against a reference. **An explicit scope limit in the abstract** makes the claim more credible. Ends with a call to action. |
| Deng et al. 2024, *Multilingual Jailbreak Challenges* (ICLR) | A **two-scenario taxonomy** (unintentional / intentional) widens the stakes. A resource-tier gradient turns a vague concern into a measurable pattern. Diagnosis plus remedy, with a quantified before/after. |
| Hanna et al. 2024, *Have Faith in Faithfulness* (COLM) | **How to deliver a negative finding:** concede strengths first, target a shared *assumption* rather than authors, report unfavourable results openly, and end on a quotable principle ("overlap is no substitute for faithfulness"). |
| Röttger et al. 2024, *XSTest* (NAACL) | Treats over-refusal as a real failure, and **both failure directions are measured** (comply with safe, refuse unsafe). A small, carefully designed set is presented as precision, not a limitation. |

**Shared pattern:**
- one to three claims, with the title stating the main one;
- abstract: one idea per sentence, a concrete number, a scope limit;
- Figure 1 carries the paper;
- captions open with the finding;
- limitations are specific, not ritual.

## 2. Writing guides (verified sources)
**Nanda, *Highly Opinionated Advice on How to Write ML Papers* (Alignment Forum):**
- Compress the work into 1–3 claims before writing.
- Intro order: context, background, contribution, evidence, impact, then bullets.
- Every figure has one takeaway; captions stand alone; don't encode meaning in red vs green.
- Red-team your own claims, and say what was predicted vs post hoc.
- *"Inform, not persuade."*
- *"Quality > Quantity."*

**Lipton & Steinhardt, *Troubling Trends in ML Scholarship* (arXiv 1807.03341).** Avoid four things:
1. speculation dressed as explanation;
2. unclear sources of gains;
3. mathiness;
4. misused language.

**Installed skills applied at each step:**
- scientific-writing: evidence binding, no invented numbers, confirmatory vs exploratory;
- venue-templates: NeurIPS 2026 kit;
- scientific-visualization and dataviz: honest encodings, a validated palette, provenance;
- peer-review and scholar-evaluation: an adversarial pass and a scored self-assessment before and after;
- citation-management and paper-lookup: every citation verified;
- statistical-power: MDE statements.

From the vendor/scientific-agent-skills set: latex-posters and scientific-slides are reserved for later deliverables; no other writing skill in the set applies.

## 3. Design system for figures: Cohere / Cohere Labs, verified
**Sources:** Pentagram's Cohere case study ("new nature", Voronoi cells; natural vs synthetic tones), cohere.com production CSS, and colours sampled from the Tiny Aya report's own figures (arXiv 2603.11510).

**Palette:**
| Role | Token (cohere.com) | Hex | Seen in Tiny Aya charts |
|---|---|---|---|
| Primary series | acrylic blue-500 | `#4c6ee6` | yes |
| Secondary / "before" | blue-325 | `#8fa6f9` | yes |
| Improvement / "after" | chart green | `#5bbf8a` | yes |
| Harm / warning accent | coral-700 (coral-500 `#ff7759` for fills) | `#ca492d` | n/a (brand accent) |
| Fourth category | quartz-700 | `#9b60aa` | n/a |
| Ink (text, axes) | Tiny Aya ink | `#16270d` | yes |
| Grid / secondary axis | green-grey | `#8a9386`, light `#c5c9c3` | yes |
| Surface | mushroom-50 / white | `#f5f4f2` / `#ffffff` | white |

- **Validated** with the dataviz checker (blue, green, coral-700, quartz-700): all colour-blind checks pass. One warning: green on white is 2.2:1 contrast, so **green series always carry direct labels**.

**Type:** CohereText falls back to Space Grotesk and then Inter; CohereMono to a monospace. Figures use **Space Grotesk** for titles and labels and **Space Mono** for tick and data labels, both OFL from Google Fonts (pending the owner's OK to download). The fallback is Inter, which is installed.

**Chart conventions (from Cohere Labs' figures):**
- white background, light horizontal gridlines, a heavy grey left and bottom axis;
- direct value labels in mono, dashed reference lines for baselines;
- a legend top-right in a box, or direct labels instead.

**What we keep from scientific practice:**
- action titles that state the finding (Cohere's titles are topical);
- intervals named in captions; provenance JSON.

## 4. Applying it to our paper
**Compressed narrative: three claims.**
1. **The gate, not the steering vector, sets how fair a repair is.** At α = 1 every gated arm makes the model about equally safe (SRR +4–6 pp). How false refusals are distributed across languages depends on the threshold rule. Per-language conformal thresholds (EORS) cut worst-language false refusal by ~25 pp at under 1 pp safety cost. CAST's sign-blind score costs low-resource languages ~3 pp of refusal, which is small, real, and predicted by Prop. 1.
2. **Measure with validated multilingual instruments, or not at all.** English keyword detectors detect 0% of refusals outside English. Judges differ on harm (Qwen3Guard κ 0.74 vs Gemini 0.43). Refusal is not safety.
3. **Clean-benchmark fairness overstates deployed fairness.** A greeting multiplies the gate's false-firing 3–10×.

**Secondary results, reported briefly with power statements:** natural drift ~1 pp; no preamble effect (safety lives in the weights); ReSta doesn't help; CircuitKIT finds no sparse circuit (tooling appendix).

**Title options:**
- *A Repair Is Only as Fair as Its Gate: Auditing Multilingual Refusal Steering in Tiny Aya* (preferred)
- *Fair Refusal Needs a Fair Gate*
- *Who Pays for Safety? Language-Fair Refusal Repair in a Small Multilingual Model*

**Abstract skeleton, one idea per sentence:**
1. context: small multilingual models, no guardrail;
2. gap: repairs judged on mean or English metrics with English instruments;
3. what we do: 10 languages, 3 tiers, 2 validated judges, 9 repair arms, paired;
4. claim 1 with the EORS number and CAST −3 pp;
5. claim 2 with κ and 0% recall;
6. claim 3 with 3–10×;
7. scope limit: one model family, gate-level brittleness, power;
8. release.

**Figure 1:** a Pareto teaser of mean safety gain vs worst-language over-refusal for every arm. One glance shows EORS dominating and the steering-only arms bunched at high over-refusal. Inset or panel (b): the prefix brittleness.

**Figure list:**
- Fig 1, the teaser;
- Fig 2, per-language false refusal, G-POOL vs EORS (dumbbell);
- Fig 3, CAST vs G-EN per-language ΔHRR, ordered by the negative-projection share (Prop. 1);
- Fig 4, the judge/instrument validity panel;
- Fig 5, the brittleness bars (original, own-language instruction, English instruction, greeting);
- Appendix: drift (current Fig 5), baseline outcomes, detection/action, CircuitKIT faithfulness curve.

**Rules for the rewrite:**
- every number via a macro;
- predicted vs post hoc labelled;
- negatives reported plainly, Hanna-style;
- limitations specific;
- no speculation as explanation (brittleness mechanism: "open");
- a scholar-evaluation score before and after.
