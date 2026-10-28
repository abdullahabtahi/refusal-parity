# Self-review log (adversarial pass on paper/main.tex)

Method: the `peer-review` skill's claim→evidence and statistics checklist, applied to our own draft. Processed locally only; no manuscript text was sent to external services. This is a working draft by the drafting agent, not an independent review.

## Pass 1: 2026-10-27 ≈07:20 (new sections: baseline, H1, gate-on-failures, ROI, circuit parity)

| # | Location | Finding | Action |
|---|---|---|---|
| 1 | Contribution 1 | "We validate three judges against human labels" — the primary local judge (Qwen3Guard) was not yet validated | Reworded to the refusal judge; TODO until the `pgp_val` Colab run lands |
| 2 | H1 paragraph | "Spearman ρ = 0.29" was an artefact: nine languages tie at zero keyword refusals, so no keyword ranking exists | Restated as "the ranking collapses (nine-way tie at zero)"; macro removed |
| 3 | Gate on failures | "not a detection failure… the model answers anyway" overclaims from a linear read-out at one layer; triggering ≠ repair | Narrowed to the linear read-out at the gating layer; the trigger-vs-repair distinction is now stated |
| 4 | Baseline | "Over-refusal is not a low-resource effect" — bn and sw are low | "does not track resource tier" |
| 5 | Baseline | "roughly 5× higher" ratio of two small proportions, no uncertainty | Labelled a point estimate and imprecise |
| 6 | Abstract (i), Contribution 1 | Measurement TODOs | Filled with the H1 result (macros) |
| 7 | Intro; ROI | "a popular open safety stack" (subjective); per-response cost scope unstated | "the open safety stack we build on"; the ROI sentence now states the 16-language prompt set |
| 8 | Abstract | "We release…" while all artifacts are private | "We will release…" (publishing is the owner's decision) |

Verified rather than changed:
- Earth's "Africa/West Asia" specialisation: model card, "Best for West Asian and African languages".
- The H1 claim "never refuses outside English": keyword-detected HRR is exactly 0.0 in all nine languages.
- Issue #3's scope matches the intro's "every refusal detector shipped with the stack".
