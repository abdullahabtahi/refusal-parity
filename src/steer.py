"""Gated steering on SafeTune's CASTModel: same CAA vectors and hooks, a different gate score.

SafeTune's faithful CAST gate scores cos(h, tanh(v (v.h))), which depends on the sign of the projection;
when a language's activations shift (the low-resource "harm score shift"), the ranking can invert
(tests/test_arms.py: AUROC 1.0 -> 0.0 on synthetic shifted languages). GateModel scores the projection
h.v instead (monotone in the harm direction) and takes a threshold that is a float or {lang: float}.

  EORS / EORS-S: tau_l = the conformal (1 - eps) quantile of language-l benign calibration scores, so every
  language gets the same benign false-firing budget eps (predictive equality). The finite-sample bound holds
  only when the direction is fitted on data disjoint from the calibration fold (EORS-S; see arms.py).
  Language comes from constrained GlotLID at inference.
"""

import torch
from safetune._refusal_helpers import _get_decoder_layers
from safetune.interventions.steer.cast import CASTModel


class GateModel(CASTModel):
    def __init__(self, model, vecs, cond, thresholds, alpha):
        super().__init__(model, vecs, condition=cond, alpha=alpha, per_prompt_gate=True)
        self.thresholds = thresholds
        self.row_langs = None  # set per batch by the caller when thresholds is per-language

    def _gate_fires(self, input_ids, **kwargs):
        got = []
        layer = _get_decoder_layers(self.model)[self.condition_layer]
        hook = layer.register_forward_hook(
            lambda _m, _i, o: got.append((o[0] if isinstance(o, tuple) else o)[:, -1, :].float())
        )
        try:
            with torch.no_grad():
                self.model.model(input_ids=input_ids, **kwargs)  # decoder only: no 262k-vocab logits
        finally:
            hook.remove()
        scores = (got[0] @ self.condition_vector.to(got[0].device).float()).cpu().tolist()
        if isinstance(self.thresholds, dict):
            fallback = max(self.thresholds.values())  # unknown language: strictest threshold, fewest false refusals
            thr = [self.thresholds.get(l, fallback) for l in self.row_langs]
        else:
            thr = [self.thresholds] * len(scores)
        return [s > t for s, t in zip(scores, thr, strict=True)], scores
