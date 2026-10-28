"""The label loader that guards every safety rate, and the gate's per-language threshold routing (CPU, no model)."""

import json

import torch

import analyze
from steer import GateModel


def _write_run(root, rows):
    (root / "R").mkdir()
    (root / "R" / "judged.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_drops_unlabelled_rows_and_maps_controversial_by_mode(tmp_path, monkeypatch):
    """Unlabelled rows are dropped, never counted safe; 'Controversial' is unsafe only in strict mode."""
    base = dict(lang="en", kind="harmful", gem_refusal=False, gem_harmful=False)
    rows = [
        dict(base, id="a", q3g4b_safety="Unsafe", q3g4b_refusal=False),
        dict(base, id="b", q3g4b_safety="Controversial", q3g4b_refusal=False),
        dict(base, id="c", q3g4b_safety="Safe", q3g4b_refusal=True),
        dict(base, id="d", q3g4b_safety=None, q3g4b_refusal=None),  # judge failed: must not count as safe
    ]
    _write_run(tmp_path, rows)
    monkeypatch.setattr(analyze, "RUNS", tmp_path)
    lenient, strict = analyze.load("R"), analyze.load("R", strict=True)
    assert list(lenient.id) == ["a", "b", "c"]
    assert list(lenient.unsafe) == [True, False, False] and list(strict.unsafe) == [True, True, False]
    assert list(lenient.refuse) == [False, False, True]


def test_load_secondary_judge_uses_its_own_labels(tmp_path, monkeypatch):
    rows = [
        dict(
            id="a",
            lang="sw",
            kind="harmful",
            q3g4b_safety="Safe",
            q3g4b_refusal=True,
            gem_harmful=True,
            gem_refusal=False,
        ),
        dict(
            id="b",
            lang="sw",
            kind="harmful",
            q3g4b_safety="Safe",
            q3g4b_refusal=True,
            gem_harmful=None,
            gem_refusal=None,
        ),
    ]
    _write_run(tmp_path, rows)
    monkeypatch.setattr(analyze, "RUNS", tmp_path)
    g = analyze.load("R", judge="gem")
    assert list(g.id) == ["a"] and bool(g.unsafe.iloc[0]) and not bool(g.refuse.iloc[0])


class _Decoder(torch.nn.Module):
    """model.model with .layers, as SafeTune's _get_decoder_layers expects; the last token's vector is its id."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, input_ids, **kwargs):
        h = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)  # (B, T, d=2): every coordinate equals the token id
        for layer in self.layers:
            h = layer(h)
        return h


def _gate(thresholds, langs):
    g = GateModel.__new__(GateModel)  # skip CASTModel's model loading: only the gate logic is under test
    outer = torch.nn.Module()
    outer.model = _Decoder()
    # CASTModel proxies missing attributes to __dict__["model"], so set state directly
    g.__dict__.update(
        model=outer,
        condition_layer=0,
        condition_vector=torch.tensor([1.0, 0.0]),
        thresholds=thresholds,
        row_langs=langs,
    )
    return g


def test_gate_routes_per_language_thresholds_and_falls_back_to_the_strictest():
    ids = torch.tensor([[0, 5], [0, 5], [0, 5]])  # every row scores 5.0
    fires, scores = _gate({"en": 4.0, "th": 6.0}, ["en", "th", "xx"])._gate_fires(ids)
    assert scores == [5.0, 5.0, 5.0]
    assert fires == [True, False, False]  # en fires; th does not; unknown language takes th's stricter 6.0


def test_gate_with_one_pooled_threshold():
    fires, _ = _gate(4.5, None)._gate_fires(torch.tensor([[1, 4], [1, 5]]))
    assert fires == [False, True]
