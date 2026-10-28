"""generate() sorts prompts by length; EORS sets per-row languages through on_batch(indices). Check that the
indices handed to on_batch are exactly the rows encoded in that batch, and that outputs return in input order."""

import torch

import generate


class Enc(dict):
    def to(self, _device):
        return self


class StubTok:
    def __init__(self):
        self.batches = []

    def __call__(self, texts, **_):
        self.batches.append(list(texts))
        return Enc(input_ids=torch.zeros(len(texts), 3, dtype=torch.long))

    def batch_decode(self, gen, **_):
        return [f"out{i}" for i in range(len(gen))]


class StubModel:
    def __init__(self, tok):
        self.tok = tok

    def generate(self, input_ids, **_):
        return torch.zeros(input_ids.shape[0], 5, dtype=torch.long)


def test_on_batch_indices_match_encoded_rows():
    texts = ["ccc", "a", "bbbbb", "dd", "eeee", "f"]  # deliberately unsorted lengths
    tok, seen = StubTok(), []
    generate.generate(tok, StubModel(tok), texts, batch=4, on_batch=lambda idx: seen.append(list(idx)))
    assert len(seen) == len(tok.batches)
    for idx, encoded in zip(seen, tok.batches, strict=True):
        assert [texts[i] for i in idx] == encoded  # on_batch indices <-> encoded rows, row by row
    assert sorted(i for idx in seen for i in idx) == list(range(len(texts)))


def test_outputs_return_in_input_order():
    texts = ["zz", "y", "xxx"]
    tok = StubTok()
    # stub decode labels by position in batch; map back through the recorded batch to check order restore
    out = generate.generate(tok, StubModel(tok), texts, batch=8)
    batch = tok.batches[0]
    assert [batch[int(o[3:])] for o in out] == texts


if __name__ == "__main__":
    test_on_batch_indices_match_encoded_rows()
    test_outputs_return_in_input_order()
    print("ok")
