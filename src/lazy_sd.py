"""A state_dict that reads tensors on demand from a model's safetensors shards.

Lets SafeTune's apply_resta (which streams one tensor at a time) run with the base and aligned
models never loaded: 3 x 6.7 GB models would not fit in 16 GB.
"""

from collections.abc import Mapping
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open


class LazyStateDict(Mapping):
    def __init__(self, repo_id):
        d = Path(snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json"]))
        self._files = {}
        for f in sorted(d.glob("*.safetensors")):
            with safe_open(f, framework="pt") as h:
                self._files.update(dict.fromkeys(h.keys(), f))

    def __getitem__(self, key):
        with safe_open(self._files[key], framework="pt") as h:
            return h.get_tensor(key)

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)


class LazyModel:
    """Duck-types the one nn.Module method apply_resta calls on base/aligned: state_dict()."""

    def __init__(self, repo_id):
        self._sd = LazyStateDict(repo_id)

    def state_dict(self):
        return self._sd
