"""Upload hf/release/ to PRIVATE Hugging Face repos and a private collection.

    .venv/bin/python scripts/upload_release.py            # build first: scripts/build_release.py

Creates the repos private (never changes the visibility of an existing repo), sets manual approval on the gated
ones, uploads each folder as one commit, and gathers them in a private collection. Making anything public is the
owner's action on the Hub.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf" / "release"
OWNER = "abdullahabtahi"
REPOS = (  # (name, repo type, gated)
    ("refusal-parity", "dataset", False),
    ("refusal-parity-responses", "dataset", True),
    ("refusal-parity-steering", "model", True),
)
COLLECTION = dict(
    title="Refusal Parity (Tiny Aya)",
    description="Is safety repair language-fair? Open labels and paper, gated responses and steering artifacts.",
)
VERSION = "v0.2.0"


def main():
    api = HfApi()
    for name, _, _ in REPOS:
        if not (OUT / name / "README.md").exists():
            sys.exit(f"{OUT / name} is not built; run scripts/build_release.py first")
    collection = api.create_collection(namespace=OWNER, private=True, exists_ok=True, **COLLECTION)
    for name, repo_type, gated in REPOS:
        repo_id = f"{OWNER}/{name}"
        if api.repo_exists(repo_id, repo_type=repo_type):
            if not api.repo_info(repo_id, repo_type=repo_type).private:
                sys.exit(f"{repo_id} is public; this script only uploads to private repos")
        else:
            api.create_repo(repo_id, repo_type=repo_type, private=True)
        if gated:
            api.update_repo_settings(repo_id, repo_type=repo_type, gated="manual")
        api.upload_folder(
            repo_id=repo_id, repo_type=repo_type, folder_path=OUT / name, commit_message=f"Release {VERSION}"
        )
        api.add_collection_item(collection.slug, item_id=repo_id, item_type=repo_type, exists_ok=True)
        print(f"uploaded {repo_id} (private{', gated: manual' if gated else ''})")
    print(f"collection (private): https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()
