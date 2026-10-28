# Security and responsible use

## Reporting a problem

Please report privately, not in a public issue, if you find:
- a security vulnerability in this code;
- harmful content released outside the gated repositories;
- a way to misuse the released steering artifacts that this project should mitigate.

Email **abdullah.abtahi@ingramin.com** with "refusal-parity security" in the subject. I aim to acknowledge reports
within a week.

## Dual-use artifacts

The steering vectors and condition directions encode Tiny Aya's refusal behaviour. Subtracted instead of added,
they can weaken refusal. They and the model responses (some of them harmful) are distributed only through gated
Hugging Face repositories with manual approval, for safety research:
- [`abdullahabtahi/refusal-parity-steering`](https://huggingface.co/abdullahabtahi/refusal-parity-steering)
- [`abdullahabtahi/refusal-parity-responses`](https://huggingface.co/datasets/abdullahabtahi/refusal-parity-responses)

Please do not re-host them without equivalent gating.

## Secrets

The code never stores credentials. Remote jobs read the Hugging Face token and SSH key from the environment
(`scripts/colab_gen.sh`). Release builds scan every published file for secret-like strings and local paths
(`scripts/build_release.py`).
