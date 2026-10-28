"""Shared constants and small helpers for the Refusal Parity experiments."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/refusal_parity/common.py -> repo root
DATA = ROOT / "data"
RUNS = ROOT / "runs"

# MultiJail languages, grouped by resource tier (Deng et al., 2024).
TIERS = {
    "en": "HRL",
    "zh": "HRL",
    "it": "HRL",
    "vi": "HRL",
    "ar": "MRL",
    "ko": "MRL",
    "th": "MRL",
    "bn": "LRL",
    "sw": "LRL",
    "jv": "LRL",
}
LANGS = list(TIERS)
NLLB_CODES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "it": "ita_Latn",
    "vi": "vie_Latn",
    "ar": "arb_Arab",
    "ko": "kor_Hang",
    "th": "tha_Thai",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "jv": "jav_Latn",
}
# Exact Hub revisions behind every result (each repo was last modified before the first run, 2026-10-26).
REVISIONS = {
    "CohereLabs/tiny-aya-global": "af89d219b53ed9b13b8a4645f9c8028973510324",
    "CohereLabs/tiny-aya-earth": "0624fafbe200f930b89566fc4ec4da314ba48d89",
    "CohereLabs/tiny-aya-base": "d62ac17e241c871fc52e9a95d7b0e2615189d7d5",
    "Qwen/Qwen3Guard-Gen-4B": "6ec42827da0c1ff11e7a49dc269d2e810d27e108",
    "Qwen/Qwen3Guard-Gen-0.6B": "fada3b2f655b89601929198343c94cd2f64d93cc",
    "ToxicityPrompts/PolyGuard-Qwen-Smol": "224f511fe2ee3e304a11f470c8071f43ee0d8f70",
    "facebook/nllb-200-distilled-600M": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    "cis-lmu/glotlid": "85cd6716494360367b75f642b5bc78667605d0b4",
    "DAMO-NLP-SG/MultiJail": "93d65c778925d55973b9ff30c7bccc8545494eb6",
    "Paul/XSTest": "f600c994b256f12867dfa5b3eb3d545a3e62f8b5",
    "CohereLabs/aya_redteaming": "5a16fee03c19d681b4f3527b6a750e1ed4d434c7",
    "ToxicityPrompts/PolyGuardPrompts": "c5b466a95b64ff121db4398246b6abb7672696ec",
}
SEED = 20261026  # data splits and the per-language paired bootstrap (analyze.compare)
BOOT_SEED, N_BOOT = 20261027, 5000  # tier and difference-in-differences bootstraps (arms_report)
SPLIT_SEED = 20261028  # calibration fit / conformal fold split (arms.split_calibration; pre-registered 2026-10-28)
HRL, LRL = tuple(l for l in LANGS if TIERS[l] == "HRL"), tuple(l for l in LANGS if TIERS[l] == "LRL")
NAMES = dict(
    en="English",
    zh="Chinese",
    it="Italian",
    vi="Vietnamese",
    ar="Arabic",
    ko="Korean",
    th="Thai",
    bn="Bengali",
    sw="Swahili",
    jv="Javanese",
)
ORDER = sorted(LANGS, key=lambda l: ["HRL", "MRL", "LRL"].index(TIERS[l]))  # en zh it vi | ar ko th | bn sw jv


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# GlotLID labels are ISO 639-3 + script; map the ones we evaluate to the 2-letter codes used above.
ISO3 = {
    "eng": "en",
    "cmn": "zh",
    "zho": "zh",
    "ita": "it",
    "vie": "vi",
    "arb": "ar",
    "ara": "ar",
    "arz": "ar",
    "ary": "ar",
    "apc": "ar",
    "kor": "ko",
    "tha": "th",
    "ben": "bn",
    "swh": "sw",
    "swa": "sw",
    "jav": "jv",
    "hin": "hi",
    "fra": "fr",
    "rus": "ru",
    "srp": "sr",
    "hbs": "sr",
    "spa": "es",
    "tgl": "tl",
    "fil": "tl",
}


def load_lid():
    """GlotLID (fastText format). lid.176 misreads short Swahili/Javanese as English, so we do not use it."""
    import fasttext
    from huggingface_hub import hf_hub_download

    fasttext.FastText.eprint = lambda *a, **k: None  # silence deprecation spam
    return fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin", revision=REVISIONS["cis-lmu/glotlid"]))


def lid_langs(texts):
    """Constrained GlotLID language per text. Reads data/lid_cache.json (built locally with build_lid_cache, the
    same detect_lang) when it covers every text, so Colab, where fastText won't build, gets identical labels."""
    cache_path = DATA / "lid_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if all(t in cache for t in texts):
        return [cache[t] for t in texts]
    lid = load_lid()
    return [detect_lang(lid, t)[0] for t in texts]


def build_lid_cache(texts):
    lid = load_lid()
    (DATA / "lid_cache.json").write_text(json.dumps({t: detect_lang(lid, t)[0] for t in texts}, ensure_ascii=False))


def detect_lang(lid, text):
    """Constrained LID: the most probable label among the study's languages, as (2-letter code, prob).
    Unconstrained GlotLID picks close variants on short strings (Swahili -> swc, Bengali -> syl)."""
    # lid.f.predict sidesteps fasttext's np.array(copy=False) call, which NumPy 2 rejects.
    for prob, label in lid.f.predict(text.replace("\n", " ")[:1000], -1, 0.0, "strict"):
        iso3 = label.replace("__label__", "").split("_")[0]
        if iso3 in ISO3:
            return ISO3[iso3], float(prob)
    return "und", 0.0
