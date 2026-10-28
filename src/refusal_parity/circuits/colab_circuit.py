"""CircuitKIT on a Colab GPU (T4: fp16; CircuitKIT's default bf16 is not native on T4).

Step 1, the parity gate CircuitKIT's docs require before trusting discovery on Tiny Aya:
    python colab_circuit.py parity prompts.json     # prompts.json: list of chat-rendered strings (render(), BOS kept)
    -> parity.json: per-prompt KL(HF || TransformerLens) on the next token; gate passes at max KL < 1e-4
Step 2, refusal-circuit discovery (EAP-IG, node level) per language; `all` runs both in one session:
    python colab_circuit.py all circuit_spec.json   # built by circuit_spec.py

Prompts are rendered locally with the real chat template (generate.render) so the circuit is studied under the
same safety preamble as generation; CircuitKIT itself feeds raw text.
"""

import json
import subprocess
import sys
from pathlib import Path

BOS = "<BOS_TOKEN>"
MODEL = "CohereLabs/tiny-aya-global"


def ensure_deps():
    try:
        import circuitkit  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "circuitkit"], check=True)


TL_SLOW = "unhooked_result = (z * w).sum(-2)"
TL_FAST = 'unhooked_result = torch.einsum("bphd,hdm->bphm", z.squeeze(-1), w[0, 0])'


def patch_tl_attn_result():
    """TransformerLens computes per-head attention outputs (needed for node-level EAP) as a broadcast product
    [b, pos, heads, d_head, d_model] then a sum: 3.3 GiB per layer at ~460 tokens, which OOMs a T4 even at batch 1
    (Kaggle v5). The einsum is the same contraction without the 5-D intermediate (~30 MB); values and gradients
    agree to fp rounding (checked locally). Must run before transformer_lens is imported; parity is re-run on the
    patched path."""
    import importlib.util

    path = (
        importlib.util.find_spec("transformer_lens").submodule_search_locations[0] + "/components/abstract_attention.py"
    )
    src = Path(path).read_text()
    if TL_FAST in src:
        return
    assert src.count(TL_SLOW) == 1, "TransformerLens changed: re-check the per-head result code before patching"
    Path(path).write_text(src.replace(TL_SLOW, TL_FAST))
    print("[patch] TransformerLens per-head result -> einsum", flush=True)


def load_hf_on_gpu(device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device).eval()


def ref_device():
    """Two GPUs (Kaggle T4 x2): the HF reference model lives on cuda:1, TransformerLens on cuda:0."""
    import torch

    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda"


def cpu_convert_ok():
    """CircuitKIT's default conversion upcasts every weight to fp32 (13.4 GB) on the default device. That fits system
    RAM on Kaggle (~29 GB) and a Colab L4 (~53 GB) but not the free T4 VM (12 GB). On the GPU it does not fit even
    an L4 next to the HF reference (Colab L4 OOM at 22 GB), so convert on the CPU whenever RAM allows."""
    import os

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    print(f"[stage] system RAM {ram / 2**30:.0f} GiB", flush=True)
    return ram > 24 * 2**30


def load_tl(hf):
    """TransformerLens model converted from GPU-resident HF weights (hf_model=...). The default path builds a second
    CPU copy and was OOM-killed on the T4 VM's 12 GB RAM; this peaks at ~2x6.7 GB on the 15 GB GPU instead."""
    import circuitkit as ck  # import before transformer_lens (CircuitKIT applies its cohere2 port)
    import torch

    if torch.cuda.device_count() > 1 or cpu_convert_ok():  # enough RAM for CircuitKIT's default CPU-side conversion
        del hf
        torch.cuda.empty_cache()
        return ck.load_model(MODEL, dtype="float16", device="cuda:0")
    # HookedTransformer.__init__ allocates its (random-init) parameters with torch's defaults: fp32 on CPU,
    # ~13.4 GB for 3.35B params, which alone exceeds the free T4 VM's 12 GB RAM. Allocate them fp16 on the GPU.
    print("[stage] TL load start", torch.cuda.memory_allocated() // 2**20, "MiB", flush=True)
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda")
    try:
        tl = ck.load_model(MODEL, dtype="float16", hf_model=hf)
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
    del hf
    torch.cuda.empty_cache()
    print("[stage] TL loaded", torch.cuda.memory_allocated() // 2**20, "MiB", flush=True)
    return tl


def parity(prompts):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = [tok(p, add_special_tokens=False, return_tensors="pt").input_ids for p in prompts]
    assert all(i[0, 0].item() == tok.bos_token_id for i in ids), "rendered prompts must start with BOS"
    hf = load_hf_on_gpu(ref_device())
    with torch.no_grad():
        ref = [F.log_softmax(hf(i.to(hf.device)).logits[0, -1].float(), -1).cpu() for i in ids]
    tl = load_tl(hf)
    out = []
    with torch.no_grad():
        for i, r in zip(ids, ref, strict=True):
            lp = F.log_softmax(
                tl(i.to(tl.cfg.device))[0, -1].float(), -1
            ).cpu()  # same ids, BOS included: no re-tokenisation
            kl = float(F.kl_div(lp, r, log_target=True, reduction="sum"))
            out.append(dict(kl=kl, top1_match=bool(lp.argmax() == r.argmax()), n_tokens=i.shape[1]))
    res = dict(
        max_kl=max(o["kl"] for o in out),
        top1_agree=sum(o["top1_match"] for o in out) / len(out),
        gate_pass=max(o["kl"] for o in out) < 1e-4,
        per_prompt=out,
        dtype="float16",
        gpu=torch.cuda.get_device_name(0),
    )
    Path("parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}))
    # TransformerLens hooks form reference cycles, so the 6.7 GB model is not freed on return; in `all` mode the
    # discovery model then loaded on top of it (Kaggle v7 OOM at 14.3 GB). Free both models explicitly.
    import gc

    del tl, hf
    gc.collect()
    torch.cuda.empty_cache()


def register(s, name):
    """Register a CircuitKIT task from one language's spec: harmful prompts paired with benign controls
    (harmful_vs_benign; answer_contrastive would use one prompt for clean and corrupt, giving zero EAP differences)."""
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task

    strip = lambda p: p[len(BOS) :] if p.startswith(BOS) else p  # TransformerLens prepends its own BOS
    ds = SafetyPromptAdapter().adapt(
        [dict(id=f"{name}{i}", prompt=strip(p)) for i, p in enumerate(s["harmful"])],
        name=name,
        refusal_token=s["refusal_token"],
        compliance_token=s["compliance_token"],
        benign_controls=[strip(p) for p in s["benign"]],
        pairing_mode="harmful_vs_benign",
    )
    register_task(NormalizedTaskSpec(ds, name=name))


def discover(spec, n_examples=32, batch_size=1):
    """spec = {lang: {harmful: [rendered], benign: [rendered], refusal_token, compliance_token}}.
    harmful_vs_benign pairing (answer_contrastive uses one prompt for clean and corrupt, so EAP's activation
    differences would be zero). Writes refusal_<lang>.pt + refusal_<lang>_scores.json (SafeTune interop fields)."""
    patch_tl_attn_result()  # before circuitkit imports transformer_lens
    import circuitkit as ck
    import torch

    print("[mem] before discovery load:", torch.cuda.memory_allocated(0) // 2**20, "MiB on cuda:0", flush=True)
    tl = load_tl(None if torch.cuda.device_count() > 1 or cpu_convert_ok() else load_hf_on_gpu())  # no HF copy needed
    print("[mem] model loaded:", torch.cuda.memory_allocated(0) // 2**20, "MiB on cuda:0", flush=True)
    summary = {}
    for lang, s in spec.items():
        name = f"refusal_{lang}"
        register(s, name)
        # batch_size=1: at ~460 tokens, TransformerLens's per-head output (z * W_O) is [b, pos, 16, 128, 2048] fp16,
        # 13.4 GiB at the default b=4 (Kaggle v3 OOM). EAP's buffers live on cfg.device, so layers can't be split
        # across the two GPUs (n_devices=2) without patching CircuitKIT.
        torch.cuda.reset_peak_memory_stats(0)
        circuit = ck.discover(
            tl,
            name,
            algorithm="eap-ig",
            level="node",
            n_examples=n_examples,
            batch_size=batch_size,
            output_path=f"{name}_discovery.pt",
        )  # one path per language
        circuit.save(f"{name}.pt")
        top = list(circuit.top_nodes(20).items())
        peak = torch.cuda.max_memory_allocated(0) / 2**30  # measured, for the paper and the upstream issue
        summary[lang] = dict(
            top20=top, n_nodes=len(circuit.scores or {}), peak_gpu_gib=round(peak, 2), gpu=torch.cuda.get_device_name(0)
        )
        print(lang, f"peak {peak:.1f} GiB; top nodes:", [n for n, _ in top[:8]], flush=True)
        Path("discover_summary.json").write_text(
            json.dumps(summary, indent=1)
        )  # per language: partial runs keep results
    import gc

    del tl
    gc.collect()
    torch.cuda.empty_cache()  # TransformerLens hooks form cycles; free before the next job loads a model


SIZES = (
    0.99,
    0.95,
    0.90,
    0.80,
    0.60,
    0.40,
    0.0,
)  # target_sparsity = share of nodes removed; 0.0 keeps all: F must be 1


def faithfulness(spec, n_examples=None):
    """Held-out faithfulness + cross-language transfer of the discovered circuits (deviations log, 12:55).
    Needs refusal_<lang>_scores.json from discover. CircuitKIT pillar 1 (causal patching):
    F = (y_circuit - y_corrupt) / (y_clean - y_corrupt), out-of-circuit nodes patched from the paired benign run,
    on spec["eval"] prompts (calibration ids never used for discovery). Writes faithfulness.json after every cell."""
    patch_tl_attn_result()
    import circuitkit  # noqa: F401  (applies the cohere2 port before transformer_lens loads)
    import torch
    from circuitkit.api import evaluate_circuit
    from circuitkit.evaluation.pillars.causal_patching import Pillar1_CausalPatching as P1

    raw, p1_run = {}, P1.run  # keep pillar 1's raw metrics: the report has only the clipped [0,1] ratio
    P1.run = staticmethod(lambda *a, **k: raw.update(p1_run(*a, **k)) or dict(raw))
    tl = load_tl(None if torch.cuda.device_count() > 1 or cpu_convert_ok() else load_hf_on_gpu())
    langs = list(spec["eval"])
    n_examples = n_examples or min(len(spec["eval"][l]["harmful"]) for l in langs)
    for lang in langs:
        register(spec["eval"][lang], f"refusal_{lang}_eval")
        node_scores = json.loads(Path(f"refusal_{lang}_scores.json").read_text())["node_scores"]
        torch.save(
            {"node_scores": node_scores}, f"refusal_{lang}_eval_scores.pt"
        )  # the side-car evaluate_circuit loads
    model_cfg = dict(
        name=MODEL, precision=str(tl.cfg.dtype).split(".")[-1], device=torch.device(tl.cfg.device).type
    )  # CircuitKIT accepts "cuda", not "cuda:0"
    rows = []
    for src in langs:
        for tgt in langs:
            for sp in SIZES:
                cfg = dict(
                    model=model_cfg,
                    pruning=dict(target_sparsity=sp, scope="both"),
                    discovery=dict(
                        algorithm="eap-ig",
                        task=f"refusal_{tgt}_eval",
                        level="node",
                        batch_size=1,
                        data_params=dict(num_examples=n_examples, seed=42),
                    ),
                )
                with_baselines = src == tgt and sp == 0.95
                if with_baselines:
                    cfg["eval"] = dict(full_faithfulness_eval=True, pillars=["patching", "baselines"])
                rep = evaluate_circuit(
                    cfg,
                    pruned_artifact_path=f"refusal_{src}.pt",
                    scores_path=f"refusal_{src}_eval_scores.pt",
                    _model=tl,
                )
                rows.append(
                    dict(
                        src=src,
                        tgt=tgt,
                        sparsity=sp,
                        patching=rep.patching_score,
                        patching_raw=rep.metadata.get("patching_raw_ratio"),
                        y_clean=raw.get("clean_score"),
                        y_corrupt=raw.get("corrupt_score"),
                        y_circuit=raw.get("raw_score"),
                        ablation_raw=rep.metadata.get("ablation_raw_ratio"),
                        baselines=rep.baseline_comparison if with_baselines else None,
                    )
                )
                raw.clear()
                print(f"[faith] {src}->{tgt} keep {1 - sp:.0%}: F={rows[-1]['patching_raw']}", flush=True)
                Path("faithfulness.json").write_text(json.dumps(rows, indent=1, default=str))


def run_all(spec):
    """One GPU session: the parity gate, then discovery. Discovery is only trusted if parity.json says gate_pass."""
    patch_tl_attn_result()
    parity(spec["parity"])
    discover(spec["discover"])


def run_full(spec):
    """Parity gate, discovery, then held-out faithfulness and transfer, in one GPU session."""
    run_all(spec)
    faithfulness(spec)


JOBS = {"parity": parity, "discover": discover, "all": run_all, "faithfulness": faithfulness, "full": run_full}

if __name__ == "__main__":
    ensure_deps()
    mode, path = sys.argv[1], sys.argv[2]
    JOBS[mode](json.loads(Path(path).read_text()))
