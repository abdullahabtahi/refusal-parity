"""MLflow tracking for every condition and the judge validation (idempotent: re-logging replaces the run).

    python track.py cond B0 --judge q3g4b          # one condition (reads runs/<cond>/judged.jsonl)
    python track.py validation                     # judge-vs-human validation
    mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

Conventions: experiment "refusal-parity"; one run per (condition, judge), tagged run_key; params = arm config;
metrics = parity summary + per-language values (e.g. SRR_sw); artifacts = judged rows, per-language CSV,
arm config, AuditKIT fingerprint; input dataset = the CuratorKIT eval set (lineage via mlflow.data).
"""

import argparse
import hashlib
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import pandas as pd

import analyze
from common import DATA, RUNS

EXPERIMENT = "refusal-parity"
mlflow.set_tracking_uri(f"sqlite:///{RUNS / 'mlflow.db'}")


def experiment_id():
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp:
        return exp.experiment_id
    return mlflow.create_experiment(EXPERIMENT, artifact_location=(RUNS / "mlartifacts").as_uri())


def fresh_run(run_key, run_name):
    """Delete any earlier run with this key so re-logging is idempotent."""
    for r in mlflow.search_runs([experiment_id()], filter_string=f"tags.run_key = '{run_key}'", output_format="list"):
        mlflow.delete_run(r.info.run_id)
    return mlflow.start_run(experiment_id=experiment_id(), run_name=run_name, tags={"run_key": run_key})


def eval_dataset():
    df = pd.read_json(DATA / "eval.jsonl", lines=True)
    digest = hashlib.sha256((DATA / "eval.jsonl").read_bytes()).hexdigest()[:12]
    return mlflow.data.from_pandas(df, source=str(DATA / "eval.jsonl"), name=f"refusal-parity-eval@{digest}"), digest


def log_condition(cond, judge):
    d = RUNS / cond
    df = analyze.load(cond, judge=judge)
    t, parity = analyze.summary(df)
    cfg = json.loads((d / "arm_config.json").read_text()) if (d / "arm_config.json").exists() else {}
    ak = json.loads((d / "auditkit.json").read_text()) if (d / "auditkit.json").exists() else {}
    ds, digest = eval_dataset()
    with fresh_run(f"{cond}|{judge}", f"{cond} [{judge}]"):
        mlflow.log_input(ds, context="evaluation")
        mlflow.set_tags(
            {
                "condition": cond,
                "judge": judge,
                "model": df["model"].iloc[0] if "model" in df else "",
                "preamble": df["preamble"].iloc[0] if "preamble" in df else "",
                "auditkit_fingerprint": ak.get("fingerprint", "n/a"),
                "eval_sha256_12": digest,
            }
        )
        mlflow.log_params(
            {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in cfg.items() if v is not None}
        )
        mlflow.log_metrics({k: float(v) for k, v in parity.items()})
        for row in t.itertuples():
            mlflow.log_metrics(
                {
                    f"SRR_{row.lang}": row.SRR,
                    f"BRR_{row.lang}": row.BRR,
                    f"HRR_{row.lang}": row.HRR,
                    f"on_lang_{row.lang}": row.on_lang,
                }
            )
        csv = d / f"per_language_{judge}.csv"
        t.to_csv(csv, index=False)
        for f in (csv, d / "arm_config.json", d / "auditkit.json", d / "judged.jsonl"):
            if f.exists():
                mlflow.log_artifact(str(f))
    print(f"logged {cond} [{judge}]  " + "  ".join(f"{k}={v:.3f}" for k, v in parity.items()))


def log_validation():
    v = pd.read_csv(RUNS / "pgp_val" / "judge_validation.csv")
    with fresh_run("judge-validation", "judge validation vs human labels (PolyGuardPrompts)"):
        mlflow.set_tags({"condition": "judge-validation", "human_labels": "ToxicityPrompts/PolyGuardPrompts"})
        for r in v.itertuples():
            key = f"{r.judge}_{r.label}_{r.lang}".replace(" ", "_").replace("(", "").replace(")", "").replace("*", "")
            mlflow.log_metrics({f"kappa__{key}": r.kappa, f"acc__{key}": r.acc})
        mlflow.log_artifact(str(RUNS / "pgp_val" / "judge_validation.csv"))
    print("logged judge validation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cond", "validation"])
    ap.add_argument("cond", nargs="?")
    ap.add_argument("--judge", default="q3g4b", choices=["q3g4b", "gem"])
    a = ap.parse_args()
    log_validation() if a.what == "validation" else log_condition(a.cond, a.judge)
