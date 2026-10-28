"""The estimators every headline number rests on (CPU, synthetic data, seconds)."""

import numpy as np
import pandas as pd

from refusal_parity.analysis import analyze, arms_report, controls_report
from refusal_parity.common import LANGS
from refusal_parity.steering import arms


def test_conformal_threshold_meets_its_budget_on_average():
    """Split conformal: P(new benign score > tau) <= eps, averaged over calibration draws (n = 60, eps = .05)."""
    rng = np.random.default_rng(0)
    eps, n = 0.05, 60
    fpr = [(rng.normal(size=20_000) > arms.conformal_threshold(rng.normal(size=n), eps)).mean() for _ in range(400)]
    assert np.mean(fpr) <= eps + 0.005  # sampling slack on 400 draws


def test_predictive_band_contains_most_realised_test_fprs():
    """The P3 band is the 97.5th percentile of the Beta-binomial test FPR: about 2.5% of draws exceed it."""
    rng = np.random.default_rng(1)
    eps, n_cal, n_test = 0.05, 60, 106
    upper = controls_report.predictive_upper(n_cal, eps, n_test)
    above = [
        (rng.normal(size=n_test) > arms.conformal_threshold(rng.normal(size=n_cal), eps)).mean() > upper
        for _ in range(4000)
    ]
    assert np.mean(above) < 0.04


def test_boot_p_is_two_sided_and_floored():
    assert arms_report.boot_p(np.full(1000, 0.3)) == 1 / 1000  # all draws on one side: floor at 1/N
    assert 0.9 < arms_report.boot_p(np.linspace(-1, 1, 1001)) <= 1.0  # centred on zero: p ~ 1


def _paired_runs(n_ids=60, shift=0.0, seed=0):
    """Two parallel runs over every language: b refuses benign prompts `shift` more often than a."""
    rng = np.random.default_rng(seed)
    rows_a, rows_b = [], []
    for lang in LANGS:
        for i in range(n_ids):
            for kind in ("harmful", "benign"):
                u = rng.random()
                base = dict(id=f"x{i}", lang=lang, kind=kind, unsafe=False)
                rows_a.append(dict(base, refuse=u < 0.2))
                rows_b.append(dict(base, refuse=u < 0.2 + (shift if kind == "benign" else 0.0)))
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def test_compare_reports_plug_in_estimates(monkeypatch):
    """Point estimates are the statistic on the observed data, not the bootstrap mean (review finding M5)."""
    monkeypatch.setattr(analyze, "N_BOOT", 300)
    a, b = _paired_runs(shift=0.1)
    _, boot = analyze.compare(a, b)
    brr = lambda x: x[x.kind == "benign"].groupby("lang").refuse.mean()
    assert np.isclose(boot["d_max_BRR"][0], brr(b).max() - brr(a).max())
    assert np.isclose(boot["d_mean_BRR"][0], brr(b).mean() - brr(a).mean())
    assert boot["d_mean_BRR"][1] <= boot["d_mean_BRR"][0] <= boot["d_mean_BRR"][2]


def test_compare_null_interval_covers_zero(monkeypatch):
    monkeypatch.setattr(analyze, "N_BOOT", 300)
    a, b = _paired_runs(shift=0.0)
    _, boot = analyze.compare(a, b)
    assert boot["d_mean_BRR"][1] <= 0 <= boot["d_mean_BRR"][2]
