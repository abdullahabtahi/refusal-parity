"""Generate Dark-Forest Swiss Editorial README data charts (fig1_teaser, fig2_eors) from runs/.

Every number, point, and Wilson interval is computed dynamically from runs/ via refusal_parity.analysis.analyze
and runs/controls_report.json (identical to src/refusal_parity/reporting/figures_results.py). Outputs HTML/SVG
sources to docs/assets/src/ and 2x-Retina PNG + provenance JSON to docs/assets/.

Usage:
    PYTHONPATH=src .venv/bin/python docs/assets/src/build_readme_charts.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from refusal_parity.analysis import analyze  # noqa: E402
from refusal_parity.common import RUNS, TIERS  # noqa: E402
from refusal_parity.reporting import figures_diagnostics  # noqa: E402

SRC_DIR = ROOT / "docs" / "assets" / "src"
OUT_DIR = ROOT / "docs" / "assets"

ARM_NAMES = {
    "E0_a100": "no repair (E0)",
    "G-POOL_a100": "G-POOL",
    "G-EN_a100": "G-EN",
    "CAST-EN_a100": "CAST-EN",
    "G-CAL_a100": "G-CAL",
    "CPOOL-S_e05_a100": "CPOOL-S",
    "EORS-S_e05_a100": "EORS-S",
    "RESTA_a100": "ReSta",
    "NEG_a100": "NEG (control)",
}


def find_chrome() -> str:
    env = os.environ.get("CHROME_BIN")
    if env and Path(env).is_file():
        return env
    for candidate in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        if candidate.startswith("/"):
            if Path(candidate).is_file():
                return candidate
        elif found := shutil.which(candidate):
            return found
    msg = "Could not find Chrome/Chromium executable. Set CHROME_BIN."
    raise FileNotFoundError(msg)


def per_lang(run: str, judge: str = "q3g4b"):
    return analyze.summary(analyze.load(run, judge=judge))[0].set_index("lang")


def common_css() -> str:
    return """
@font-face {
  font-family: 'Space Grotesk';
  src: url('../../../paper/fonts/SpaceGrotesk-Regular.ttf') format('truetype');
  font-weight: 400;
}
@font-face {
  font-family: 'Space Grotesk';
  src: url('../../../paper/fonts/SpaceGrotesk-Medium.ttf') format('truetype');
  font-weight: 500;
}
@font-face {
  font-family: 'Space Grotesk';
  src: url('../../../paper/fonts/SpaceGrotesk-Bold.ttf') format('truetype');
  font-weight: 700;
}
@font-face {
  font-family: 'Space Mono';
  src: url('../../../paper/fonts/SpaceMono-Regular.ttf') format('truetype');
  font-weight: 400;
}
@font-face {
  font-family: 'Space Mono';
  src: url('../../../paper/fonts/SpaceMono-Bold.ttf') format('truetype');
  font-weight: 700;
}

:root {
  --bg: #0c1809;
  --surface: #10200c;
  --surface-alt: #0d1b0a;
  --rule: #263d1f;
  --rule-subtle: #1b2e15;
  --ink: #eef3ec;
  --ink-secondary: #9cb095;
  --ink-muted: #687d61;
  --lime: #b6f012;
  --sage: #789a6d;
  --ochre: #d48c6a;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  width: 1420px;
  background: var(--bg);
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
  padding: 20px;
  -webkit-font-smoothing: antialiased;
}
.blueprint {
  border: 1px solid var(--rule);
  background: var(--surface);
}
.masthead {
  display: flex;
  flex-direction: column;
  padding: 16px 24px;
  border-bottom: 1px solid var(--rule);
  background: var(--surface-alt);
  gap: 6px;
}
.masthead-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.masthead-kicker {
  font-family: 'Space Mono', monospace;
  font-size: 10.5px;
  color: var(--lime);
  letter-spacing: 0.06em;
}
.masthead-title {
  font-size: 17.5px;
  font-weight: 500;
  letter-spacing: -0.015em;
  color: var(--ink);
  line-height: 1.34;
}
.masthead-right {
  font-family: 'Space Mono', monospace;
  font-size: 10.5px;
  color: var(--ink-secondary);
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.masthead-right span.muted { color: var(--ink-muted); }
.masthead-right span.sep { color: var(--rule); margin: 0 6px; }
.panel-index {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 20px;
  border-bottom: 1px solid var(--rule-subtle);
  font-family: 'Space Mono', monospace;
  font-size: 10.2px;
  letter-spacing: 0.03em;
  background: var(--surface-alt);
  white-space: nowrap;
}
.panel-num { color: var(--lime); font-weight: 700; }
.panel-sub { color: var(--ink-secondary); }
.plot-canvas {
  background-color: var(--surface);
  background-image: radial-gradient(var(--rule) 0.75px, transparent 0.75px);
  background-size: 14px 14px;
  padding: 14px 20px 16px 20px;
}
.footer-strip {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 20px;
  border-top: 1px solid var(--rule);
  background: var(--bg);
  font-family: 'Space Mono', monospace;
  font-size: 10.5px;
  color: var(--ink-secondary);
}
.legend-group {
  display: flex;
  align-items: center;
  gap: 22px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
"""


def build_fig1_teaser() -> tuple[str, dict]:
    pts = {}
    for run in ARM_NAMES:
        t = per_lang(run)
        pts[run] = (float(100 * t.BRR.max()), float(100 * t.SRR.mean()), str(t.BRR.idxmax()))

    arms_b = [
        ("E0", "E0_a100", "E0_greet_a100"),
        ("CPOOL-S", "CPOOL-S_e10_a100", "CPOOL-S_e10_greet_a100"),
        ("EORS-S", "EORS-S_e10_a100", "EORS-S_e10_greet_a100"),
    ]
    greet = {}
    for name, plain, greeted in arms_b:
        x0, x1 = float(100 * per_lang(plain).BRR.mean()), float(100 * per_lang(greeted).BRR.mean())
        greet[name] = (x0, x1)

    lrl_delta = float(json.loads((RUNS / "controls_report.json").read_text())["end_to_end"]["0.05"]["lrl_hrr"]["delta"])
    c_, e_ = pts["CPOOL-S_e05_a100"], pts["EORS-S_e05_a100"]
    e0_rows = analyze.load("E0_a100").query("lang == 'en'")
    n_h, n_b = int((e0_rows.kind == "harmful").sum()), int((e0_rows.kind == "benign").sum())
    gated = [v for k, v in pts.items() if k not in ("E0_a100", "RESTA_a100", "NEG_a100")]
    lo_s, hi_s = min(v[1] for v in gated), max(v[1] for v in gated)
    lo_b, hi_b = min(v[0] for v in gated), max(v[0] for v in gated)

    title_line1 = (
        f"At the same budget, per-language thresholds cut worst-language over-refusal {c_[0]:.1f}% → {e_[0]:.1f}%;"
    )
    title_line2 = f"with a benign greeting prefix, mean benign refusal reaches {greet['EORS-S'][1]:.0f}% under either rule (ε=.10)"

    caption = (
        "(a) Every repair arm on Tiny Aya Earth (A100, α=1), judged by Qwen3Guard-4B on the test split "
        f"({n_h} harmful MultiJail and {n_b} benign translated-XSTest prompts per language, 10 languages): mean safe-response rate against the "
        "worst language's benign refusal rate. Arrows: from the unrepaired model (E0) to one pooled split-conformal "
        "threshold (CPOOL-S, ε=.05), and from it to per-language split-conformal thresholds at the same budget "
        "(EORS-S); the two share steering vector, condition direction and calibration data. Paired bootstrap "
        "intervals are in Tables 1 and 2. (b) End-to-end greeting test (pre-registered secondary; ε=.10): mean "
        'benign refusal without and with the English prefix "Good morning." in every language.'
    )
    alt = (
        f"Left: scatter of repair arms near {lo_s:.0f}-{hi_s:.0f}% safety, spread from about {lo_b:.0f}% to "
        f"{hi_b:.0f}% worst-language benign refusal; arrows run from the unrepaired model to CPOOL-S and on to "
        "EORS-S, which has the lowest worst-language refusal. Right: three dumbbells; the greeting raises mean "
        f"benign refusal from about {greet['E0'][0]:.0f}% to {greet['E0'][1]:.0f}% for the unrepaired model and "
        f"from about {greet['EORS-S'][0]:.0f}% to {greet['EORS-S'][1]:.0f}% for both threshold rules."
    )
    meta = dict(
        caption=caption,
        alt_text=alt,
        provenance=dict(
            points=pts,
            greeting=greet,
            judge="q3g4b",
            source="runs/*_a100/judged.jsonl (incl. *_greet_a100)",
            script="docs/assets/src/build_readme_charts.py:build_fig1_teaser",
        ),
    )

    # Coordinate mappers for Panel A (SVG viewBox 760 x 320)
    # Plot box: x in [64, 730], y in [18, 266]
    # Data domain: xlim = (10, 55), ylim = (88, 101.6)
    def ax_x(val: float) -> float:
        return 64.0 + (val - 10.0) / (55.0 - 10.0) * (730.0 - 64.0)

    def ax_y(val: float) -> float:
        return 266.0 - (val - 88.0) / (101.6 - 88.0) * (266.0 - 18.0)

    svg_a_parts = []
    # Horizontal grid lines (88 to 100 step 2)
    for y_tick in range(88, 101, 2):
        py = ax_y(y_tick)
        col = "#263d1f" if y_tick == 88 else "#1b2e15"
        svg_a_parts.append(f'<line x1="64" y1="{py:.1f}" x2="730" y2="{py:.1f}" stroke="{col}" stroke-width="1"/>')
        svg_a_parts.append(
            f'<text x="54" y="{py + 3.5:.1f}" text-anchor="end" font-family="Space Mono" '
            f'font-size="10.5" fill="#9cb095">{y_tick}</text>'
        )

    # Vertical grid lines & ticks (10 to 50 step 10)
    for x_tick in range(10, 51, 10):
        px = ax_x(x_tick)
        col = "#263d1f" if x_tick == 10 else "#1b2e15"
        svg_a_parts.append(
            f'<line x1="{px:.1f}" y1="18" x2="{px:.1f}" y2="266" stroke="{col}" '
            f'stroke-width="1" stroke-dasharray="{"none" if x_tick == 10 else "2,3"}"/>'
        )
        svg_a_parts.append(
            f'<text x="{px:.1f}" y="285" text-anchor="middle" font-family="Space Mono" '
            f'font-size="10.5" fill="#9cb095">{x_tick}%</text>'
        )

    # Axis titles
    svg_a_parts.append(
        '<text x="397" y="310" text-anchor="middle" font-family="Space Grotesk" '
        'font-size="12" font-weight="500" fill="#eef3ec">'
        "Worst-language false refusal of benign prompts (%)</text>"
    )
    svg_a_parts.append(
        '<text x="16" y="142" text-anchor="middle" transform="rotate(-90 16 142)" '
        'font-family="Space Grotesk" font-size="12" font-weight="500" fill="#eef3ec">'
        "Mean safe-response rate (%)</text>"
    )

    # Trajectory arrows: E0 -> CPOOL-S -> EORS-S
    for src_k, dst_k, col_arr in (
        ("E0_a100", "CPOOL-S_e05_a100", "#9cb095"),
        ("CPOOL-S_e05_a100", "EORS-S_e05_a100", "#b6f012"),
    ):
        x1, y1 = ax_x(pts[src_k][0]), ax_y(pts[src_k][1])
        x2, y2 = ax_x(pts[dst_k][0]), ax_y(pts[dst_k][1])
        dx, dy = x2 - x1, y2 - y1
        dist = float(np.hypot(dx, dy))
        ux, uy = dx / dist, dy / dist
        sx, sy = x1 + ux * 11, y1 + uy * 11
        ex, ey = x2 - ux * 12, y2 - uy * 12
        # Arrowhead
        px1, py1 = ex - ux * 7 - uy * 4, ey - uy * 7 + ux * 4
        px2, py2 = ex - ux * 7 + uy * 4, ey - uy * 7 - ux * 4
        svg_a_parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{col_arr}" stroke-width="1.6"/>'
        )
        svg_a_parts.append(
            f'<polygon points="{ex:.1f},{ey:.1f} {px1:.1f},{py1:.1f} {px2:.1f},{py2:.1f}" fill="{col_arr}"/>'
        )

    # Label offsets (dx, dy, anchor) carefully tuned for 760x320 canvas
    label_pos = {
        "E0_a100": (12, 16, "start", "#eef3ec", "500"),
        "CPOOL-S_e05_a100": (12, 15, "start", "#eef3ec", "700"),
        "EORS-S_e05_a100": (-12, 15, "end", "#b6f012", "700"),
        "G-CAL_a100": (-10, -10, "end", "#9cb095", "400"),
        "CAST-EN_a100": (0, -12, "middle", "#9cb095", "400"),
        "G-POOL_a100": (11, -3, "start", "#9cb095", "400"),
        "G-EN_a100": (11, 12, "start", "#9cb095", "400"),
        "RESTA_a100": (12, 4, "start", "#d48c6a", "700"),
        "NEG_a100": (12, 4, "start", "#687d61", "400"),
    }

    for run, (xv, yv, _) in pts.items():
        px, py = ax_x(xv), ax_y(yv)
        if run == "E0_a100":
            svg_a_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="#10200c" stroke="#eef3ec" stroke-width="2.2"/>'
            )
        elif run == "CPOOL-S_e05_a100":
            svg_a_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="#10200c" stroke="#eef3ec" stroke-width="2.2"/>'
            )
            svg_a_parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="#eef3ec"/>')
        elif run == "EORS-S_e05_a100":
            svg_a_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="#b6f012" stroke="#0c1809" stroke-width="1.5"/>'
            )
        elif run == "RESTA_a100":
            svg_a_parts.append(
                f'<rect x="{px - 6:.1f}" y="{py - 6:.1f}" width="12" height="12" '
                'fill="#d48c6a" stroke="#0c1809" stroke-width="1.2"/>'
            )
        elif run == "NEG_a100":
            svg_a_parts.append(
                f'<rect x="{px - 5.5:.1f}" y="{py - 5.5:.1f}" width="11" height="11" '
                'fill="#687d61" stroke="#0c1809" stroke-width="1.2"/>'
            )
        else:
            svg_a_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="#789a6d" stroke="#0c1809" stroke-width="1.2"/>'
            )

        dx, dy, anchor, lcol, fw = label_pos[run]
        svg_a_parts.append(
            f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}" '
            f'font-family="Space Mono" font-size="10.5" font-weight="{fw}" fill="{lcol}">{ARM_NAMES[run]}</text>'
        )

    # Callout box for LRL harmful-prompt refusal trade-off
    bx, by = ax_x(39.0) - 126, ax_y(92.0) - 26
    lrl_str = f"{100 * lrl_delta:+.1f}".replace("-", "−")
    svg_a_parts.append(
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="252" height="50" fill="#0d1b0a" stroke="#d48c6a" stroke-width="1"/>'
    )
    svg_a_parts.append(
        f'<text x="{bx + 126:.1f}" y="{by + 19:.1f}" text-anchor="middle" '
        'font-family="Space Mono" font-size="10" font-weight="700" fill="#d48c6a">'
        f"EORS-S vs CPOOL-S TRADE-OFF: {lrl_str} pts</text>"
    )
    svg_a_parts.append(
        f'<text x="{bx + 126:.1f}" y="{by + 36:.1f}" text-anchor="middle" '
        'font-family="Space Grotesk" font-size="10.5" fill="#9cb095">'
        "Low-resource harmful-prompt refusal (ε=.05)</text>"
    )

    # Coordinate mappers for Panel B (SVG viewBox 530 x 320)
    # Plot box: x in [82, 502], y in [18, 266]
    # Data domain: xlim = (0, 85)
    def bx_x(val: float) -> float:
        return 82.0 + (val / 85.0) * (502.0 - 82.0)

    row_ys = [68.0, 148.0, 228.0]
    svg_b_parts = []

    for x_tick in range(0, 81, 20):
        px = bx_x(x_tick)
        col = "#263d1f" if x_tick == 0 else "#1b2e15"
        svg_b_parts.append(
            f'<line x1="{px:.1f}" y1="18" x2="{px:.1f}" y2="266" stroke="{col}" '
            f'stroke-width="1" stroke-dasharray="{"none" if x_tick == 0 else "2,3"}"/>'
        )
        svg_b_parts.append(
            f'<text x="{px:.1f}" y="285" text-anchor="middle" font-family="Space Mono" '
            f'font-size="10.5" fill="#9cb095">{x_tick}%</text>'
        )

    svg_b_parts.append('<line x1="82" y1="266" x2="502" y2="266" stroke="#263d1f" stroke-width="1"/>')
    svg_b_parts.append(
        '<text x="292" y="310" text-anchor="middle" font-family="Space Grotesk" '
        'font-size="12" font-weight="500" fill="#eef3ec">'
        'Mean benign refusal (%) — original vs. with "Good morning."</text>'
    )

    for i, (name, _, _) in enumerate(arms_b):
        ry = row_ys[i]
        x0, x1 = greet[name]
        px0, px1 = bx_x(x0), bx_x(x1)
        label_col = "#b6f012" if name == "EORS-S" else "#eef3ec"
        svg_b_parts.append(
            f'<text x="70" y="{ry + 4:.1f}" text-anchor="end" font-family="Space Mono" '
            f'font-size="11" font-weight="700" fill="{label_col}">{name}</text>'
        )
        # Connecting bar
        svg_b_parts.append(
            f'<line x1="{px0:.1f}" y1="{ry:.1f}" x2="{px1:.1f}" y2="{ry:.1f}" stroke="#263d1f" stroke-width="5"/>'
        )
        # Original dot (sage/lime)
        orig_fill = "#b6f012" if name == "EORS-S" else "#789a6d"
        svg_b_parts.append(
            f'<circle cx="{px0:.1f}" cy="{ry:.1f}" r="6.5" fill="{orig_fill}" stroke="#0c1809" stroke-width="1.5"/>'
        )
        # Greeted dot (ochre/terracotta)
        svg_b_parts.append(
            f'<circle cx="{px1:.1f}" cy="{ry:.1f}" r="6.5" fill="#d48c6a" stroke="#0c1809" stroke-width="1.5"/>'
        )
        if x1 - x0 < 12:
            svg_b_parts.append(
                f'<text x="{px1 + 12:.1f}" y="{ry + 4:.1f}" font-family="Space Mono" '
                f'font-size="10.5" fill="#eef3ec">{x0:.0f}% → {x1:.0f}%</text>'
            )
        else:
            svg_b_parts.append(
                f'<text x="{px0:.1f}" y="{ry - 12:.1f}" text-anchor="middle" font-family="Space Mono" '
                f'font-size="10.5" fill="#9cb095">{x0:.0f}%</text>'
            )
            svg_b_parts.append(
                f'<text x="{px1 + 12:.1f}" y="{ry + 4:.1f}" font-family="Space Mono" '
                f'font-size="10.5" font-weight="700" fill="#d48c6a">{x1:.0f}%</text>'
            )

    # Inline legend inside Panel B top-right
    svg_b_parts.append(
        '<rect x="334" y="24" width="160" height="48" fill="#0d1b0a" stroke="#263d1f" stroke-width="1"/>'
    )
    svg_b_parts.append('<circle cx="350" cy="39" r="5" fill="#789a6d"/>')
    svg_b_parts.append(
        '<text x="364" y="43" font-family="Space Mono" font-size="10" fill="#eef3ec">Original prompt</text>'
    )
    svg_b_parts.append('<circle cx="350" cy="58" r="5" fill="#d48c6a"/>')
    svg_b_parts.append(
        '<text x="364" y="62" font-family="Space Mono" font-size="10" fill="#d48c6a">+ "Good morning."</text>'
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{common_css()}
.two-col {{
  display: grid;
  grid-template-columns: 1.42fr 1fr;
}}
.panel {{
  border-right: 1px solid var(--rule);
  display: flex;
  flex-direction: column;
}}
.panel:last-child {{ border-right: none; }}
</style>
</head>
<body>
<div class="blueprint">
  <div class="masthead">
    <div class="masthead-top">
      <div class="masthead-kicker">FIGURE 01 // SAFETY–OVER-REFUSAL FRONTIER &amp; GREETING STRESS TEST</div>
      <div class="masthead-right">
        <span class="muted">MODEL:</span> TINY AYA EARTH<span class="sep">/</span>
        <span class="muted">JUDGE:</span> QWEN3GUARD-4B<span class="sep">/</span>
        <span class="muted">SPLIT:</span> TEST ({n_h} MULTIJAIL · {n_b} XSTEST / LANG)
      </div>
    </div>
    <div class="masthead-title">{title_line1}<br>{title_line2}</div>
  </div>

  <div class="two-col">
    <div class="panel">
      <div class="panel-index">
        <span class="panel-num">01A // GATED REPAIRS (ε = 0.05)</span>
        <span class="panel-sub">SIMILAR SAFETY · DIFFERENT WORST-LANGUAGE OVER-REFUSAL</span>
      </div>
      <div class="plot-canvas">
        <svg viewBox="0 0 760 320" width="100%" height="320">
          {"".join(svg_a_parts)}
        </svg>
      </div>
    </div>

    <div class="panel">
      <div class="panel-index">
        <span class="panel-num">01B // GREETING TEST (ε = 0.10)</span>
        <span class="panel-sub">GREETING BREAKS BOTH THRESHOLD RULES</span>
      </div>
      <div class="plot-canvas">
        <svg viewBox="0 0 530 320" width="100%" height="320">
          {"".join(svg_b_parts)}
        </svg>
      </div>
    </div>
  </div>

  <div class="footer-strip">
    <div class="legend-group">
      <span class="legend-item">
        <svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#10200c" stroke="#eef3ec" stroke-width="2"/></svg>
        <span>E0 (Unrepaired)</span>
      </span>
      <span class="legend-item">
        <svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#10200c" stroke="#eef3ec" stroke-width="2"/><circle cx="7" cy="7" r="2" fill="#eef3ec"/></svg>
        <span>CPOOL-S (One pooled threshold)</span>
      </span>
      <span class="legend-item">
        <svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="#b6f012"/></svg>
        <span style="color:#b6f012; font-weight:700;">EORS-S (Per-language thresholds)</span>
      </span>
      <span class="legend-item">
        <svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#789a6d"/></svg>
        <span>Max-F1 Gated Arms</span>
      </span>
      <span class="legend-item">
        <svg width="14" height="14"><rect x="2" y="2" width="10" height="10" fill="#d48c6a"/></svg>
        <span>ReSta (Weight-space)</span>
      </span>
    </div>
    <div style="color:var(--ink-muted);">SOURCE: runs/*_a100/judged.jsonl · 10 LANGUAGES</div>
  </div>
</div>
</body>
</html>
"""
    return html, meta


def build_fig2_eors() -> tuple[str, dict]:
    runs = [
        ("E0_a100", "no repair (E0)"),
        ("CPOOL-S_e05_a100", "one pooled threshold (CPOOL-S)"),
        ("EORS-S_e05_a100", "per-language thresholds (EORS-S)"),
    ]
    tabs = {r: per_lang(r) for r, _ in runs}
    g, e = tabs["CPOOL-S_e05_a100"], tabs["EORS-S_e05_a100"]
    dh = (e.HRR - g.HRR).sort_values()
    losers = [figures_diagnostics.NAMES[l] for l in dh.index if dh[l] <= -0.05]

    title_line1 = (
        f"At a matched false-refusal budget (ε=.05), per-language thresholds narrow over-refusal "
        f"(worst {100 * g.BRR.max():.1f}% → {100 * e.BRR.max():.1f}%)"
    )
    title_line2 = f"and lower harmful-prompt refusal where benign prompts score high ({', '.join(losers)})"

    caption = (
        "Tiny Aya Earth on the A100 (α=1). CPOOL-S and EORS-S share steering vector, condition direction, calibration "
        "fold and false-refusal budget (ε=.05); they differ only in whether the conformal threshold is pooled over "
        "languages or set per language. E0: unrepaired. Qwen3Guard-4B labels; 95% Wilson intervals; languages "
        "grouped high / medium / low resource. Paired changes with bootstrap intervals are in Table 2."
    )
    alt = (
        "Two dot plots by language. Benign refusal: EORS-S is lower than CPOOL-S in Thai, Javanese and Bengali and "
        "slightly higher in high-resource languages. Harmful refusal: EORS-S is lower in Javanese, Bengali and Thai "
        "and higher in Italian and English."
    )
    meta = dict(
        caption=caption,
        alt_text=alt,
        provenance=dict(
            source="runs/{E0,CPOOL-S_e05,EORS-S_e05}_a100/judged.jsonl",
            script="docs/assets/src/build_readme_charts.py:build_fig2_eors",
        ),
    )

    def render_dot_panel(col_name: str, xlim: tuple[float, float], show_labels: bool, show_tiers: bool) -> str:
        # SVG viewBox 650 x 350
        x0_px = 118.0 if show_labels else 28.0
        x1_px = 592.0 if show_tiers else 622.0
        y0_px, y1_px = 18.0, 302.0
        lo, hi = xlim

        def px(val: float) -> float:
            return x0_px + (val - lo) / (hi - lo) * (x1_px - x0_px)

        def py(idx: float) -> float:
            return y0_px + (idx + 0.5) / 10.0 * (y1_px - y0_px)

        parts = []
        # Vertical grid lines & ticks (step 10)
        for tick in range(int(lo), int(hi) + 1, 10):
            tx = px(tick)
            col = "#263d1f" if tick in (int(lo), int(hi)) else "#1b2e15"
            parts.append(
                f'<line x1="{tx:.1f}" y1="{y0_px}" x2="{tx:.1f}" y2="{y1_px}" stroke="{col}" '
                f'stroke-width="1" stroke-dasharray="{"none" if tick == int(lo) else "2,3"}"/>'
            )
            parts.append(
                f'<text x="{tx:.1f}" y="322" text-anchor="middle" font-family="Space Mono" '
                f'font-size="10.5" fill="#9cb095">{tick}%</text>'
            )

        parts.append(f'<line x1="{x0_px}" y1="{y1_px}" x2="{x1_px}" y2="{y1_px}" stroke="#263d1f" stroke-width="1"/>')

        # Tier dividers after row 3 (index 3.5) and row 6 (index 6.5)
        for div_idx in (3.5, 6.5):
            dy = py(div_idx)
            parts.append(
                f'<line x1="{x0_px}" y1="{dy:.1f}" x2="{x1_px}" y2="{dy:.1f}" stroke="#263d1f" stroke-width="1.2"/>'
            )

        # Language rows
        for idx, lang in enumerate(figures_diagnostics.ORDER):
            ry = py(idx)
            if show_labels:
                tier = TIERS[lang]
                t_col = "#b6f012" if tier == "LRL" else "#9cb095"
                parts.append(
                    f'<text x="{x0_px - 12:.1f}" y="{ry + 4:.1f}" text-anchor="end" '
                    f'font-family="Space Grotesk" font-size="12" font-weight="500" fill="#eef3ec">'
                    f"{figures_diagnostics.NAMES[lang]}</text>"
                )
                parts.append(
                    f'<text x="6" y="{ry + 3.5:.1f}" font-family="Space Mono" font-size="9.5" fill="{t_col}">'
                    f"{lang.upper()}</text>"
                )

            vals = [float(100 * tabs[r].loc[lang, col_name]) for r, _ in runs]
            parts.append(
                f'<line x1="{px(min(vals)):.1f}" y1="{ry:.1f}" x2="{px(max(vals)):.1f}" y2="{ry:.1f}" '
                'stroke="#1b2e15" stroke-width="4"/>'
            )

            # Plot each of the 3 arms with slight vertical offset (-5px, 0, +5px) and 95% Wilson CI
            offsets = (-5.2, 0.0, 5.2)
            for (r, _), off in zip(runs, offsets, strict=True):
                row = tabs[r].loc[lang]
                v = float(100 * row[col_name])
                v_lo = float(100 * row[f"{col_name}_lo"])
                v_hi = float(100 * row[f"{col_name}_hi"])
                cy = ry + off
                vx, vx_lo, vx_hi = px(v), px(v_lo), px(v_hi)

                if r == "E0_a100":
                    parts.append(
                        f'<line x1="{vx_lo:.1f}" y1="{cy:.1f}" x2="{vx_hi:.1f}" y2="{cy:.1f}" '
                        'stroke="#687d61" stroke-width="1.2"/>'
                    )
                    parts.append(
                        f'<rect x="{vx - 4:.1f}" y="{cy - 4:.1f}" width="8" height="8" '
                        'fill="#687d61" stroke="#0c1809" stroke-width="0.8"/>'
                    )
                elif r == "CPOOL-S_e05_a100":
                    parts.append(
                        f'<line x1="{vx_lo:.1f}" y1="{cy:.1f}" x2="{vx_hi:.1f}" y2="{cy:.1f}" '
                        'stroke="#9cb095" stroke-width="1.3"/>'
                    )
                    # Open diamond marker
                    pts_d = f"{vx:.1f},{cy - 5.5:.1f} {vx + 5.5:.1f},{cy:.1f} {vx:.1f},{cy + 5.5:.1f} {vx - 5.5:.1f},{cy:.1f}"
                    parts.append(f'<polygon points="{pts_d}" fill="#10200c" stroke="#eef3ec" stroke-width="1.8"/>')
                else:  # EORS-S_e05_a100
                    parts.append(
                        f'<line x1="{vx_lo:.1f}" y1="{cy:.1f}" x2="{vx_hi:.1f}" y2="{cy:.1f}" '
                        'stroke="#b6f012" stroke-width="1.5"/>'
                    )
                    parts.append(
                        f'<circle cx="{vx:.1f}" cy="{cy:.1f}" r="4.8" fill="#b6f012" '
                        'stroke="#0c1809" stroke-width="1.0"/>'
                    )

        if show_tiers:
            for mid_idx, tier_label, t_col in (
                (1.5, "HIGH TIER", "#9cb095"),
                (5.0, "MEDIUM TIER", "#9cb095"),
                (8.0, "LOW TIER", "#b6f012"),
            ):
                ty = py(mid_idx)
                parts.append(
                    f'<text x="618" y="{ty:.1f}" text-anchor="middle" '
                    f'transform="rotate(90 618 {ty:.1f})" font-family="Space Mono" '
                    f'font-size="9.5" font-weight="700" letter-spacing="0.06em" fill="{t_col}">{tier_label}</text>'
                )

        axis_label = (
            "Benign prompts refused (%) — lower is less over-refusal"
            if col_name == "BRR"
            else "Harmful prompts refused (%) — higher is stronger protection"
        )
        parts.append(
            f'<text x="{(x0_px + x1_px) / 2:.1f}" y="344" text-anchor="middle" '
            'font-family="Space Grotesk" font-size="12" font-weight="500" fill="#eef3ec">'
            f"{axis_label}</text>"
        )
        return "".join(parts)

    svg_brr = render_dot_panel("BRR", (0, 50), show_labels=True, show_tiers=False)
    svg_hrr = render_dot_panel("HRR", (40, 100), show_labels=False, show_tiers=True)

    tiers_line = " · ".join(  # from the study's tier map, never typed
        f"{name} ({', '.join(lang.upper() for lang in TIERS if TIERS[lang] == tier)})"
        for tier, name in (("HRL", "HIGH"), ("MRL", "MEDIUM"), ("LRL", "LOW"))
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{common_css()}
.two-col {{
  display: grid;
  grid-template-columns: 1.08fr 1fr;
}}
.panel {{
  border-right: 1px solid var(--rule);
  display: flex;
  flex-direction: column;
}}
.panel:last-child {{ border-right: none; }}
</style>
</head>
<body>
<div class="blueprint">
  <div class="masthead">
    <div class="masthead-top">
      <div class="masthead-kicker">FIGURE 02 // PER-LANGUAGE REFUSAL PARITY BY RESOURCE TIER (ε = 0.05)</div>
      <div class="masthead-right">
        <span class="muted">INTERVALS:</span> 95% WILSON<span class="sep">/</span>
        <span class="muted">CALIBRATION:</span> SPLIT-CONFORMAL (ε=0.05)
      </div>
    </div>
    <div class="masthead-title">{title_line1}<br>{title_line2}</div>
  </div>

  <div class="two-col">
    <div class="panel">
      <div class="panel-index">
        <span class="panel-num">02A // BENIGN PROMPTS REFUSED (%)</span>
        <span class="panel-sub">FALSE-REFUSAL PARITY ACROSS LANGUAGES</span>
      </div>
      <div class="plot-canvas">
        <svg viewBox="0 0 650 352" width="100%" height="352">
          {svg_brr}
        </svg>
      </div>
    </div>

    <div class="panel">
      <div class="panel-index">
        <span class="panel-num">02B // HARMFUL PROMPTS REFUSED (%)</span>
        <span class="panel-sub">PROTECTION COST IN HIGH-SCORING LOW-RESOURCE TIERS</span>
      </div>
      <div class="plot-canvas">
        <svg viewBox="0 0 650 352" width="100%" height="352">
          {svg_hrr}
        </svg>
      </div>
    </div>
  </div>

  <div class="footer-strip">
    <div class="legend-group">
      <span class="legend-item">
        <svg width="14" height="14"><rect x="3" y="3" width="8" height="8" fill="#687d61"/></svg>
        <span>No repair (E0)</span>
      </span>
      <span class="legend-item">
        <svg width="16" height="16"><polygon points="8,2 14,8 8,14 2,8" fill="#10200c" stroke="#eef3ec" stroke-width="1.8"/></svg>
        <span>One pooled threshold (CPOOL-S)</span>
      </span>
      <span class="legend-item">
        <svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="#b6f012"/></svg>
        <span style="color:#b6f012; font-weight:700;">Per-language thresholds (EORS-S)</span>
      </span>
    </div>
    <div style="color:var(--ink-muted);">TIERS: {tiers_line}</div>
  </div>
</div>
</body>
</html>
"""
    return html, meta


def render_html_to_png(html_path: Path, out_png: Path) -> None:
    chrome = find_chrome()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_png = Path(tmp.name)

    try:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--force-device-scale-factor=2",
                "--window-size=1420,700",
                f"--screenshot={tmp_png}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        im = Image.open(tmp_png).convert("RGB")
        arr = np.array(im)
        bg_rgb = arr[-1, -1]
        diff = np.max(np.abs(arr.astype(int) - bg_rgb.astype(int)), axis=2)
        ys, _ = np.where(diff > 2)
        y_max = min(im.height, int(ys.max()) + 40)
        im_cropped = im.crop((0, 0, im.width, y_max))

        im_q = im_cropped.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        im_q.save(out_png, optimize=True)
        kb = out_png.stat().st_size / 1024
        print(f"Rendered {out_png.name}: {im_cropped.size} -> {kb:.1f} KB")
    finally:
        if tmp_png.exists():
            tmp_png.unlink()


def main() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, builder in (("fig1_teaser", build_fig1_teaser), ("fig2_eors", build_fig2_eors)):
        html, meta = builder()
        html_path = SRC_DIR / f"{name}.html"
        png_path = OUT_DIR / f"{name}.png"
        json_path = OUT_DIR / f"{name}.json"

        html_path.write_text(html)
        json_path.write_text(json.dumps(meta, indent=1) + "\n")
        render_html_to_png(html_path, png_path)


if __name__ == "__main__":
    main()
