#!/bin/zsh
# Steering-strength sweep on Earth (calibration split only, as preregistered): S-POOL at several alphas on the
# first 20 calibration ids per (kind, lang) = 400 responses each; alpha 0 is the same-subset baseline.
# Waits for runs/mech_earth/calib_acts.pt and for no other model job on the Mac GPU. Gemini judges each run
# in the background while the next one generates. Pick alpha with src/refusal_parity/steering/choose_alpha.py afterwards.
# Usage: scripts/alpha_sweep.sh [alphas...]      default: 0 0.5 1 2 4
set -u
cd "${0:A:h}/../src"
PY=../.venv/bin/python
if (( $# )); then ALPHAS=("$@"); else ALPHAS=(0 0.5 1 2 4); fi
pat="mech"".py --model"   # assembled so this script's own command line never matches
until [[ -s ../runs/mech_earth/calib_acts.pt ]] && ! pgrep -f "$pat" >/dev/null; do sleep 30; done
echo "=== $(date +%H:%M) sweep start: alphas ${ALPHAS[*]}"
for al in $ALPHAS; do
  tag=a$al; dir=../runs/S-POOL_$tag
  [[ -s $dir/responses.jsonl ]] || $PY -u -m refusal_parity.generation.run_arm --arm S-POOL --model CohereLabs/tiny-aya-earth \
      --acts ../runs/mech_earth/calib_acts.pt --split calib --limit 20 --alpha $al --tag $tag 2>&1 \
      | grep -E "responses ->|Error|Traceback|assert"
  [[ -s $dir/responses.jsonl ]] || { echo "FAILED generate $tag"; continue; }
  ( $PY -u -m refusal_parity.scoring.gemini judge $dir/responses.jsonl $dir/gemini_judged.jsonl --concurrency 12 > $dir/gemini.log 2>&1 ) &
  echo "=== $(date +%H:%M) $tag generated"
done
wait
echo "SWEEP DONE"
