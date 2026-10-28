#!/bin/zsh
# After scripts/colab_score.sh <arm> returns: install the Colab labels, add language ID locally, merge the Gemini
# judge, and log both judges to MLflow. The same steps as B0 (research/DIARY.md §21), in one tested path.
# Usage: scripts/finalize_arm.sh <arm>         e.g. scripts/finalize_arm.sh E0
set -eu
cd "${0:A:h}/../src"
PY=../.venv/bin/python
ARM=$1; D=../runs/$ARM
[[ -s $D/colab/judged.jsonl ]] || { echo "no $D/colab/judged.jsonl: did colab_score.sh finish?"; exit 1; }
n_resp=$(wc -l < $D/responses.jsonl); n_judged=$(wc -l < $D/colab/judged.jsonl)
(( n_resp == n_judged )) || { echo "row mismatch: $n_resp responses vs $n_judged judged"; exit 1; }
cp $D/colab/judged.jsonl $D/colab/auditkit.json $D/
$PY -m refusal_parity.scoring.fill_lang $D/judged.jsonl 2>&1 | grep -v -i warning | tail -1
if [[ -s $D/gemini_judged.jsonl ]]; then
  $PY -m refusal_parity.scoring.merge_gemini $D | tail -1
  judges=(q3g4b gem)
else
  echo "no Gemini labels for $ARM: logging the Qwen3Guard judge only"; judges=(q3g4b)
fi
for j in $judges; do $PY -u -m refusal_parity.track cond $ARM --judge $j 2>&1 | grep -E "^logged|Error"; done
