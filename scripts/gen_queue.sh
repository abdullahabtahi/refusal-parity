#!/bin/zsh
# Generation-only queue for the Mac GPU: run arms one at a time; Gemini judges each in the background (cloud).
# Qwen3Guard scoring is done off the Mac (scripts/kaggle_score.py or colab_score.sh), then scripts/finalize_arm.sh.
# (queue.sh also scores on the Mac, which takes ~4.5 h per arm with swap; this skips that.)
# Usage: scripts/gen_queue.sh "NAME|run_arm.py args" ...     NAME = run dir under runs/ (arm, or arm_tag)
set -u
cd "${0:A:h}/../src"
PY=../.venv/bin/python
for spec in "$@"; do
  name=${spec%%|*}; args=${spec#*|}; dir=../runs/$name
  echo "=== $(date +%H:%M) $name :: $args"
  if [[ ! -s $dir/responses.jsonl ]]; then
    eval "$PY -u -m refusal_parity.generation.run_arm $args" 2>&1 | grep -E "responses ->|Error|Traceback|assert"
    [[ -s $dir/responses.jsonl ]] || { echo "FAILED generate $name"; continue; }
  fi
  [[ -s $dir/gemini_judged.jsonl ]] || \
    ( $PY -u -m refusal_parity.scoring.gemini judge $dir/responses.jsonl $dir/gemini_judged.jsonl --concurrency 12 > $dir/gemini.log 2>&1 ) &
  echo "=== $(date +%H:%M) $name generated"
done
wait
echo "GEN QUEUE DONE"
