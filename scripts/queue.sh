#!/bin/zsh
# Run arms one at a time on the Mac GPU: generate (AuditKIT + Tiny Aya) -> local judges (AuditKIT);
# Gemini judging runs in the background in the cloud.
# Usage: scripts/queue.sh "B0|--arm B0 --preamble default" "E0|--arm E0 --model CohereLabs/tiny-aya-earth" ...
# The part before | is the run directory name (and --tag if it differs from --arm).
set -u
cd "${0:A:h}/../src"
PY=../.venv/bin/python
for spec in "$@"; do
  name=${spec%%|*}; args=${spec#*|}
  dir=../runs/$name
  echo "=== $(date +%H:%M) $name :: $args"
  if [[ ! -s $dir/responses.jsonl ]]; then
    eval "$PY -u -m refusal_parity.generation.run_arm $args" 2>&1 | grep -vE "Warning|redirects|Loading weights"
    [[ -s $dir/responses.jsonl ]] || { echo "FAILED generate $name"; continue; }
  fi
  ( $PY -u -m refusal_parity.scoring.gemini judge $dir/responses.jsonl $dir/gemini_judged.jsonl --concurrency 12 > $dir/gemini.log 2>&1 ) &
  gem_pid=$!
  $PY -u -m refusal_parity.scoring.audit score $dir/responses.jsonl 2>&1 | grep -E "AuditKIT run|Error|Traceback" || echo "FAILED score $name"
  $PY -u -m refusal_parity.track cond $name --judge q3g4b 2>&1 | grep -E "^logged|Error"
  # Gemini finishes in the cloud; merge and log it when ready, without blocking the next arm on the GPU.
  ( while kill -0 $gem_pid 2>/dev/null; do sleep 20; done; $PY -m refusal_parity.scoring.merge_gemini $dir && $PY -u -m refusal_parity.track cond $name --judge gem 2>&1 | grep -E "^logged|Error" ) &
  echo "=== $(date +%H:%M) $name done"
done
wait
echo "QUEUE DONE"
