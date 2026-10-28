#!/bin/zsh
# Idempotent: finalize every run whose Qwen3Guard labels have landed (runs/<run>/colab/judged.jsonl), and re-finalize a
# run when its complete Gemini labels arrive after the first pass. Safe to call repeatedly (e.g. from a watcher).
# Usage: scripts/finalize_ready.sh [run ...]      (default: every run with colab/judged.jsonl)
set -u
cd "${0:A:h}/.."
runs=(${@:-${${(f)"$(ls -d runs/*/colab/judged.jsonl(N) | cut -d/ -f2)"}}})
for r in $runs; do
  d=runs/$r
  [[ -s $d/colab/judged.jsonl ]] || continue
  gem_ready=0; [[ -s $d/gemini_judged.jsonl ]] && (( $(wc -l < $d/gemini_judged.jsonl) == $(wc -l < $d/responses.jsonl) )) && gem_ready=1
  has_gem=0; [[ -s $d/judged.jsonl ]] && head -1 $d/judged.jsonl | grep -q '"gem_harmful"' && has_gem=1
  if [[ ! -s $d/judged.jsonl || $d/colab/judged.jsonl -nt $d/judged.jsonl ]] || (( gem_ready && ! has_gem )); then
    (( gem_ready )) || mv $d/gemini_judged.jsonl $d/gemini_judged.partial.jsonl 2>/dev/null  # merge only complete labels
    echo "== finalize $r (gemini: $gem_ready)"; scripts/finalize_arm.sh $r
    [[ -f $d/gemini_judged.partial.jsonl ]] && mv $d/gemini_judged.partial.jsonl $d/gemini_judged.jsonl
  fi
done
