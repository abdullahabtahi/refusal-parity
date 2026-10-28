#!/bin/zsh
# Score one arm's responses on a Colab T4 with the unchanged audit.py (AuditKIT + Qwen3Guard judges).
# The judges are ungated, so no HF token is uploaded. The VM is always stopped.
# Usage: scripts/colab_score.sh <arm>     e.g. scripts/colab_score.sh B0
set -eu
cd "${0:A:h}/.."
C="$PWD/.venv/bin/colab"; ARM=$1; S=score-$ARM
K=${COLAB_SSH_KEY:?set COLAB_SSH_KEY to the private key whose public half the Colab session trusts}
R() { ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 \
        -o ProxyCommand="'$C' ssh --proxy-mode -s $S -i $K" root@colab "$@"; }
trap '$C stop -s $S </dev/null >/dev/null 2>&1 || true; echo "[driver] session stopped"' EXIT
trap 'exit 130' INT TERM

$C new -s $S --gpu T4 </dev/null
for t in 1 2 3 4 5 6; do R true && break || { echo "[driver] ssh not ready (try $t)"; sleep 15; }; done  # sshd lags READY
R "mkdir -p /content/src /content/runs/$ARM"
bundle=$(mktemp -t colab_src).tgz; tar czf $bundle --exclude __pycache__ --exclude '*.pyc' src/refusal_parity
$C upload -s $S $bundle /content/src.tgz </dev/null >/dev/null && R "cd /content && tar xzf src.tgz"
$C upload -s $S runs/$ARM/responses.jsonl /content/runs/$ARM/responses.jsonl </dev/null
# fds detached: otherwise the job inherits this SSH session's channel and the launch call blocks until it ends
R "cd /content; (pip install -q 'transformers==5.20.2' 'safetune==0.1.10' 'auditkit[transformers]==1.1.3' 'accelerate==1.15.0' > install.log 2>&1 && cd src && JUDGE_BATCH=4 python -u -m refusal_parity.scoring.audit score ../runs/$ARM/responses.jsonl --no-lid > ../score.log 2>&1) </dev/null >/dev/null 2>&1 & echo \$! > /content/job.pid; echo started"
done_=0
for i in $(seq 1 240); do  # up to 6 h: polling starts at launch now, and B0 (3,150 rows, two judges, bf16) took ~3 h
  sleep 90
  st=$(R 'kill -0 $(cat /content/job.pid) 2>/dev/null && echo RUNNING || echo DONE; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; tail -c 300 /content/score.log 2>/dev/null || tail -c 200 /content/install.log; true' 2>&1) || st="PROBE-FAILED"
  (( i % 10 == 1 )) || [[ $st == DONE* ]] && echo "[poll $i] $(date +%H:%M) $(echo $st | tr '\n' ' ' | cut -c1-260)"
  [[ $st == DONE* ]] && { done_=1; break; }
done
(( done_ )) || echo "[driver] TIMEOUT: job still running after the poll window; outputs below may be partial"
R 'tail -c 1500 /content/score.log' || true
mkdir -p runs/$ARM/colab
for f in judged.jsonl auditkit.json; do $C download -s $S /content/runs/$ARM/$f runs/$ARM/colab/$f </dev/null || true; done
ls -la runs/$ARM/colab
