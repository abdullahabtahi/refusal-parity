#!/bin/zsh
# Run a colab_circuit.py job detached on a Colab VM (nohup over SSH) and poll its log + RAM + GPU memory,
# instead of `colab exec`, whose 10 s read timeout drops long jobs. Token by `colab upload` only; VM always stopped.
# Usage: [GPU=L4] [POLLS=120] [OUT=runs/circuitkit/<dir>] scripts/colab_ssh_job.sh <mode> <spec.json> <output-to-fetch...>
#   GPU: T4 (free tier) | L4 | A100 (Colab Pro); POLLS x 2 min is the poll window; outputs land in OUT.
#   UPLOAD="a b": extra input files, copied to /content/<basename> (e.g. discovered circuits for faithfulness).
set -eu
cd "${0:A:h}/.."
C="$PWD/.venv/bin/colab"; S=circuitkit-job
K=${COLAB_SSH_KEY:?set COLAB_SSH_KEY to the private key whose public half the Colab session trusts}
MODE=$1; SPEC=$2; shift 2
GPU=${GPU:-T4}; POLLS=${POLLS:-45}; OUT=${OUT:-runs/circuitkit}
R() { ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 \
        -o ProxyCommand="'$C' ssh --proxy-mode -s $S -i $K" root@colab "$@"; }
trap '$C stop -s $S </dev/null >/dev/null 2>&1 || true; echo "[driver] session stopped"' EXIT
trap 'exit 130' INT TERM  # zsh skips EXIT traps on signals; route them through exit so the VM is always stopped

$C new -s $S --gpu $GPU </dev/null
for t in 1 2 3 4 5 6; do R true && break || { echo "[driver] ssh not ready (try $t)"; sleep 15; }; done  # sshd lags READY
R 'mkdir -p /root/.cache/huggingface'
$C upload -s $S ~/.cache/huggingface/token /root/.cache/huggingface/token </dev/null
$C upload -s $S src/colab_circuit.py /content/colab_circuit.py </dev/null
$C upload -s $S $SPEC /content/spec.json </dev/null
for f in ${=UPLOAD:-}; do $C upload -s $S $f /content/${f:t} </dev/null; done
# The job must not inherit the SSH session's fds, or this connection stays open (one SSH per runtime) and polls block.
R "cd /content; (pip install -q 'circuitkit==0.2.0' 'transformer-lens==3.8.0' > install.log 2>&1 && python -u colab_circuit.py $MODE spec.json > run.log 2>&1) </dev/null >/dev/null 2>&1 & echo \$! > job.pid; echo started"
for i in $(seq 1 $POLLS); do
  sleep 120
  st=$(R 'kill -0 $(cat /content/job.pid) 2>/dev/null && echo RUNNING || echo DONE; free -g | awk "/Mem/{print \"ram_used_GB=\"\$3}"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -2 /content/run.log; dmesg 2>/dev/null | grep -ci "killed process" || true' 2>&1) || st="PROBE-FAILED"
  echo "[poll $i] $(date +%H:%M) $(echo $st | tr '\n' ' ')"
  [[ $st == DONE* ]] && break
done
R 'tail -40 /content/run.log' || true
mkdir -p $OUT
for f in "$@" run.log; do $C download -s $S /content/$f $OUT/$f </dev/null || true; done
ls -la $OUT
