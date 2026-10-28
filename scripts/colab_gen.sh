#!/bin/zsh
# Generate arms on a Colab GPU with the unchanged run_arm.py, timing each arm; the Mac-side twin is gen_queue.sh.
# HF token by `colab upload` only (Tiny Aya is gated); the VM is always stopped.
# Usage: COLAB_SSH_KEY=<private key> [GPU=A100] [SESSION=gen-job] [ACTS=runs/mech_earth/calib_acts.pt] \
#          scripts/colab_gen.sh "NAME|run_arm args" ...
#   Other jobs (e.g. Qwen3Guard scoring): SCRIPT=audit.py JOB_ENV="JUDGE_BATCH=4" FETCH="judged.jsonl auditkit.json"
#   OUTSUB=colab INPUTS="runs/X/responses.jsonl ..." ACTS=none  "X|score ../runs/X/responses.jsonl --no-lid"
#   NAME = run dir under runs/ (use --tag so A100 runs never overwrite the Mac's). Outputs: runs/NAME/, timing in
#   runs/NAME/colab_timing.json. Keep --batch equal to the Mac's (16) so hardware is the only difference.
set -eu
cd "${0:A:h}/.."
C="$PWD/.venv/bin/colab"; S=${SESSION:-gen-job}; GPU=${GPU:-A100}; ACTS=${ACTS:-runs/mech_earth/calib_acts.pt}
SCRIPT=${SCRIPT:-run_arm.py}; JOB_ENV=${JOB_ENV:-}; FETCH=${FETCH:-responses.jsonl arm_config.json}; OUTSUB=${OUTSUB:-}; INPUTS=${INPUTS:-}
out_dir() { echo runs/$1${OUTSUB:+/$OUTSUB}; }  # where a finished job's outputs land locally
K=${COLAB_SSH_KEY:?set COLAB_SSH_KEY to the private key whose public half the Colab session trusts}
R() { ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -o ProxyCommand="'$C' ssh --proxy-mode -s $S -i $K" root@colab "$@"; }
# Large files (activations, ~290 MB) exceed the Colab upload API (HTTP 500); copy them over the same SSH channel.
P() { scp -q -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
        -o ProxyCommand="'$C' ssh --proxy-mode -s $S -i $K" "$1" root@colab:"$2"; }
trap '$C stop -s $S </dev/null >/dev/null 2>&1 || true; echo "[driver] session stopped"' EXIT
trap 'exit 130' INT TERM

# Resume: arms whose responses already landed locally are skipped (a dropout re-runs only what is missing).
specs=(); for spec in "$@"; do [[ -s $(out_dir ${spec%%|*})/${FETCH%% *} ]] && echo "[driver] skip ${spec%%|*}: done" || specs+=("$spec"); done
(( ${#specs} )) || { echo "[driver] nothing to run"; trap - EXIT; exit 0; }
set -- "${specs[@]}"
$C new -s $S --gpu $GPU </dev/null
for t in 1 2 3 4 5 6; do R true && break || { echo "[driver] ssh not ready (try $t)"; sleep 15; }; done
retry() { for k in 1 2 3; do "$@" && return 0; echo "[driver] retry $k: ${1:t}"; sleep 20; done; return 1; }
retry R "mkdir -p /root/.cache/huggingface /content/src /content/data /content/runs /content/${ACTS:h}" || exit 1
retry $C upload -s $S ~/.cache/huggingface/token /root/.cache/huggingface/token </dev/null || { echo "[driver] token upload failed"; exit 1; }
# Code + data as one archive over scp: ~30 single-file uploads through the Colab API failed on a flaky link.
bundle=$(mktemp -t colab_bundle).tgz; tar czf $bundle src/*.py data/eval.jsonl data/lid_cache.json ${=INPUTS}
retry P $bundle /content/bundle.tgz && retry R "cd /content && tar xzf bundle.tgz" || { echo "[driver] bundle upload failed"; exit 1; }
[[ $ACTS == none ]] || retry P $ACTS /content/$ACTS || { echo "[driver] scp failed: $ACTS"; exit 1; }
# One script file runs every arm in sequence and records wall time per arm (model load included, as on the Mac).
# Written locally and uploaded, so no $(...) passes through nested remote quoting (it would expand at launch).
job=$(mktemp -t colab_gen); {
  echo 'cd /content/src && mkdir -p /content/runs'
  echo "pip install -q 'transformers==5.20.2' 'safetune==0.1.10' 'auditkit[transformers]==1.1.3' 'accelerate==1.15.0' > ../install.log 2>&1"  # = runs/mac_versions.txt
  echo "pip freeze | grep -iE '^(torch|transformers|safetune|auditkit|accelerate)==' > ../runs/versions.txt"
  for spec in "$@"; do
    name=${spec%%|*}; args=${spec#*|}
    echo "t0=\$(date +%s); $JOB_ENV python -u $SCRIPT $args > ../$name.log 2>&1; rc=\$?; t1=\$(date +%s)"
    echo "echo \"{\\\"name\\\": \\\"$name\\\", \\\"rc\\\": \$rc, \\\"seconds\\\": \$((t1-t0)), \\\"gpu\\\": \\\"$GPU\\\"}\" > ../runs/$name.timing.json"
  done
} > $job
cat $job
retry P $job /content/gen.sh || { echo "[driver] job upload failed"; exit 1; }
R "cd /content; (bash gen.sh) </dev/null >/dev/null 2>&1 & echo \$! > /content/job.pid; echo started"
fetch() {  # one finished arm's outputs -> runs/<name>/
  local name=$1 d=$(out_dir $1); mkdir -p $d
  for f in ${=FETCH}; do $C download -s $S /content/runs/$name/$f $d/$f </dev/null >/dev/null 2>&1 || true; done
  $C download -s $S /content/runs/$name.timing.json $d/colab_timing.json </dev/null >/dev/null 2>&1 || true
  $C download -s $S /content/$name.log $d/colab_run.log </dev/null >/dev/null 2>&1 || true
  echo "[driver] fetched $name: $(cat $d/colab_timing.json 2>/dev/null) rows=$(wc -l < $d/${FETCH%% *} 2>/dev/null || echo 0)"
}
typeset -A fetched
# Outputs leave the VM as each arm finishes: an offline Mac lets Colab reclaim the runtime (2026-10-27, six arms lost).
for i in $(seq 1 120); do  # up to 4 h
  sleep 120
  st=$(R 'test -f /content/job.pid || echo LOST; kill -0 $(cat /content/job.pid 2>/dev/null) 2>/dev/null && echo RUNNING || echo DONE; ls /content/runs 2>/dev/null | grep timing.json; true' 2>&1) || st="PROBE-FAILED"
  echo "[poll $i] $(date +%H:%M) $(echo $st | tr '\n' ' ')"
  [[ $st == LOST* ]] && { echo "[driver] runtime was replaced (job.pid gone): outputs not yet fetched are lost; re-run to resume"; exit 1; }
  for spec in "$@"; do
    name=${spec%%|*}
    [[ $st == *"$name.timing.json"* && -z ${fetched[$name]:-} ]] && { fetch $name; fetched[$name]=1; }
  done
  [[ $st == DONE* ]] && break
done
mkdir -p runs/${1%%|*}
$C download -s $S /content/runs/versions.txt runs/${1%%|*}/colab_versions.txt </dev/null || true  # vs the Mac's
for spec in "$@"; do name=${spec%%|*}; [[ -z ${fetched[$name]:-} ]] && fetch $name; done
