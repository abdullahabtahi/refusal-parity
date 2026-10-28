#!/bin/zsh
# CircuitKIT parity gate on a Colab T4. The HF token travels by `colab upload` (file transfer, not recorded in
# exec history); the VM is always stopped on exit.
set -eu
cd "${0:A:h}/.."
C=.venv/bin/colab
S=circuitkit-parity
trap 'kill ${W:-0} 2>/dev/null; $C stop -s $S </dev/null >/dev/null 2>&1 || true; echo "[driver] session stopped"' EXIT

$C new -s $S --gpu T4 </dev/null
# Watchdog: a crashed kernel leaves `colab exec` waiting forever; stopping the session releases it.
( sleep ${WATCHDOG_SECS:-3600}; $C stop -s $S ) </dev/null >/dev/null 2>&1 &
W=$!  # by PID: `kill %1` does not work in a non-interactive shell, and the watchdog must not hold our stdout
echo 'import os; os.makedirs("/root/.cache/huggingface", exist_ok=True)' | $C exec -s $S
$C upload -s $S ~/.cache/huggingface/token /root/.cache/huggingface/token </dev/null
$C upload -s $S src/colab_circuit.py /content/colab_circuit.py </dev/null
$C upload -s $S scratch_colab/parity_prompts.json /content/parity_prompts.json </dev/null
$C install -s $S circuitkit </dev/null
echo 'import os, sys; os.chdir("/content"); sys.argv = ["colab_circuit.py", "parity", "parity_prompts.json"]; exec(open("colab_circuit.py").read())' | $C exec -s $S
mkdir -p runs/circuitkit
$C download -s $S /content/parity.json runs/circuitkit/parity.json </dev/null
echo "[driver] parity.json downloaded"
