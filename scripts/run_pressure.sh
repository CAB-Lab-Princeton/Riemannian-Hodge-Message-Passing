#!/usr/bin/env bash
# AirfRANS pressure study on the conforming near-field meshes (tasks AFP* of rhmp/tasks/airfrans.py).  Protocol of every
# run: seed SEED (42), EPOCHS (100) epochs, Adam 1e-3 with cosine decay to 1e-5, weight decay 1e-5, gradient clipping 1,
# batches of 8 meshes, checkpoint selected by the validation R2 (simulations 630..699), reported numbers =
# result.json["test100"] (simulations 850..949).  Runs: runs/airfrans/<task>/<name>_s<seed>/.
#
#   bash scripts/run_pressure.sh final                 # the final RHMP configuration (FINAL_TASK, FINAL_ARGS below)
#   bash scripts/run_pressure.sh ablations             # leave-one-out ablations of the final configuration
#   bash scripts/run_pressure.sh variants              # log U input (Reynolds number), Cp loss, angle-of-attack scalar
#   bash scripts/run_pressure.sh t8                    # the final configuration on the original T8 construction (T8i)
#   bash scripts/run_pressure.sh baselines             # the 11 baselines on AFP (node (sdf, U, a) -> raw p)
#   bash scripts/run_pressure.sh controls              # strongest baselines: Cp target, positions, both, physics input
#   bash scripts/run_pressure.sh seeds                 # seeds 1 and 2 of the final RHMP and of the baselines
#   bash scripts/run_pressure.sh reference             # P1 potential flow + Kutta + Bernoulli without learning
#   bash scripts/run_pressure.sh eval [run dirs]       # physical-unit metrics, fine / coarse, rotations, wall, lift
#   DUMP=1 bash scripts/run_pressure.sh eval [runs]    # ... and prediction dumps <task>/preds/<run>.npz
#   bash scripts/run_pressure.sh analysis [run dir]    # learned Kutta metric / circulation of a projection model
#   bash scripts/run_pressure.sh collect               # runs/airfrans/results_pressure.json + Markdown table
#   bash scripts/run_pressure.sh train <task> <name> [trainer flags...]     # one RHMP run
#   bash scripts/run_pressure.sh base <task> <model> [trainer flags...]     # one baseline (legacy node inputs)
#   bash scripts/run_pressure.sh queue <task:model> ...                     # baselines one after the other
#
# Environment: EPOCHS, SEED, BUDGET (parameter budget of the baselines in millions), PAR (parallel RHMP runs of a
# group, default 1), MIN_FREE_MB (a queue item waits until the visible GPU has this much free memory, default 20000),
# TAG (suffix of a baseline's run name), DEVICE (e.g. cpu: appended as --device), EVAL_DEVICE.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
OUT="${OUT:-runs/airfrans}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
PAR="${PAR:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
mkdir -p "$OUT" runs/logs

# ---- the final RHMP configuration --------------------------------------------------------------------------------
# task AFPi0: vertex sdf; edge inflow 1-form theta_e = u_inf . (x_dst - x_src) (odd, amplitude column) + mean sdf (even);
# face mean sdf (even).  Dynamic similarity: inputs divided by the RMS of theta, output multiplied by |u_inf|^2 (Galerkin
# energy of theta).  Stack: projection layer (metric Hodge projection with the learned edge metric, harmonic cochain
# with the learned Kutta metric, vertex potential; sparse direct solves) + three polynomial layers, homogeneous gates.
# Readout: node MLP with 16 Galerkin energy densities of projections of the edge features.
FINAL_TASK="${FINAL_TASK:-AFPi0}"
SCALE="--amplitude 1:1 --amplitude-power 2 --amplitude-energy"
STACK="--gate homog --lift-vertex --lift-context 16 --project-solver direct"
FINAL_C="${FINAL_C:-118}"                      # 88,921 parameters
FINAL_ARGS="--C $FINAL_C --layers project,poly,poly,poly $STACK $SCALE --readout-energy 16"
# Baseline widths: the registry rule gives every baseline the same width for any RHMP count in (88,901, 90,129]
# (SchNet changes at 88,901, GCN at 90,129), so one budget in that band serves the final count.
BUDGET="${BUDGET:-0.089813}"
BASELINES="gem_cnn gauge_cnn mgn cw_net gcn gat schnet egnn mpsn sccnn clifford_smpn"
STRONG="gem_cnn gauge_cnn mgn cw_net"

wait_gpu() {   # wait until the visible GPU has MIN_FREE_MB free
  [ "${DEVICE:-}" = cpu ] && return
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n 1)
    [ "${free:-0}" -ge "$MIN_FREE_MB" ] && break
    sleep 30
  done
}

pwait() {      # keep at most PAR background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
}

train() {      # train <task> <name> <trainer args...>
  local task="$1" name="$2"
  shift 2
  local dir="$OUT/$task/${name}_s$SEED"
  mkdir -p "$dir"
  $PY -u -m rhmp.train --task "$task" --native --epochs "$EPOCHS" --seed "$SEED" --out "$dir" \
    ${DEVICE:+--device "$DEVICE"} "$@" >> "$dir/train.log" 2>&1
  echo "$(date '+%F %T') $task $name s$SEED exit=$? $(grep 'best val' "$dir/train.log" | tail -n 1)"
}

base() {       # base <task> <model> <trainer args...>: baseline, node inputs, width matched to BUDGET
  local task="$1" model="$2"
  shift 2
  local dir="$OUT/$task/${model}${TAG:-}_s$SEED"
  mkdir -p "$dir"
  $PY -u -m rhmp.train --task "$task" --legacy --model "$model" --param-budget "$BUDGET" --epochs "$EPOCHS" \
    --seed "$SEED" --out "$dir" "$@" >> "$dir/train.log" 2>&1
  echo "$(date '+%F %T') $task $model${TAG:-} s$SEED exit=$? $(grep 'best val' "$dir/train.log" | tail -n 1)"
}

cmd="${1:?usage: run_pressure.sh final|ablations|variants|t8|baselines|controls|seeds|reference|eval|analysis|collect|train|base|queue}"
shift
case "$cmd" in
train) train "$@" ;;
base) base "$@" ;;
queue)         # items task:model (baselines), one after the other, each waiting for free GPU memory
  for item in "$@"; do
    IFS=: read -r task model <<< "$item"
    wait_gpu
    base "$task" "$model"
  done
  ;;
final)
  train "$FINAL_TASK" rhmp $FINAL_ARGS
  ;;
ablations)     # leave-one-out (same C; the parameter count changes with the removed part)
  train "$FINAL_TASK" abl_nosolve --C $FINAL_C --layers poly,poly,poly,poly $STACK $SCALE --readout-energy 16 & pwait
  train "$FINAL_TASK" abl_noenergy --C $FINAL_C --layers project,poly,poly,poly $STACK $SCALE & pwait
  # no U^2 scaling: vertex (sdf, U, a), inflow 1-form as an ordinary odd input, plain normalised-pressure output
  train AFPiU abl_noscale --C $FINAL_C --layers project,poly,poly,poly $STACK --readout-energy 16 & pwait
  # no inflow 1-form: vertex (U, sdf, a), U^2 scaling from the scalar U
  train AFPs abl_noform --C $FINAL_C --layers project,poly,poly,poly $STACK --amplitude 0:1 --amplitude-power 2 \
    --readout-energy 16 & pwait
  train "$FINAL_TASK" abl_nokutta --C $FINAL_C --layers project,poly,poly,poly $STACK $SCALE --readout-energy 16 \
    --no-kutta & pwait
  train "$FINAL_TASK" abl_normgate --C $FINAL_C --layers project,poly,poly,poly --lift-vertex --lift-context 16 \
    --project-solver direct $SCALE --readout-energy 16 & pwait
  wait
  ;;
variants)
  train AFPiR var_logU $FINAL_ARGS & pwait                                   # weak Reynolds-number dependence
  train AFPi var_aoa $FINAL_ARGS & pwait                                     # + angle of attack as a vertex scalar
  # trained on the pressure coefficient (every simulation with the same weight): nondimensional output
  train AFPcp var_cp --C $FINAL_C --layers project,poly,poly,poly $STACK --amplitude 1:1 --amplitude-power 0 \
    --amplitude-energy --readout-energy 16 & pwait
  wait
  ;;
t8)
  train T8i rhmp $FINAL_ARGS
  ;;
baselines)
  for m in ${*:-$BASELINES}; do wait_gpu; base AFP "$m"; done
  ;;
controls)      # the strongest baselines with (a) the Cp target, (b) chord-frame positions, both, and the physics input
  for t in AFPcp AFPxy AFPcpxy AFPpf; do
    for m in $STRONG; do wait_gpu; base "$t" "$m"; done
  done
  ;;
seeds)
  for s in 1 2; do
    SEED=$s train "$FINAL_TASK" rhmp $FINAL_ARGS & pwait
  done
  wait
  for s in 1 2; do
    for m in $BASELINES; do wait_gpu; SEED=$s base AFP "$m"; done
  done
  ;;
reference)
  $PY scripts/airfrans_potential_reference.py --ids 850:950 --fit-ids 0:630:9 --out "$OUT/potential_ref_test100.json"
  ;;
eval)          # eval [run dirs...]: physical-unit metrics, zero-shot sets, rotations, wall / lift (+ dumps with DUMP=1)
  runs="${*:-$(ls -d "$OUT"/AFP*/*_s* 2>/dev/null)}"
  for r in $runs; do
    [ -f "$r/result.json" ] || continue
    if [ -f "$r/pressure_metrics.json" ] && { [ -z "${DUMP:-}" ] || [ -f "$(dirname "$r")/preds/$(basename "$r").npz" ]; }; then
      continue
    fi
    $PY -u scripts/airfrans_pressure_eval.py "$r" --force ${DUMP:+--dump} ${EVAL_DEVICE:+--device "$EVAL_DEVICE"} \
      > "$r/pressure_eval.log" 2>&1 && echo "$(date '+%F %T') evaluated $r" || echo "FAILED $r (see $r/pressure_eval.log)"
  done
  ;;
analysis)
  $PY -u scripts/airfrans_pressure_analysis.py "${1:-$OUT/$FINAL_TASK/rhmp_s$SEED}" --device "${EVAL_DEVICE:-cpu}"
  ;;
collect)
  $PY scripts/airfrans_pressure_collect.py --root "$OUT" --md "$@"
  ;;
*) echo "unknown command $cmd"; exit 2 ;;
esac
