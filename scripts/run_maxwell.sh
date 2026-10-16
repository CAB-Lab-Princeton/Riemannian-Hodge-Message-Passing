#!/usr/bin/env bash
# Experiments of the nonlinear magnetostatics tasks MXS and MXA (rhmp/tasks/maxwell.py, generators
# datasets/generators/gen_maxwell.py and gen_maxwell_aniso.py).  Every training run uses the fixed-mesh protocol of rhmp.train: seed 42, 100
# epochs, Adam 1e-3 with cosine decay to 1e-5, weight decay 1e-5, clipping 1, batch 64, sequential 70/15/15 split,
# model selection on validation R2, test100 = samples 4250..4349.  Runs are skipped when their result is complete.
#
#   bash scripts/run_maxwell.sh data                         # data sets (MXS, MXS_ood15, MXS_fine, MXS_coarse) + selftest
#   bash scripts/run_maxwell.sh data_aniso                   # MXA, MXA_ood15, MXA_fine, MXA_coarse, MXA_heldout + selftest
#   bash scripts/run_maxwell.sh train <variant> [extra trainer flags...]       # one RHMP configuration (list below)
#   bash scripts/run_maxwell.sh baselines [models...]        # node baselines (legacy inputs), one after the other
#   bash scripts/run_maxwell.sh queue models...              # baselines started in the background, each as soon as the
#                                                            # GPU has MIN_FREE_MB (14000) MiB free (shared GPUs);
#                                                            # <model>_rotaug = rotation-augmented training,
#                                                            # train:<variant> = an RHMP variant
#   bash scripts/run_maxwell.sh eval [run dirs...]           # divergence, rotations, timing, prediction dumps
#   bash scripts/run_maxwell.sh augment <gem_cnn|gauge_cnn>  # rotation-augmented baseline training
#   bash scripts/run_maxwell.sh tables                       # results.json + Markdown tables of all runs
#   bash scripts/run_maxwell.sh recovery <run dir>           # learned reluctivity vs the true B-H law
#
# RHMP variants (run directory runs/maxwell/MXS/<variant>_s42):
#   solver_T_J4      physical solve, Picard J = 4, tensor metric, material columns only in the metric (2.8K params)
#   solver_T_J8      the same with J = 8
#   solver_T_J2      the same with J = 2
#   solver_T_J{0,2,4}_nogeo   the solver variants with metric heads that see no geometric descriptors (material and
#                    state invariants only: a constitutive law without mesh-specific cues)
#   solver_D_J4      diagonal metric
#   solver_T_J0      one linear solve (no state dependence)
#   solver_T_frozen  frozen reference metric (H = Galerkin star of the unit tensor; homogeneous operator)
#   stack_T_J3       three polynomial layers + a Picard solve layer (J = 3, tensor metric), C = 64, MLP lifting and head
#   poly4            the general four-layer model without a solve (C = 128, 90.7K params)
#   poly4_frozen     poly4 with frozen metrics (dec_fixed control)
# Main task MXA (TASK=MXA; anisotropic steel, several layout families; material columns = 7 edge + 5 face columns):
#   solver_T_J2_nogeo  physical solve, Picard J = 2, tensor metric, heads without geometric descriptors
#   solver_T_J2        the same with geometric descriptors
#   solver_T_J0_nogeo  one linear solve (no state dependence)
#   solver_D_J2_nogeo  diagonal metric (cannot represent misaligned anisotropic tensors)
#   solver_T_J2_noorient  task MXA_noorient: no rolling-direction columns (ablation of the orientation input)
#   solver_T_J4_nogeo  four Picard solves: the RHMP model of the paper (results/maxwell_magnetostatics/rhmp_s*)
#   solver_T_J2_nogeo_init  J = 2 with a learned starting operating point of the iteration (--picard-init learned)
#   solver_T_frozen    frozen reference metric
#   stack_T_J2         three polynomial layers + a Picard solve layer (J = 2, tensor metric), C = 64
#   poly4              the general four-layer model (C = 128, 91.9K params: the baseline budget)
# Secondary task MXE (TASK=MXE; electrostatics of an ECT sensor, linear): esolver_T (one solve, tensor metric,
# electrode potentials as Dirichlet data through solve_bc='lift'), esolver_D (diagonal metric), esolver_T_frozen,
# esolver_T_nolift (homogeneous Dirichlet solve: control), poly4.
#
# Environment: TASK (MXS | MXA | MXE), OUT (runs/maxwell), EPOCHS (100), SEED (42), PY (python3), BUDGET (baseline
# parameter budget in millions; default = the poly4 model of the task: 0.090709 MXS, 0.091861 MXA, 0.091605 MXE;
# baselines: the 11 paper baselines + mgn, MeshGraphNet with 15 processor steps, a reference row), WORKERS (48), EVAL_BATCH
# (16: batch of the validation / test passes; memory only).  An interrupted run resumes from last.pt; a run whose
# training finished but whose final evaluation failed is completed by re-running the same command.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
OUT="${OUT:-runs/maxwell}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
TASK="${TASK:-MXS}"
case "$TASK" in
  MXE) BUDGET="${BUDGET:-0.091605}" ;;
  MXA) BUDGET="${BUDGET:-0.091861}" ;;
  *)   BUDGET="${BUDGET:-0.090709}" ;;
esac
BASELINES="gcn gat schnet egnn mpsn sccnn gauge_cnn gem_cnn cw_net clifford_smpn deeponet"
[[ "$TASK" == MXA ]] && BASELINES="$BASELINES mgn"
EDGE_PROTOCOL="mpsn sccnn clifford_smpn"
if [[ "$TASK" == MXA ]]; then MAT="1:7,2:5"; else MAT="1:3,2:3"; fi
SOLVER="--solver-mode --layers solve --material $MAT --log-range 5 --C 4 --solve-precond twolevel --solve-iters 128 --solve-tol 1e-5"
ESOLVER="--solver-mode --layers solve --material 1:5,2:5 --log-range 5 --C 4 --solve-precond twolevel --solve-iters 128 --solve-tol 1e-5"
EVAL_BATCH="${EVAL_BATCH:-16}"            # evaluation batch (memory of the refined-mesh test sets; no effect on results)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT/$TASK" runs/logs
cmd="${1:?usage: run_maxwell.sh data|data_aniso|train|baselines|eval|augment|tables|recovery ...}"
shift

variant_args() {
  case "$1" in
    solver_T_J4)     echo "$SOLVER --metric-type tensor --solve-picard 4" ;;
    solver_T_J8)     echo "$SOLVER --metric-type tensor --solve-picard 8" ;;
    solver_T_J2)     echo "$SOLVER --metric-type tensor --solve-picard 2" ;;
    solver_T_J2_nogeo) echo "$SOLVER --metric-type tensor --solve-picard 2 --no-metric-geo" ;;
    solver_T_J4_nogeo) echo "$SOLVER --metric-type tensor --solve-picard 4 --no-metric-geo" ;;
    solver_T_J0_nogeo) echo "$SOLVER --metric-type tensor --no-metric-geo" ;;
    solver_D_J4)     echo "$SOLVER --solve-picard 4" ;;
    solver_T_J0)     echo "$SOLVER --metric-type tensor" ;;
    solver_T_frozen) echo "$SOLVER --metric-type tensor --no-learn-metric" ;;
    solver_T_J2_nogeo_noorient|solver_T_J2_noorient)
                     echo "${SOLVER/--material $MAT/--material 1:5,2:5} --metric-type tensor --solve-picard 2 --no-metric-geo" ;;
    solver_D_J2_nogeo) echo "$SOLVER --solve-picard 2 --no-metric-geo" ;;
    solver_T_J2_nogeo_init) echo "$SOLVER --metric-type tensor --solve-picard 2 --no-metric-geo --picard-init learned" ;;
    stack_T_J2)      echo "--layers poly,poly,poly,solve --C 64 --metric-type tensor --log-range 5 --solve-picard 2 --solve-precond twolevel --solve-iters 64 --solve-tol 1e-4 --no-metric-geo" ;;
    stack_T_J3)      echo "--layers poly,poly,poly,solve --C 64 --metric-type tensor --log-range 5 --solve-picard 3 --solve-precond twolevel --solve-iters 64 --solve-tol 1e-4" ;;
    poly4)           echo "--C 128 --layers 4" ;;
    poly4_frozen)    echo "--C 128 --layers 4 --no-learn-metric" ;;
    esolver_T)        echo "$ESOLVER --metric-type tensor --solve-bc lift" ;;
    esolver_D)        echo "$ESOLVER --solve-bc lift" ;;
    esolver_T_frozen) echo "$ESOLVER --metric-type tensor --solve-bc lift --no-learn-metric" ;;
    esolver_T_nolift) echo "$ESOLVER --metric-type tensor" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

case "$cmd" in
data)
  $PY -u datasets/generators/gen_maxwell.py data --workers "${WORKERS:-48}"
  $PY -u datasets/generators/gen_maxwell.py selftest --workers 20
  $PY -u datasets/generators/gen_maxwell.py estat --workers "${WORKERS:-48}"
  $PY -u datasets/generators/gen_maxwell.py estat --selftest
  ;;
data_aniso)
  $PY -u datasets/generators/gen_maxwell_aniso.py data --workers "${WORKERS:-48}"
  $PY -u datasets/generators/gen_maxwell_aniso.py selftest --workers 12
  ;;
train)
  variant="${1:?variant}"; shift
  args="$(variant_args "$variant")" || exit 2
  dir="$OUT/$TASK/${variant}_s${SEED}"
  task="$TASK"
  [[ "$variant" == *_noorient ]] && task="${TASK}_noorient"      # rolling-direction columns dropped (ablation)
  mkdir -p "$dir"
  $PY -u -m rhmp.train --task "$task" --native $args --epochs "$EPOCHS" --seed "$SEED" --eval-batch "$EVAL_BATCH" \
    --out "$dir" "$@" \
    >> "$dir/train.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') $TASK $variant exit=$rc -> $dir"
  ;;
baselines)
  models="${*:-$BASELINES}"
  for m in $models; do
    base="${m%_node}"
    dir="$OUT/$TASK/${m}_s${SEED}"
    opts=()
    if [[ " $EDGE_PROTOCOL " == *" $m "* ]]; then opts=(--model-opts '{"head": "v1_edge"}'); fi   # paper protocol
    if [[ "$m" == *_node ]]; then opts=(--model-opts '{"head": "node"}'); fi                     # node-head variant
    mkdir -p "$dir"
    $PY -u -m rhmp.train --task "$TASK" --legacy --model "$base" --param-budget "$BUDGET" "${opts[@]}" \
      --epochs "$EPOCHS" --seed "$SEED" --eval-batch "$EVAL_BATCH" --out "$dir" >> "$dir/train.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') $TASK $m exit=$rc -> $dir"
  done
  ;;
queue)
  gi="${CUDA_VISIBLE_DEVICES:-0}"
  for m in "$@"; do
    until [[ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gi%%,*}" | head -1)" -gt "${MIN_FREE_MB:-14000}" ]]; do
      sleep 60
    done
    echo "$(date '+%F %T') start $TASK $m"
    if [[ "$m" == train:* ]]; then bash "$0" train "${m#train:}" &
    elif [[ "$m" == *_rotaug ]]; then bash "$0" augment "${m%_rotaug}" &
    else bash "$0" baselines "$m" &
    fi
    sleep "${QUEUE_PAUSE:-300}"                    # let the run allocate its memory before the next check
  done
  wait
  ;;
eval)
  runs="${*:-$(ls -d "$OUT/$TASK"/*_s"$SEED" 2>/dev/null)}"
  for r in $runs; do
    [[ -f "$r/best.pt" ]] || { echo "skip $r (no best.pt)"; continue; }
    $PY -u scripts/maxwell_eval.py "$r" --dump --batch "$EVAL_BATCH" ${EVAL_OPTS:-} > "$r/maxwell_eval.log" 2>&1 \
      && echo "evaluated $r" || echo "FAILED $r (see $r/maxwell_eval.log)"
  done
  ;;
augment)
  m="${1:?gem_cnn|gauge_cnn}"; shift
  dir="$OUT/$TASK/${m}_rotaug_s${SEED}"
  mkdir -p "$dir"
  $PY -u scripts/maxwell_augment.py --task "$TASK" --model "$m" --param-budget "$BUDGET" --epochs "$EPOCHS" --seed "$SEED" \
    --out "$dir" "$@" > "$dir/train.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') $TASK $m rotation-augmented exit=$rc -> $dir"
  ;;
tables)
  $PY scripts/maxwell_tables.py --root "$OUT/$TASK" "$@"
  ;;
recovery)
  $PY -u scripts/maxwell_eval.py "${1:?run dir}" --recovery-only
  ;;
*)
  echo "unknown command $cmd"; exit 2;;
esac
