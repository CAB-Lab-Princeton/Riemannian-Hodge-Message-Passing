#!/usr/bin/env bash
# Trains the models of the paper's tables (Table 2, Table 3 and the cross-seed appendix): the RHMP configuration of
# each of the seven tasks and the width-matched baselines, with the common protocol of rhmp.train (100 epochs, Adam
# 1e-3 with cosine decay to 1e-5, weight decay 1e-5, gradient clipping 1, batches of 64 samples or 8 meshes, the
# checkpoint with the best validation R2; reported numbers = result.json["test100"], the first 100 test samples).
# The trainer flags below are the ones stored in results/<task>/<run>/config.json of the shipped runs.
#
#   bash scripts/train_paper_models.sh rhmp <task>                  # the RHMP model of one task
#   bash scripts/train_paper_models.sh baselines <task> [models...] # the baselines of one task (default: all of the task)
#   bash scripts/train_paper_models.sh all                          # everything, one run after the other
#   SEED=1 bash scripts/train_paper_models.sh all                   # another seed (the paper: 42, 1, 2)
#   DRY=1 bash scripts/train_paper_models.sh all                    # print the commands only
#
# Tasks (name used here: task codes of rhmp.tasks, RHMP / baselines; data files):
#   ns          NS vorticity on a stretched periodic grid      T1gml_mean / T1g     datasets/v2/T1_cns_vorticity_graded.pkl
#   torus       anisotropic transport on a torus               T2a                  datasets/v2/T2a.pt
#   ellipsoid   ellipsoid surface flow                         T3g / T3             datasets/T3_ellipsoid_coexact.pkl
#   maxwell     nonlinear anisotropic magnetostatics           MXA                  datasets/v2/MXA*.pt
#   wilson      U(1) Wilson loop                               T6                   datasets/T6_wilson_loop.pkl, v2/T6_native.pt
#   yang_mills  SU(2) Yang-Mills                               T7                   datasets/T7_yang_mills_su2.pkl, v2/T7_native.pt
#   yang_mills_face  SU(2) Yang-Mills with the face target     T7f                  (RHMP only; seed 42 in the paper)
#   airfrans    AirfRANS pressure on body-fitted meshes        AFPi0 / AFP          datasets/v2/AFM*.pt
#
# Baselines: gcn gat schnet egnn mpsn sccnn gauge_cnn gem_cnn cw_net clifford_smpn deeponet (fno on ns only, deeponet
# not on airfrans), node inputs (--legacy); mpsn / sccnn / clifford_smpn use their v1 edge protocol (edge-averaged
# target, rhmp.baselines.registry).  Widths are matched to the parameter budget --param-budget (millions; the registry
# rule takes the first width within [1, 1.2] x budget) where a budget is given, and to the parameter count of the
# default RHMP configuration of the task otherwise (wilson, yang_mills, ellipsoid).
#
# Output: runs/<task directory>/<model>_s<seed>/ (config.json, history.json, best.pt, last.pt, result.json), the same
# layout as results/.  A run resumes if interrupted and is skipped when its result.json is complete.  Environment:
# SEED (42), EPOCHS (100), OUT (runs), EXTRA (extra trainer flags, e.g. "--device cpu").
set -uo pipefail
set -f                       # the JSON option values contain brackets: no pathname expansion
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
OUT="${OUT:-runs}"
EXTRA="${EXTRA:-}"

NSHEAD="--gate homog --cochain-head context --lift-context 16 --lift-vertex --readout-depth 3 --readout-spatial --readout-skip --readout-wedge 32 --readout-hidden 32"
ALL_BASELINES="gcn gat schnet egnn mpsn sccnn gauge_cnn gem_cnn cw_net clifford_smpn deeponet"

# spec <task name>: sets DIR (output directory), RHMP_TASK, RHMP_ARGS, BASE_TASK, BASE_ARGS, MODELS
spec() {
  case "$1" in
    ns)         DIR=ns_vorticity; RHMP_TASK=T1gml_mean
                RHMP_ARGS="--C 102 --layers project,poly,poly,poly $NSHEAD --project-solver dense"
                BASE_TASK=T1g; BASE_ARGS="--param-budget 0.088868"; MODELS="$ALL_BASELINES fno" ;;
    torus)      DIR=torus_transport; RHMP_TASK=T2a
                RHMP_ARGS="--layers project,poly,resolvent,poly --metric-type tensor --tensor-param full --resolvent-iters 12 --metric-ref 1:3 --material 1:2,2:2 --transport {\"inputs\":[0],\"order\":2,\"metric\":\"star\"} --log-range 5.0 --project-solver dense"
                BASE_TASK=T2a; BASE_ARGS="--param-budget 0.0375"; MODELS="$ALL_BASELINES" ;;
    ellipsoid)  DIR=ellipsoid_flow; RHMP_TASK=T3g
                RHMP_ARGS="--lifting linear --readout-head linear --fit-readout"
                BASE_TASK=T3; BASE_ARGS=""; MODELS="$ALL_BASELINES" ;;
    maxwell)    DIR=maxwell_magnetostatics; RHMP_TASK=MXA
                RHMP_ARGS="--solver-mode --layers solve --C 4 --metric-type tensor --material 1:7,2:5 --log-range 5.0 --solve-precond twolevel --solve-iters 128 --solve-tol 1e-05 --solve-picard 4 --no-metric-geo --eval-batch 16"
                BASE_TASK=MXA; BASE_ARGS="--param-budget 0.091861 --eval-batch 16"; MODELS="$ALL_BASELINES" ;;
    wilson)     DIR=wilson_loop; RHMP_TASK=T6
                RHMP_ARGS="--layers project,poly,poly,poly --project-solver dense"
                BASE_TASK=T6; BASE_ARGS=""; MODELS="$ALL_BASELINES" ;;
    yang_mills) DIR=yang_mills; RHMP_TASK=T7
                RHMP_ARGS="--C 120 --layers project,poly,poly,poly $NSHEAD --project-solver dense"
                BASE_TASK=T7; BASE_ARGS=""; MODELS="$ALL_BASELINES" ;;
    yang_mills_face) DIR=yang_mills_face; RHMP_TASK=T7f
                RHMP_ARGS="--C 120 --layers project,poly,poly,poly $NSHEAD --project-solver dense"
                BASE_TASK=T7f; BASE_ARGS=""; MODELS="" ;;
    airfrans)   DIR=airfrans_pressure; RHMP_TASK=AFPi0
                RHMP_ARGS="--C 118 --layers project,poly,poly,poly --gate homog --lift-context 16 --lift-vertex --amplitude 1:1 --amplitude-power 2 --amplitude-energy --readout-energy 16 --project-solver direct"
                BASE_TASK=AFP; BASE_ARGS="--param-budget 0.089813"; MODELS="gcn gat schnet egnn mpsn sccnn gauge_cnn gem_cnn cw_net clifford_smpn" ;;
    *) echo "unknown task $1 (ns torus ellipsoid maxwell wilson yang_mills yang_mills_face airfrans)"; exit 2 ;;
  esac
}

# per-model exceptions: on the Maxwell task the three edge-protocol baselines are built with the v1 edge head
# explicitly; on the torus task MPSN gets the budget 0.036135 (with 0.0375 the width rule overshoots the 20 % band)
model_extra() {
  case "$1:$2" in
    maxwell:mpsn|maxwell:sccnn|maxwell:clifford_smpn) echo '--model-opts {"head":"v1_edge"}' ;;
    torus:mpsn) echo '--param-budget 0.036135' ;;
  esac
}

run() {   # run <output dir> <trainer args...>
  local dir="$1"; shift
  if [[ "${DRY:-0}" == "1" ]]; then printf "python3 -m rhmp.train"; printf " %q" "$@" --epochs "$EPOCHS" --seed "$SEED" --out "$dir" $EXTRA; echo; return; fi
  local log="$OUT/logs/$(basename "$(dirname "$dir")")_$(basename "$dir").log"
  mkdir -p "$dir" "$OUT/logs"
  echo "=== $(date '+%F %T') $dir"
  python3 -u -m rhmp.train "$@" --epochs "$EPOCHS" --seed "$SEED" --out "$dir" $EXTRA > "$log" 2>&1
  echo "    exit $? ; $(grep 'best val' "$log" | tail -n 1)"
}

rhmp_run() {   # rhmp_run <task name>
  spec "$1"
  # shellcheck disable=SC2086
  run "$OUT/$DIR/rhmp_s$SEED" --task "$RHMP_TASK" --model rhmp $RHMP_ARGS
}

baseline_runs() {   # baseline_runs <task name> [models...]
  local t="$1"; shift
  spec "$t"
  for m in ${*:-$MODELS}; do
    # shellcheck disable=SC2086
    local args="$BASE_ARGS" extra; extra="$(model_extra "$t" "$m")"
    [[ "$extra" == --param-budget* ]] && args="$extra" && extra=""
    run "$OUT/$DIR/${m}_s$SEED" --task "$BASE_TASK" --legacy --model "$m" $args $extra
  done
}

case "${1:-}" in
  rhmp)       rhmp_run "${2:?task}" ;;
  baselines)  baseline_runs "${2:?task}" "${@:3}" ;;
  all)        for t in ns torus ellipsoid maxwell wilson yang_mills yang_mills_face airfrans; do
                rhmp_run "$t"; baseline_runs "$t"
              done ;;
  *)          sed -n '2,31p' "$0"; exit 1 ;;
esac
