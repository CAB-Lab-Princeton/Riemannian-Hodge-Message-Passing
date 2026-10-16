#!/usr/bin/env bash
# Generates the data sets in datasets/v2/ (git-ignored; see datasets/README.md for every file, its generator and the
# inputs it needs).  By default the generated sets of the paper's seven tasks:
#   T6_native.pt, T7_native.pt            edge cochains of the Wilson-loop / Yang-Mills sets, regenerated from the original
#                                         pickles with the CUDA random number generator (a GPU is needed to reproduce them)
#   T2v.pt, T2a.pt, T2a_valmesh.pt        torus transport: the velocities of the original generator and the anisotropic task
#   MXA*.pt                               nonlinear anisotropic magnetostatics (+ self-test of the solver)
#   T1_cns_vorticity_graded(_val).pkl     NS vorticity on the stretched grid: only when SRC is the PDEBench HDF5 file
#   AFM*.pt                               AirfRANS body-fitted meshes: only when AIRFRANS_ROOT is the AirfRANS Dataset directory
# All CPU generators are seeded and deterministic (the T6 / T7 regeneration reproduces the original samples with the CUDA
# RNG, see datasets/README.md).  Options: EXTRA=1 adds the heterogeneous-Poisson and tetrahedral sets (HP_*, TET_k100),
# SUITE=1 the extension suite (SURF, DYN, HP_qual_*; docs/TASK_SUITE_DETAILS.md), ANISO=1 the anisotropy suite (AHP,
# ASURF, ACURL, ADARCY; docs/ANISO_TASKS.md); SKIP_PAPER=1 skips the paper sets, SKIP_NATIVE=1 only the T6 / T7
# regeneration.  WORKERS (default 32) processes for the CPU generators.
#   bash scripts/gen_datasets.sh                                                   # the paper's generated sets
#   SRC=/data/pdebench/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
#   AIRFRANS_ROOT=/data/airfrans/Dataset bash scripts/gen_datasets.sh              # + the NS grid and AirfRANS sets
#   SUITE=1 ANISO=1 EXTRA=1 bash scripts/gen_datasets.sh                           # everything
set -euo pipefail
cd "$(dirname "$0")/.."
W="${WORKERS:-32}"
if [[ "${SKIP_PAPER:-0}" != "1" ]]; then
  if [[ "${SKIP_NATIVE:-0}" != "1" ]]; then
    python3 -u datasets/generators/gen_T6_native.py                               # needs datasets/T6_wilson_loop.pkl, GPU
    python3 -u datasets/generators/gen_T7_native.py                               # needs datasets/T7_yang_mills_su2.pkl, GPU
  fi
  python3 -u datasets/generators/gen_T2v.py                                       # needs datasets/T2_torus_advection_diffusion.pkl
  python3 -u datasets/generators/gen_T2a.py --workers "$W"                        # needs T2v.pt and pyvista
  python3 -u datasets/generators/gen_T2a.py --val-meshes datasets/v2/T2a_valmesh.pt --workers "$W"
  python3 -u datasets/generators/gen_maxwell_aniso.py data --workers "$W"
  python3 -u datasets/generators/gen_maxwell_aniso.py selftest --workers 12
  if [[ -n "${SRC:-}" ]]; then
    python3 -u datasets/generators/gen_T1_graded.py --src "$SRC"                  # needs datasets/T1_cns_vorticity.pkl, h5py
    python3 -u datasets/generators/gen_T1_graded.py --src "$SRC" --zero-shot-samples val
  else
    echo "SRC not set: skipping the stretched-grid NS set (datasets/v2/T1_cns_vorticity_graded.pkl)"
  fi
  if [[ -n "${AIRFRANS_ROOT:-}" ]]; then
    python3 -u datasets/generators/gen_airfrans.py mesh --workers "$W"            # needs the airfrans package and pyvista
  else
    echo "AIRFRANS_ROOT not set: skipping the AirfRANS meshes (datasets/v2/AFM*.pt)"
  fi
fi
if [[ "${EXTRA:-0}" == "1" ]]; then          # heterogeneous Poisson and tetrahedral sets (docs/TASK_SUITE.md)
  for k in 10 100 1000; do
    python3 -u datasets/generators/gen_HP.py --kappa "$k" --n 5000 --n-fine 500 --workers "$W"
  done
  python3 -u datasets/generators/gen_HP.py --kappa 100 --aniso --n 5000 --n-fine 500 --workers "$W"   # HP_k100_aniso
  for r in 10 100; do                                                                                 # HP_k100_aniso<r>
    python3 -u datasets/generators/gen_HP.py --kappa 100 --aniso-max "$r" --n 5000 --n-fine 500 --workers "$W"
  done
  # HP_k10000 for the contrast sweep: 1000 samples, the fields and meshes of the first 1000 samples of the HP_k* sets
  python3 -u datasets/generators/gen_HP.py --kappa 10000 --n 1000 --n-fine 0 --workers "$W"
  python3 -u datasets/generators/gen_TET.py --kappa 100 --n 3000 --n-fine 200 --workers "$W"
fi
if [[ "${SUITE:-0}" == "1" ]]; then          # extension suite (docs/TASK_SUITE_DETAILS.md); gen_qual needs HP_k100.pt
  python3 -u datasets/generators/gen_surf.py --workers "$W"
  python3 -u datasets/generators/gen_dyn.py --workers "$W"
  python3 -u datasets/generators/gen_qual.py --workers "$W"
fi
if [[ "${ANISO:-0}" == "1" ]]; then          # anisotropy suite (docs/ANISO_TASKS.md)
  python3 -u datasets/generators/gen_aniso.py --tasks AHP ASURF ACURL ADARCY --workers "$W"
  python3 -u datasets/generators/gen_aniso.py --analyse 12 --workers "$W"      # oracle representability report
fi
ls -la datasets/v2/
