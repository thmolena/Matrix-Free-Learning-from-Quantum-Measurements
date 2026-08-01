#!/usr/bin/env python
"""Reproduce the manuscript's experimental and controlled-study evidence.

This is the canonical source-tree entry point. It first runs the experimental
Ramsey analysis and then the six controlled studies sequentially in a single
process, writing LaTeX tables, PDF line/bar figures, and evidence JSON directly
into ``code/manuscript_assets``. Recompiling ``main.tex`` afterwards picks up
the regenerated numbers automatically.

Usage
-----
    python scripts/reference_run_all.py            # full config -- reproduces the paper (~35 min)
    python scripts/reference_run_all.py --quick    # fast smoke test (2 seeds, few epochs, ~1 min)
    python scripts/reference_run_all.py --full     # explicit alias for the full config

Threading / OpenMP
------------------
On macOS + conda (and some Linux setups) PyTorch and NumPy/SciPy can load two
copies of the OpenMP runtime, which on this workload causes hard crashes
("OMP: Error #15", "pthread_mutex_init failed") or silent hangs when >= 3 BLAS
threads are used or several processes run at once. We therefore pin everything
to a SINGLE thread and force MKL to a sequential backend BEFORE importing torch.
The matrices here are tiny (d = 2..8), so single-threaded costs almost nothing.
See ctpcpinn/foundations.py for the reproduction and theory map. You can
override with N>1 via the environment, but only do so if you have confirmed a
single OpenMP runtime.

Determinism
-----------
Each experiment is repeated over the fixed seeds [0, 1, 2, 3, 4] and reports
mean +/- 95% confidence interval. Results are deterministic per seed; only the
wall-clock TIMES in experiment 5 are machine dependent (their ratios are stable).
"""

import os

# --- OpenMP / threading safety: set BEFORE importing numpy/scipy/torch. --------
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
import time
from pathlib import Path

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

import torch
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

from ctpcpinn.config import QUICK_CONFIG, FULL_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Reproduce CPTP-Compiler-PINN results")
    parser.add_argument('--quick', action='store_true', help='Fast smoke test')
    parser.add_argument('--full', action='store_true', help='Full publication run (default)')
    args = parser.parse_args()

    if args.quick:
        config = dict(QUICK_CONFIG)
        print("Running in QUICK mode (smoke test; NOT publication quality)...")
    else:
        config = dict(FULL_CONFIG)
        print("Running in FULL mode (reproduces the manuscript)...")

    asset_dir = os.path.join(CODE_DIR, 'manuscript_assets')
    config['figures_dir'] = os.path.join(asset_dir, 'figures')
    config['tables_dir'] = os.path.join(asset_dir, 'tables')
    Path(config['figures_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['tables_dir']).mkdir(parents=True, exist_ok=True)

    print(f"  threads={torch.get_num_threads()}  seeds={config.get('seeds')}  "
          f"epochs={config.get('n_epochs')}")
    print(f"  figures -> {config['figures_dir']}")
    print(f"  tables  -> {config['tables_dir']}")

    from ctpcpinn.real_data import reproduce as reproduce_real_data
    reproduce_real_data(asset_dir)
    print(f"  experimental Ramsey evidence -> {asset_dir}")

    from ctpcpinn.experiments import exp1_qubit_inverse
    from ctpcpinn.experiments import exp2_sparse_measurements
    from ctpcpinn.experiments import exp3_qutrit_leakage
    from ctpcpinn.experiments import exp4_two_qubit_gate
    from ctpcpinn.experiments import exp5_compiler_ablation
    from ctpcpinn.experiments import exp6_spectral_truncation

    experiments = [
        ("Experiment 1: single-qubit identification", exp1_qubit_inverse.run),
        ("Experiment 2: sparse-measurement ablation", exp2_sparse_measurements.run),
        ("Experiment 3: fast qutrit reconstruction (spectral frame)", exp3_qutrit_leakage.run),
        ("Experiment 4: two-qubit gate generalization", exp4_two_qubit_gate.run),
        ("Experiment 5: dense-vs-structured compiler", exp5_compiler_ablation.run),
        ("Experiment 6: symmetric Fourier generator approximation", exp6_spectral_truncation.run),
    ]

    total_start = time.time()
    results = {}
    for name, run_fn in experiments:
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        exp_start = time.time()
        try:
            results[name] = run_fn(dict(config))
            print(f"  Completed in {time.time() - exp_start:.1f}s")
        except Exception as e:  # keep going so one failure does not lose the rest
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None

    print(f"\n{'=' * 60}")
    print(f"  All experiments completed in {time.time() - total_start:.1f}s")
    print(f"  Now rebuild the PDF:  cd .. && tectonic -X compile main.tex")
    print(f"{'=' * 60}")
    return results


if __name__ == '__main__':
    main()
