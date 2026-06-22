#!/usr/bin/env python
"""Reproduce EVERY table and figure in the manuscript, deterministically.

This is the canonical entry point. It runs the five experiments sequentially in
a single process and writes the LaTeX table files and PDF figures directly into
``submission/tables`` and ``submission/figures``. Recompiling ``main.tex``
afterwards picks up the regenerated numbers automatically.

Usage
-----
    python run_all.py            # full config -- reproduces the paper (~35 min)
    python run_all.py --quick    # fast smoke test (2 seeds, few epochs, ~1 min)
    python run_all.py --full     # explicit alias for the full config

Threading / OpenMP
------------------
On macOS + conda (and some Linux setups) PyTorch and NumPy/SciPy can load two
copies of the OpenMP runtime, which on this workload causes hard crashes
("OMP: Error #15", "pthread_mutex_init failed") or silent hangs when >= 3 BLAS
threads are used or several processes run at once. We therefore pin everything
to a SINGLE thread and force MKL to a sequential backend BEFORE importing torch.
The matrices here are tiny (d = 2..8), so single-threaded costs almost nothing.
See REPRODUCE.txt section 5 for details. You can override with N>1 via the
environment, but only do so if you have confirmed a single OpenMP runtime.

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

    # Resolve output paths to submission/figures and submission/tables.
    submission_dir = os.path.dirname(CODE_DIR)              # .../submission
    config['figures_dir'] = os.path.join(submission_dir, 'figures')
    config['tables_dir'] = os.path.join(submission_dir, 'tables')
    Path(config['figures_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['tables_dir']).mkdir(parents=True, exist_ok=True)

    print(f"  threads={torch.get_num_threads()}  seeds={config.get('seeds')}  "
          f"epochs={config.get('n_epochs')}")
    print(f"  figures -> {config['figures_dir']}")
    print(f"  tables  -> {config['tables_dir']}")

    from ctpcpinn.experiments import exp1_qubit_inverse
    from ctpcpinn.experiments import exp2_sparse_measurements
    from ctpcpinn.experiments import exp3_qutrit_leakage
    from ctpcpinn.experiments import exp4_two_qubit_gate
    from ctpcpinn.experiments import exp5_compiler_ablation

    experiments = [
        ("Experiment 1: single-qubit identification", exp1_qubit_inverse.run),
        ("Experiment 2: sparse-measurement ablation", exp2_sparse_measurements.run),
        ("Experiment 3: fast qutrit reconstruction", exp3_qutrit_leakage.run),
        ("Experiment 4: two-qubit gate generalization", exp4_two_qubit_gate.run),
        ("Experiment 5: dense-vs-structured compiler", exp5_compiler_ablation.run),
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
