#!/usr/bin/env python
"""Run one experiment at full configuration into manuscript_assets.

Usage: python scripts/reference_run_one.py <exp1|exp2|exp3|exp4|exp5> [n_threads]

Used to run the multi-seed experiments concurrently (one process each) with a
capped thread count so they do not oversubscribe the CPU.
"""
import sys, os, time
# Robust threading config: avoid the duplicate-OpenMP mutex storm that crashes
# (or risks corrupting) results when several torch processes run concurrently on
# macOS conda. Single-threaded BLAS/OMP per process + MKL sequential removes the
# second OpenMP runtime; small matrices do not benefit from threads anyway, so we
# get speed from running several independent processes instead.
name = sys.argv[1]
n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = str(n_threads)
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import torch
torch.set_num_threads(n_threads)

from ctpcpinn.config import FULL_CONFIG
cfg = dict(FULL_CONFIG)
cfg['figures_dir'] = os.path.join(CODE_DIR, 'manuscript_assets', 'figures')
cfg['tables_dir'] = os.path.join(CODE_DIR, 'manuscript_assets', 'tables')

mods = {
    'exp1': 'ctpcpinn.experiments.exp1_qubit_inverse',
    'exp2': 'ctpcpinn.experiments.exp2_sparse_measurements',
    'exp3': 'ctpcpinn.experiments.exp3_qutrit_leakage',
    'exp4': 'ctpcpinn.experiments.exp4_two_qubit_gate',
    'exp5': 'ctpcpinn.experiments.exp5_compiler_ablation',
}
import importlib
mod = importlib.import_module(mods[name])
t0 = time.time()
mod.run(cfg)
print(f"[{name}] done in {time.time()-t0:.1f}s")
