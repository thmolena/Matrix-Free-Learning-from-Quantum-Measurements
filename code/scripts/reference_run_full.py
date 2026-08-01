#!/usr/bin/env python
"""Regenerate every manuscript figure and table at the full configuration.

Writes figure PDFs and LaTeX tables into ``code/manuscript_assets``.
Single-threaded for OpenMP safety; each experiment is isolated so one failure
does not lose the rest.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
import traceback
from pathlib import Path

import torch
torch.set_num_threads(1)

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from ctpcpinn.config import FULL_CONFIG

config = dict(FULL_CONFIG)
config['figures_dir'] = str(CODE / 'manuscript_assets' / 'figures')
config['tables_dir'] = str(CODE / 'manuscript_assets' / 'tables')
Path(config['figures_dir']).mkdir(parents=True, exist_ok=True)
Path(config['tables_dir']).mkdir(parents=True, exist_ok=True)

from ctpcpinn.experiments import (exp1_qubit_inverse, exp2_sparse_measurements,
                                  exp3_qutrit_leakage, exp4_two_qubit_gate,
                                  exp5_compiler_ablation, exp6_spectral_truncation)
EXPS = [
    ("exp1 single-qubit", exp1_qubit_inverse.run),
    ("exp2 sparse", exp2_sparse_measurements.run),
    ("exp3 qutrit spectral", exp3_qutrit_leakage.run),
    ("exp4 two-qubit", exp4_two_qubit_gate.run),
    ("exp5 compiler", exp5_compiler_ablation.run),
    ("exp6 spectral truncation", exp6_spectral_truncation.run),
]
t0 = time.time()
for name, fn in EXPS:
    print(f"\n===== {name} =====", flush=True)
    s = time.time()
    try:
        fn(dict(config))
        print(f"  {name} done in {time.time()-s:.0f}s", flush=True)
    except Exception as e:
        print(f"  {name} FAILED: {e}", flush=True)
        traceback.print_exc()
print(f"\nALL DONE in {time.time()-t0:.0f}s", flush=True)
