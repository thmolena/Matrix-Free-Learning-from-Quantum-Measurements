"""Command-line reproduction of the manuscript's experimental and simulation evidence.

Installed as the console script ``ctpcpinn-reproduce`` (see pyproject.toml). It
runs the experimental Ramsey analysis followed by six controlled studies and
writes LaTeX tables, PDF line/bar figures, and evidence JSON into an output
directory you choose.

Examples
--------
    ctpcpinn-reproduce                       # full run -> ./ctpcpinn_results
    ctpcpinn-reproduce --output-dir out      # full run -> out/{tables,figures}
    ctpcpinn-reproduce --quick               # fast smoke test (2 seeds)
    ctpcpinn-reproduce --experiment exp1     # a single experiment

Determinism
-----------
Every experiment is repeated over the FIXED seeds [0, 1, 2, 3, 4] and reports
mean +/- 95% confidence interval. With the pinned dependencies (see
pyproject.toml) and single-threaded execution, the table values are reproduced
exactly. The wall-clock TIMES in experiment 5 are machine dependent (their ratios
and the analytic memory columns are not).
"""

import os

# OpenMP / threading safety MUST be configured before importing numpy/torch.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import time
from pathlib import Path

import torch

from .config import QUICK_CONFIG, FULL_CONFIG

EXPERIMENTS = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6']
_TITLES = {
    'exp1': 'single-qubit identification',
    'exp2': 'sparse-measurement ablation',
    'exp3': 'fast qutrit reconstruction (spectral frame)',
    'exp4': 'two-qubit gate generalization',
    'exp5': 'dense-vs-structured compiler',
    'exp6': 'symmetric Fourier generator approximation',
}


def _load(name):
    import importlib
    return importlib.import_module(f'ctpcpinn.experiments.{name}_' + {
        'exp1': 'qubit_inverse', 'exp2': 'sparse_measurements',
        'exp3': 'qutrit_leakage', 'exp4': 'two_qubit_gate',
        'exp5': 'compiler_ablation', 'exp6': 'spectral_truncation'}[name])


def _versions():
    import numpy, scipy, matplotlib
    return (f"python {os.sys.version.split()[0]} | torch {torch.__version__} | "
            f"numpy {numpy.__version__} | scipy {scipy.__version__} | "
            f"matplotlib {matplotlib.__version__}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='ctpcpinn-reproduce',
        description='Reproduce the manuscript tables and figures.')
    p.add_argument('--output-dir', default='ctpcpinn_results',
                   help='where to write tables/ and figures/ (default: ./ctpcpinn_results)')
    p.add_argument('--in-place', action='store_true',
                   help='write directly into code/manuscript_assets/{tables,figures}')
    p.add_argument('--quick', action='store_true', help='fast smoke test (NOT publication quality)')
    p.add_argument('--experiment', choices=EXPERIMENTS, default=None,
                   help='run only this experiment (default: all)')
    p.add_argument('--threads', type=int, default=1,
                   help='BLAS/OMP threads (default 1; >1 only if a single OpenMP runtime is confirmed)')
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    config = dict(QUICK_CONFIG if args.quick else FULL_CONFIG)

    out = (Path(__file__).resolve().parents[1] / 'manuscript_assets'
           if args.in_place else Path(args.output_dir).resolve())
    config['figures_dir'] = str(out / 'figures')
    config['tables_dir'] = str(out / 'tables')
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (out / 'tables').mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CPTP-Compiler-PINNs reproduction")
    print(f"  env:     {_versions()}")
    print(f"  mode:    {'QUICK (smoke test)' if args.quick else 'FULL (reproduces the manuscript)'}")
    print(f"  threads: {torch.get_num_threads()}   seeds: {config.get('seeds')}   "
          f"epochs: {config.get('n_epochs')}")
    print(f"  output:  {out}")
    print("=" * 70)

    if not args.experiment:
        from .real_data import reproduce as reproduce_real_data
        print("\n--- experimental Ramsey evidence ---")
        reproduce_real_data(out)

    todo = [args.experiment] if args.experiment else EXPERIMENTS
    t0 = time.time()
    for name in todo:
        print(f"\n--- {name}: {_TITLES[name]} ---")
        exp_start = time.time()
        _load(name).run(dict(config))
        print(f"    done in {time.time() - exp_start:.1f}s")

    print("\n" + "=" * 70)
    print(f"Finished {len(todo)} experiment(s) in {time.time() - t0:.1f}s")
    print(f"Tables  -> {out / 'tables'}")
    print(f"Figures -> {out / 'figures'}")
    print("These are the exact table/figure files used in the manuscript "
          "(code/manuscript_assets/tables and code/manuscript_assets/figures).")
    print("=" * 70)


if __name__ == '__main__':
    main()
