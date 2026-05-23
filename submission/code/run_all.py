#!/usr/bin/env python
"""Run all experiments for CPTP-Compiler-PINNs paper.

Usage:
    python submission/code/run_all.py           # default (moderate) config
    python submission/code/run_all.py --quick   # fast smoke test
    python submission/code/run_all.py --full    # publication-quality (slower)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Ensure the code directory is on the path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import QUICK_CONFIG, FULL_CONFIG, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run CPTP-Compiler-PINNs experiments")
    parser.add_argument('--quick', action='store_true', help='Fast smoke test')
    parser.add_argument('--full', action='store_true', help='Full publication run')
    args = parser.parse_args()

    if args.quick:
        config = QUICK_CONFIG.copy()
        print("Running in QUICK mode (smoke test)...")
    elif args.full:
        config = FULL_CONFIG.copy()
        print("Running in FULL mode (publication-quality)...")
    else:
        config = DEFAULT_CONFIG.copy()
        print("Running in DEFAULT mode...")

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(CODE_DIR))
    submission_dir = os.path.join(project_root, 'submission')
    config['figures_dir'] = os.path.join(submission_dir, 'figures')
    config['tables_dir'] = os.path.join(submission_dir, 'tables')

    Path(config['figures_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['tables_dir']).mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Import experiments
    from experiments import exp1_qubit_inverse
    from experiments import exp2_sparse_measurements
    from experiments import exp3_qutrit_leakage
    from experiments import exp4_two_qubit_gate
    from experiments import exp5_compiler_ablation

    experiments = [
        ("Experiment 1: Qubit Inverse Problem", exp1_qubit_inverse.run),
        ("Experiment 2: Sparse Measurements", exp2_sparse_measurements.run),
        ("Experiment 3: Qutrit Leakage", exp3_qutrit_leakage.run),
        ("Experiment 4: Two-Qubit Gate", exp4_two_qubit_gate.run),
        ("Experiment 5: Compiler Ablation", exp5_compiler_ablation.run),
    ]

    results = {}
    for name, run_fn in experiments:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        exp_start = time.time()
        try:
            result = run_fn(config)
            results[name] = result
            elapsed = time.time() - exp_start
            print(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  All experiments completed in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  Figures saved to: {config['figures_dir']}")
    print(f"  Tables saved to: {config['tables_dir']}")

    return results


if __name__ == '__main__':
    main()
