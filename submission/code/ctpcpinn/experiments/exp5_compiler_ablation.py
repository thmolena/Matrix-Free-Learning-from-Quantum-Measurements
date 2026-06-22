"""Experiment 5: Compiler ablation — dense vs structured Liouvillian."""

import sys
import os

import numpy as np
import time
from pathlib import Path

from ctpcpinn.operators import (pauli_x, pauli_y, pauli_z, identity,
                                 sigma_minus, kron_n, destroy)
from ctpcpinn.compiler import QuantumModelIR, CompiledLindbladModel, benchmark_residual_modes
from ctpcpinn.metrics import memory_estimate_dense_vs_structured, WallClockTimer
from ctpcpinn.plotting import plot_compiler_scaling
from ctpcpinn.utils import set_seed


def _make_system(d: int):
    """Create a d-dimensional test system with random H and a few Lindblad ops."""
    rng = np.random.default_rng(42)
    # Random Hermitian Hamiltonian
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (A + A.conj().T) / 2

    # m = min(d, 3) Lindblad operators (lowering-like)
    m = min(d, 3)
    Ls = []
    for k in range(m):
        L = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        L *= 0.1  # keep rates moderate
        Ls.append(L)

    return H, Ls, m


def run(config: dict = None):
    """Run experiment 5: compiler scaling ablation."""
    if config is None:
        config = {}

    set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    verbose = config.get('verbose', True)
    n_batch = config.get('n_batch', 100)
    # Timing is noisy; use many repeats and report the median (warm-up excluded).
    n_repeats = config.get('n_repeats_timing', max(config.get('n_repeats', 5), 30))

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    dims = [2, 3, 4, 6, 8]
    times_dense = []
    times_structured = []
    memory_data = []

    for d in dims:
        if verbose:
            print(f"  Benchmarking d={d}...")

        H, Ls, m = _make_system(d)
        params = {}

        def h_fn(params, t, H=H):
            return H

        def ls_fn(params, t, Ls=Ls):
            return Ls

        ir = QuantumModelIR(
            dimension=d,
            hamiltonian_fn=h_fn,
            lindblad_ops_fn=ls_fn,
            name=f"random_d{d}"
        )

        result = benchmark_residual_modes(ir, params, n_points=n_batch, n_repeats=n_repeats)
        times_dense.append(result['dense']['median_time_s'])
        times_structured.append(result['structured']['median_time_s'])

        mem = memory_estimate_dense_vs_structured(d, m)
        memory_data.append(mem)

        if verbose:
            print(f"    Dense: {result['dense']['median_time_s']:.4f}s, "
                  f"Structured: {result['structured']['median_time_s']:.4f}s, "
                  f"Ratio: {result['dense']['median_time_s'] / max(result['structured']['median_time_s'], 1e-10):.2f}x")

    # Plot
    plot_compiler_scaling(dims, times_dense, times_structured,
                          os.path.join(figures_dir, 'exp5_compiler_scaling.pdf'))

    # Table
    table_path = os.path.join(tables_dir, 'exp5_runtime_memory.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Experiment 5: compiler runtime and memory scaling. Times are the "
                r"median residual-evaluation wall time over " + f"{n_repeats}" +
                r" repeats (warm-up excluded) for a batch of " + f"{n_batch}" +
                r" states; memory is the analytic storage for the dense $d^2\times d^2$ "
                r"Liouvillian versus the structured operators. CPU, single machine.}" + "\n")
        f.write(r"\label{tab:exp5}" + "\n")
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$d$ & Dense (s) & Structured (s) & Speedup & Dense (MB) & Structured (MB) \\" + "\n")
        f.write(r"\midrule" + "\n")
        for i, d in enumerate(dims):
            speedup = times_dense[i] / max(times_structured[i], 1e-10)
            f.write(f"{d} & {times_dense[i]:.4f} & {times_structured[i]:.4f} & "
                    f"{speedup:.1f}$\\times$ & "
                    f"{memory_data[i]['dense_MB']:.3f} & {memory_data[i]['structured_MB']:.5f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 5 complete.")

    return {'dims': dims, 'times_dense': times_dense, 'times_structured': times_structured,
            'memory_data': memory_data}


if __name__ == '__main__':
    run()
