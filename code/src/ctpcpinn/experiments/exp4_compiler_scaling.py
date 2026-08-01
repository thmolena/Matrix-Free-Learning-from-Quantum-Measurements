"""Experiment 4: Compiler scaling — dense vs structured Liouvillian benchmarks.

This experiment compares dense superoperator evaluation with structured
matrix-free residual evaluation on random dense Lindblad systems across
increasing Hilbert-space dimensions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from pathlib import Path


def _make_random_system(d, m, rng):
    """Create a random d-dimensional Lindblad system with m jump operators."""
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (A + A.conj().T) / 2
    Ls = []
    for _ in range(m):
        L = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        L *= 0.1
        Ls.append(L)
    return H, Ls


def _build_dense_liouvillian(H, Ls, rates):
    """Build the full d^2 x d^2 Liouvillian superoperator (column-major convention)."""
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)
    L_super = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L, gamma in zip(Ls, rates):
        Ldag_L = L.conj().T @ L
        L_super += gamma * (
            np.kron(L.conj(), L)
            - 0.5 * np.kron(I, Ldag_L)
            - 0.5 * np.kron(Ldag_L.T, I)
        )
    return L_super


def _eval_dense_residual(L_super, rho):
    """Evaluate residual using dense superoperator matvec (column-major)."""
    d = int(np.sqrt(L_super.shape[0]))
    rho_vec = rho.flatten(order='F')
    drho_vec = L_super @ rho_vec
    return drho_vec.reshape(d, d, order='F')


def _eval_structured_residual(H, Ls, rates, rho):
    """Evaluate L[rho] directly without forming full superoperator."""
    drho = -1j * (H @ rho - rho @ H)
    for L, gamma in zip(Ls, rates):
        Ldag_L = L.conj().T @ L
        drho += gamma * (L @ rho @ L.conj().T - 0.5 * (Ldag_L @ rho + rho @ Ldag_L))
    return drho


def run(config: dict = None):
    """Run experiment 4: compiler scaling benchmark."""
    if config is None:
        config = {}

    np.random.seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'manuscript_assets/figures')
    tables_dir = config.get('tables_dir', 'manuscript_assets/tables')
    verbose = config.get('verbose', True)
    n_repeats = config.get('n_repeats', 5)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Dimensions and jump-operator counts matching the paper
    dims = [2, 4, 8, 16, 24]
    ms = [2, 4, 4, 4, 4]

    results = []

    for d, m in zip(dims, ms):
        if verbose:
            print(f"  Benchmarking d={d}, m={m}...")

        H, Ls = _make_random_system(d, m, rng)
        rates = [1.0] * m

        # Random test state (valid density matrix)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho_test = A @ A.conj().T
        rho_test /= np.trace(rho_test)

        # Dense: build and evaluate
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            L_super = _build_dense_liouvillian(H, Ls, rates)
            _ = _eval_dense_residual(L_super, rho_test)
        dense_total = (time.perf_counter() - t0) / n_repeats

        # Structured: evaluate directly
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            _ = _eval_structured_residual(H, Ls, rates, rho_test)
        struct_total = (time.perf_counter() - t0) / n_repeats

        # Memory estimates (bytes -> MB, using 1e6 convention for paper consistency)
        dense_mem_bytes = 2 * (d ** 4) * 8  # complex128 d^2 x d^2 matrix
        struct_mem_bytes = 2 * (m + 1) * (d ** 2) * 8  # H + m Lindblad ops, complex128
        dense_mem_mb = dense_mem_bytes / 1e6
        struct_mem_mb = struct_mem_bytes / 1e6

        # Verify equivalence
        res_dense = _eval_dense_residual(L_super, rho_test)
        res_struct = _eval_structured_residual(H, Ls, rates, rho_test)
        rel_err = np.linalg.norm(res_dense - res_struct) / max(np.linalg.norm(res_dense), 1e-15)

        results.append({
            'd': d, 'm': m,
            'dense_mem_mb': dense_mem_mb,
            'struct_mem_mb': struct_mem_mb,
            'dense_ms': dense_total * 1000,
            'struct_ms': struct_total * 1000,
            'rel_err': rel_err,
        })

        if verbose:
            speedup = dense_total / max(struct_total, 1e-15)
            print(f"    Dense: {dense_total*1000:.4f}ms, Struct: {struct_total*1000:.4f}ms, "
                  f"Speedup: {speedup:.2f}x, RelErr: {rel_err:.1e}")

    # Write data file (matching paper's filecontents format)
    dat_path = os.path.join(tables_dir, 'cptp_qst_compiler_scaling.dat')
    with open(dat_path, 'w') as f:
        f.write("d denseMem structMem denseMs structMs\n")
        for r in results:
            f.write(f"{r['d']} {r['dense_mem_mb']:.6g} {r['struct_mem_mb']:.6g} "
                    f"{r['dense_ms']:.7g} {r['struct_ms']:.7g}\n")

    # Write LaTeX table
    table_path = os.path.join(tables_dir, 'exp4_compiler_scaling.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Dense-superoperator versus structured residual evaluation.}" + "\n")
        f.write(r"\label{tab:compiler_scaling}" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{lccccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"$d$ & $m$ & Dense MB & Structured MB & Dense eval ms & "
                r"Structured eval ms & Eval speedup & Relative error \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in results:
            speedup = r['dense_ms'] / max(r['struct_ms'], 1e-15)
            f.write(f"{r['d']} & {r['m']} & {r['dense_mem_mb']:.3f} & "
                    f"{r['struct_mem_mb']:.3f} & {r['dense_ms']:.3f} & "
                    f"{r['struct_ms']:.3f} & {speedup:.2f} & "
                    f"{r['rel_err']:.1e} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 4 complete.")

    return results


if __name__ == '__main__':
    run()
