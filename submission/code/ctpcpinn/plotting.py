"""Plotting utilities for publication-quality figures."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Use LaTeX-style fonts
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6.5, 4.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_parameter_recovery(true_params: dict, learned_params_cptp: dict,
                            learned_params_baseline: dict, save_path: str):
    """Bar chart comparing true vs learned parameters for CPTP and baseline."""
    _ensure_dir(save_path)
    names = list(true_params.keys())
    n = len(names)

    true_vals = [true_params[k] for k in names]
    cptp_vals = [learned_params_cptp.get(k, 0) for k in names]
    base_vals = [learned_params_baseline.get(k, 0) for k in names]

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(x - width, true_vals, width, label='True')
    ax.bar(x, cptp_vals, width, label='CPTP-PINN')
    ax.bar(x + width, base_vals, width, label='Unconstrained')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Parameter value')
    ax.legend()
    ax.set_title('Parameter Recovery')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_state_fidelity(t_grid: np.ndarray, fid_cptp: np.ndarray,
                        fid_baseline: np.ndarray, save_path: str):
    """Plot state fidelity over time for CPTP vs baseline."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_grid, fid_cptp, '-', linewidth=1.5, label='CPTP-PINN')
    ax.plot(t_grid, fid_baseline, '--', linewidth=1.5, label='Unconstrained')
    ax.set_xlabel('Time')
    ax.set_ylabel('State Fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.set_title('State Fidelity vs. Time')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_sparse_measurement_ablation(fractions: list, errors_cptp: list,
                                      errors_baseline: list, save_path: str,
                                      ylabel: str = 'Mean Trace Distance'):
    """Plot error vs observation fraction."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(fractions, errors_cptp, 'o-', linewidth=1.5, label='CPTP-PINN')
    ax.plot(fractions, errors_baseline, 's--', linewidth=1.5, label='Unconstrained')
    ax.set_xlabel('Observation Fraction')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title('Sparse Measurement Robustness')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_qutrit_leakage(t_grid: np.ndarray, pops_exact: np.ndarray,
                        pops_pred: np.ndarray, save_path: str):
    """Plot qutrit population dynamics (exact vs predicted)."""
    _ensure_dir(save_path)
    d = pops_exact.shape[1]
    labels = [rf'$|{i}\rangle$' for i in range(d)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for i in range(d):
        ax1.plot(t_grid, pops_exact[:, i], linewidth=1.5, label=labels[i])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Population')
    ax1.set_title('Exact')
    ax1.legend()

    for i in range(d):
        ax2.plot(t_grid, pops_pred[:, i], linewidth=1.5, label=labels[i])
    ax2.set_xlabel('Time')
    ax2.set_title('CPTP-PINN Prediction')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_gate_fidelity(t_grid: np.ndarray, fidelities: dict, save_path: str):
    """Plot gate/process fidelity proxy for multiple input states."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for state_name, fid in fidelities.items():
        ax.plot(t_grid, fid, linewidth=1.5, label=state_name)
    ax.set_xlabel('Time')
    ax.set_ylabel('State Fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    ax.set_title('Two-Qubit Gate: State Fidelity Proxy')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_compiler_scaling(dims: list, times_dense: list, times_structured: list,
                          save_path: str):
    """Plot runtime scaling: structured vs dense Liouvillian."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogy(dims, times_dense, 'o-', linewidth=1.5, label='Dense Liouvillian')
    ax.semilogy(dims, times_structured, 's-', linewidth=1.5, label='Structured')
    ax.set_xlabel('Hilbert Space Dimension $d$')
    ax.set_ylabel('Time per batch (s)')
    ax.legend()
    ax.set_title('Compiler Scaling: Dense vs. Structured Residual')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
