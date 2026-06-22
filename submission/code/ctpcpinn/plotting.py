"""Plotting utilities for publication-quality figures."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Nature/NMI figure style: sans-serif, colorblind-safe palette, embedded fonts,
# no on-figure titles (panel descriptions go in the caption), no gridlines.
# Okabe-Ito qualitative palette (colorblind-safe; avoids red/green confusion).
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#CC79A7',
             '#56B4E9', '#D55E00', '#F0E442', '#000000']
plt.rcParams.update({
    'font.size': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'axes.linewidth': 0.6,
    'axes.prop_cycle': plt.cycler(color=OKABE_ITO),
    'axes.grid': False,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,   # embed editable TrueType text
    'ps.fonttype': 42,
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
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
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
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
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
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
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


def plot_state_fidelity_multiseed(t_grid: np.ndarray, series: dict, save_path: str):
    """Plot state fidelity over time for several named methods (one seed each)."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    styles = ['-', '--', '-.', ':']
    for k, (name, fid) in enumerate(series.items()):
        ax.plot(t_grid, fid, styles[k % len(styles)], linewidth=1.5, label=name)
    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('State fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_parameter_recovery_multiseed(param_order: list, err_by_method: dict,
                                      save_path: str):
    """Grouped bar chart of mean relative parameter error (log-y) with 95% CI
    error bars, one group of bars per parameter."""
    from .stats import aggregate
    _ensure_dir(save_path)
    methods = list(err_by_method.keys())
    n_p = len(param_order)
    n_m = len(methods)
    width = 0.8 / n_m
    x = np.arange(n_p)
    plabels = {'omega': r'$\omega$', 'Omega': r'$\Omega$',
               'gamma1': r'$\gamma_1$', 'gamma_phi': r'$\gamma_\phi$'}

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for j, m in enumerate(methods):
        means = [aggregate(err_by_method[m][p])['mean'] for p in param_order]
        cis = [aggregate(err_by_method[m][p])['ci'] for p in param_order]
        ax.bar(x + (j - (n_m - 1) / 2) * width, means, width, label=m,
               yerr=cis, capsize=2, error_kw={'elinewidth': 0.8})
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([plabels.get(p, p) for p in param_order])
    ax.set_ylabel('Relative parameter error')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_sparse_ablation_multiseed(fractions: list, results_by_method: dict,
                                   save_path: str, ylabel: str = 'Mean trace distance'):
    """Error vs observation fraction for several methods, with 95% CI bands."""
    from .stats import aggregate
    _ensure_dir(save_path)
    markers = {'CPTP-PINN (ours)': 'o-', 'Soft-penalty PINN': '^--', 'Unconstrained': 's:'}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for name, per_frac in results_by_method.items():
        means = np.array([aggregate(per_frac[i])['mean'] for i in range(len(fractions))])
        cis = np.array([aggregate(per_frac[i])['ci'] for i in range(len(fractions))])
        ax.plot(fractions, means, markers.get(name, 'o-'), linewidth=1.5, label=name)
        ax.fill_between(fractions, means - cis, means + cis, alpha=0.2)
    ax.set_xlabel('Observation fraction')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_qutrit_fidelity_distribution(fids_plain: list, fids_fourier: list,
                                      save_path: str):
    """Per-seed reconstruction fidelity for plain vs Fourier qutrit networks,
    with mean and 95% CI overlaid. Visualizes the across-seed unreliability."""
    from .stats import aggregate
    _ensure_dir(save_path)
    data = [list(fids_plain), list(fids_fourier)]
    labels = ['Plain MLP', 'Fourier features']
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for i, (vals, name) in enumerate(zip(data, labels)):
        x = np.full(len(vals), i + 1, dtype=float)
        # small deterministic jitter (no RNG) for visibility
        jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.03
        ax.scatter(x + jit, vals, s=28, alpha=0.7, zorder=3,
                   label='per-seed' if i == 0 else None)
        a = aggregate(vals)
        ax.errorbar(i + 1, a['mean'], yerr=a['ci'], fmt='_', color='k',
                    markersize=22, capsize=5, elinewidth=1.4, zorder=4,
                    label='mean $\\pm$ 95% CI' if i == 0 else None)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean state fidelity')
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_gate_fidelity_multiseed(t_grid: np.ndarray, fidelities: dict, save_path: str):
    """Two-qubit gate: per-state fidelity over time (representative seed)."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for state_name, fid in fidelities.items():
        ax.plot(t_grid, fid, linewidth=1.5, label=state_name)
    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('State fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_compiler_scaling(dims: list, times_dense: list, times_structured: list,
                          save_path: str):
    """Plot runtime scaling: structured vs dense Liouvillian."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.semilogy(dims, times_dense, 'o-', linewidth=1.5, label='Dense Liouvillian')
    ax.semilogy(dims, times_structured, 's-', linewidth=1.5, label='Structured')
    ax.set_xlabel('Hilbert Space Dimension $d$')
    ax.set_ylabel('Time per batch (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
