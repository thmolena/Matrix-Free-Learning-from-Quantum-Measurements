"""Journal-neutral vector figure generation for line and bar charts.

Design rules applied here:
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family) and a sans-serif math font
    (``mathtext.fontset = dejavusans``).
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures, drawn by
    ``panel_label``; single-PDF panels that are packed side-by-side in main.tex
    also carry their own letter so each sub-PDF is self-labelled.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
    2011): safe under deuteranopia/protanopia, avoids the red-green trap. Any
    heatmap uses a perceptually uniform map (viridis/cividis).
  * Error bars / shaded 95% confidence-interval bands wherever a mean is plotted;
    the caption states n (five seeds) and that the interval is a 95% CI.
  * Top/right spines removed for an uncluttered frame.

The numbers are deterministic via fixed seeds + single-threaded execution, and
SOURCE_DATE_EPOCH is pinned so the PDF byte-stamp is reproducible.
"""

import os

# Determinism: pin the build epoch BEFORE importing matplotlib so its PDF backend
# stamps a fixed CreationDate, making every figure PDF byte-identical across runs
# (the underlying numbers are already deterministic via fixed seeds + 1 thread).
os.environ.setdefault('SOURCE_DATE_EPOCH', '1700000000')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path

# --- Okabe-Ito colour-blind-safe qualitative palette (Wong, Nat. Methods 2011).
PUBLICATION_PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#000000',  # black
]
# Backwards-compatible alias (older imports / call sites).
OKABE_ITO = PUBLICATION_PALETTE

# Portable column widths in inches.
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20


def apply_publication_style() -> None:
    """Install journal-neutral matplotlib defaults (idempotent)."""
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,   # embed TrueType so text stays selectable/editable
        'ps.fonttype': 42,
        'svg.hashsalt': 'ctpcpinn',  # deterministic element IDs across runs
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',  # keep in-figure math sans-serif
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'lines.linewidth': 1.2,
        'lines.markersize': 3.0,
        'legend.frameon': False,
        'axes.prop_cycle': cycler(color=PUBLICATION_PALETTE),
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })


# Apply once on import so module-level defaults are consistent even for any
# caller that forgets to call apply_publication_style() explicitly.
apply_publication_style()


def panel_label(ax, letter: str, x: float = -0.22, y: float = 1.04) -> None:
    """Bold lower-case panel label in the upper-left."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom', ha='right')


def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment 1 -- single-qubit identification.
# ---------------------------------------------------------------------------
def plot_parameter_recovery(true_params: dict, learned_params_cptp: dict,
                            learned_params_baseline: dict, save_path: str):
    """Bar chart comparing true vs learned parameters for CPTP and baseline."""
    apply_publication_style()
    _ensure_dir(save_path)
    names = list(true_params.keys())
    n = len(names)
    true_vals = [true_params[k] for k in names]
    cptp_vals = [learned_params_cptp.get(k, 0) for k in names]
    base_vals = [learned_params_baseline.get(k, 0) for k in names]
    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.8))
    ax.bar(x - width, true_vals, width, label='True', color=PUBLICATION_PALETTE[7])
    ax.bar(x, cptp_vals, width, label='CPTP-PINN', color=PUBLICATION_PALETTE[0])
    ax.bar(x + width, base_vals, width, label='Unconstrained', color=PUBLICATION_PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Parameter value')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_state_fidelity(t_grid: np.ndarray, fid_cptp: np.ndarray,
                        fid_baseline: np.ndarray, save_path: str):
    """Plot state fidelity over time for CPTP vs baseline."""
    apply_publication_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    ax.plot(t_grid, fid_cptp, '-', label='CPTP-PINN')
    ax.plot(t_grid, fid_baseline, '--', label='Unconstrained')
    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('State fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_parameter_recovery_multiseed(param_order: list, err_by_method: dict,
                                      save_path: str):
    """Grouped bar chart of mean relative parameter error (log-y) with 95% CI
    error bars (n = 5 seeds), one group of bars per parameter. Panel label 'a'."""
    from .stats import aggregate
    apply_publication_style()
    _ensure_dir(save_path)
    methods = list(err_by_method.keys())
    n_p = len(param_order)
    n_m = len(methods)
    width = 0.8 / n_m
    x = np.arange(n_p)
    plabels = {'omega': r'$\omega$', 'Omega': r'$\Omega$',
               'gamma1': r'$\gamma_1$', 'gamma_phi': r'$\gamma_\phi$'}
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    for j, m in enumerate(methods):
        means = [aggregate(err_by_method[m][p])['mean'] for p in param_order]
        cis = [aggregate(err_by_method[m][p])['ci'] for p in param_order]
        ax.bar(x + (j - (n_m - 1) / 2) * width, means, width, label=m,
               yerr=cis, capsize=2, color=PUBLICATION_PALETTE[j % len(PUBLICATION_PALETTE)],
               error_kw={'elinewidth': 0.7, 'capthick': 0.7})
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([plabels.get(p, p) for p in param_order])
    ax.set_ylabel('Relative parameter error')
    ax.legend(ncol=2, handlelength=1.2, columnspacing=1.0)
    # Panel letter for this sub-PDF is placed by the LaTeX minipanel (\textbf{a}).
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_state_fidelity_multiseed(t_grid: np.ndarray, series: dict, save_path: str):
    """Plot state fidelity over time for several named methods (representative
    seed). Panel label 'b'."""
    apply_publication_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    styles = ['-', '--', '-.', ':']
    for k, (name, fid) in enumerate(series.items()):
        ax.plot(t_grid, fid, styles[k % len(styles)],
                color=PUBLICATION_PALETTE[k % len(PUBLICATION_PALETTE)], label=name)
    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('State fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    # Panel letter for this sub-PDF is placed by the LaTeX minipanel (\textbf{b}).
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 2 -- sparse-measurement ablation.
# ---------------------------------------------------------------------------
def plot_sparse_measurement_ablation(fractions: list, errors_cptp: list,
                                     errors_baseline: list, save_path: str,
                                     ylabel: str = 'Mean trace distance'):
    """Plot error vs observation fraction (legacy single-seed entry point)."""
    apply_publication_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    ax.plot(fractions, errors_cptp, 'o-', label='CPTP-PINN', color=PUBLICATION_PALETTE[0])
    ax.plot(fractions, errors_baseline, 's--', label='Unconstrained', color=PUBLICATION_PALETTE[1])
    ax.set_xlabel('Observation fraction')
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_sparse_ablation_multiseed(fractions: list, results_by_method: dict,
                                   save_path: str, ylabel: str = 'Mean trace distance'):
    """Error vs observation fraction for several methods, with shaded 95% CI
    bands (n = 5 seeds). Panel label 'a'."""
    from .stats import aggregate
    apply_publication_style()
    _ensure_dir(save_path)
    markers = {'CPTP-PINN (ours)': 'o-', 'Soft-penalty PINN': '^--', 'Unconstrained': 's:'}
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    for k, (name, per_frac) in enumerate(results_by_method.items()):
        means = np.array([aggregate(per_frac[i])['mean'] for i in range(len(fractions))])
        cis = np.array([aggregate(per_frac[i])['ci'] for i in range(len(fractions))])
        color = PUBLICATION_PALETTE[k % len(PUBLICATION_PALETTE)]
        ax.plot(fractions, means, markers.get(name, 'o-'), color=color, label=name)
        ax.fill_between(fractions, np.maximum(means - cis, 1e-12), means + cis,
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel('Observation fraction')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.legend()
    # Panel letter for this sub-PDF is placed by the LaTeX minipanel (\textbf{a}).
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 3 -- qutrit leakage.
# ---------------------------------------------------------------------------
def plot_qutrit_leakage(t_grid: np.ndarray, pops_exact: np.ndarray,
                        pops_pred: np.ndarray, save_path: str):
    """Plot qutrit population dynamics (exact vs predicted), two panels a,b."""
    apply_publication_style()
    _ensure_dir(save_path)
    d = pops_exact.shape[1]
    labels = [rf'$|{i}\rangle$' for i in range(d)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.8), sharey=True)
    for i in range(d):
        ax1.plot(t_grid, pops_exact[:, i], color=PUBLICATION_PALETTE[i % len(PUBLICATION_PALETTE)],
                 label=labels[i])
    ax1.set_xlabel('Time (arb. units)')
    ax1.set_ylabel('Population')
    ax1.legend()
    panel_label(ax1, 'a')
    for i in range(d):
        ax2.plot(t_grid, pops_pred[:, i], color=PUBLICATION_PALETTE[i % len(PUBLICATION_PALETTE)],
                 label=labels[i])
    ax2.set_xlabel('Time (arb. units)')
    ax2.legend()
    panel_label(ax2, 'b')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_qutrit_fidelity_distribution(fids_plain: list, fids_fourier: list,
                                      save_path: str):
    """Bar chart of mean qutrit fidelity with 95% CI over seeds."""
    from .stats import aggregate
    apply_publication_style()
    _ensure_dir(save_path)
    data = [list(fids_plain), list(fids_fourier)]
    labels = ['Plain MLP', 'Fourier\nfeatures']
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    summaries = [aggregate(vals) for vals in data]
    ax.bar(
        np.arange(1, len(data) + 1),
        [item['mean'] for item in summaries],
        yerr=[item['ci'] for item in summaries],
        capsize=4,
        color=PUBLICATION_PALETTE[:len(data)],
        edgecolor='black',
        linewidth=0.4,
    )
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean state fidelity')
    # Panel letter for this sub-PDF is placed by the LaTeX minipanel (\textbf{b}).
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_qutrit_fidelity_comparison(fids_by_method: dict, save_path: str):
    """Bar chart of mean qutrit fidelity and 95% CI for each method."""
    from .stats import aggregate
    apply_publication_style()
    _ensure_dir(save_path)
    methods = list(fids_by_method.keys())
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    summaries = [aggregate(list(fids_by_method[name])) for name in methods]
    ax.bar(
        np.arange(1, len(methods) + 1),
        [item['mean'] for item in summaries],
        yerr=[item['ci'] for item in summaries],
        capsize=4,
        color=[PUBLICATION_PALETTE[i % len(PUBLICATION_PALETTE)] for i in range(len(methods))],
        edgecolor='black',
        linewidth=0.4,
    )
    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean state fidelity')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_spectral_truncation(levels: list, fidelities: list, oob_weights: list,
                             save_path: str):
    """Symmetric Fourier approximation of the interaction-picture generator:
    mean state fidelity of the propagated level-N truncated generator (left axis)
    and the out-of-band spectral weight / certificate bound (right axis, log) as a
    function of the retained temporal-frequency level N. The full Fourier series
    recovers the exact generator under the stated convergence assumptions. Raw
    finite partial sums preserve Hermiticity and trace but need not be GKSL."""
    apply_publication_style()
    _ensure_dir(save_path)
    levels = np.asarray(levels, dtype=float)
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.8))
    infid = 1.0 - np.asarray(fidelities, dtype=float)
    infid = np.clip(infid, 1e-16, None)
    ax.semilogy(levels, infid, 'o-', color=PUBLICATION_PALETTE[0],
                label='mean infidelity $1-F$')
    ax.set_xlabel(r'Fourier-truncation level $N$ (symmetric band)')
    ax.set_ylabel('Mean infidelity $1-F$')
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.semilogy(levels, np.clip(oob_weights, 1e-16, None), 's--',
                 color=PUBLICATION_PALETTE[1], label='out-of-band weight')
    ax2.set_ylabel('Out-of-band spectral weight', color=PUBLICATION_PALETTE[1])
    ax2.tick_params(axis='y', colors=PUBLICATION_PALETTE[1])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 4 -- two-qubit dissipative gate.
# ---------------------------------------------------------------------------
def plot_gate_fidelity(t_grid: np.ndarray, fidelities: dict, save_path: str):
    """Plot gate/process fidelity proxy for multiple input states."""
    apply_publication_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    for k, (state_name, fid) in enumerate(fidelities.items()):
        ax.plot(t_grid, fid, color=PUBLICATION_PALETTE[k % len(PUBLICATION_PALETTE)], label=state_name)
    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('State fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_gate_fidelity_multiseed(t_grid: np.ndarray, fidelities: dict, save_path: str):
    """Two-qubit gate: per-state fidelity over time (representative seed).
    Panel label 'c'."""
    apply_publication_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    for k, (state_name, fid) in enumerate(fidelities.items()):
        ax.plot(t_grid, fid, color=PUBLICATION_PALETTE[k % len(PUBLICATION_PALETTE)], label=state_name)
    ax.set_xlabel('Time (arb. units)')
    ax.set_ylabel('State fidelity')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', ncol=2)
    # Panel letter for this sub-PDF is placed by the LaTeX minipanel (\textbf{c}).
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


# ---------------------------------------------------------------------------
# Experiment 5 -- dense-vs-structured compiler scaling.
# ---------------------------------------------------------------------------
def plot_compiler_scaling(dims: list, times_dense: list, times_structured: list,
                          save_path: str):
    """Plot runtime scaling: structured vs dense Liouvillian (median over
    repeated timings, log-y)."""
    apply_publication_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.8))
    ax.semilogy(dims, times_dense, 'o-', label='Dense Liouvillian', color=PUBLICATION_PALETTE[1])
    ax.semilogy(dims, times_structured, 's-', label='Structured', color=PUBLICATION_PALETTE[0])
    ax.set_xlabel(r'Hilbert-space dimension $d$')
    ax.set_ylabel('Time per batch (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
