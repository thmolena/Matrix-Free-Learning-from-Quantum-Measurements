"""Figure generation, styled to Nature Machine Intelligence (NMI) conventions.

Design rules applied here (Nature Portfolio artwork & formatting guidance):
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
  * Top/right spines removed for an uncluttered Nature-style frame.

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
NMI_PALETTE = [
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
OKABE_ITO = NMI_PALETTE

# Nature column widths in inches (single column 89 mm, double column 183 mm).
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent)."""
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
        'axes.prop_cycle': cycler(color=NMI_PALETTE),
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })


# Apply once on import so module-level defaults are NMI-compliant even for any
# caller that forgets to call apply_nmi_style() explicitly.
apply_nmi_style()


def panel_label(ax, letter: str, x: float = -0.22, y: float = 1.04) -> None:
    """Bold lower-case panel label in the upper-left (Nature convention)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom', ha='right')


def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 -- method-overview schematic (NMI 'Figure 1' convention).
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc, ec='#222222', fontsize=6.6):
    """Draw a rounded method-schematic box with centred text. Returns the
    midpoints of the right and left edges (for arrow attachment)."""
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle='round,pad=0.010,rounding_size=0.02',
        linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fontsize, zorder=5)
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1, color='#444444'):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color=color,
                                shrinkA=2, shrinkB=2))


def plot_schematic(save_path: str) -> str:
    """Programmatic method-overview schematic of the CPTP-Compiler-PINN pipeline.

    Left-to-right: a parameterized open-quantum (Lindblad/GKSL) specification is
    lowered by a matrix-free compiler to a residual kernel; a structure-
    preserving residual PINN (Cholesky density net + CPTP generator) is trained
    against the physics residual and the data; physical metrics are read out.
    No in-plot title -- the description is in the LaTeX caption.
    """
    apply_nmi_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    blue, orange, teal, grey, green = (
        '#D6E6F2', '#FBE6D4', '#D6EFE3', '#ECECEC', '#E6F2D6')

    # Top row: the two structure-preserving network branches feeding the loss.
    y_top, h = 0.62, 0.26
    (r_dens, _) = _box(ax, (0.010, y_top), 0.205, h,
                       'density net\n$\\rho_\\phi=A_\\phi A_\\phi^\\dagger/\\mathrm{Tr}(\\cdot)$\n'
                       'Hermitian, PSD, trace 1', blue)
    (r_gen, _) = _box(ax, (0.010, 0.12), 0.205, h,
                      'CPTP generator\n'
                      '$\\gamma_k=\\mathrm{softplus}(\\xi_k)\\geq 0$\n'
                      '$C=BB^\\dagger$ PSD', orange)

    # Middle: parameterized Lindblad/GKSL spec -> matrix-free compiler.
    (r_spec, l_spec) = _box(ax, (0.270, 0.37), 0.205, h,
                            'open-quantum spec\n$\\dot\\rho=\\mathcal{L}_\\Theta[\\rho]$\n'
                            '(Lindblad / GKSL)', green)
    (r_comp, l_comp) = _box(ax, (0.530, 0.37), 0.190, h,
                            'matrix-free\ncompiler\ndense $\\,|\\,$ structured', teal)

    # Right: physics-informed loss -> outputs / metrics.
    (r_loss, l_loss) = _box(ax, (0.775, 0.37), 0.215, h,
                            'physics-informed loss\n'
                            '$\\mathcal{J}_{\\rm data}+\\mathcal{J}_{\\rm phys}+\\mathcal{J}_0$\n'
                            '$\\to$ fidelity, trace dist.,\npositivity, residual', grey)

    mid_y = 0.50
    # density/generator branches feed the spec/compiler chain.
    _arrow(ax, r_dens, (0.270, mid_y + 0.02))
    _arrow(ax, r_gen, (0.270, mid_y - 0.02))
    _arrow(ax, r_spec, l_comp)
    _arrow(ax, r_comp, l_loss)

    ax.text(0.50, 0.955,
            'structure-preserving residual PINN for open quantum dynamics',
            ha='center', va='center', fontsize=7.2, color='#333333')
    ax.text(0.50, 0.035,
            'dense $\\leftrightarrow$ structured residual agree to floating-point precision'
            '   $\\bullet$   physicality guaranteed at every step (Theorem 1)',
            ha='center', va='center', fontsize=6.0, color='#555555')
    fig.savefig(save_path, format='pdf')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Experiment 1 -- single-qubit identification.
# ---------------------------------------------------------------------------
def plot_parameter_recovery(true_params: dict, learned_params_cptp: dict,
                            learned_params_baseline: dict, save_path: str):
    """Bar chart comparing true vs learned parameters for CPTP and baseline."""
    apply_nmi_style()
    _ensure_dir(save_path)
    names = list(true_params.keys())
    n = len(names)
    true_vals = [true_params[k] for k in names]
    cptp_vals = [learned_params_cptp.get(k, 0) for k in names]
    base_vals = [learned_params_baseline.get(k, 0) for k in names]
    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.8))
    ax.bar(x - width, true_vals, width, label='True', color=NMI_PALETTE[7])
    ax.bar(x, cptp_vals, width, label='CPTP-PINN', color=NMI_PALETTE[0])
    ax.bar(x + width, base_vals, width, label='Unconstrained', color=NMI_PALETTE[1])
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
    apply_nmi_style()
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
    apply_nmi_style()
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
               yerr=cis, capsize=2, color=NMI_PALETTE[j % len(NMI_PALETTE)],
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
    apply_nmi_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    styles = ['-', '--', '-.', ':']
    for k, (name, fid) in enumerate(series.items()):
        ax.plot(t_grid, fid, styles[k % len(styles)],
                color=NMI_PALETTE[k % len(NMI_PALETTE)], label=name)
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
    apply_nmi_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    ax.plot(fractions, errors_cptp, 'o-', label='CPTP-PINN', color=NMI_PALETTE[0])
    ax.plot(fractions, errors_baseline, 's--', label='Unconstrained', color=NMI_PALETTE[1])
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
    apply_nmi_style()
    _ensure_dir(save_path)
    markers = {'CPTP-PINN (ours)': 'o-', 'Soft-penalty PINN': '^--', 'Unconstrained': 's:'}
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    for k, (name, per_frac) in enumerate(results_by_method.items()):
        means = np.array([aggregate(per_frac[i])['mean'] for i in range(len(fractions))])
        cis = np.array([aggregate(per_frac[i])['ci'] for i in range(len(fractions))])
        color = NMI_PALETTE[k % len(NMI_PALETTE)]
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
    apply_nmi_style()
    _ensure_dir(save_path)
    d = pops_exact.shape[1]
    labels = [rf'$|{i}\rangle$' for i in range(d)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.8), sharey=True)
    for i in range(d):
        ax1.plot(t_grid, pops_exact[:, i], color=NMI_PALETTE[i % len(NMI_PALETTE)],
                 label=labels[i])
    ax1.set_xlabel('Time (arb. units)')
    ax1.set_ylabel('Population')
    ax1.legend()
    panel_label(ax1, 'a')
    for i in range(d):
        ax2.plot(t_grid, pops_pred[:, i], color=NMI_PALETTE[i % len(NMI_PALETTE)],
                 label=labels[i])
    ax2.set_xlabel('Time (arb. units)')
    ax2.legend()
    panel_label(ax2, 'b')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_qutrit_fidelity_distribution(fids_plain: list, fids_fourier: list,
                                      save_path: str):
    """Per-seed reconstruction fidelity for plain vs Fourier qutrit networks,
    with the mean and 95% CI overlaid (n = 5 seeds). Visualizes the across-seed
    unreliability. Panel label 'b'."""
    from .stats import aggregate
    apply_nmi_style()
    _ensure_dir(save_path)
    data = [list(fids_plain), list(fids_fourier)]
    labels = ['Plain MLP', 'Fourier\nfeatures']
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    for i, (vals, name) in enumerate(zip(data, labels)):
        x = np.full(len(vals), i + 1, dtype=float)
        jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.03  # deterministic
        ax.scatter(x + jit, vals, s=22, alpha=0.7, zorder=3,
                   color=NMI_PALETTE[i], label='per-seed' if i == 0 else None)
        a = aggregate(vals)
        ax.errorbar(i + 1, a['mean'], yerr=a['ci'], fmt='_', color='k',
                    markersize=20, capsize=4, elinewidth=1.2, zorder=4,
                    label='mean $\\pm$ 95% CI' if i == 0 else None)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean state fidelity')
    ax.legend(loc='lower left')
    # Panel letter for this sub-PDF is placed by the LaTeX minipanel (\textbf{b}).
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_qutrit_fidelity_comparison(fids_by_method: dict, save_path: str):
    """Per-seed reconstruction fidelity for several qutrit density networks
    (plain MLP, Fourier features, spectral frame), with the mean and 95% CI
    overlaid (n = 5 seeds). The spectral-frame (interaction-picture) network is
    the proposed cure for spectral bias. Panel label 'b'."""
    from .stats import aggregate
    apply_nmi_style()
    _ensure_dir(save_path)
    methods = list(fids_by_method.keys())
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    for i, name in enumerate(methods):
        vals = list(fids_by_method[name])
        x = np.full(len(vals), i + 1, dtype=float)
        jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.04
        ax.scatter(x + jit, vals, s=20, alpha=0.7, zorder=3,
                   color=NMI_PALETTE[i % len(NMI_PALETTE)],
                   label='per-seed' if i == 0 else None)
        a = aggregate(vals)
        ax.errorbar(i + 1, a['mean'], yerr=a['ci'], fmt='_', color='k',
                    markersize=18, capsize=4, elinewidth=1.2, zorder=4,
                    label='mean $\\pm$ 95% CI' if i == 0 else None)
    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel('Mean state fidelity')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_spectral_truncation(levels: list, fidelities: list, oob_weights: list,
                             save_path: str):
    """Operator-system spectral truncation of the interaction-picture generator:
    mean state fidelity of the propagated level-N truncated generator (left axis)
    and the out-of-band spectral weight / certificate bound (right axis, log) as a
    function of the retained Bohr-band level N. The full band recovers the exact
    generator; N=0 is the secular (Davies) generator and N=1 the rotating-wave
    approximation."""
    apply_nmi_style()
    _ensure_dir(save_path)
    levels = np.asarray(levels, dtype=float)
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.8))
    infid = 1.0 - np.asarray(fidelities, dtype=float)
    infid = np.clip(infid, 1e-16, None)
    ax.semilogy(levels, infid, 'o-', color=NMI_PALETTE[0],
                label='mean infidelity $1-F$')
    ax.set_xlabel(r'Spectral-truncation level $N$ (retained Bohr band)')
    ax.set_ylabel('Mean infidelity $1-F$')
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.semilogy(levels, np.clip(oob_weights, 1e-16, None), 's--',
                 color=NMI_PALETTE[1], label='out-of-band weight')
    ax2.set_ylabel('Out-of-band spectral weight', color=NMI_PALETTE[1])
    ax2.tick_params(axis='y', colors=NMI_PALETTE[1])
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
    apply_nmi_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    for k, (state_name, fid) in enumerate(fidelities.items()):
        ax.plot(t_grid, fid, color=NMI_PALETTE[k % len(NMI_PALETTE)], label=state_name)
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
    apply_nmi_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.8))
    for k, (state_name, fid) in enumerate(fidelities.items()):
        ax.plot(t_grid, fid, color=NMI_PALETTE[k % len(NMI_PALETTE)], label=state_name)
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
    apply_nmi_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.8))
    ax.semilogy(dims, times_dense, 'o-', label='Dense Liouvillian', color=NMI_PALETTE[1])
    ax.semilogy(dims, times_structured, 's-', label='Structured', color=NMI_PALETTE[0])
    ax.set_xlabel(r'Hilbert-space dimension $d$')
    ax.set_ylabel('Time per batch (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
