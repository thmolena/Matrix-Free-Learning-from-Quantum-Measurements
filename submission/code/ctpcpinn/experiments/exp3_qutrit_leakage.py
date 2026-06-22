"""Experiment 3: qutrit (3-level) leakage reconstruction (multi-seed).

Tests whether a deterministic Fourier time-feature encoding lets the
structure-preserving density network reconstruct the fast coherent-plus-
dissipative dynamics of a transmon-like qutrit, where a plain-MLP density
network suffers from spectral bias. A plain MLP (no Fourier features) and a
Fourier-feature network are trained on identical data over an independent set of
seeds; the reported mean state fidelity is mean +/- 95% CI over the seeds.
Generator-rate identification in this fast regime is harder and is left as
future work (see the sensitivity-Gramian discussion in the main text).
"""

import sys
import os

import numpy as np
import torch
from pathlib import Path

from ctpcpinn.operators import projector
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, PositiveParameter
from ctpcpinn.losses import data_loss, physics_residual_loss, initial_condition_loss
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn.plotting import plot_qutrit_fidelity_distribution
from ctpcpinn.stats import aggregate, fmt_pm
from ctpcpinn.utils import (set_seed, numpy_to_torch_complex,
                            torch_to_numpy_complex, torch_lindblad_rhs)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run(config: dict = None):
    if config is None:
        config = {}
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 3000)
    n_time_points = config.get('n_time_points', 100)
    w_phys = config.get('w_phys', 1.0)
    n_colloc = config.get('n_colloc', 120)
    fourier_features = config.get('fourier_features', 48)
    seeds = config.get('seeds', DEFAULT_SEEDS)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    d = 3
    T = 4.0
    E1 = 2.0 * np.pi * config.get('E1', 2.0)
    E2 = 2.0 * np.pi * config.get('E2', 3.8)
    H = E1 * projector(d, 1, 1) + E2 * projector(d, 2, 2)
    gamma_10, gamma_21, gamma_phi = 0.5, 0.2, 0.1
    L_10 = np.sqrt(gamma_10) * projector(d, 0, 1)
    L_21 = np.sqrt(gamma_21) * projector(d, 1, 2)
    L_phi = np.sqrt(gamma_phi) * np.diag([0, 1, 2]).astype(np.complex128)

    def H_fn(t):
        return H

    def Ls_fn(t):
        return [L_10, L_21, L_phi]

    psi0 = np.array([0.8, 0.5, 0.3], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())
    t_grid = np.linspace(0, T, n_time_points)
    rhos_exact = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)

    P0, P1, P2 = projector(d, 0, 0), projector(d, 1, 1), projector(d, 2, 2)
    X01 = projector(d, 0, 1) + projector(d, 1, 0)
    Y01 = -1j * (projector(d, 0, 1) - projector(d, 1, 0))
    X12 = projector(d, 1, 2) + projector(d, 2, 1)
    Y12 = -1j * (projector(d, 1, 2) - projector(d, 2, 1))
    observables = {'P0': P0, 'P1': P1, 'P2': P2,
                   'X01': X01, 'Y01': Y01, 'X12': X12, 'Y12': Y12}

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in observables.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    rho0_torch = numpy_to_torch_complex(rho0)
    H_torch = numpy_to_torch_complex(H)
    P01_torch = numpy_to_torch_complex(projector(d, 0, 1))
    P12_torch = numpy_to_torch_complex(projector(d, 1, 2))
    diag_torch = numpy_to_torch_complex(np.diag([0, 1, 2]).astype(np.complex128))
    t_max = float(t_grid[-1])
    obs_names = list(obs_torch.keys())

    def lindblad_rhs_fn(rho_batch, params):
        L1 = torch.sqrt(params['gamma_10']()) * P01_torch
        L2 = torch.sqrt(params['gamma_21']()) * P12_torch
        L3 = torch.sqrt(params['gamma_phi']()) * diag_torch
        return torch_lindblad_rhs(rho_batch, H_torch, [L1, L2, L3])

    def train_model(ff, seed):
        set_seed(seed)
        measurements = generate_measurements(rhos_exact, observables,
                                             noise_std=0.02, seed=seed)
        target = torch.stack([torch.tensor(measurements[k], dtype=torch.float32)
                              for k in obs_names], dim=-1)
        model = DensityMatrixNet(d=3, hidden_dim=96, n_layers=4,
                                 fourier_features=ff, fourier_period=T)
        rate_params = {'gamma_10': PositiveParameter(0.3),
                       'gamma_21': PositiveParameter(0.1),
                       'gamma_phi': PositiveParameter(0.05)}
        params = list(model.parameters())
        for p in rate_params.values():
            params.extend(p.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(n_epochs):
            opt.zero_grad()
            rho_pred = model(t_data.clone().requires_grad_(True))
            pred = torch.stack([torch.einsum('bij,ji->b', rho_pred, obs_torch[k]).real
                                for k in obs_names], dim=-1)
            d_loss = data_loss(pred, target)
            t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
            p_loss = physics_residual_loss(model(t_colloc), t_colloc,
                                           lambda rho: lindblad_rhs_fn(rho, rate_params))
            ic_loss = initial_condition_loss(model(torch.zeros(1)), rho0_torch)
            (1.0 * d_loss + w_phys * p_loss + 10.0 * ic_loss).backward()
            opt.step()
        with torch.no_grad():
            rhos_pred = torch_to_numpy_complex(model(t_data))
        return rhos_pred, state_fidelity_over_time(rhos_pred, rhos_exact)

    fids_plain, fids_fourier = [], []
    rep_pred_f = None
    for si, seed in enumerate(seeds):
        if verbose:
            print(f"  seed {seed} ({si+1}/{len(seeds)})...")
        _, fp = train_model(0, seed)
        rhos_pred_f, ff_ = train_model(fourier_features, seed)
        fids_plain.append(float(np.mean(fp)))
        fids_fourier.append(float(np.mean(ff_)))
        if si == 0:
            rep_pred_f = rhos_pred_f

    # Per-seed fidelity distribution: visualizes the across-seed unreliability
    # and that the Fourier advantage does not survive seed averaging.
    plot_qutrit_fidelity_distribution(
        fids_plain, fids_fourier, os.path.join(figures_dir, 'exp3_qutrit_leakage.pdf'))

    table_path = os.path.join(tables_dir, 'exp3_leakage_results.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n\\centering\n")
        f.write(r"\caption{Experiment 3: qutrit reconstruction mean state fidelity with a "
                r"plain versus a Fourier-feature density network. Mean $\pm$ 95\% CI over "
                + f"{len(seeds)} seeds." + r"}" + "\n")
        f.write(r"\label{tab:exp3}" + "\n")
        f.write(r"\begin{tabular}{lc}" + "\n\\toprule\n")
        f.write(r"Density network & Mean state fidelity \\" + "\n\\midrule\n")
        f.write(r"Plain MLP & " + fmt_pm(fids_plain, digits=4) + r" \\" + "\n")
        f.write(r"Fourier time-features & " + fmt_pm(fids_fourier, digits=4) + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

    if verbose:
        print(f"  Plain MLP: {fmt_pm(fids_plain)} | Fourier: {fmt_pm(fids_fourier)}")
        print("  Experiment 3 complete.")

    return {'fids_plain': fids_plain, 'fids_fourier': fids_fourier, 'seeds': seeds}


if __name__ == '__main__':
    run()
