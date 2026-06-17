"""Experiment 3: qutrit (3-level) leakage reconstruction.

Demonstrates that Fourier time-features let the structure-preserving density
network reconstruct the coherent-plus-dissipative dynamics of a fast multi-level
(qutrit) system, where a plain-MLP density network suffers from spectral bias.
Generator (rate) identification in this fast regime is harder and is left as
future work; see the discussion of the sensitivity Gramian in the main text.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from pathlib import Path

from ctpcpinn.operators import projector
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, PositiveParameter
from ctpcpinn.losses import data_loss, physics_residual_loss, initial_condition_loss
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn.plotting import plot_qutrit_leakage
from ctpcpinn.utils import (set_seed, numpy_to_torch_complex,
                            torch_to_numpy_complex, torch_lindblad_rhs)


def run(config: dict = None):
    """Run experiment 3: qutrit reconstruction, plain vs Fourier time-features."""
    if config is None:
        config = {}

    set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 1500)
    n_time_points = config.get('n_time_points', 120)
    w_phys = config.get('w_phys', 1.0)
    n_colloc = config.get('n_colloc', 120)
    fourier_features = config.get('fourier_features', 48)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    d = 3  # qutrit

    # Anharmonic qutrit Hamiltonian E1|1><1| + E2|2><2| (with E2 < 2 E1). The
    # energies set the coherent (Bohr) oscillation frequencies; over the window
    # t in [0, T] they produce several oscillations -- enough that a plain
    # time-MLP density network suffers from spectral bias.
    T = 4.0
    E1 = 2.0 * np.pi * config.get('E1', 2.0)
    E2 = 2.0 * np.pi * config.get('E2', 3.8)
    H = E1 * projector(d, 1, 1) + E2 * projector(d, 2, 2)

    gamma_10, gamma_21, gamma_phi = 0.5, 0.2, 0.1
    L_10 = np.sqrt(gamma_10) * projector(d, 0, 1)   # |0><1|, relaxation
    L_21 = np.sqrt(gamma_21) * projector(d, 1, 2)   # |1><2|, leakage
    L_phi = np.sqrt(gamma_phi) * np.diag([0, 1, 2]).astype(np.complex128)  # dephasing

    def H_fn(t):
        return H

    def Ls_fn(t):
        return [L_10, L_21, L_phi]

    psi0 = np.array([0.8, 0.5, 0.3], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())

    t_grid = np.linspace(0, T, n_time_points)
    rhos_exact = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)

    # Observe populations and coherence quadratures (informationally complete).
    P0, P1, P2 = projector(d, 0, 0), projector(d, 1, 1), projector(d, 2, 2)
    X01 = projector(d, 0, 1) + projector(d, 1, 0)
    Y01 = -1j * (projector(d, 0, 1) - projector(d, 1, 0))
    X12 = projector(d, 1, 2) + projector(d, 2, 1)
    Y12 = -1j * (projector(d, 1, 2) - projector(d, 2, 1))
    observables = {'P0': P0, 'P1': P1, 'P2': P2,
                   'X01': X01, 'Y01': Y01, 'X12': X12, 'Y12': Y12}
    measurements = generate_measurements(rhos_exact, observables, noise_std=0.02)

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in observables.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    meas_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in measurements.items()}
    rho0_torch = numpy_to_torch_complex(rho0)
    H_torch = numpy_to_torch_complex(H)
    P01_torch = numpy_to_torch_complex(projector(d, 0, 1))
    P12_torch = numpy_to_torch_complex(projector(d, 1, 2))
    diag_torch = numpy_to_torch_complex(np.diag([0, 1, 2]).astype(np.complex128))
    t_max = float(t_grid[-1])
    obs_names = list(obs_torch.keys())
    target_tensor = torch.stack([meas_torch[k] for k in obs_names], dim=-1)

    def lindblad_rhs_fn(rho_batch, params):
        L1 = torch.sqrt(params['gamma_10']()) * P01_torch
        L2 = torch.sqrt(params['gamma_21']()) * P12_torch
        L3 = torch.sqrt(params['gamma_phi']()) * diag_torch
        return torch_lindblad_rhs(rho_batch, H_torch, [L1, L2, L3])

    def train_model(ff):
        """Train a density network with ff Fourier features; return predicted
        trajectory and per-time state fidelity against the exact dynamics."""
        set_seed(config.get('seed', 42))
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
            d_loss = data_loss(pred, target_tensor)
            t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
            p_loss = physics_residual_loss(model(t_colloc), t_colloc,
                                           lambda rho: lindblad_rhs_fn(rho, rate_params))
            ic_loss = initial_condition_loss(model(torch.zeros(1)), rho0_torch)
            (1.0 * d_loss + w_phys * p_loss + 10.0 * ic_loss).backward()
            opt.step()
        with torch.no_grad():
            rhos_pred = torch_to_numpy_complex(model(t_data))
        return rhos_pred, state_fidelity_over_time(rhos_pred, rhos_exact)

    if verbose:
        print("  Training plain-MLP density net (no Fourier features)...")
    _, fids_plain = train_model(0)
    if verbose:
        print("  Training Fourier-MLP density net...")
    rhos_pred_f, fids_fourier = train_model(fourier_features)

    fid_plain = float(np.mean(fids_plain))
    fid_fourier = float(np.mean(fids_fourier))
    if verbose:
        print(f"  Mean state fidelity: plain MLP = {fid_plain:.4f}, "
              f"Fourier MLP = {fid_fourier:.4f}")

    # Population plot for the Fourier model.
    pops_exact = np.stack([[np.real(np.trace(P @ rhos_exact[i])) for P in [P0, P1, P2]]
                           for i in range(len(t_grid))])
    pops_pred = np.stack([[np.real(np.trace(P @ rhos_pred_f[i])) for P in [P0, P1, P2]]
                          for i in range(len(t_grid))])
    plot_qutrit_leakage(t_grid, pops_exact, pops_pred,
                        os.path.join(figures_dir, 'exp3_qutrit_leakage.pdf'))

    # Table: reconstruction fidelity, plain vs Fourier.
    table_path = os.path.join(tables_dir, 'exp3_leakage_results.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Experiment 3: qutrit reconstruction fidelity with a plain "
                r"versus a Fourier-feature density network.}" + "\n")
        f.write(r"\label{tab:exp3}" + "\n")
        f.write(r"\begin{tabular}{lc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Density network & Mean state fidelity \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(f"Plain MLP & {fid_plain:.4f} \\\\\n")
        f.write(f"Fourier time-features & {fid_fourier:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 3 complete.")

    return {'fid_plain': fid_plain, 'fid_fourier': fid_fourier,
            'fidelities': fids_fourier}


if __name__ == '__main__':
    run()
