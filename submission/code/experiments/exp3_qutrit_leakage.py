"""Experiment 3: Qutrit (3-level) leakage model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ctpcpinn.operators import identity, projector, destroy
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, PositiveParameter
from ctpcpinn.losses import data_loss, physics_residual_loss, initial_condition_loss
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn.plotting import plot_qutrit_leakage
from ctpcpinn.utils import set_seed, numpy_to_torch_complex, torch_to_numpy_complex, torch_lindblad_rhs


def run(config: dict = None):
    """Run experiment 3: qutrit leakage."""
    if config is None:
        config = {}

    set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 1500)
    n_time_points = config.get('n_time_points', 80)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    d = 3  # qutrit

    # Transmon-like anharmonic Hamiltonian: E1|1><1| + E2|2><2|
    E1 = 2.0 * np.pi * 5.0   # GHz (scaled)
    E2 = 2.0 * np.pi * 9.5   # anharmonicity: E2 < 2*E1

    H = E1 * projector(d, 1, 1) + E2 * projector(d, 2, 2)

    # Lindblad: relaxation 1->0, leakage 2->1, dephasing
    gamma_10 = 0.5    # decay |1> -> |0|
    gamma_21 = 0.2    # leakage |2> -> |1>
    gamma_phi = 0.1   # dephasing

    a = destroy(d)
    L_10 = np.sqrt(gamma_10) * projector(d, 0, 1)  # |0><1|
    L_21 = np.sqrt(gamma_21) * projector(d, 1, 2)  # |1><2|
    # Dephasing: sqrt(gamma_phi) * diag(0, 1, 2)
    L_phi = np.sqrt(gamma_phi) * np.diag([0, 1, 2]).astype(np.complex128)

    true_params = {'gamma_10': gamma_10, 'gamma_21': gamma_21, 'gamma_phi': gamma_phi}

    def H_fn(t):
        return H

    def Ls_fn(t):
        return [L_10, L_21, L_phi]

    # Initial state: superposition with some |2> leakage component
    psi0 = np.array([0.8, 0.5, 0.3], dtype=np.complex128)
    psi0 /= np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())

    t_grid = np.linspace(0, 4.0, n_time_points)
    rhos_exact = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)

    # Observables: populations
    P0 = projector(d, 0, 0)
    P1 = projector(d, 1, 1)
    P2 = projector(d, 2, 2)
    observables = {'P0': P0, 'P1': P1, 'P2': P2}
    measurements = generate_measurements(rhos_exact, observables, noise_std=0.01)

    # CPTP-PINN for qutrit
    if verbose:
        print("  Training qutrit CPTP-PINN...")
    model = DensityMatrixNet(d=3, hidden_dim=96, n_layers=4)
    g10_param = PositiveParameter(0.3)
    g21_param = PositiveParameter(0.1)
    gphi_param = PositiveParameter(0.05)

    rate_params = {'gamma_10': g10_param, 'gamma_21': g21_param, 'gamma_phi': gphi_param}

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in observables.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    meas_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in measurements.items()}
    rho0_torch = numpy_to_torch_complex(rho0)
    H_torch = numpy_to_torch_complex(H)
    P01_torch = numpy_to_torch_complex(projector(d, 0, 1))
    P12_torch = numpy_to_torch_complex(projector(d, 1, 2))
    diag_torch = numpy_to_torch_complex(np.diag([0, 1, 2]).astype(np.complex128))

    t_max = float(t_grid[-1])

    def lindblad_rhs_fn(rho_batch, params):
        g10 = params['gamma_10']()
        g21 = params['gamma_21']()
        gphi = params['gamma_phi']()
        L1 = torch.sqrt(g10) * P01_torch
        L2 = torch.sqrt(g21) * P12_torch
        L3 = torch.sqrt(gphi) * diag_torch
        return torch_lindblad_rhs(rho_batch, H_torch, [L1, L2, L3])

    all_params = list(model.parameters())
    for p in rate_params.values():
        all_params.extend(p.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    obs_names = list(obs_torch.keys())
    target_tensor = torch.stack([meas_torch[k] for k in obs_names], dim=-1)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        t_grad = t_data.clone().requires_grad_(True)
        rho_pred = model(t_grad)

        pred_list = []
        for obs_name in obs_names:
            O = obs_torch[obs_name]
            expect = torch.einsum('bij,ji->b', rho_pred, O).real
            pred_list.append(expect)
        pred_obs = torch.stack(pred_list, dim=-1)
        d_loss = data_loss(pred_obs, target_tensor)

        n_colloc = 50
        t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
        rho_colloc = model(t_colloc)
        p_loss = physics_residual_loss(rho_colloc, t_colloc.unsqueeze(-1),
                                       lambda rho: lindblad_rhs_fn(rho, rate_params))

        rho_pred_0 = model(torch.zeros(1))
        ic_loss = initial_condition_loss(rho_pred_0, rho0_torch)

        loss = 1.0 * d_loss + 0.5 * p_loss + 10.0 * ic_loss
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch:4d} | loss={loss.item():.4e}")

    # Evaluate
    with torch.no_grad():
        rhos_pred = torch_to_numpy_complex(model(t_data))

    # Extract populations
    pops_exact = np.stack([np.real(np.trace(P @ rhos_exact[i]) )
                           for P in [P0, P1, P2]
                           for i in range(len(t_grid))]).reshape(3, -1).T
    pops_pred = np.stack([np.real(np.trace(P @ rhos_pred[i]))
                          for P in [P0, P1, P2]
                          for i in range(len(t_grid))]).reshape(3, -1).T

    learned_rates = {
        'gamma_10': g10_param.value,
        'gamma_21': g21_param.value,
        'gamma_phi': gphi_param.value,
    }

    fids = state_fidelity_over_time(rhos_pred, rhos_exact)

    if verbose:
        print(f"  Learned rates: {learned_rates}")
        print(f"  True rates: {true_params}")
        print(f"  Mean fidelity: {np.mean(fids):.4f}")

    # Plot
    plot_qutrit_leakage(t_grid, pops_exact, pops_pred,
                        os.path.join(figures_dir, 'exp3_qutrit_leakage.pdf'))

    # Table
    table_path = os.path.join(tables_dir, 'exp3_leakage_results.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Experiment 3: Qutrit leakage rate recovery.}" + "\n")
        f.write(r"\label{tab:exp3}" + "\n")
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Rate & True & Learned & Rel.\ Error \\" + "\n")
        f.write(r"\hline" + "\n")
        for k in true_params:
            rel_err = abs(true_params[k] - learned_rates[k]) / true_params[k]
            f.write(f"$\\{k.replace('_', r'\_')}$ & {true_params[k]:.3f} & "
                    f"{learned_rates[k]:.3f} & {rel_err:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"Mean Fidelity & --- & {np.mean(fids):.4f} & --- \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 3 complete.")

    return {'learned_rates': learned_rates, 'fidelities': fids}


if __name__ == '__main__':
    run()
