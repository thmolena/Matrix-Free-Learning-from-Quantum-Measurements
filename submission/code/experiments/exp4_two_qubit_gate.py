"""Experiment 4: Two-qubit gate with dissipation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ctpcpinn.operators import (pauli_x, pauli_y, pauli_z, identity,
                                 sigma_minus, kron_n)
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, PositiveParameter
from ctpcpinn.losses import data_loss, physics_residual_loss, initial_condition_loss
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn.plotting import plot_gate_fidelity
from ctpcpinn.utils import set_seed, numpy_to_torch_complex, torch_to_numpy_complex, torch_lindblad_rhs


def run(config: dict = None):
    """Run experiment 4: two-qubit gate with noise."""
    if config is None:
        config = {}

    set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 1200)
    n_time_points = config.get('n_time_points', 60)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    d = 4  # two qubits
    I2 = identity(2)
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()

    # Two-qubit operators
    XX = kron_n([sx, sx])
    YY = kron_n([sy, sy])
    ZI = kron_n([sz, I2])
    IZ = kron_n([I2, sz])
    smI = kron_n([sm, I2])
    Ism = kron_n([I2, sm])
    szI = kron_n([sz, I2])
    Isz = kron_n([I2, sz])

    # Parameters
    J = 2.0 * np.pi * 0.05    # coupling strength
    Delta1 = 2.0 * np.pi * 5.0
    Delta2 = 2.0 * np.pi * 4.8
    gamma1 = 0.1               # amplitude damping qubit 1
    gamma2 = 0.08              # amplitude damping qubit 2
    gamma_phi1 = 0.05          # dephasing qubit 1
    gamma_phi2 = 0.04          # dephasing qubit 2

    true_params = {
        'gamma1': gamma1, 'gamma2': gamma2,
        'gamma_phi1': gamma_phi1, 'gamma_phi2': gamma_phi2,
    }

    def H_fn(t):
        # Time-dependent coupling (Gaussian pulse)
        Jt = J * np.exp(-(t - 1.5) ** 2 / (2 * 0.3 ** 2))
        return Jt * (XX + YY) + 0.5 * Delta1 * ZI + 0.5 * Delta2 * IZ

    def Ls_fn(t):
        return [
            np.sqrt(gamma1) * smI,
            np.sqrt(gamma2) * Ism,
            np.sqrt(gamma_phi1 / 2) * szI,
            np.sqrt(gamma_phi2 / 2) * Isz,
        ]

    # Test with multiple initial states
    init_states = {
        '|00>': np.array([1, 0, 0, 0], dtype=np.complex128),
        '|01>': np.array([0, 1, 0, 0], dtype=np.complex128),
        '|10>': np.array([0, 0, 1, 0], dtype=np.complex128),
        '|+0>': np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2),
    }

    t_grid = np.linspace(0, 3.0, n_time_points)
    all_fidelities = {}

    # Use |00> for training
    psi_train = init_states['|00>']
    rho0_train = np.outer(psi_train, psi_train.conj())
    rhos_exact_train = solve_lindblad_trajectory(H_fn, Ls_fn, rho0_train, t_grid)

    # Observables for d=4: local Pauli expectations
    obs_ops = {
        'ZI': ZI, 'IZ': IZ, 'XX': XX, 'YY': YY,
    }
    measurements = generate_measurements(rhos_exact_train, obs_ops, noise_std=0.02)

    # Train CPTP-PINN
    if verbose:
        print("  Training two-qubit CPTP-PINN...")
    # Fourier time-features (period = window T = 3) for the fast detuning dynamics.
    model = DensityMatrixNet(d=4, hidden_dim=128, n_layers=4,
                             fourier_features=config.get('fourier_features', 32),
                             fourier_period=3.0)
    g1_p = PositiveParameter(0.2)
    g2_p = PositiveParameter(0.2)
    gp1_p = PositiveParameter(0.1)
    gp2_p = PositiveParameter(0.1)
    J_param = nn.Parameter(torch.tensor(0.2))

    rate_params = {
        'gamma1': g1_p, 'gamma2': g2_p,
        'gamma_phi1': gp1_p, 'gamma_phi2': gp2_p,
        'J': J_param,
    }

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in obs_ops.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    meas_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in measurements.items()}
    rho0_torch = numpy_to_torch_complex(rho0_train)

    XX_t = numpy_to_torch_complex(XX)
    YY_t = numpy_to_torch_complex(YY)
    ZI_t = numpy_to_torch_complex(ZI)
    IZ_t = numpy_to_torch_complex(IZ)
    smI_t = numpy_to_torch_complex(smI)
    Ism_t = numpy_to_torch_complex(Ism)
    szI_t = numpy_to_torch_complex(szI)
    Isz_t = numpy_to_torch_complex(Isz)

    t_max = float(t_grid[-1])

    def lindblad_rhs_fn(rho_batch, params):
        g1 = params['gamma1']()
        g2 = params['gamma2']()
        gp1 = params['gamma_phi1']()
        gp2 = params['gamma_phi2']()
        # Use fixed Delta, learned J
        J_val = params['J']
        H = J_val * (XX_t + YY_t) + 0.5 * Delta1 * ZI_t + 0.5 * Delta2 * IZ_t
        L1 = torch.sqrt(g1) * smI_t
        L2 = torch.sqrt(g2) * Ism_t
        L3 = torch.sqrt(gp1 / 2) * szI_t
        L4 = torch.sqrt(gp2 / 2) * Isz_t
        return torch_lindblad_rhs(rho_batch, H, [L1, L2, L3, L4])

    all_params_list = list(model.parameters()) + [J_param]
    for p in [g1_p, g2_p, gp1_p, gp2_p]:
        all_params_list.extend(p.parameters())
    optimizer = torch.optim.Adam(all_params_list, lr=1e-3)

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

        n_colloc = 40
        t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
        rho_colloc = model(t_colloc)
        p_loss = physics_residual_loss(rho_colloc, t_colloc,
                                       lambda rho: lindblad_rhs_fn(rho, rate_params))

        rho_pred_0 = model(torch.zeros(1))
        ic_loss = initial_condition_loss(rho_pred_0, rho0_torch)

        loss = 1.0 * d_loss + 0.3 * p_loss + 10.0 * ic_loss
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 400 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch:4d} | loss={loss.item():.4e}")

    # Build the learned generator (learned coupling + rates, with the known detunings)
    # and propagate it from each held-out initial state. The neural rho_theta(t)
    # represents only the training trajectory; generalization to new initial states
    # is a property of the recovered Lindblad generator, so that is what we test.
    J_learned = float(J_param.detach())
    g1L, g2L, gp1L, gp2L = g1_p.value, g2_p.value, gp1_p.value, gp2_p.value

    def H_learned_fn(t):
        return J_learned * (XX + YY) + 0.5 * Delta1 * ZI + 0.5 * Delta2 * IZ

    def Ls_learned_fn(t):
        return [
            np.sqrt(g1L) * smI,
            np.sqrt(g2L) * Ism,
            np.sqrt(gp1L / 2) * szI,
            np.sqrt(gp2L / 2) * Isz,
        ]

    for state_name, psi in init_states.items():
        rho0_eval = np.outer(psi, psi.conj())
        rhos_exact_eval = solve_lindblad_trajectory(H_fn, Ls_fn, rho0_eval, t_grid)

        if state_name == '|00>':
            # Trained initial state: read the network trajectory directly.
            with torch.no_grad():
                rhos_pred_eval = torch_to_numpy_complex(model(t_data))
        else:
            # Held-out initial states: propagate the learned generator.
            rhos_pred_eval = solve_lindblad_trajectory(
                H_learned_fn, Ls_learned_fn, rho0_eval, t_grid)

        fids = state_fidelity_over_time(rhos_pred_eval, rhos_exact_eval)
        all_fidelities[state_name] = fids

    # Gate fidelity proxy: average over input states
    mean_gate_fid = np.mean([np.mean(f) for f in all_fidelities.values()])

    if verbose:
        print(f"  Mean gate fidelity proxy: {mean_gate_fid:.4f}")
        print(f"  Learned gamma1={g1_p.value:.4f} (true={gamma1})")
        print(f"  Learned gamma2={g2_p.value:.4f} (true={gamma2})")

    # Plot
    plot_gate_fidelity(t_grid, all_fidelities,
                       os.path.join(figures_dir, 'exp4_gate_fidelity.pdf'))

    # Table
    table_path = os.path.join(tables_dir, 'exp4_gate_results.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Experiment 4: Two-qubit gate fidelity proxy.}" + "\n")
        f.write(r"\label{tab:exp4}" + "\n")
        f.write(r"\begin{tabular}{lcc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Initial State & Mean Fidelity & Final Fidelity \\" + "\n")
        f.write(r"\hline" + "\n")
        state_tex = {'|00>': r'$\lvert 00\rangle$', '|01>': r'$\lvert 01\rangle$',
                     '|10>': r'$\lvert 10\rangle$', '|+0>': r'$\lvert {+}0\rangle$'}
        for name, fid in all_fidelities.items():
            f.write(f"{state_tex.get(name, name)} & {np.mean(fid):.4f} & {fid[-1]:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(f"Average & {mean_gate_fid:.4f} & --- \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 4 complete.")

    return {'fidelities': all_fidelities, 'mean_gate_fidelity': mean_gate_fid}


if __name__ == '__main__':
    run()
