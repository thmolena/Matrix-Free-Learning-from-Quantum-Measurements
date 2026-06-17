"""Experiment 1: Qubit inverse problem — recover Lindblad rates from data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, identity, sigma_minus
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, UnconstrainedDensityNet, PositiveParameter
from ctpcpinn.losses import data_loss, physics_residual_loss, initial_condition_loss, total_loss
from ctpcpinn.metrics import (state_fidelity_over_time, trace_distance_over_time,
                               parameter_relative_error, positivity_violation_rate)
from ctpcpinn.plotting import plot_parameter_recovery, plot_state_fidelity
from ctpcpinn.utils import set_seed, numpy_to_torch_complex, torch_to_numpy_complex, torch_lindblad_rhs


def run(config: dict = None):
    """Run experiment 1: qubit inverse problem."""
    if config is None:
        config = {}

    set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 1500)
    n_time_points = config.get('n_time_points', 80)
    noise_std = config.get('noise_std', 0.02)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    # --- True system parameters ---
    omega = 2.0 * np.pi * 0.5   # qubit frequency
    Omega = 2.0 * np.pi * 0.1   # drive strength
    gamma1 = 0.3                 # amplitude damping rate
    gamma_phi = 0.15             # dephasing rate

    true_params = {'omega': omega, 'Omega': Omega, 'gamma1': gamma1, 'gamma_phi': gamma_phi}

    # Hamiltonian
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()

    def H_fn(t):
        return 0.5 * omega * sz + 0.5 * Omega * sx

    def Ls_fn(t):
        return [np.sqrt(gamma1) * sm, np.sqrt(gamma_phi / 2) * sz]

    # Initial state: |+> = (|0> + |1>)/sqrt(2)
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

    # Generate exact trajectory
    t_grid = np.linspace(0, 3.0, n_time_points)
    rhos_exact = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)

    # Generate measurements
    observables = {'sx': sx, 'sy': sy, 'sz': sz}
    measurements = generate_measurements(rhos_exact, observables, noise_std=noise_std)

    # --- CPTP-PINN model ---
    if verbose:
        print("  Training CPTP-PINN...")
    model_cptp = DensityMatrixNet(d=2, hidden_dim=96, n_layers=4)

    # Learnable positive rates
    gamma1_param = PositiveParameter(init_value=0.5)
    gamma_phi_param = PositiveParameter(init_value=0.5)
    omega_param = nn.Parameter(torch.tensor(2.0))
    Omega_param = nn.Parameter(torch.tensor(1.0))

    rate_params_cptp = {
        'gamma1': gamma1_param,
        'gamma_phi': gamma_phi_param,
        'omega': omega_param,
        'Omega': Omega_param,
    }

    # Convert to torch
    obs_torch = {k: numpy_to_torch_complex(v) for k, v in observables.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    meas_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in measurements.items()}
    rho0_torch = numpy_to_torch_complex(rho0)
    sm_torch = numpy_to_torch_complex(sm)
    sz_torch = numpy_to_torch_complex(sz)

    def lindblad_rhs_fn_cptp(rho_batch, params):
        g1 = params['gamma1']()
        gphi = params['gamma_phi']()
        om = params['omega']
        Om = params['Omega']
        H = 0.5 * om * sz_torch + 0.5 * Om * obs_torch['sx']
        L1 = torch.sqrt(g1) * sm_torch
        L2 = torch.sqrt(gphi / 2) * sz_torch
        return torch_lindblad_rhs(rho_batch, H, [L1, L2])

    # Training
    all_params_cptp = list(model_cptp.parameters()) + [omega_param, Omega_param]
    all_params_cptp += list(gamma1_param.parameters()) + list(gamma_phi_param.parameters())
    optimizer = torch.optim.Adam(all_params_cptp, lr=1e-3)

    obs_names = ['sx', 'sy', 'sz']
    target_tensor = torch.stack([meas_torch[k] for k in obs_names], dim=-1)
    t_max = float(t_grid[-1])

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        t_grad = t_data.clone().requires_grad_(True)
        rho_pred = model_cptp(t_grad)

        # Data loss
        pred_list = []
        for obs_name in obs_names:
            O = obs_torch[obs_name]
            expect = torch.einsum('bij,ji->b', rho_pred, O).real
            pred_list.append(expect)
        pred_obs = torch.stack(pred_list, dim=-1)
        d_loss = data_loss(pred_obs, target_tensor)

        # Physics loss
        n_colloc = 60
        t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
        rho_colloc = model_cptp(t_colloc)
        p_loss = physics_residual_loss(rho_colloc, t_colloc,
                                       lambda rho: lindblad_rhs_fn_cptp(rho, rate_params_cptp))

        # IC loss
        rho_pred_0 = model_cptp(torch.zeros(1))
        ic_loss = initial_condition_loss(rho_pred_0, rho0_torch)

        loss = 1.0 * d_loss + 0.5 * p_loss + 10.0 * ic_loss
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch:4d} | loss={loss.item():.4e} data={d_loss.item():.4e} "
                  f"phys={p_loss.item():.4e}")

    # Extract learned parameters
    learned_cptp = {
        'omega': float(omega_param.data),
        'Omega': float(Omega_param.data),
        'gamma1': gamma1_param.value,
        'gamma_phi': gamma_phi_param.value,
    }

    # --- Baseline unconstrained model ---
    if verbose:
        print("  Training Unconstrained baseline...")
    model_base = UnconstrainedDensityNet(d=2, hidden_dim=96, n_layers=4)
    gamma1_b = PositiveParameter(init_value=0.5)
    gamma_phi_b = PositiveParameter(init_value=0.5)
    omega_b = nn.Parameter(torch.tensor(2.0))
    Omega_b = nn.Parameter(torch.tensor(1.0))

    rate_params_base = {'gamma1': gamma1_b, 'gamma_phi': gamma_phi_b,
                        'omega': omega_b, 'Omega': Omega_b}

    def lindblad_rhs_fn_base(rho_batch, params):
        g1 = params['gamma1']()
        gphi = params['gamma_phi']()
        om = params['omega']
        Om = params['Omega']
        H = 0.5 * om * sz_torch + 0.5 * Om * obs_torch['sx']
        L1 = torch.sqrt(g1) * sm_torch
        L2 = torch.sqrt(gphi / 2) * sz_torch
        return torch_lindblad_rhs(rho_batch, H, [L1, L2])

    all_params_base = list(model_base.parameters()) + [omega_b, Omega_b]
    all_params_base += list(gamma1_b.parameters()) + list(gamma_phi_b.parameters())
    optimizer_b = torch.optim.Adam(all_params_base, lr=1e-3)

    for epoch in range(n_epochs):
        optimizer_b.zero_grad()

        t_grad = t_data.clone().requires_grad_(True)
        rho_pred = model_base(t_grad)

        pred_list = []
        for obs_name in obs_names:
            O = obs_torch[obs_name]
            expect = torch.einsum('bij,ji->b', rho_pred, O).real
            pred_list.append(expect)
        pred_obs = torch.stack(pred_list, dim=-1)
        d_loss = data_loss(pred_obs, target_tensor)

        n_colloc = 60
        t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
        rho_colloc = model_base(t_colloc)
        p_loss = physics_residual_loss(rho_colloc, t_colloc,
                                       lambda rho: lindblad_rhs_fn_base(rho, rate_params_base))

        rho_pred_0 = model_base(torch.zeros(1))
        ic_loss = initial_condition_loss(rho_pred_0, rho0_torch)

        loss = 1.0 * d_loss + 0.5 * p_loss + 10.0 * ic_loss
        loss.backward()
        optimizer_b.step()

    learned_base = {
        'omega': float(omega_b.data),
        'Omega': float(Omega_b.data),
        'gamma1': gamma1_b.value,
        'gamma_phi': gamma_phi_b.value,
    }

    # --- Evaluate ---
    with torch.no_grad():
        rhos_cptp = torch_to_numpy_complex(model_cptp(t_data))
        rhos_base = torch_to_numpy_complex(model_base(t_data))

    fid_cptp = state_fidelity_over_time(rhos_cptp, rhos_exact)
    fid_base = state_fidelity_over_time(rhos_base, rhos_exact)
    err_cptp = parameter_relative_error(true_params, learned_cptp)
    err_base = parameter_relative_error(true_params, learned_base)

    if verbose:
        print(f"  CPTP-PINN mean fidelity: {np.mean(fid_cptp):.4f}")
        print(f"  Baseline mean fidelity: {np.mean(fid_base):.4f}")
        print(f"  CPTP param errors: {err_cptp}")
        print(f"  Baseline param errors: {err_base}")

    # --- Save figures ---
    plot_parameter_recovery(true_params, learned_cptp, learned_base,
                            os.path.join(figures_dir, 'exp1_parameter_recovery.pdf'))
    plot_state_fidelity(t_grid, fid_cptp, fid_base,
                        os.path.join(figures_dir, 'exp1_state_fidelity.pdf'))

    # --- Save table ---
    table_path = os.path.join(tables_dir, 'exp1_parameter_errors.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Experiment 1: Parameter recovery relative errors.}" + "\n")
        f.write(r"\label{tab:exp1}" + "\n")
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Parameter & True Value & CPTP-PINN Error & Unconstrained Error \\" + "\n")
        f.write(r"\hline" + "\n")
        param_tex = {'omega': r'$\omega$', 'Omega': r'$\Omega$',
                     'gamma1': r'$\gamma_1$', 'gamma_phi': r'$\gamma_\phi$'}
        for k in true_params:
            f.write(f"{param_tex.get(k, k)} & {true_params[k]:.4f} & {err_cptp[k]:.4f} & {err_base[k]:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"Mean Fidelity & --- & " + f"{np.mean(fid_cptp):.4f} & {np.mean(fid_base):.4f}" + r" \\" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 1 complete.")

    return {
        'fid_cptp': fid_cptp, 'fid_base': fid_base,
        'err_cptp': err_cptp, 'err_base': err_base,
        'learned_cptp': learned_cptp, 'learned_base': learned_base,
    }


if __name__ == '__main__':
    run()
