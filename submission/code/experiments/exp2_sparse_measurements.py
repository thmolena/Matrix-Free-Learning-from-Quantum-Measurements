"""Experiment 2: Sparse measurement ablation study."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, sigma_minus
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, UnconstrainedDensityNet, PositiveParameter
from ctpcpinn.losses import data_loss, physics_residual_loss, initial_condition_loss
from ctpcpinn.metrics import trace_distance_over_time, parameter_relative_error
from ctpcpinn.plotting import plot_sparse_measurement_ablation
from ctpcpinn.utils import set_seed, numpy_to_torch_complex, torch_to_numpy_complex, torch_lindblad_rhs


def _train_model(model, rate_params, obs_torch, t_data, target_tensor, mask,
                 rho0_torch, lindblad_rhs_fn, n_epochs, t_max, sm_torch, sz_torch):
    """Helper to train a single model with given mask."""
    all_params = list(model.parameters())
    for p in rate_params.values():
        if isinstance(p, nn.Module):
            all_params.extend(p.parameters())
        elif isinstance(p, nn.Parameter):
            all_params.append(p)

    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    obs_names = list(obs_torch.keys())

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
        d_loss = data_loss(pred_obs, target_tensor, mask)

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


def run(config: dict = None):
    """Run experiment 2: sparse measurement ablation."""
    if config is None:
        config = {}

    set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 1000)
    n_time_points = config.get('n_time_points', 80)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    # System setup (same as exp1)
    omega = 2.0 * np.pi * 0.5
    Omega = 2.0 * np.pi * 0.1
    gamma1 = 0.3
    gamma_phi = 0.15
    true_params = {'omega': omega, 'Omega': Omega, 'gamma1': gamma1, 'gamma_phi': gamma_phi}

    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()

    def H_fn(t):
        return 0.5 * omega * sz + 0.5 * Omega * sx

    def Ls_fn(t):
        return [np.sqrt(gamma1) * sm, np.sqrt(gamma_phi / 2) * sz]

    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    t_grid = np.linspace(0, 3.0, n_time_points)
    rhos_exact = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)

    observables = {'sx': sx, 'sy': sy, 'sz': sz}
    measurements = generate_measurements(rhos_exact, observables, noise_std=0.02)

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in observables.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    meas_torch = {k: torch.tensor(v, dtype=torch.float32) for k, v in measurements.items()}
    rho0_torch = numpy_to_torch_complex(rho0)
    sm_torch = numpy_to_torch_complex(sm)
    sz_torch = numpy_to_torch_complex(sz)
    t_max = float(t_grid[-1])

    obs_names = ['sx', 'sy', 'sz']
    target_tensor = torch.stack([meas_torch[k] for k in obs_names], dim=-1)
    N = n_time_points
    n_obs = 3

    fractions = [1.0, 0.5, 0.25, 0.1]
    results_cptp = []
    results_base = []

    for frac in fractions:
        if verbose:
            print(f"  Fraction = {frac:.2f}")

        # Create observation mask
        rng = np.random.default_rng(42)
        mask_np = (rng.random((N, n_obs)) < frac).astype(np.float32)
        mask_np[0, :] = 1.0  # Always observe first point
        mask = torch.tensor(mask_np)

        # CPTP model
        set_seed(42)
        model_c = DensityMatrixNet(d=2, hidden_dim=64, n_layers=3)
        g1_c = PositiveParameter(0.5)
        gp_c = PositiveParameter(0.5)
        om_c = nn.Parameter(torch.tensor(2.0))
        Om_c = nn.Parameter(torch.tensor(1.0))
        rp_c = {'gamma1': g1_c, 'gamma_phi': gp_c, 'omega': om_c, 'Omega': Om_c}

        def lind_fn_c(rho, params):
            g1 = params['gamma1']()
            gp = params['gamma_phi']()
            om = params['omega']
            Om = params['Omega']
            H = 0.5 * om * sz_torch + 0.5 * Om * obs_torch['sx']
            L1 = torch.sqrt(g1) * sm_torch
            L2 = torch.sqrt(gp / 2) * sz_torch
            return torch_lindblad_rhs(rho, H, [L1, L2])

        _train_model(model_c, rp_c, obs_torch, t_data, target_tensor, mask,
                     rho0_torch, lind_fn_c, n_epochs, t_max, sm_torch, sz_torch)

        # Baseline
        set_seed(42)
        model_b = UnconstrainedDensityNet(d=2, hidden_dim=64, n_layers=3)
        g1_b = PositiveParameter(0.5)
        gp_b = PositiveParameter(0.5)
        om_b = nn.Parameter(torch.tensor(2.0))
        Om_b = nn.Parameter(torch.tensor(1.0))
        rp_b = {'gamma1': g1_b, 'gamma_phi': gp_b, 'omega': om_b, 'Omega': Om_b}

        def lind_fn_b(rho, params):
            g1 = params['gamma1']()
            gp = params['gamma_phi']()
            om = params['omega']
            Om = params['Omega']
            H = 0.5 * om * sz_torch + 0.5 * Om * obs_torch['sx']
            L1 = torch.sqrt(g1) * sm_torch
            L2 = torch.sqrt(gp / 2) * sz_torch
            return torch_lindblad_rhs(rho, H, [L1, L2])

        _train_model(model_b, rp_b, obs_torch, t_data, target_tensor, mask,
                     rho0_torch, lind_fn_b, n_epochs, t_max, sm_torch, sz_torch)

        # Evaluate
        with torch.no_grad():
            rhos_c = torch_to_numpy_complex(model_c(t_data))
            rhos_b = torch_to_numpy_complex(model_b(t_data))

        td_c = trace_distance_over_time(rhos_c, rhos_exact)
        td_b = trace_distance_over_time(rhos_b, rhos_exact)
        results_cptp.append(np.mean(td_c))
        results_base.append(np.mean(td_b))

    # Plot
    plot_sparse_measurement_ablation(fractions, results_cptp, results_base,
                                      os.path.join(figures_dir, 'exp2_sparse_measurements.pdf'))

    # Table
    table_path = os.path.join(tables_dir, 'exp2_sparse_ablation.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Experiment 2: Mean trace distance vs.\ observation fraction.}" + "\n")
        f.write(r"\label{tab:exp2}" + "\n")
        f.write(r"\begin{tabular}{lcc}" + "\n")
        f.write(r"\hline" + "\n")
        f.write(r"Fraction & CPTP-PINN & Unconstrained \\" + "\n")
        f.write(r"\hline" + "\n")
        for i, frac in enumerate(fractions):
            f.write(f"{frac:.2f} & {results_cptp[i]:.4f} & {results_base[i]:.4f} \\\\\n")
        f.write(r"\hline" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 2 complete.")

    return {'fractions': fractions, 'cptp': results_cptp, 'baseline': results_base}


if __name__ == '__main__':
    run()
