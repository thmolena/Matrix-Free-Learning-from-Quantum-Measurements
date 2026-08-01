"""Experiment 2: sparse-measurement ablation (multi-seed, with soft-penalty).

Randomly retains a fraction of the observation entries and measures how
trajectory recovery degrades. Three methods are compared at each fraction --
the hard-constrained CPTP-PINN, a soft-penalty PINN (unconstrained network plus
an eigenvalue-negativity penalty, the soft alternative of Sec. 3.3), and a fully
unconstrained network -- each over an independent set of random seeds. This is
where the architectural constraint matters most: with few observations the
soft/unconstrained surrogates can leave the physical set, while the Cholesky map
cannot. All numbers are mean +/- 95% CI over the seeds.
"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, sigma_minus
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, UnconstrainedDensityNet, PositiveParameter
from ctpcpinn.losses import (data_loss, physics_residual_loss, initial_condition_loss,
                             positivity_penalty)
from ctpcpinn.metrics import trace_distance_over_time
from ctpcpinn.plotting import plot_sparse_ablation_multiseed
from ctpcpinn.stats import aggregate, fmt_pm
from ctpcpinn.utils import (set_seed, numpy_to_torch_complex, torch_to_numpy_complex,
                            torch_lindblad_rhs)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = [1.0, 0.5, 0.25, 0.1]
LAMBDA_POS = 10.0


def _train(kind, seed, model, rate_params, obs_torch, t_data, target_tensor, mask,
           rho0_torch, rhs_fn, n_epochs, t_max):
    all_params = list(model.parameters())
    for p in rate_params.values():
        if isinstance(p, nn.Module):
            all_params.extend(p.parameters())
        elif isinstance(p, nn.Parameter):
            all_params.append(p)
    opt = torch.optim.Adam(all_params, lr=1e-3)
    obs_names = list(obs_torch.keys())
    for _ in range(n_epochs):
        opt.zero_grad()
        rho_pred = model(t_data.clone().requires_grad_(True))
        pred = torch.stack([torch.einsum('bij,ji->b', rho_pred, obs_torch[k]).real
                            for k in obs_names], dim=-1)
        d_loss = data_loss(pred, target_tensor, mask)
        t_col = torch.linspace(0, t_max, 50).requires_grad_(True)
        rho_col = model(t_col)
        p_loss = physics_residual_loss(rho_col, t_col, lambda r: rhs_fn(r, rate_params))
        ic_loss = initial_condition_loss(model(torch.zeros(1)), rho0_torch)
        loss = 1.0 * d_loss + 0.5 * p_loss + 10.0 * ic_loss
        if kind == 'soft':
            loss = loss + LAMBDA_POS * positivity_penalty(rho_col)
        loss.backward()
        opt.step()


def run(config: dict = None):
    if config is None:
        config = {}
    figures_dir = config.get('figures_dir', 'manuscript_assets/figures')
    tables_dir = config.get('tables_dir', 'manuscript_assets/tables')
    n_epochs = config.get('n_epochs', 3000)
    n_time_points = config.get('n_time_points', 100)
    noise_std = config.get('noise_std', 0.02)
    seeds = config.get('seeds', DEFAULT_SEEDS)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    omega = 2.0 * np.pi * 0.5
    Omega = 2.0 * np.pi * 0.1
    gamma1, gamma_phi = 0.3, 0.15
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
    obs_names = ['sx', 'sy', 'sz']

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in observables.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    rho0_torch = numpy_to_torch_complex(rho0)
    sm_torch = numpy_to_torch_complex(sm)
    sz_torch = numpy_to_torch_complex(sz)
    t_max = float(t_grid[-1])
    N = n_time_points

    def rhs_fn(rho, params):
        H = 0.5 * params['omega'] * sz_torch + 0.5 * params['Omega'] * obs_torch['sx']
        L1 = torch.sqrt(params['gamma1']()) * sm_torch
        L2 = torch.sqrt(params['gamma_phi']() / 2) * sz_torch
        return torch_lindblad_rhs(rho, H, [L1, L2])

    methods = ['cptp', 'soft', 'unconstrained']
    label = {'cptp': 'CPTP-PINN (ours)', 'soft': 'Soft-penalty PINN',
             'unconstrained': 'Unconstrained'}
    # per method: list (over fractions) of list (over seeds) of mean trace distance
    td = {m: [[] for _ in FRACTIONS] for m in methods}

    for seed in seeds:
        if verbose:
            print(f"  seed {seed}...")
        measurements = generate_measurements(rhos_exact, observables,
                                             noise_std=noise_std, seed=seed)
        target_tensor = torch.stack(
            [torch.tensor(measurements[k], dtype=torch.float32) for k in obs_names], dim=-1)
        for fi, frac in enumerate(FRACTIONS):
            rng = np.random.default_rng(1000 * seed + fi)
            mask_np = (rng.random((N, len(obs_names))) < frac).astype(np.float32)
            mask_np[0, :] = 1.0
            mask = torch.tensor(mask_np)

            for kind in methods:
                set_seed(seed)
                if kind == 'cptp':
                    model = DensityMatrixNet(d=2, hidden_dim=64, n_layers=3)
                else:
                    model = UnconstrainedDensityNet(d=2, hidden_dim=64, n_layers=3)
                rp = {'gamma1': PositiveParameter(0.5), 'gamma_phi': PositiveParameter(0.5),
                      'omega': nn.Parameter(torch.tensor(2.0)),
                      'Omega': nn.Parameter(torch.tensor(1.0))}
                _train(kind, seed, model, rp, obs_torch, t_data, target_tensor, mask,
                       rho0_torch, rhs_fn, n_epochs, t_max)
                with torch.no_grad():
                    rhos = torch_to_numpy_complex(model(t_data))
                td[kind][fi].append(float(np.mean(trace_distance_over_time(rhos, rhos_exact))))

    # Figure
    plot_sparse_ablation_multiseed(
        FRACTIONS, {label[m]: td[m] for m in methods},
        os.path.join(figures_dir, 'exp2_sparse_measurements.pdf'))

    # Table
    table_path = os.path.join(tables_dir, 'exp2_sparse_ablation.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n\\centering\n")
        f.write(r"\caption{Experiment 2: mean trace distance to the exact trajectory "
                r"versus retained observation fraction. Mean $\pm$ 95\% CI over "
                + f"{len(seeds)} seeds." + r"}" + "\n")
        f.write(r"\label{tab:exp2}" + "\n")
        f.write(r"\begin{tabular}{lccc}" + "\n\\toprule\n")
        f.write(r"Fraction & CPTP-PINN (ours) & Soft-penalty PINN & Unconstrained \\"
                + "\n\\midrule\n")
        for fi, frac in enumerate(FRACTIONS):
            f.write(f"{frac:.2f} & " + " & ".join(fmt_pm(td[m][fi], digits=4) for m in methods)
                    + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

    if verbose:
        for m in methods:
            print(f"  {label[m]}: " + ", ".join(
                f"f={FRACTIONS[fi]}:{aggregate(td[m][fi])['mean']:.4f}" for fi in range(len(FRACTIONS))))
        print("  Experiment 2 complete.")

    return {'fractions': FRACTIONS, 'td': td, 'seeds': seeds}


if __name__ == '__main__':
    run()
