"""Experiment 1: single-qubit inverse problem (multi-seed, with baselines).

Recovers the four generator parameters (omega, Omega, gamma_1, gamma_phi) of a
driven, damped qubit from noisy observable data. Four methods are compared, each
repeated over an independent set of random seeds (different network
initialisation *and* different measurement-noise realisation per seed):

  * CPTP-PINN          -- hard Cholesky density constraint (this work);
  * Soft-penalty PINN  -- unconstrained net + eigenvalue-negativity penalty
                          (the soft alternative argued against in Sec. 3.3);
  * Unconstrained      -- unconstrained net, no positivity control;
  * Classical LSQ      -- nonlinear least-squares fit of the four parameters
                          with the exact model structure (Levenberg-Marquardt),
                          a strong model-based reference.

All reported numbers are mean +/- 95% CI (Student-t) over the seeds and are
written directly into tables/exp1_parameter_errors.tex.
"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.optimize import least_squares

from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, identity, sigma_minus
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, UnconstrainedDensityNet, PositiveParameter
from ctpcpinn.losses import (data_loss, physics_residual_loss, initial_condition_loss,
                             positivity_penalty)
from ctpcpinn.metrics import (state_fidelity_over_time, parameter_relative_error,
                              positivity_violation_rate)
from ctpcpinn.plotting import plot_parameter_recovery_multiseed, plot_state_fidelity_multiseed
from ctpcpinn.stats import aggregate, fmt_pm
from ctpcpinn.utils import (set_seed, numpy_to_torch_complex, torch_to_numpy_complex,
                            torch_lindblad_rhs)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
PARAM_ORDER = ['omega', 'Omega', 'gamma1', 'gamma_phi']
LAMBDA_POS = 10.0  # soft positivity-penalty weight (steel-manned)


def _build_system():
    omega = 2.0 * np.pi * 0.5
    Omega = 2.0 * np.pi * 0.1
    gamma1, gamma_phi = 0.3, 0.15
    true_params = {'omega': omega, 'Omega': Omega, 'gamma1': gamma1, 'gamma_phi': gamma_phi}
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()

    def H_fn(t):
        return 0.5 * omega * sz + 0.5 * Omega * sx

    def Ls_fn(t):
        return [np.sqrt(gamma1) * sm, np.sqrt(gamma_phi / 2) * sz]

    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    return true_params, H_fn, Ls_fn, rho0, (sx, sy, sz, sm)


def _train_pinn(kind, seed, n_epochs, t_data, target_tensor, obs_torch, obs_names,
                rho0_torch, sz_torch, sm_torch, t_max):
    """Train one PINN (kind in {'cptp','soft','unconstrained'}). Returns
    (learned_params, rhos_pred)."""
    set_seed(seed)
    if kind == 'cptp':
        model = DensityMatrixNet(d=2, hidden_dim=96, n_layers=4)
    else:
        model = UnconstrainedDensityNet(d=2, hidden_dim=96, n_layers=4)

    g1 = PositiveParameter(0.5)
    gphi = PositiveParameter(0.5)
    om = nn.Parameter(torch.tensor(2.0))
    Om = nn.Parameter(torch.tensor(1.0))

    def rhs(rho):
        H = 0.5 * om * sz_torch + 0.5 * Om * obs_torch['sx']
        L1 = torch.sqrt(g1()) * sm_torch
        L2 = torch.sqrt(gphi() / 2) * sz_torch
        return torch_lindblad_rhs(rho, H, [L1, L2])

    params = list(model.parameters()) + [om, Om] + list(g1.parameters()) + list(gphi.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    for _ in range(n_epochs):
        opt.zero_grad()
        rho_pred = model(t_data.clone().requires_grad_(True))
        pred_obs = torch.stack([torch.einsum('bij,ji->b', rho_pred, obs_torch[k]).real
                                for k in obs_names], dim=-1)
        d_loss = data_loss(pred_obs, target_tensor)
        t_col = torch.linspace(0, t_max, 60).requires_grad_(True)
        rho_col = model(t_col)
        p_loss = physics_residual_loss(rho_col, t_col, rhs)
        ic_loss = initial_condition_loss(model(torch.zeros(1)), rho0_torch)
        loss = 1.0 * d_loss + 0.5 * p_loss + 10.0 * ic_loss
        if kind == 'soft':
            loss = loss + LAMBDA_POS * positivity_penalty(rho_col)
        loss.backward()
        opt.step()

    learned = {'omega': float(om.data), 'Omega': float(Om.data),
               'gamma1': g1.value, 'gamma_phi': gphi.value}
    with torch.no_grad():
        rhos_pred = torch_to_numpy_complex(model(t_data))
    return learned, rhos_pred


def _least_squares_fit(measurements, observables, rho0, t_grid):
    """Strong classical baseline: fit (omega, Omega, gamma1, gamma_phi) by
    nonlinear least squares using the exact Lindblad model structure."""
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()
    obs_names = ['sx', 'sy', 'sz']
    y = np.concatenate([measurements[k] for k in obs_names])

    def predict(theta):
        om, Om, g1, gphi = theta
        def H_fn(t):
            return 0.5 * om * sz + 0.5 * Om * sx
        def Ls_fn(t):
            return [np.sqrt(max(g1, 0.0)) * sm, np.sqrt(max(gphi, 0.0) / 2) * sz]
        rhos = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)
        preds = [np.array([np.real(np.trace(observables[k] @ rhos[i]))
                           for i in range(len(t_grid))]) for k in obs_names]
        return np.concatenate(preds)

    theta0 = np.array([2.0, 1.0, 0.5, 0.5])
    res = least_squares(lambda th: predict(th) - y, theta0,
                        bounds=([-np.inf, -np.inf, 0.0, 0.0], np.inf), method='trf')
    om, Om, g1, gphi = res.x
    learned = {'omega': float(om), 'Omega': float(Om),
               'gamma1': float(g1), 'gamma_phi': float(gphi)}
    def H_fn(t):
        return 0.5 * om * sz + 0.5 * Om * sx
    def Ls_fn(t):
        return [np.sqrt(max(g1, 0.0)) * sm, np.sqrt(max(gphi, 0.0) / 2) * sz]
    rhos = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)
    return learned, rhos


def run(config: dict = None):
    if config is None:
        config = {}
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 3000)
    n_time_points = config.get('n_time_points', 100)
    noise_std = config.get('noise_std', 0.02)
    seeds = config.get('seeds', DEFAULT_SEEDS)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    true_params, H_fn, Ls_fn, rho0, (sx, sy, sz, sm) = _build_system()
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

    methods = ['cptp', 'soft', 'unconstrained', 'lsq']
    # Accumulators: per-method lists across seeds.
    err = {m: {p: [] for p in PARAM_ORDER} for m in methods}
    fid = {m: [] for m in methods}
    viol = {m: [] for m in methods}
    # Keep representative (first-seed) trajectories for the figures.
    rep = {}

    for si, seed in enumerate(seeds):
        if verbose:
            print(f"  seed {seed} ({si+1}/{len(seeds)})...")
        measurements = generate_measurements(rhos_exact, observables,
                                             noise_std=noise_std, seed=seed)
        target_tensor = torch.stack(
            [torch.tensor(measurements[k], dtype=torch.float32) for k in obs_names], dim=-1)

        for kind in ['cptp', 'soft', 'unconstrained']:
            learned, rhos_pred = _train_pinn(
                kind, seed, n_epochs, t_data, target_tensor, obs_torch, obs_names,
                rho0_torch, sz_torch, sm_torch, t_max)
            e = parameter_relative_error(true_params, learned)
            for p in PARAM_ORDER:
                err[kind][p].append(e[p])
            f = state_fidelity_over_time(rhos_pred, rhos_exact)
            fid[kind].append(float(np.mean(f)))
            viol[kind].append(positivity_violation_rate(rhos_pred))
            if si == 0:
                rep[kind] = {'fid_t': f, 'rhos': rhos_pred}

        # Classical least-squares baseline.
        learned_lsq, rhos_lsq = _least_squares_fit(measurements, observables, rho0, t_grid)
        e = parameter_relative_error(true_params, learned_lsq)
        for p in PARAM_ORDER:
            err['lsq'][p].append(e[p])
        f = state_fidelity_over_time(rhos_lsq, rhos_exact)
        fid['lsq'].append(float(np.mean(f)))
        viol['lsq'].append(positivity_violation_rate(rhos_lsq))
        if si == 0:
            rep['lsq'] = {'fid_t': f, 'rhos': rhos_lsq}

    # --- Figures ---
    plot_state_fidelity_multiseed(
        t_grid, {'CPTP-PINN (ours)': rep['cptp']['fid_t'],
                 'Soft-penalty PINN': rep['soft']['fid_t'],
                 'Unconstrained': rep['unconstrained']['fid_t']},
        os.path.join(figures_dir, 'exp1_state_fidelity.pdf'))
    plot_parameter_recovery_multiseed(
        PARAM_ORDER, {'CPTP-PINN (ours)': err['cptp'], 'Soft-penalty PINN': err['soft'],
                      'Unconstrained': err['unconstrained'], 'Classical LSQ': err['lsq']},
        os.path.join(figures_dir, 'exp1_parameter_recovery.pdf'))

    # --- Table (methods as rows) ---
    label = {'cptp': r'CPTP-PINN (ours)', 'soft': r'Soft-penalty PINN',
             'unconstrained': r'Unconstrained', 'lsq': r'Classical LSQ'}
    ptex = {'omega': r'$\omega$', 'Omega': r'$\Omega$',
            'gamma1': r'$\gamma_1$', 'gamma_phi': r'$\gamma_\phi$'}
    table_path = os.path.join(tables_dir, 'exp1_parameter_errors.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n\\centering\n")
        f.write(r"\caption{Experiment 1: single-qubit parameter-recovery relative error, "
                r"mean state fidelity, and positivity-violation rate (fraction of time "
                r"points with a negative eigenvalue). Values are mean $\pm$ 95\% CI over "
                + f"{len(seeds)} seeds. "
                r"True values: $\omega=3.1416$, $\Omega=0.6283$, $\gamma_1=0.3000$, "
                r"$\gamma_\phi=0.1500$. The classical least-squares fit uses the exact "
                r"model structure and is the strongest method on this full-data task; the "
                r"PINN advantage appears under sparse measurements (Table~\ref{tab:exp2}).}" + "\n")
        f.write(r"\label{tab:exp1}" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{lcccccc}" + "\n\\hline\n")
        f.write(r"Method & $\omega$ err & $\Omega$ err & $\gamma_1$ err & "
                r"$\gamma_\phi$ err & Mean fidelity & Pos.\ viol. \\" + "\n\\hline\n")
        for m in methods:
            cells = [fmt_pm(err[m][p], digits=3) for p in PARAM_ORDER]
            vrate = aggregate(viol[m])['mean']
            f.write(f"{label[m]} & " + " & ".join(cells) + " & "
                    + fmt_pm(fid[m], digits=4) + f" & {vrate:.3f}" + r" \\" + "\n")
        f.write(r"\hline" + "\n\\end{tabular}}\n\\end{table}\n")

    if verbose:
        for m in methods:
            print(f"  {label[m]}: fid={fmt_pm(fid[m])}, "
                  f"omega_err={fmt_pm(err[m]['omega'],3)}, viol={aggregate(viol[m])['mean']:.3f}")
        print("  Experiment 1 complete.")

    return {'err': err, 'fid': fid, 'viol': viol, 'seeds': seeds}


if __name__ == '__main__':
    run()
