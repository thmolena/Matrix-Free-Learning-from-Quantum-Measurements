"""Experiment 3: fast qutrit (leakage) reconstruction -- the spectral-frame cure.

A transmon-like three-level system has fast coherent (Bohr-frequency) oscillations
on top of slow dissipative decay. A smooth-in-time density network suffers from
spectral bias, and -- as earlier versions of this study reported -- neither a plain
MLP nor a deterministic Fourier-feature encoding reconstructs the dynamics reliably
across seeds.

This experiment adds the SPECTRAL-FRAME (interaction-picture eigen-operator)
density network: it learns only the slowly varying envelope
``rho_tilde_E(t) = U(t)^dag rho(t) U(t)`` in the Hamiltonian eigenbasis through the
same hard Cholesky map, and supplies the fast oscillation analytically via the
known U(t) = exp(-i H t). Because the trainable target is slow, spectral bias is
removed at the architecture level. Three otherwise-identical density networks are
compared over an independent set of seeds:

  * Plain MLP (no Fourier features)            -- baseline
  * Fourier time-features (K = 48)             -- baseline
  * Spectral frame (interaction picture)       -- this work

We report mean state fidelity AND the relative recovery error of the three
dissipative rates (gamma_10, gamma_21, gamma_phi), each as mean +/- 95% CI over the
seeds. The spectral frame is expected to be both reliable (small CI) and accurate
on the rates, because the derivative of the slow envelope is well conditioned under
measurement noise.
"""

import os
from pathlib import Path

import numpy as np
import torch

from ctpcpinn.operators import projector
from ctpcpinn.solvers import solve_lindblad_trajectory, generate_measurements
from ctpcpinn.models import DensityMatrixNet, SpectralDensityNet, PositiveParameter
from ctpcpinn.losses import (data_loss, physics_residual_loss,
                             initial_condition_loss, interaction_residual_loss)
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn import spectral as sp
from ctpcpinn.plotting import plot_qutrit_fidelity_comparison
from ctpcpinn.stats import aggregate, fmt_pm
from ctpcpinn.utils import (set_seed, numpy_to_torch_complex,
                            torch_to_numpy_complex, torch_lindblad_rhs)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
METHODS = ['Plain MLP', 'Fourier', 'Spectral (ours)']
TRUE_RATES = {'gamma_10': 0.5, 'gamma_21': 0.2, 'gamma_phi': 0.1}


def run(config: dict = None):
    if config is None:
        config = {}
    figures_dir = config.get('figures_dir', 'manuscript_assets/figures')
    tables_dir = config.get('tables_dir', 'manuscript_assets/tables')
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
    J_10 = projector(d, 0, 1)                     # unit-rate jump operators
    J_21 = projector(d, 1, 2)
    J_phi = np.diag([0, 1, 2]).astype(np.complex128)
    L_10 = np.sqrt(gamma_10) * J_10
    L_21 = np.sqrt(gamma_21) * J_21
    L_phi = np.sqrt(gamma_phi) * J_phi
    E_eig, V_eig = sp.eigh_hamiltonian(H)         # for the rate readout

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
    E_torch = torch.tensor(np.linalg.eigvalsh(H).real, dtype=torch.float32)
    t_max = float(t_grid[-1])
    obs_names = list(obs_torch.keys())

    def lab_lindblad_rhs(rho_batch, params):
        L1 = torch.sqrt(params['gamma_10']()) * P01_torch
        L2 = torch.sqrt(params['gamma_21']()) * P12_torch
        L3 = torch.sqrt(params['gamma_phi']()) * diag_torch
        return torch_lindblad_rhs(rho_batch, H_torch, [L1, L2, L3])

    def interaction_rhs(rho_tilde, t, params):
        L1 = torch.sqrt(params['gamma_10']()) * P01_torch
        L2 = torch.sqrt(params['gamma_21']()) * P12_torch
        L3 = torch.sqrt(params['gamma_phi']()) * diag_torch
        # H diagonal -> eigenbasis = computational basis; jump ops already in it.
        return sp.torch_interaction_rhs(rho_tilde, t, E_torch, [L1, L2, L3], None)

    def train_model(method, seed):
        set_seed(seed)
        measurements = generate_measurements(rhos_exact, observables,
                                             noise_std=0.02, seed=seed)
        target = torch.stack([torch.tensor(measurements[k], dtype=torch.float32)
                              for k in obs_names], dim=-1)
        rate_params = {'gamma_10': PositiveParameter(0.3),
                       'gamma_21': PositiveParameter(0.1),
                       'gamma_phi': PositiveParameter(0.05)}
        spectral = (method == 'Spectral (ours)')
        if spectral:
            model = SpectralDensityNet(H, hidden_dim=96, n_layers=4,
                                       fourier_features=0, fourier_period=T)
        else:
            ff = fourier_features if method == 'Fourier' else 0
            model = DensityMatrixNet(d=3, hidden_dim=96, n_layers=4,
                                     fourier_features=ff, fourier_period=T)
        params = list(model.parameters())
        for p in rate_params.values():
            params.extend(p.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(n_epochs):
            opt.zero_grad()
            if spectral:
                tt = t_data.clone().requires_grad_(True)
                rho_lab = model.rho_lab(tt)
                pred = torch.stack([torch.einsum('bij,ji->b', rho_lab, obs_torch[k]).real
                                    for k in obs_names], dim=-1)
                d_loss = data_loss(pred, target)
                t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
                env = model(t_colloc)
                p_loss = interaction_residual_loss(
                    env, t_colloc, lambda rt, t: interaction_rhs(rt, t, rate_params))
                ic_loss = initial_condition_loss(model.rho_lab(torch.zeros(1)), rho0_torch)
            else:
                rho_pred = model(t_data.clone().requires_grad_(True))
                pred = torch.stack([torch.einsum('bij,ji->b', rho_pred, obs_torch[k]).real
                                    for k in obs_names], dim=-1)
                d_loss = data_loss(pred, target)
                t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
                p_loss = physics_residual_loss(
                    model(t_colloc), t_colloc,
                    lambda rho: lab_lindblad_rhs(rho, rate_params))
                ic_loss = initial_condition_loss(model(torch.zeros(1)), rho0_torch)
            (1.0 * d_loss + w_phys * p_loss + 10.0 * ic_loss).backward()
            opt.step()
        t_fine = torch.linspace(0, t_max, 150)
        with torch.no_grad():
            if spectral:
                rhos_pred = torch_to_numpy_complex(model.rho_lab(t_data))
                lab_fine = torch_to_numpy_complex(model.rho_lab(t_fine))
            else:
                rhos_pred = torch_to_numpy_complex(model(t_data))
                lab_fine = torch_to_numpy_complex(model(t_fine))
        fid = state_fidelity_over_time(rhos_pred, rhos_exact)
        # Unified rate readout: rotate each method's learned trajectory to the
        # interaction-picture envelope, then fit the linear-in-rates generator by
        # the noise-robust integral (matrix-exponential) least squares. The
        # spectral frame yields an accurate envelope, so its rates are recoverable;
        # the unreconstructed baselines yield corrupted envelopes and rates.
        tf = t_fine.numpy()
        env_fine = np.stack([sp.interaction_envelope(lab_fine[i], tf[i], E_eig, V_eig)
                             for i in range(len(tf))])
        g = sp.fit_rates_integral(env_fine, tf, [J_10, J_21, J_phi])
        learned = {'gamma_10': g[0], 'gamma_21': g[1], 'gamma_phi': g[2]}
        rate_err = np.mean([abs(learned[k] - TRUE_RATES[k]) / TRUE_RATES[k]
                            for k in TRUE_RATES])
        dom_err = np.mean([abs(g[0] - 0.5) / 0.5, abs(g[1] - 0.2) / 0.2])
        return float(np.mean(fid)), float(rate_err), float(dom_err)

    fids = {m: [] for m in METHODS}
    rate_errs = {m: [] for m in METHODS}
    dom_errs = {m: [] for m in METHODS}
    for si, seed in enumerate(seeds):
        if verbose:
            print(f"  seed {seed} ({si+1}/{len(seeds)})...")
        for m in METHODS:
            fid, rerr, derr = train_model(m, seed)
            fids[m].append(fid)
            rate_errs[m].append(rerr)
            dom_errs[m].append(derr)
            if verbose:
                print(f"    {m:16s} fid={fid:.4f}  rate_err={rerr:.3f}  dom_rate_err={derr:.3f}")

    plot_qutrit_fidelity_comparison(
        fids, os.path.join(figures_dir, 'exp3_qutrit_leakage.pdf'))

    table_path = os.path.join(tables_dir, 'exp3_leakage_results.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n\\centering\n")
        f.write(r"\caption{Experiment 3: fast qutrit reconstruction. Mean state fidelity and mean "
                r"relative recovery error of the dominant decay rates ($\gamma_{10},\gamma_{21}$) "
                r"for a plain MLP, a Fourier-feature density network, and the spectral-frame "
                r"(interaction-picture) density network. Rates are read out by an integral "
                r"least-squares fit of the linear-in-rates generator to the learned "
                r"interaction-picture envelope; this readout is only meaningful for a reconstructed "
                r"trajectory, so it is reported only for methods reaching mean fidelity $\ge0.9$ "
                r"(`---' otherwise). Mean $\pm$ 95\% CI over " + f"{len(seeds)} seeds." + r"}" + "\n")
        f.write(r"\label{tab:exp3}" + "\n")
        f.write(r"\begin{tabular}{lcc}" + "\n\\toprule\n")
        f.write(r"Density network & Mean state fidelity & Dominant-rate error \\"
                + "\n\\midrule\n")
        for m in METHODS:
            mean_fid = float(np.mean(fids[m]))
            rate_cell = fmt_pm(dom_errs[m], digits=3) if mean_fid >= 0.9 else r"---"
            f.write(f"{m} & " + fmt_pm(fids[m], digits=4) + " & " + rate_cell + r" \\" + "\n")
        f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

    if verbose:
        for m in METHODS:
            print(f"  {m}: fid={fmt_pm(fids[m])} | dom_rate={fmt_pm(dom_errs[m], digits=3)} "
                  f"| all_rate={fmt_pm(rate_errs[m], digits=3)}")
        print("  Experiment 3 complete.")

    return {'fids': fids, 'rate_errs': rate_errs, 'dom_errs': dom_errs,
            'seeds': seeds, 'methods': METHODS}


if __name__ == '__main__':
    run()
