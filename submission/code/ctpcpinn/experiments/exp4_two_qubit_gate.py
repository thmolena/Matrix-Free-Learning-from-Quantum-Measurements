"""Experiment 4: two-qubit dissipative gate, generalization across initial states.

The network is trained on the |00> trajectory only. The scientifically
meaningful test is whether the *learned generator* (known detunings, learned
coupling and dissipative rates) -- rather than the single trained trajectory --
predicts held-out initial states. We propagate the recovered Lindbladian from
|01>, |10>, and |+0> and compare to the exact dynamics, over an independent set
of seeds. Reported fidelities are mean +/- 95% CI over the seeds.
"""

import sys
import os

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
from ctpcpinn.plotting import plot_gate_fidelity_multiseed
from ctpcpinn.stats import aggregate, fmt_pm
from ctpcpinn.utils import (set_seed, numpy_to_torch_complex, torch_to_numpy_complex,
                            torch_lindblad_rhs)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
STATE_ORDER = ['|00>', '|01>', '|10>', '|+0>']
STATE_TEX = {'|00>': r'$\lvert 00\rangle$', '|01>': r'$\lvert 01\rangle$',
             '|10>': r'$\lvert 10\rangle$', '|+0>': r'$\lvert {+}0\rangle$'}
# matplotlib mathtext-safe labels (no \lvert) for figure legends
STATE_PLOT = {'|00>': r'$|00\rangle$', '|01>': r'$|01\rangle$',
              '|10>': r'$|10\rangle$', '|+0>': r'$|{+}0\rangle$'}


def run(config: dict = None):
    if config is None:
        config = {}
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    n_epochs = config.get('n_epochs', 3000)
    n_time_points = config.get('n_time_points', 60)
    fourier_features = config.get('fourier_features', 32)
    seeds = config.get('seeds', DEFAULT_SEEDS)
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    I2 = identity(2)
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()
    XX = kron_n([sx, sx]); YY = kron_n([sy, sy])
    ZI = kron_n([sz, I2]); IZ = kron_n([I2, sz])
    smI = kron_n([sm, I2]); Ism = kron_n([I2, sm])
    szI = kron_n([sz, I2]); Isz = kron_n([I2, sz])

    J = 2.0 * np.pi * 0.05
    Delta1 = 2.0 * np.pi * 5.0
    Delta2 = 2.0 * np.pi * 4.8
    gamma1, gamma2, gamma_phi1, gamma_phi2 = 0.1, 0.08, 0.05, 0.04

    def H_fn(t):
        Jt = J * np.exp(-(t - 1.5) ** 2 / (2 * 0.3 ** 2))
        return Jt * (XX + YY) + 0.5 * Delta1 * ZI + 0.5 * Delta2 * IZ

    def Ls_fn(t):
        return [np.sqrt(gamma1) * smI, np.sqrt(gamma2) * Ism,
                np.sqrt(gamma_phi1 / 2) * szI, np.sqrt(gamma_phi2 / 2) * Isz]

    init_states = {
        '|00>': np.array([1, 0, 0, 0], dtype=np.complex128),
        '|01>': np.array([0, 1, 0, 0], dtype=np.complex128),
        '|10>': np.array([0, 0, 1, 0], dtype=np.complex128),
        '|+0>': np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2),
    }
    t_grid = np.linspace(0, 3.0, n_time_points)
    t_max = float(t_grid[-1])

    psi_train = init_states['|00>']
    rho0_train = np.outer(psi_train, psi_train.conj())
    rhos_exact_train = solve_lindblad_trajectory(H_fn, Ls_fn, rho0_train, t_grid)
    obs_ops = {'ZI': ZI, 'IZ': IZ, 'XX': XX, 'YY': YY}

    obs_torch = {k: numpy_to_torch_complex(v) for k, v in obs_ops.items()}
    t_data = torch.tensor(t_grid, dtype=torch.float32)
    rho0_torch = numpy_to_torch_complex(rho0_train)
    XX_t, YY_t = numpy_to_torch_complex(XX), numpy_to_torch_complex(YY)
    ZI_t, IZ_t = numpy_to_torch_complex(ZI), numpy_to_torch_complex(IZ)
    smI_t, Ism_t = numpy_to_torch_complex(smI), numpy_to_torch_complex(Ism)
    szI_t, Isz_t = numpy_to_torch_complex(szI), numpy_to_torch_complex(Isz)
    obs_names = list(obs_torch.keys())

    # Pre-solve exact held-out trajectories (seed independent).
    exact_eval = {name: solve_lindblad_trajectory(H_fn, Ls_fn,
                  np.outer(psi, psi.conj()), t_grid) for name, psi in init_states.items()}

    def train_one(seed):
        set_seed(seed)
        measurements = generate_measurements(rhos_exact_train, obs_ops, noise_std=0.02, seed=seed)
        target = torch.stack([torch.tensor(measurements[k], dtype=torch.float32)
                              for k in obs_names], dim=-1)
        model = DensityMatrixNet(d=4, hidden_dim=128, n_layers=4,
                                 fourier_features=fourier_features, fourier_period=3.0)
        g1, g2 = PositiveParameter(0.2), PositiveParameter(0.2)
        gp1, gp2 = PositiveParameter(0.1), PositiveParameter(0.1)
        J_param = nn.Parameter(torch.tensor(0.2))
        rate_params = {'gamma1': g1, 'gamma2': g2, 'gamma_phi1': gp1,
                       'gamma_phi2': gp2, 'J': J_param}

        def rhs(rho):
            H = J_param * (XX_t + YY_t) + 0.5 * Delta1 * ZI_t + 0.5 * Delta2 * IZ_t
            L1 = torch.sqrt(g1()) * smI_t
            L2 = torch.sqrt(g2()) * Ism_t
            L3 = torch.sqrt(gp1() / 2) * szI_t
            L4 = torch.sqrt(gp2() / 2) * Isz_t
            return torch_lindblad_rhs(rho, H, [L1, L2, L3, L4])

        params = list(model.parameters()) + [J_param]
        for p in [g1, g2, gp1, gp2]:
            params.extend(p.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        for _ in range(n_epochs):
            opt.zero_grad()
            rho_pred = model(t_data.clone().requires_grad_(True))
            pred = torch.stack([torch.einsum('bij,ji->b', rho_pred, obs_torch[k]).real
                                for k in obs_names], dim=-1)
            d_loss = data_loss(pred, target)
            t_col = torch.linspace(0, t_max, 40).requires_grad_(True)
            p_loss = physics_residual_loss(model(t_col), t_col, lambda r: rhs(r))
            ic_loss = initial_condition_loss(model(torch.zeros(1)), rho0_torch)
            (1.0 * d_loss + 0.3 * p_loss + 10.0 * ic_loss).backward()
            opt.step()

        # Build learned generator and propagate held-out states.
        J_l = float(J_param.detach())
        g1L, g2L, gp1L, gp2L = g1.value, g2.value, gp1.value, gp2.value

        def H_l(t):
            return J_l * (XX + YY) + 0.5 * Delta1 * ZI + 0.5 * Delta2 * IZ

        def Ls_l(t):
            return [np.sqrt(g1L) * smI, np.sqrt(g2L) * Ism,
                    np.sqrt(gp1L / 2) * szI, np.sqrt(gp2L / 2) * Isz]

        fids = {}
        for name, psi in init_states.items():
            if name == '|00>':
                with torch.no_grad():
                    rhos_pred = torch_to_numpy_complex(model(t_data))
            else:
                rhos_pred = solve_lindblad_trajectory(H_l, Ls_l,
                            np.outer(psi, psi.conj()), t_grid)
            fids[name] = state_fidelity_over_time(rhos_pred, exact_eval[name])
        return fids

    mean_fid = {s: [] for s in STATE_ORDER}
    final_fid = {s: [] for s in STATE_ORDER}
    avg_fid = []
    rep_fids = None
    for si, seed in enumerate(seeds):
        if verbose:
            print(f"  seed {seed} ({si+1}/{len(seeds)})...")
        fids = train_one(seed)
        for s in STATE_ORDER:
            mean_fid[s].append(float(np.mean(fids[s])))
            final_fid[s].append(float(fids[s][-1]))
        avg_fid.append(float(np.mean([np.mean(fids[s]) for s in STATE_ORDER])))
        if si == 0:
            rep_fids = fids

    plot_gate_fidelity_multiseed(t_grid, {STATE_PLOT[s]: rep_fids[s] for s in STATE_ORDER},
                                 os.path.join(figures_dir, 'exp4_gate_fidelity.pdf'))

    table_path = os.path.join(tables_dir, 'exp4_gate_results.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[ht]" + "\n\\centering\n")
        f.write(r"\caption{Experiment 4: two-qubit dissipative gate. State fidelity for the "
                r"trained $\lvert 00\rangle$ trajectory and for three held-out initial states "
                r"obtained by propagating the learned generator. Mean $\pm$ 95\% CI over "
                + f"{len(seeds)} seeds." + r"}" + "\n")
        f.write(r"\label{tab:exp4}" + "\n")
        f.write(r"\begin{tabular}{lcc}" + "\n\\toprule\n")
        f.write(r"Initial state & Mean fidelity & Final fidelity \\" + "\n\\midrule\n")
        for s in STATE_ORDER:
            tag = STATE_TEX[s] + (r" (trained)" if s == '|00>' else r" (held-out)")
            f.write(f"{tag} & " + fmt_pm(mean_fid[s], digits=4) + " & "
                    + fmt_pm(final_fid[s], digits=4) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"Average & " + fmt_pm(avg_fid, digits=4) + r" & --- \\" + "\n")
        f.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

    if verbose:
        for s in STATE_ORDER:
            print(f"  {s}: mean={fmt_pm(mean_fid[s])}")
        print(f"  Average: {fmt_pm(avg_fid)}")
        print("  Experiment 4 complete.")

    return {'mean_fid': mean_fid, 'final_fid': final_fid, 'avg_fid': avg_fid, 'seeds': seeds}


if __name__ == '__main__':
    run()
