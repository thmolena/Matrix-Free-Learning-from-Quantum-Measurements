"""Experiment 6: operator-system spectral truncation of the open-system generator.

This experiment realizes the central new theoretical object of the manuscript: the
*spectral truncation* of a Lindblad/GKSL generator, lifted from the static
C*-algebraic / operator-system setting (Connes--van Suijlekom; spectral truncation
kernels) to a dynamical, completely-positive generator.

A driven, dissipative two-qubit system is moved into the interaction picture set
by its large local detunings. There the generator L_tilde(t) is almost periodic;
on the window [0, T] it has a Fourier series L_tilde(t) = sum_k e^{i k w0 t} L_k.
The level-N spectral truncation keeps the |k| <= N band -- a projection onto a
finite operator system of slow Bohr eigen-operators. We propagate the level-N
truncated generator (re-rotating the envelope back to the lab frame) and report:

  * the mean state fidelity of the truncated trajectory to the exact one;
  * the out-of-band spectral weight sum_{|k|>N} ||L_k||, which upper-bounds the
    generator-approximation error (the a-posteriori truncation certificate).

N = 0 is the secular (Davies) generator and N = 1 the rotating-wave approximation;
as N grows the truncated generator converges to the exact generator and the
fidelity reaches floating-point precision -- the structured/dense kernels are the
full-band special case. This is deterministic (no training, no seeds).
"""

import os
from pathlib import Path

import numpy as np

from ctpcpinn.operators import (pauli_x, pauli_y, pauli_z, identity,
                                sigma_minus, kron_n)
from ctpcpinn.solvers import solve_lindblad_trajectory
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn import spectral as sp
from ctpcpinn.plotting import plot_spectral_truncation

# Levels reported (retained Bohr-band index). 0 = secular, 1 = RWA, ... -> exact.
DEFAULT_LEVELS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]
_LABEL = {0: 'secular (Davies)', 1: 'rotating-wave', }


def run(config: dict = None):
    if config is None:
        config = {}
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    verbose = config.get('verbose', True)
    levels = config.get('trunc_levels', DEFAULT_LEVELS)
    n_grid = config.get('trunc_fft_grid', 4096)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    # --- Driven dissipative two-qubit system (multi-harmonic interaction frame).
    I2 = identity(2)
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    sm = sigma_minus()
    XX = kron_n([sx, sx]); YY = kron_n([sy, sy])
    ZI = kron_n([sz, I2]); IZ = kron_n([I2, sz])
    smI = kron_n([sm, I2]); Ism = kron_n([I2, sm])

    Delta1 = 2.0 * np.pi * 5.0
    Delta2 = 2.0 * np.pi * 4.8
    J0 = 2.0 * np.pi * 0.4            # strong exchange -> non-trivial harmonics
    gamma1, gamma2 = 0.1, 0.08
    H0 = 0.5 * Delta1 * ZI + 0.5 * Delta2 * IZ     # frame Hamiltonian (diagonal)

    def Hc_fn(t):                     # residual coherent coupling (not in the frame)
        return J0 * np.exp(-(t - 1.5) ** 2 / (2 * 0.3 ** 2)) * (XX + YY)

    def H_full(t):
        return H0 + Hc_fn(t)

    def Ls_fn(t):
        return [np.sqrt(gamma1) * smI, np.sqrt(gamma2) * Ism]

    d = 4
    T = 3.0
    psi = np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2)   # |+0>
    rho0 = np.outer(psi, psi.conj())
    t_grid = np.linspace(0, T, 80)

    rhos_exact = solve_lindblad_trajectory(H_full, Ls_fn, rho0, t_grid)

    # Interaction-frame spectrum and Fourier modes of the generator.
    E, V = sp.eigh_hamiltonian(H0)
    modes, norms = sp.generator_fourier_modes(E, V, Hc_fn, Ls_fn, T, n_grid=n_grid)

    max_level = (n_grid // 2)
    fids, oob, max_infid = [], [], []
    for N in levels:
        Nc = min(N, max_level)
        traj = sp.propagate_truncated(modes, Nc, E, V, rho0, t_grid, T)
        f = state_fidelity_over_time(traj, rhos_exact)
        fids.append(float(np.mean(f)))
        max_infid.append(float(np.max(1.0 - f)))
        oob.append(sp.spectral_truncation_error(norms, Nc))
        if verbose:
            print(f"  N={N:3d}  mean_fid={fids[-1]:.6f}  max_infid={max_infid[-1]:.2e}"
                  f"  out-of-band weight={oob[-1]:.3e}")

    plot_spectral_truncation(levels, fids, oob,
                             os.path.join(figures_dir, 'exp6_spectral_truncation.pdf'))

    # Table: a representative subset of levels.
    show = [0, 1, 2, 4, 8, 16]
    table_path = os.path.join(tables_dir, 'exp6_spectral_truncation.tex')
    with open(table_path, 'w') as fh:
        fh.write(r"\begin{table}[ht]" + "\n\\centering\n")
        fh.write(r"\caption{Experiment 6: operator-system spectral truncation of the "
                 r"interaction-picture generator. Mean and worst-case state fidelity of the "
                 r"level-$N$ truncated generator propagated against the exact trajectory, and "
                 r"the out-of-band spectral weight $\sum_{|k|>N}\lVert L_k\rVert$ that bounds "
                 r"the generator-approximation error. $N{=}0$ is the secular (Davies) generator "
                 r"and $N{=}1$ the rotating-wave approximation; the full band recovers the exact "
                 r"generator. Deterministic (no seeds).}" + "\n")
        fh.write(r"\label{tab:exp6}" + "\n")
        fh.write(r"\begin{tabular}{llccc}" + "\n\\toprule\n")
        fh.write(r"$N$ & Interpretation & Mean fidelity & Max infidelity & Out-of-band weight \\"
                 + "\n\\midrule\n")
        lev_to_idx = {N: i for i, N in enumerate(levels)}
        for N in show:
            if N not in lev_to_idx:
                continue
            i = lev_to_idx[N]
            interp = _LABEL.get(N, 'partial band')
            fh.write(f"{N} & {interp} & {fids[i]:.6f} & {max_infid[i]:.2e} & "
                     f"{oob[i]:.3e} " + r"\\" + "\n")
        fh.write(r"\bottomrule" + "\n\\end{tabular}\n\\end{table}\n")

    if verbose:
        print("  Experiment 6 complete.")

    return {'levels': levels, 'fids': fids, 'oob': oob, 'max_infid': max_infid,
            'mode_norms': norms}


if __name__ == '__main__':
    run()
