"""Experiment 2: Residual certificate — verify Theorem 4 (trace-norm bound).

This experiment takes the exact qubit GKSL trajectory, forms physical
perturbations by mixing with the maximally mixed state, computes residuals,
and verifies that the a-posteriori certificate upper-bounds the actual
trace distance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path
from scipy.linalg import expm

# Import directly from submodules to avoid triggering torch import via __init__.py
from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, sigma_minus, identity


def _dense_liouvillian(H, Ls, rates):
    """Build dense d^2 x d^2 Liouvillian superoperator (column-major convention)."""
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)
    L_super = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L, gamma in zip(Ls, rates):
        Ldag_L = L.conj().T @ L
        L_super += gamma * (
            np.kron(L.conj(), L)
            - 0.5 * np.kron(I, Ldag_L)
            - 0.5 * np.kron(Ldag_L.T, I)
        )
    return L_super


def _lindblad_rhs(rho, H, Ls, rates):
    """Evaluate L[rho] directly (structured mode)."""
    drho = -1j * (H @ rho - rho @ H)
    for L, gamma in zip(Ls, rates):
        Ldag_L = L.conj().T @ L
        drho += gamma * (L @ rho @ L.conj().T - 0.5 * (Ldag_L @ rho + rho @ Ldag_L))
    return drho


def _trace_norm(A):
    """Trace norm = sum of singular values (for Hermitian: sum |eigenvalues|)."""
    return np.sum(np.abs(np.linalg.eigvalsh(0.5 * (A + A.conj().T))))


def _set_seed(seed):
    np.random.seed(seed)


def run(config: dict = None):
    """Run experiment 2: residual certificate verification."""
    if config is None:
        config = {}

    _set_seed(config.get('seed', 42))
    figures_dir = config.get('figures_dir', 'submission/figures')
    tables_dir = config.get('tables_dir', 'submission/tables')
    verbose = config.get('verbose', True)

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    # --- System parameters matching the paper ---
    omega = np.pi          # qubit frequency
    Omega_drive = 0.3 * np.pi    # drive strength
    gamma1 = 0.35          # amplitude damping
    gamma_phi = 0.18       # dephasing

    sx = pauli_x()
    sy = pauli_y()
    sz = pauli_z()
    sm = sigma_minus()
    I2 = identity(2)

    H = 0.5 * omega * sz + 0.5 * Omega_drive * sx
    L1 = sm           # amplitude damping jump
    L2 = sz / np.sqrt(2)  # dephasing jump (sqrt included in rate)
    Ls = [L1, L2]
    rates = [gamma1, gamma_phi]

    # Initial state |+><+|
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

    # Time grid
    T = 5.0
    N_grid = 601
    t_grid = np.linspace(0, T, N_grid)
    dt = t_grid[1] - t_grid[0]

    # Solve exact trajectory via dense Liouvillian matrix exponential (column-major)
    L_super = _dense_liouvillian(H, Ls, rates)
    d = 2
    rho0_vec = rho0.flatten(order='F')
    rhos_exact = np.zeros((N_grid, d, d), dtype=np.complex128)
    for i, t in enumerate(t_grid):
        rho_vec = expm(L_super * t) @ rho0_vec
        rhos_exact[i] = rho_vec.reshape(d, d, order='F')

    # Perturbation amplitudes
    eps_values = [0.01, 0.025, 0.05, 0.1]
    results = []

    for eps in eps_values:
        # Perturbed trajectory: rho_eps(t) = (1 - a(t)) rho(t) + a(t) I/2
        # where a(t) = eps * sin^2(pi*t/T)
        a_t = eps * np.sin(np.pi * t_grid / T) ** 2
        rhos_pert = np.zeros_like(rhos_exact)
        for i in range(N_grid):
            rhos_pert[i] = (1 - a_t[i]) * rhos_exact[i] + a_t[i] * I2 / 2

        # Compute derivative of perturbed trajectory via finite differences
        drho_dt = np.zeros_like(rhos_pert)
        # Central differences for interior points
        for i in range(1, N_grid - 1):
            drho_dt[i] = (rhos_pert[i + 1] - rhos_pert[i - 1]) / (2 * dt)
        # Forward/backward at endpoints
        drho_dt[0] = (rhos_pert[1] - rhos_pert[0]) / dt
        drho_dt[-1] = (rhos_pert[-1] - rhos_pert[-2]) / dt

        # Compute residual r(t) = d rho_eps/dt - L[rho_eps(t)]
        residual_trace_norms = np.zeros(N_grid)
        for i in range(N_grid):
            L_rho = _lindblad_rhs(rhos_pert[i], H, Ls, rates)
            r_i = drho_dt[i] - L_rho
            residual_trace_norms[i] = _trace_norm(r_i)

        # Actual max trace distance
        actual_td = np.zeros(N_grid)
        for i in range(N_grid):
            diff = rhos_pert[i] - rhos_exact[i]
            actual_td[i] = 0.5 * _trace_norm(diff)
        max_td = np.max(actual_td)

        # Certificate: (1/2) ||rho_eps(0) - rho(0)||_1 + (1/2) integral_0^t ||r(s)||_1 ds
        # At t=0: a(0) = 0, so rho_eps(0) = rho(0), initial error = 0
        # Certificate at time t: (1/2) * integral_0^t ||r(s)||_1 ds
        # We want the max certificate over all t
        cumulative_cert = np.zeros(N_grid)
        for i in range(1, N_grid):
            # Trapezoidal integration of (1/2) ||r(s)||_1
            cumulative_cert[i] = cumulative_cert[i - 1] + 0.5 * dt * (
                residual_trace_norms[i - 1] + residual_trace_norms[i]) / 2
        cert = np.max(cumulative_cert)

        results.append({
            'eps': eps,
            'max_td': max_td,
            'cert': cert,
            'ratio': cert / max_td if max_td > 0 else float('inf'),
        })

        if verbose:
            print(f"  eps={eps:.3f}: max TD={max_td:.7f}, cert={cert:.7f}, "
                  f"ratio={cert / max_td:.2f}")

    # Write data file (matching paper's filecontents format)
    dat_path = os.path.join(tables_dir, 'cptp_qst_residual_certificate.dat')
    with open(dat_path, 'w') as f:
        f.write("eps maxTD cert\n")
        for r in results:
            f.write(f"{r['eps']} {r['max_td']:.7g} {r['cert']:.7g}\n")

    # Write LaTeX table
    table_path = os.path.join(tables_dir, 'exp2_residual_certificate.tex')
    with open(table_path, 'w') as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{A posteriori residual certificate for physical perturbations "
                r"of the exact qubit trajectory.}" + "\n")
        f.write(r"\label{tab:residual_certificate}" + "\n")
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Perturbation amplitude & Max trace distance & Certificate & "
                r"Certificate / error \\" + "\n")
        f.write(r"\midrule" + "\n")
        for r in results:
            f.write(f"{r['eps']:.3f} & {r['max_td']:.4f} & {r['cert']:.4f} & "
                    f"{r['ratio']:.2f} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    if verbose:
        print("  Experiment 2 complete.")

    return results


if __name__ == '__main__':
    run()
