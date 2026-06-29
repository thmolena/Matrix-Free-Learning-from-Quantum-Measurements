"""Invariants of the spectral frame and the operator-system spectral truncation."""
import numpy as np
import torch

from ctpcpinn.operators import projector
from ctpcpinn.solvers import solve_lindblad_trajectory
from ctpcpinn.models import SpectralDensityNet
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn import spectral as sp


def _qutrit():
    d = 3
    E1, E2 = 2 * np.pi * 2.0, 2 * np.pi * 3.8
    H = E1 * projector(d, 1, 1) + E2 * projector(d, 2, 2)
    Ls = [np.sqrt(0.5) * projector(d, 0, 1), np.sqrt(0.2) * projector(d, 1, 2),
          np.sqrt(0.1) * np.diag([0, 1, 2]).astype(np.complex128)]
    psi0 = np.array([0.8, 0.5, 0.3], dtype=np.complex128); psi0 /= np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())
    return d, H, Ls, rho0


def test_interaction_roundtrip_is_exact():
    d, H, Ls, rho0 = _qutrit()
    t_grid = np.linspace(0, 4.0, 50)
    rhos = solve_lindblad_trajectory(lambda t: H, lambda t: Ls, rho0, t_grid)
    E, V = sp.eigh_hamiltonian(H)
    for i, t in enumerate(t_grid):
        env = sp.interaction_envelope(rhos[i], t, E, V)
        back = sp.lab_from_envelope(env, t, E, V)
        assert np.max(np.abs(back - rhos[i])) < 1e-10


def test_spectral_density_net_lab_state_is_physical():
    d, H, Ls, rho0 = _qutrit()
    torch.manual_seed(0)
    net = SpectralDensityNet(H, hidden_dim=24, n_layers=2)
    with torch.no_grad():
        rho_lab = net.rho_lab(torch.linspace(0, 4.0, 6)).numpy()
    for r in rho_lab:
        assert np.max(np.abs(r - r.conj().T)) < 1e-4
        assert abs(np.trace(r) - 1.0) < 1e-4
        assert np.linalg.eigvalsh(0.5 * (r + r.conj().T)).min() >= -1e-5


def test_full_band_truncation_recovers_exact():
    d, H, Ls, rho0 = _qutrit()
    T = 4.0
    t_grid = np.linspace(0, T, 60)
    rhos = solve_lindblad_trajectory(lambda t: H, lambda t: Ls, rho0, t_grid)
    E, V = sp.eigh_hamiltonian(H)
    modes, norms = sp.generator_fourier_modes(E, V, None, lambda t: Ls, T, n_grid=1024)
    traj = sp.propagate_truncated(modes, 64, E, V, rho0, t_grid, T)
    assert np.mean(state_fidelity_over_time(traj, rhos)) > 0.999
    # out-of-band weight decreases as more modes are retained
    assert sp.spectral_truncation_error(norms, 8) <= sp.spectral_truncation_error(norms, 1)


def test_integral_rate_readout_on_exact_envelope():
    d, H, Ls, rho0 = _qutrit()
    J = [projector(d, 0, 1), projector(d, 1, 2), np.diag([0, 1, 2]).astype(np.complex128)]
    t_grid = np.linspace(0, 4.0, 80)
    rhos = solve_lindblad_trajectory(lambda t: H, lambda t: Ls, rho0, t_grid)
    E, V = sp.eigh_hamiltonian(H)
    env = np.stack([sp.interaction_envelope(rhos[i], t_grid[i], E, V)
                    for i in range(len(t_grid))])
    g = sp.fit_rates_integral(env, t_grid, J)
    assert np.allclose(g, [0.5, 0.2, 0.1], atol=1e-2)
