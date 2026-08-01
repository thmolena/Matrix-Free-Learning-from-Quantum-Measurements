"""Invariants and counterexamples for the spectral frame/Fourier approximation."""
import numpy as np
import pytest
import torch

from ctpcpinn.operators import projector
from ctpcpinn.solvers import solve_lindblad_trajectory
from ctpcpinn.models import SpectralDensityNet
from ctpcpinn.metrics import state_fidelity_over_time
from ctpcpinn.lindblad import liouvillian_dense, vectorize_rho, unvectorize_rho
from ctpcpinn.operators import sigma_minus
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


def test_symmetric_fourier_truncation_need_not_be_gksl():
    """Paired Fourier modes preserve trace/Hermiticity, but not positivity.

    The nonnegative amplitude-damping rate gamma(t)=(1+cos(t))^2 has the
    first-order symmetric Fourier partial sum 3/2+2 cos(t), which equals -1/2
    at t=pi.  A forward Euler step of that truncated generator sends the
    excited state to a matrix with a negative eigenvalue.  This exact example
    guards against describing raw superoperator truncation as GKSL preserving.
    """
    d = 2
    n_grid = 256
    period = 2.0 * np.pi
    times = np.arange(n_grid) * period / n_grid
    zero_hamiltonian = np.zeros((d, d), dtype=np.complex128)
    lowering = sigma_minus()
    sampled = np.stack([
        liouvillian_dense(
            zero_hamiltonian,
            [np.sqrt((1.0 + np.cos(t)) ** 2) * lowering],
        )
        for t in times
    ])
    modes = np.fft.fft(sampled, axis=0) / n_grid
    first_band = sp._truncate_modes(modes, 1)
    truncated_at_pi = sp.truncated_generator_at(first_band, np.pi, period)

    # The symmetric partial sum still preserves Hermiticity and trace.
    probe = np.array([[0.2, 0.3 + 0.1j], [0.3 - 0.1j, 0.8]])
    derivative = unvectorize_rho(
        truncated_at_pi @ vectorize_rho(probe), d
    )
    assert np.trace(derivative) == pytest.approx(0.0, abs=1e-13)
    assert np.allclose(derivative, derivative.conj().T, atol=1e-13)

    # But its rate is -1/2, so even an arbitrarily small positive step is not
    # positivity preserving on the excited state.
    excited = np.diag([0.0, 1.0]).astype(np.complex128)
    h = 0.1
    stepped = unvectorize_rho(
        (np.eye(d * d) + h * truncated_at_pi) @ vectorize_rho(excited), d
    )
    assert np.linalg.eigvalsh(0.5 * (stepped + stepped.conj().T)).min() < -0.049


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
