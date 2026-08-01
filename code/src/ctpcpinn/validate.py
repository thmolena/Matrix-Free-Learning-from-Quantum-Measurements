#!/usr/bin/env python
"""Quick validation script (no pytest required)."""

import sys
import os

import numpy as np

def test_operators():
    from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, identity, sigma_minus, commutator
    sx = pauli_x()
    sy = pauli_y()
    sz = pauli_z()
    assert sx.shape == (2, 2)
    assert np.allclose(sx @ sx, np.eye(2))
    assert np.allclose(sy @ sy, np.eye(2))
    assert np.allclose(sz @ sz, np.eye(2))
    # [sx, sy] = 2i sz
    assert np.allclose(commutator(sx, sy), 2j * sz)
    print("  [PASS] operators")

def test_lindblad_trace():
    from ctpcpinn.operators import pauli_z, sigma_minus
    from ctpcpinn.lindblad import lindblad_rhs, check_density_matrix
    sz = pauli_z()
    sm = sigma_minus()
    H = 0.5 * sz
    Ls = [np.sqrt(0.3) * sm, np.sqrt(0.1) * sz]
    rng = np.random.default_rng(42)
    for d in [2, 3, 4]:
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Hd = (A + A.conj().T) / 2
        Lsd = [0.1 * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))]
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)
        drho = lindblad_rhs(rho, Hd, Lsd)
        assert abs(np.trace(drho)) < 1e-10, f"d={d}: Tr(drho)={np.trace(drho)}"
    print("  [PASS] lindblad trace preservation")

def test_density_net():
    import torch
    from ctpcpinn.models import DensityMatrixNet, PositiveParameter
    torch.manual_seed(42)
    for d in [2, 3, 4]:
        model = DensityMatrixNet(d=d, hidden_dim=32, n_layers=2)
        t = torch.linspace(0, 1, 5)
        with torch.no_grad():
            rho = model(t)
        rho_np = rho.numpy()
        for i in range(5):
            # Hermiticity
            herm_err = np.max(np.abs(rho_np[i] - rho_np[i].conj().T))
            assert herm_err < 1e-5, f"d={d} Herm err={herm_err}"
            # Trace
            tr = np.trace(rho_np[i])
            assert abs(tr - 1.0) < 1e-5, f"d={d} Trace err={abs(tr-1)}"
            # PSD
            eigvals = np.linalg.eigvalsh(rho_np[i])
            assert np.min(eigvals) >= -1e-6, f"d={d} Min eig={np.min(eigvals)}"
    print("  [PASS] DensityMatrixNet (Hermitian, PSD, trace-1)")

def test_positive_param():
    import torch
    from ctpcpinn.models import PositiveParameter
    for init in [0.01, 0.1, 1.0, 5.0]:
        pp = PositiveParameter(init_value=init)
        val = pp()
        assert val.item() > 0, f"softplus gave non-positive for init={init}"
    print("  [PASS] PositiveParameter always positive")

def test_compiler():
    from ctpcpinn.operators import pauli_z, sigma_minus
    from ctpcpinn.compiler import QuantumModelIR, CompiledLindbladModel
    sz = pauli_z()
    sm = sigma_minus()
    def h_fn(params, t): return 0.5 * sz
    def ls_fn(params, t): return [np.sqrt(0.3) * sm]
    ir = QuantumModelIR(dimension=2, hamiltonian_fn=h_fn, lindblad_ops_fn=ls_fn, observables={'sz': sz})
    compiled_d = CompiledLindbladModel(ir, mode='dense')
    compiled_s = CompiledLindbladModel(ir, mode='structured')
    rng = np.random.default_rng(42)
    A = rng.standard_normal((2,2)) + 1j * rng.standard_normal((2,2))
    rho = A @ A.conj().T / np.trace(A @ A.conj().T)
    drho_d = compiled_d.rhs(0.0, rho, {})
    drho_s = compiled_s.rhs(0.0, rho, {})
    assert np.allclose(drho_d, drho_s, atol=1e-12), "Dense and structured disagree!"
    print("  [PASS] Compiler dense/structured agree")

def test_spectral_frame():
    """Spectral-frame (interaction-picture) machinery: physicality of the lab
    state, exact lab<->envelope equivalence, and numerical convergence of the
    symmetric Fourier approximation to its sampled full-band representation."""
    import torch
    from ctpcpinn.operators import projector
    from ctpcpinn.solvers import solve_lindblad_trajectory
    from ctpcpinn.models import SpectralDensityNet
    from ctpcpinn.metrics import state_fidelity_over_time
    from ctpcpinn import spectral as sp

    d = 3
    E1, E2 = 2 * np.pi * 2.0, 2 * np.pi * 3.8
    H = E1 * projector(d, 1, 1) + E2 * projector(d, 2, 2)
    Ls = [np.sqrt(0.5) * projector(d, 0, 1), np.sqrt(0.2) * projector(d, 1, 2),
          np.sqrt(0.1) * np.diag([0, 1, 2]).astype(np.complex128)]
    psi0 = np.array([0.8, 0.5, 0.3], dtype=np.complex128); psi0 /= np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())
    t_grid = np.linspace(0, 4.0, 60)
    rhos_exact = solve_lindblad_trajectory(lambda t: H, lambda t: Ls, rho0, t_grid)
    E, V = sp.eigh_hamiltonian(H)

    # (a) lab <-> interaction-picture roundtrip is exact.
    err = max(np.max(np.abs(sp.lab_from_envelope(
        sp.interaction_envelope(rhos_exact[i], t_grid[i], E, V), t_grid[i], E, V)
        - rhos_exact[i])) for i in range(len(t_grid)))
    assert err < 1e-10, f"interaction-picture roundtrip err={err}"

    # (b) the spectral-frame network's lab state is a valid density matrix.
    torch.manual_seed(0)
    net = SpectralDensityNet(H, hidden_dim=32, n_layers=3)
    with torch.no_grad():
        rho_lab = net.rho_lab(torch.linspace(0, 4.0, 8)).numpy()
    for r in rho_lab:
        assert np.max(np.abs(r - r.conj().T)) < 1e-4
        assert abs(np.trace(r) - 1.0) < 1e-4
        assert np.linalg.eigvalsh(0.5 * (r + r.conj().T)).min() >= -1e-5
    print("  [PASS] spectral frame (physical lab state, exact IP roundtrip)")

    # (c) a sufficiently wide Fourier band reproduces the reference trajectory.
    modes, norms = sp.generator_fourier_modes(E, V, None, lambda t: Ls, 4.0, n_grid=1024)
    traj = sp.propagate_truncated(modes, 64, E, V, rho0, t_grid, 4.0)
    assert np.mean(state_fidelity_over_time(traj, rhos_exact)) > 0.999
    print("  [PASS] symmetric Fourier approximation -> sampled full band")


def test_solver():
    from ctpcpinn.operators import pauli_z, sigma_minus
    from ctpcpinn.solvers import solve_lindblad_trajectory
    from ctpcpinn.lindblad import check_density_matrix
    sz = pauli_z()
    sm = sigma_minus()
    def H_fn(t): return 0.5 * sz
    def Ls_fn(t): return [np.sqrt(0.3) * sm]
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    t_grid = np.linspace(0, 2, 20)
    rhos = solve_lindblad_trajectory(H_fn, Ls_fn, rho0, t_grid)
    assert rhos.shape == (20, 2, 2)
    for i in range(20):
        result = check_density_matrix(rhos[i])
        assert result['trace_error'] < 1e-8
        assert result['min_eigenvalue'] >= -1e-8
    print("  [PASS] ODE solver produces valid states")

def main():
    """Run all dependency-free invariant checks (console script ctpcpinn-validate)."""
    import os
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    print("Running validation tests...")
    test_operators()
    test_lindblad_trace()
    test_density_net()
    test_positive_param()
    test_compiler()
    test_spectral_frame()
    test_solver()
    print("\nAll tests PASSED.")
    return 0


if __name__ == '__main__':
    main()
