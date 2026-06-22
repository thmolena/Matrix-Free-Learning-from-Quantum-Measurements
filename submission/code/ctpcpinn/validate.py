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
    test_solver()
    print("\nAll tests PASSED.")
    return 0


if __name__ == '__main__':
    main()
