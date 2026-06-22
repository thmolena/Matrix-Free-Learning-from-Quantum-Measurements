"""Test compiler shapes and mode agreement."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from ctpcpinn.operators import pauli_z, sigma_minus, identity
from ctpcpinn.compiler import QuantumModelIR, CompiledLindbladModel


@pytest.fixture
def qubit_ir():
    """Simple qubit system IR for testing."""
    sz = pauli_z()
    sm = sigma_minus()

    def h_fn(params, t):
        return 0.5 * sz

    def ls_fn(params, t):
        return [np.sqrt(0.3) * sm]

    ir = QuantumModelIR(
        dimension=2,
        hamiltonian_fn=h_fn,
        lindblad_ops_fn=ls_fn,
        observables={'sz': sz},
        name='test_qubit'
    )
    return ir


class TestCompilerShapes:
    """Test that compiler outputs have correct shapes."""

    def test_rhs_shape(self, qubit_ir):
        compiled = CompiledLindbladModel(qubit_ir, mode='structured')
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        drho = compiled.rhs(0.0, rho, {})
        assert drho.shape == (2, 2)
        assert drho.dtype == np.complex128

    def test_measurement_output(self, qubit_ir):
        compiled = CompiledLindbladModel(qubit_ir)
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=np.complex128)
        result = compiled.measurement(rho)
        assert 'sz' in result
        assert isinstance(result['sz'], float)

    def test_residual_batch_shape(self, qubit_ir):
        compiled = CompiledLindbladModel(qubit_ir)
        N = 5
        t_batch = np.linspace(0, 1, N)
        rho_batch = np.zeros((N, 2, 2), dtype=np.complex128)
        rho_batch[:, 0, 0] = 1.0
        drho_batch = np.zeros_like(rho_batch)

        residuals = compiled.residual_batch(t_batch, rho_batch, drho_batch, {})
        assert residuals.shape == (N,)

    def test_dense_structured_agree(self, qubit_ir):
        """Dense and structured modes should give the same RHS."""
        compiled_dense = CompiledLindbladModel(qubit_ir, mode='dense')
        compiled_struct = CompiledLindbladModel(qubit_ir, mode='structured')

        rng = np.random.default_rng(42)
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        rho = A @ A.conj().T / np.trace(A @ A.conj().T)

        drho_dense = compiled_dense.rhs(0.5, rho, {})
        drho_struct = compiled_struct.rhs(0.5, rho, {})

        assert np.allclose(drho_dense, drho_struct, atol=1e-12), \
            f"Max disagreement: {np.max(np.abs(drho_dense - drho_struct))}"

    def test_memory_estimate(self, qubit_ir):
        compiled = CompiledLindbladModel(qubit_ir)
        mem = compiled.memory_estimate()
        assert 'dense_liouvillian_bytes' in mem
        assert 'structured_operator_bytes' in mem
        assert mem['dense_liouvillian_bytes'] > mem['structured_operator_bytes']


class TestHigherDimensions:
    """Test compiler with higher dimensions."""

    @pytest.mark.parametrize("d", [3, 4, 5])
    def test_dense_structured_agree_higher_d(self, d):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H = (A + A.conj().T) / 2
        Ls = [0.1 * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
              for _ in range(2)]

        def h_fn(params, t, H=H):
            return H

        def ls_fn(params, t, Ls=Ls):
            return Ls

        ir = QuantumModelIR(dimension=d, hamiltonian_fn=h_fn, lindblad_ops_fn=ls_fn)
        compiled_d = CompiledLindbladModel(ir, mode='dense')
        compiled_s = CompiledLindbladModel(ir, mode='structured')

        B = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = B @ B.conj().T / np.trace(B @ B.conj().T)

        drho_d = compiled_d.rhs(0.0, rho, {})
        drho_s = compiled_s.rhs(0.0, rho, {})

        assert np.allclose(drho_d, drho_s, atol=1e-10)
