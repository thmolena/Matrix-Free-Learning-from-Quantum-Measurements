"""Test that Lindblad RHS preserves trace."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from ctpcpinn.operators import pauli_x, pauli_y, pauli_z, sigma_minus, identity
from ctpcpinn.lindblad import lindblad_rhs, check_density_matrix


class TestLindbladTrace:
    """Test trace preservation of the Lindblad equation."""

    def test_trace_of_drho_is_zero(self):
        """Tr(d rho/dt) = 0 for any valid rho, H, Ls."""
        sz = pauli_z()
        sm = sigma_minus()
        H = 0.5 * sz
        Ls = [np.sqrt(0.3) * sm, np.sqrt(0.1) * sz]

        # Random valid density matrix
        rng = np.random.default_rng(42)
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        rho = A @ A.conj().T
        rho /= np.trace(rho)

        drho = lindblad_rhs(rho, H, Ls)
        trace_drho = np.trace(drho)
        assert abs(trace_drho) < 1e-12, f"Tr(drho/dt) = {trace_drho}"

    def test_trace_preservation_many_random(self):
        """Test trace preservation for many random systems."""
        rng = np.random.default_rng(123)
        for d in [2, 3, 4]:
            for _ in range(10):
                # Random H
                A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                H = (A + A.conj().T) / 2

                # Random Ls
                n_ops = rng.integers(1, 4)
                Ls = [0.1 * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
                      for _ in range(n_ops)]

                # Random rho
                B = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
                rho = B @ B.conj().T
                rho /= np.trace(rho)

                drho = lindblad_rhs(rho, H, Ls)
                assert abs(np.trace(drho)) < 1e-10

    def test_check_density_matrix_valid(self):
        """check_density_matrix returns small errors for valid states."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        rho = A @ A.conj().T
        rho /= np.trace(rho)

        result = check_density_matrix(rho)
        assert result['trace_error'] < 1e-12
        assert result['min_eigenvalue'] >= -1e-12
        assert result['hermiticity_error'] < 1e-12
