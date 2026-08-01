"""Test CPTP generator properties."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy.linalg import expm
from ctpcpinn.operators import pauli_z, sigma_minus, identity
from ctpcpinn.lindblad import liouvillian_dense, vectorize_rho, unvectorize_rho


class TestCPTPGenerator:
    """Test that the Liouvillian generates a CPTP map."""

    def test_trace_preservation_generator(self):
        """The Liouvillian should satisfy: sum_j L[j, :] = 0 for each column block
        that enforces trace preservation, i.e., Tr(e^{Lt} rho) = 1."""
        sz = pauli_z()
        sm = sigma_minus()
        H = 0.5 * sz
        gamma = 0.3
        Ls = [np.sqrt(gamma) * sm]

        L_super = liouvillian_dense(H, Ls)
        d = 2

        # Check: for any rho, Tr(L[rho]) = 0
        # This means vec(I)^T @ L_super = 0 (identity vectorized as trace functional)
        I_vec = vectorize_rho(identity(d))
        # The trace functional in column-stacking is: sum of diagonal elements
        # trace(rho) = sum_k rho[k,k] = sum_k vec[k*d + k] for Fortran order
        trace_vec = np.zeros(d * d, dtype=np.complex128)
        for k in range(d):
            trace_vec[k * d + k] = 1.0

        # Trace preservation: trace_vec @ L_super = 0
        result = trace_vec @ L_super
        assert np.allclose(result, 0, atol=1e-12), \
            f"Trace preservation violated: max error {np.max(np.abs(result))}"

    def test_cptp_map_positivity(self):
        """e^{Lt} applied to valid rho should give valid rho."""
        sz = pauli_z()
        sm = sigma_minus()
        H = 0.5 * sz
        Ls = [np.sqrt(0.3) * sm, np.sqrt(0.1) * sz]

        L_super = liouvillian_dense(H, Ls)
        d = 2

        # Random valid initial state
        rng = np.random.default_rng(42)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho0 = A @ A.conj().T
        rho0 /= np.trace(rho0)

        # Evolve for various times
        for t in [0.1, 0.5, 1.0, 5.0]:
            vec_rho_t = expm(L_super * t) @ vectorize_rho(rho0)
            rho_t = unvectorize_rho(vec_rho_t, d)

            # Check trace
            assert abs(np.trace(rho_t) - 1.0) < 1e-10, \
                f"Trace error at t={t}: {abs(np.trace(rho_t) - 1)}"

            # Check Hermiticity
            assert np.allclose(rho_t, rho_t.conj().T, atol=1e-10), \
                f"Hermiticity violated at t={t}"

            # Check positivity
            eigvals = np.linalg.eigvalsh(rho_t)
            assert np.min(eigvals) >= -1e-10, \
                f"Negative eigenvalue at t={t}: {np.min(eigvals)}"

    def test_positive_rates_give_valid_generator(self):
        """Softplus-based positive rates should always give a valid Lindblad generator."""
        import torch
        from ctpcpinn.models import PositiveParameter

        for raw_val in [-5.0, -1.0, 0.0, 1.0, 10.0]:
            pp = PositiveParameter(init_value=1.0)
            pp.raw.data = torch.tensor(raw_val)
            gamma = pp()
            assert gamma.item() > 0, f"softplus({raw_val}) gave non-positive value"

    def test_primitive_path_lindbladian_can_have_zero_instantaneous_contraction(self):
        """Primitivity does not imply a negative trace-norm logarithmic norm.

        Bidirectional jumps on the connected four-vertex path define a primitive
        Lindbladian with the full-rank stationary state I/4.  On diagonal
        operators its restriction is the classical path generator.  For
        X=diag(1,0,0,-1), ||(I+hL)X||_1=||X||_1 for every 0<h<1, so the
        instantaneous logarithmic norm is zero, not strictly negative.
        """
        d = 4
        jumps = []
        for target, source in ((0, 1), (1, 0), (1, 2),
                               (2, 1), (2, 3), (3, 2)):
            jump = np.zeros((d, d), dtype=np.complex128)
            jump[target, source] = 1.0
            jumps.append(jump)

        generator = liouvillian_dense(
            np.zeros((d, d), dtype=np.complex128), jumps
        )
        stationary = identity(d) / d
        assert np.linalg.norm(generator @ vectorize_rho(stationary)) < 1e-12

        # The zero eigenvalue is simple and every other mode decays, which is
        # the finite-dimensional primitivity check used by the manuscript.
        eigenvalues = np.linalg.eigvals(generator)
        assert np.count_nonzero(np.abs(eigenvalues) < 1e-10) == 1
        assert np.max(eigenvalues.real[np.abs(eigenvalues) >= 1e-10]) < -0.5

        direction = np.diag([1.0, 0.0, 0.0, -1.0]).astype(np.complex128)
        initial_norm = np.linalg.norm(direction, ord="nuc")
        eye_super = np.eye(d * d, dtype=np.complex128)
        for h in (1e-8, 1e-5, 1e-2, 0.1):
            stepped = unvectorize_rho(
                (eye_super + h * generator) @ vectorize_rho(direction), d
            )
            assert np.linalg.norm(stepped, ord="nuc") == pytest.approx(
                initial_norm, abs=2e-12
            )

    def test_identity_in_operator_list_breaks_kossakowski_only_if_direction(self):
        """A noncanonical basis makes the GKS coefficient matrix nonunique.

        With F=I, its dissipator is identically zero.  Thus the negative scalar
        coefficient C=[-1] still represents the zero (hence CPTP) generator.
        The equivalence ``C PSD iff GKSL`` therefore requires a canonical
        traceless Hilbert--Schmidt orthonormal basis; PSD remains sufficient for
        any fixed operator list.
        """
        d = 3
        rho = np.array(
            [[0.4, 0.1j, 0.0], [-0.1j, 0.35, 0.02], [0.0, 0.02, 0.25]],
            dtype=np.complex128,
        )
        ident = identity(d)
        identity_dissipator = (
            ident @ rho @ ident
            - 0.5 * (ident @ ident @ rho + rho @ ident @ ident)
        )
        coefficient = -1.0
        assert coefficient < 0.0
        assert np.allclose(coefficient * identity_dissipator, 0.0, atol=1e-14)
