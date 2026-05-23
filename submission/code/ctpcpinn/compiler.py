"""Intermediate representation and compiler for Lindblad models."""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from .lindblad import lindblad_rhs, liouvillian_dense, vectorize_rho, unvectorize_rho


@dataclass
class QuantumModelIR:
    """Intermediate representation for an open quantum system model.

    Attributes:
        dimension: Hilbert space dimension d
        hamiltonian_fn: callable(params) -> H (d,d) complex array
        lindblad_ops_fn: callable(params) -> list of (d,d) Lindblad operators
        observables: dict mapping name -> (d,d) observable operator
        name: human-readable name
    """
    dimension: int
    hamiltonian_fn: Callable
    lindblad_ops_fn: Callable
    observables: dict = field(default_factory=dict)
    name: str = "unnamed"


class CompiledLindbladModel:
    """Compiled Lindblad model supporting two residual evaluation modes.

    Modes:
        'dense': construct full d^2 x d^2 Liouvillian, apply via matvec
        'structured': apply Lindblad terms directly without materializing Liouvillian
    """

    def __init__(self, ir: QuantumModelIR, mode: str = 'structured'):
        """Compile a QuantumModelIR into an evaluable model.

        Args:
            ir: QuantumModelIR instance
            mode: 'dense' or 'structured'
        """
        self.ir = ir
        self.mode = mode
        self.d = ir.dimension

    def rhs(self, t: float, rho: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate d rho/dt at a given time and state.

        Args:
            t: time
            rho: density matrix (d, d)
            params: dict of model parameters

        Returns:
            drho_dt: (d, d) complex array
        """
        H = self.ir.hamiltonian_fn(params, t)
        Ls = self.ir.lindblad_ops_fn(params, t)

        if self.mode == 'dense':
            L_super = liouvillian_dense(H, Ls)
            vec_rho = vectorize_rho(rho)
            vec_drho = L_super @ vec_rho
            return unvectorize_rho(vec_drho, self.d)
        else:
            return lindblad_rhs(rho, H, Ls)

    def rhs_vectorized(self, t: float, vec_rho: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate d vec(rho)/dt for ODE solvers.

        Args:
            t: time
            vec_rho: vectorized density matrix (d^2,)
            params: dict of model parameters

        Returns:
            d vec(rho)/dt: (d^2,) complex array
        """
        rho = unvectorize_rho(vec_rho, self.d)
        drho = self.rhs(t, rho, params)
        return vectorize_rho(drho)

    def measurement(self, rho: np.ndarray, obs_names: Optional[list] = None) -> dict:
        """Compute expectation values of observables.

        Args:
            rho: density matrix (d, d)
            obs_names: subset of observable names (None = all)

        Returns:
            dict mapping observable name -> expectation value
        """
        names = obs_names or list(self.ir.observables.keys())
        results = {}
        for name in names:
            O = self.ir.observables[name]
            results[name] = float(np.real(np.trace(O @ rho)))
        return results

    def residual_batch(self, t_batch: np.ndarray, rho_batch: np.ndarray,
                       drho_dt_batch: np.ndarray, params: dict) -> np.ndarray:
        """Compute residual ||drho/dt - L[rho]||_F^2 over a batch.

        Args:
            t_batch: (N,) time points
            rho_batch: (N, d, d) density matrices
            drho_dt_batch: (N, d, d) time derivatives
            params: model parameters

        Returns:
            residuals: (N,) Frobenius norm squared of residual at each point
        """
        N = len(t_batch)
        residuals = np.zeros(N)
        for i in range(N):
            rhs_val = self.rhs(t_batch[i], rho_batch[i], params)
            diff = drho_dt_batch[i] - rhs_val
            residuals[i] = np.real(np.sum(diff * diff.conj()))
        return residuals

    def memory_estimate(self) -> dict:
        """Estimate memory usage for dense vs structured modes."""
        d = self.d
        dense_bytes = (d ** 4) * 16  # complex128
        # Structured: store H (d^2) + m Lindblad ops (m * d^2)
        structured_bytes = (d ** 2) * 16  # Hamiltonian
        return {
            'dense_liouvillian_bytes': dense_bytes,
            'structured_operator_bytes': structured_bytes,
            'dense_liouvillian_MB': dense_bytes / (1024 ** 2),
            'structured_operator_MB': structured_bytes / (1024 ** 2),
        }


def benchmark_residual_modes(ir: QuantumModelIR, params: dict,
                              n_points: int = 100, n_repeats: int = 5) -> dict:
    """Benchmark dense vs structured residual evaluation.

    Returns:
        dict with timing and throughput for each mode
    """
    d = ir.dimension
    t_batch = np.linspace(0, 1, n_points)
    rho_batch = np.zeros((n_points, d, d), dtype=np.complex128)
    # Initialize with random valid density matrices
    for i in range(n_points):
        A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rho_batch[i] = A @ A.conj().T / np.trace(A @ A.conj().T)

    drho_dt_batch = np.zeros_like(rho_batch)

    results = {}
    for mode in ['dense', 'structured']:
        compiled = CompiledLindbladModel(ir, mode=mode)
        times_list = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            compiled.residual_batch(t_batch, rho_batch, drho_dt_batch, params)
            elapsed = time.perf_counter() - start
            times_list.append(elapsed)
        mean_time = np.mean(times_list)
        results[mode] = {
            'mean_time_s': mean_time,
            'evals_per_sec': n_points / mean_time if mean_time > 0 else float('inf'),
        }

    return results
