"""Evaluation metrics for CPTP-Compiler-PINNs."""

import numpy as np
import time
from .lindblad import fidelity_qubit_or_general, trace_distance, check_density_matrix


def state_fidelity_over_time(rhos_pred: np.ndarray, rhos_exact: np.ndarray) -> np.ndarray:
    """Compute fidelity F(rho_pred, rho_exact) at each time step.

    Args:
        rhos_pred: (N, d, d) predicted density matrices
        rhos_exact: (N, d, d) exact density matrices

    Returns:
        fidelities: (N,) array
    """
    N = rhos_pred.shape[0]
    fids = np.zeros(N)
    for i in range(N):
        fids[i] = fidelity_qubit_or_general(rhos_pred[i], rhos_exact[i])
    return fids


def trace_distance_over_time(rhos_pred: np.ndarray, rhos_exact: np.ndarray) -> np.ndarray:
    """Compute trace distance T(rho_pred, rho_exact) at each time step."""
    N = rhos_pred.shape[0]
    dists = np.zeros(N)
    for i in range(N):
        dists[i] = trace_distance(rhos_pred[i], rhos_exact[i])
    return dists


def parameter_relative_error(true_params: dict, learned_params: dict) -> dict:
    """Compute relative error |p_true - p_learned| / |p_true| for each parameter."""
    errors = {}
    for key in true_params:
        p_true = true_params[key]
        p_learned = learned_params.get(key, 0.0)
        if abs(p_true) > 1e-12:
            errors[key] = abs(p_true - p_learned) / abs(p_true)
        else:
            errors[key] = abs(p_learned)
    return errors


def positivity_violation_rate(rhos: np.ndarray, tol: float = -1e-8) -> float:
    """Fraction of time steps where min eigenvalue < tol."""
    N = rhos.shape[0]
    violations = 0
    for i in range(N):
        eigvals = np.linalg.eigvalsh(rhos[i])
        if np.min(eigvals) < tol:
            violations += 1
    return violations / N


def trace_error_over_time(rhos: np.ndarray) -> np.ndarray:
    """Compute |Tr(rho) - 1| at each time step."""
    N = rhos.shape[0]
    errs = np.zeros(N)
    for i in range(N):
        errs[i] = abs(np.trace(rhos[i]) - 1.0)
    return errs


def hermiticity_error_over_time(rhos: np.ndarray) -> np.ndarray:
    """Compute ||rho - rho^dag||_max at each time step."""
    N = rhos.shape[0]
    errs = np.zeros(N)
    for i in range(N):
        errs[i] = np.max(np.abs(rhos[i] - rhos[i].conj().T))
    return errs


class WallClockTimer:
    """Simple context manager for timing code blocks."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def memory_estimate_dense_vs_structured(d: int, m: int) -> dict:
    """Estimate memory for dense Liouvillian vs structured evaluation.

    Args:
        d: Hilbert space dimension
        m: number of Lindblad operators

    Returns:
        dict with memory estimates in bytes and MB
    """
    bytes_per_complex = 16  # complex128

    dense_elements = d ** 4
    dense_bytes = dense_elements * bytes_per_complex

    # Structured: H (d^2) + m Lindblad ops (m * d^2) + rho (d^2)
    structured_elements = d ** 2 + m * d ** 2 + d ** 2
    structured_bytes = structured_elements * bytes_per_complex

    return {
        'd': d,
        'm': m,
        'dense_bytes': dense_bytes,
        'structured_bytes': structured_bytes,
        'dense_MB': dense_bytes / (1024 ** 2),
        'structured_MB': structured_bytes / (1024 ** 2),
        'ratio': dense_bytes / max(structured_bytes, 1),
    }
