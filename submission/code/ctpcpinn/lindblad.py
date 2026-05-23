"""Lindblad master equation utilities."""

import numpy as np
from scipy.linalg import sqrtm
from .operators import commutator, anticommutator


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, Ls: list) -> np.ndarray:
    """Compute d rho/dt = -i[H, rho] + sum_k D[L_k](rho).

    Args:
        rho: density matrix (d, d)
        H: Hamiltonian (d, d)
        Ls: list of Lindblad operators [(d, d), ...]

    Returns:
        drho_dt: (d, d) complex array
    """
    drho = -1j * commutator(H, rho)
    for L in Ls:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        drho += L @ rho @ Ldag - 0.5 * anticommutator(LdagL, rho)
    return drho


def vectorize_rho(rho: np.ndarray) -> np.ndarray:
    """Vectorize density matrix using column-stacking (Fortran order).

    rho (d, d) -> vec (d^2,)
    """
    return rho.flatten(order='F')


def unvectorize_rho(vec: np.ndarray, d: int) -> np.ndarray:
    """Unvectorize to density matrix.

    vec (d^2,) -> rho (d, d)
    """
    return vec.reshape((d, d), order='F')


def liouvillian_dense(H: np.ndarray, Ls: list) -> np.ndarray:
    """Construct the full Liouvillian superoperator as a d^2 x d^2 matrix.

    L_super @ vec(rho) = vec(d rho/dt)

    Uses column-stacking vectorization.
    """
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)

    # Hamiltonian part: -i(H kron I - I kron H^T)
    L_super = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Dissipator
    for L in Ls:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        # L kron L* - 0.5 (LdagL kron I + I kron LdagL^T)
        L_super += (np.kron(L.conj(), L)
                    - 0.5 * np.kron(I, LdagL)
                    - 0.5 * np.kron(LdagL.T, I))

    return L_super


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance T(rho, sigma) = 0.5 * Tr|rho - sigma|."""
    diff = rho - sigma
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigvals))


def fidelity_qubit_or_general(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Quantum state fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    For pure state sigma = |psi><psi|, F = <psi|rho|psi>.
    Uses scipy.linalg.sqrtm for general case.
    """
    sqrt_rho = sqrtm(rho)
    # Ensure Hermiticity of intermediate
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm(product)
    # Fidelity
    fid = np.real(np.trace(sqrt_product)) ** 2
    return float(np.clip(fid, 0.0, 1.0))


def check_density_matrix(rho: np.ndarray) -> dict:
    """Check physical validity of a density matrix.

    Returns:
        dict with 'trace_error', 'min_eigenvalue', 'hermiticity_error'
    """
    trace_err = abs(np.trace(rho) - 1.0)
    eigvals = np.linalg.eigvalsh(rho)
    min_eig = float(np.min(eigvals))
    herm_err = float(np.max(np.abs(rho - rho.conj().T)))
    return {
        'trace_error': float(trace_err),
        'min_eigenvalue': min_eig,
        'hermiticity_error': herm_err,
    }
