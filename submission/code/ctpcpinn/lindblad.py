"""The physics: the Lindblad/GKSL master equation (THEORY.txt sections 4-6, 13).

An open quantum system's state rho(t) (a density matrix) evolves by the
Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation

    d rho/dt = -i [H, rho]                              (coherent / unitary part)
               + sum_k ( L_k rho L_k^dagger
                         - (1/2){ L_k^dagger L_k, rho } )    (dissipation)

where H = H^dagger is the Hamiltonian and the L_k are jump operators carrying the
noise (relaxation, dephasing, leakage). This module implements:

  * lindblad_rhs(rho, H, Ls): evaluates the right-hand side directly from H and
    the jump operators -- the "structured", matrix-free route, cost O(m d^3);
  * liouvillian_dense(H, Ls): builds the equivalent d^2 x d^2 "Liouvillian"
    superoperator via Kronecker products, so that d vec(rho)/dt = L vec(rho) --
    the "dense" route, cost/stores O(d^4). The two agree exactly; the compiler
    (compiler.py) chooses between them, and experiment 5 benchmarks the gap;
  * trace_distance and fidelity_qubit_or_general: the two standard ways to
    measure how far apart two states are (used to score the learned trajectory);
  * check_density_matrix: the physical-validity diagnostics (Hermiticity error,
    minimum eigenvalue, trace error) defining the set D(H).

A key property used by the residual certificate (Theorem 4): the Lindblad
generator is trace preserving, so Tr(d rho/dt) = 0, and e^{tL} is completely
positive and trace preserving (CPTP) exactly when every rate is >= 0.
"""

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


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """Hermitian matrix square root via eigendecomposition.

    Numerically stable for rank-deficient (e.g. pure-state) inputs, unlike
    ``scipy.linalg.sqrtm`` which warns and loses accuracy on singular matrices.
    Negative eigenvalues from round-off are clipped to zero.
    """
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w.real, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T


def fidelity_qubit_or_general(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann state fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    For pure state sigma = |psi><psi|, F = <psi|rho|psi>. Implemented with a
    Hermitian eigendecomposition so the result is stable for pure and
    rank-deficient states (no LinAlg warnings, no spurious imaginary parts).
    """
    sqrt_rho = _psd_sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    product = 0.5 * (product + product.conj().T)
    w = np.linalg.eigvalsh(product)
    w = np.clip(w.real, 0.0, None)
    fid = float(np.sum(np.sqrt(w)) ** 2)
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
