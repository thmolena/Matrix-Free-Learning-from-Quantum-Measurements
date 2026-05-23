"""Quantum operators in numpy complex128."""

import numpy as np
from functools import reduce


def pauli_x() -> np.ndarray:
    """Pauli X (sigma_x) matrix."""
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def pauli_y() -> np.ndarray:
    """Pauli Y (sigma_y) matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def pauli_z() -> np.ndarray:
    """Pauli Z (sigma_z) matrix."""
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def identity(d: int) -> np.ndarray:
    """Identity matrix of dimension d."""
    return np.eye(d, dtype=np.complex128)


def destroy(d: int) -> np.ndarray:
    """Bosonic annihilation operator (lowering) for d-level system."""
    a = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def create(d: int) -> np.ndarray:
    """Bosonic creation operator (raising) for d-level system."""
    return destroy(d).conj().T


def projector(d: int, i: int, j: int) -> np.ndarray:
    """|i><j| projector in d-dimensional Hilbert space."""
    p = np.zeros((d, d), dtype=np.complex128)
    p[i, j] = 1.0
    return p


def kron_n(ops: list) -> np.ndarray:
    """Tensor product of a list of operators."""
    return reduce(np.kron, ops)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """{A, B} = AB + BA."""
    return A @ B + B @ A


def sigma_minus() -> np.ndarray:
    """Lowering operator |0><1| for a qubit."""
    return projector(2, 1, 0).T  # |0><1|
    # Actually sigma_- = |1><0| maps |0>->0, |1>->|0>
    # In standard basis |0>=[1,0], |1>=[0,1]: sigma_- = [[0,1],[0,0]]
    # return np.array([[0, 1], [0, 0]], dtype=np.complex128)


def sigma_plus() -> np.ndarray:
    """Raising operator |1><0| for a qubit."""
    return sigma_minus().conj().T


# Fix sigma_minus to standard convention
def sigma_minus() -> np.ndarray:
    """sigma_- = |0><1| (lowers excited to ground)."""
    return np.array([[0, 1], [0, 0]], dtype=np.complex128)


def sigma_plus() -> np.ndarray:
    """sigma_+ = |1><0| (raises ground to excited)."""
    return np.array([[0, 0], [1, 0]], dtype=np.complex128)
