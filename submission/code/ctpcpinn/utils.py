"""Utility functions."""

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_to_torch_complex(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy complex128 array to torch cfloat tensor."""
    return torch.from_numpy(arr.astype(np.complex64)).to(torch.cfloat)


def torch_to_numpy_complex(t: torch.Tensor) -> np.ndarray:
    """Convert torch complex tensor to numpy complex128."""
    return t.detach().cpu().numpy().astype(np.complex128)


def make_hermitian(M: torch.Tensor) -> torch.Tensor:
    """Symmetrize a batch of matrices: (M + M^dag) / 2."""
    return 0.5 * (M + M.conj().transpose(-2, -1))


def torch_trace(M: torch.Tensor) -> torch.Tensor:
    """Batch trace: M is (..., d, d), returns (...)."""
    return torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)


def torch_commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """[A, B] = AB - BA for batched matrices."""
    return torch.bmm(A, B) - torch.bmm(B, A)


def torch_anticommutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """{A, B} = AB + BA for batched matrices."""
    return torch.bmm(A, B) + torch.bmm(B, A)


def torch_lindblad_rhs(rho: torch.Tensor, H: torch.Tensor,
                       Ls: list) -> torch.Tensor:
    """Compute Lindblad RHS in PyTorch for batched rho.

    Args:
        rho: (batch, d, d) complex tensor
        H: (d, d) complex tensor (Hamiltonian)
        Ls: list of (d, d) complex tensors (Lindblad operators)

    Returns:
        drho_dt: (batch, d, d) complex tensor
    """
    batch = rho.shape[0]
    d = rho.shape[1]

    # Expand H for batched multiplication
    H_batch = H.unsqueeze(0).expand(batch, d, d)

    # -i[H, rho]
    drho = -1j * (torch.bmm(H_batch, rho) - torch.bmm(rho, H_batch))

    # Dissipator
    for L in Ls:
        L_batch = L.unsqueeze(0).expand(batch, d, d)
        Ldag = L.conj().T.unsqueeze(0).expand(batch, d, d)
        LdagL = (L.conj().T @ L).unsqueeze(0).expand(batch, d, d)

        drho = drho + (torch.bmm(torch.bmm(L_batch, rho), Ldag)
                       - 0.5 * torch.bmm(LdagL, rho)
                       - 0.5 * torch.bmm(rho, LdagL))

    return drho
