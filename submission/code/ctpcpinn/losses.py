"""The physics-informed loss (THEORY.txt sections 9-10).

A physics-informed neural network (PINN) is trained to satisfy a differential
equation by penalizing the equation's RESIDUAL at sampled collocation points, in
addition to fitting data. The total objective is

    J = lambda_data * J_data + lambda_phys * J_phys + lambda_0 * J_0

  * data_loss (J_data): mean squared error between predicted observables
    Tr[O_j rho_phi(t_i)] and the measurements, with an optional binary MASK so
    that missing entries are simply skipped (this is how the sparse-measurement
    experiment works).
  * physics_residual_loss (J_phys): the heart of the method. It computes the time
    derivative d rho_phi/dt by AUTOMATIC DIFFERENTIATION of the network output
    with respect to its input t, then penalizes the Frobenius norm of the
    Lindblad residual || d rho_phi/dt - L_Theta[rho_phi] ||^2. A small residual
    certifies a small trajectory error in trace norm (Theorem 4).
  * initial_condition_loss (J_0): anchors rho_phi(0) to the known initial state.
  * positivity_penalty: the SOFT alternative to the hard Cholesky constraint --
    it penalizes negative eigenvalues after the fact. Used only by the
    soft-penalty baseline, precisely to demonstrate that a penalty is weaker than
    making illegal states unrepresentable (Experiments 1-2).

Gradients of J with respect to both the network parameters and the generator
parameters are obtained by backpropagation; the optimizer is Adam.
"""

import torch
import torch.nn as nn
from typing import Optional


def data_loss(predicted_obs: torch.Tensor, observed_vals: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE loss between predicted and observed expectation values.

    Args:
        predicted_obs: (N, n_obs) predicted observables
        observed_vals: (N, n_obs) measured values
        mask: (N, n_obs) binary mask for sparse observations (1 = observed)

    Returns:
        scalar loss
    """
    diff_sq = (predicted_obs - observed_vals) ** 2
    if mask is not None:
        diff_sq = diff_sq * mask
        return diff_sq.sum() / (mask.sum() + 1e-10)
    return diff_sq.mean()


def physics_residual_loss(rho_t: torch.Tensor, t: torch.Tensor,
                          lindblad_rhs_fn) -> torch.Tensor:
    """Physics-informed residual: ||d rho/dt - L[rho]||_F^2.

    Computes d rho/dt via autograd and compares to the Lindblad RHS.

    Args:
        rho_t: (N, d, d) complex density matrices (from DensityMatrixNet)
        t: (N, 1) time tensor with requires_grad=True
        lindblad_rhs_fn: callable(rho_batch) -> (N, d, d) Lindblad RHS

    Returns:
        scalar residual loss
    """
    # Compute time derivative via autograd
    # rho_t must be computed from t with grad enabled
    N, d, _ = rho_t.shape

    # Take real and imaginary parts for autograd
    rho_real = rho_t.real
    rho_imag = rho_t.imag

    # Compute gradients
    drho_real_dt = torch.zeros_like(rho_real)
    drho_imag_dt = torch.zeros_like(rho_imag)

    for i in range(d):
        for j in range(d):
            grad_real = torch.autograd.grad(
                rho_real[:, i, j].sum(), t,
                create_graph=True, retain_graph=True
            )[0]
            drho_real_dt[:, i, j] = grad_real.squeeze(-1)

            grad_imag = torch.autograd.grad(
                rho_imag[:, i, j].sum(), t,
                create_graph=True, retain_graph=True
            )[0]
            drho_imag_dt[:, i, j] = grad_imag.squeeze(-1)

    drho_dt = torch.complex(drho_real_dt, drho_imag_dt)

    # Evaluate Lindblad RHS
    lind_rhs = lindblad_rhs_fn(rho_t)  # (N, d, d)

    # Frobenius norm squared of residual
    residual = drho_dt - lind_rhs
    loss = (residual.real ** 2 + residual.imag ** 2).sum(dim=(-2, -1)).mean()
    return loss


def positivity_penalty(rho_t: torch.Tensor) -> torch.Tensor:
    """Soft positivity penalty: mean over the batch of sum_i ReLU(-lambda_i(rho)).

    This is the *soft* alternative to the hard Cholesky constraint: it penalizes
    negative eigenvalues after the fact instead of making them unrepresentable.
    Used by the soft-penalty PINN baseline (Sec. 3.3 of the manuscript). The
    eigenvalues are taken of the Hermitian part of rho so the penalty is well
    defined even when the network output drifts away from Hermiticity.

    Args:
        rho_t: (N, d, d) complex tensor of (approximate) density matrices.

    Returns:
        scalar penalty (zero iff every matrix in the batch is PSD).
    """
    rho_herm = 0.5 * (rho_t + rho_t.conj().transpose(-2, -1))
    eigvals = torch.linalg.eigvalsh(rho_herm)  # (N, d), ascending, real
    return torch.relu(-eigvals).sum(dim=-1).mean()


def initial_condition_loss(rho_pred_0: torch.Tensor,
                           rho_0: torch.Tensor) -> torch.Tensor:
    """Loss enforcing initial condition.

    Args:
        rho_pred_0: (1, d, d) or (d, d) predicted rho at t=0
        rho_0: (d, d) target initial state

    Returns:
        scalar Frobenius norm squared
    """
    if rho_pred_0.dim() == 3:
        rho_pred_0 = rho_pred_0.squeeze(0)
    diff = rho_pred_0 - rho_0
    return (diff.real ** 2 + diff.imag ** 2).sum()


def regularization_loss(params_dict: dict, weight_decay: float = 1e-4) -> torch.Tensor:
    """L2 regularization on model parameters."""
    reg = torch.tensor(0.0)
    for p in params_dict.values():
        if p.requires_grad:
            reg = reg + (p ** 2).sum()
    return weight_decay * reg


def total_loss(data_l: torch.Tensor, physics_l: torch.Tensor,
               ic_l: torch.Tensor, reg_l: torch.Tensor,
               w_data: float = 1.0, w_physics: float = 1.0,
               w_ic: float = 10.0, w_reg: float = 1e-4) -> torch.Tensor:
    """Weighted sum of all loss components."""
    return (w_data * data_l + w_physics * physics_l +
            w_ic * ic_l + w_reg * reg_l)
