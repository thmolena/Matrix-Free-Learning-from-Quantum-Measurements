"""Loss functions for CPTP-Compiler-PINNs."""

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
