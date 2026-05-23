"""Training loop for CPTP-Compiler-PINNs."""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional
from .models import DensityMatrixNet, PositiveParameter
from .losses import data_loss, physics_residual_loss, initial_condition_loss, total_loss


def train_pinn(model: DensityMatrixNet,
               rate_params: dict,
               observables_torch: dict,
               t_data: torch.Tensor,
               measurements_data: dict,
               rho0: torch.Tensor,
               lindblad_rhs_fn,
               config: dict,
               mask: Optional[torch.Tensor] = None) -> dict:
    """Train a CPTP-PINN model.

    Args:
        model: DensityMatrixNet instance
        rate_params: dict of name -> PositiveParameter or nn.Parameter
        observables_torch: dict of obs_name -> (d,d) torch tensor
        t_data: (N,) tensor of observation times
        measurements_data: dict of obs_name -> (N,) tensor of measured values
        rho0: (d,d) complex target initial state
        lindblad_rhs_fn: callable(rho_batch, rate_params) -> (batch, d, d)
        config: training configuration dict
        mask: optional (N, n_obs) binary mask for sparse observations

    Returns:
        log: dict with loss history and timing
    """
    # Config defaults
    n_epochs = config.get('n_epochs', 2000)
    lr = config.get('lr', 1e-3)
    w_data = config.get('w_data', 1.0)
    w_physics = config.get('w_physics', 1.0)
    w_ic = config.get('w_ic', 10.0)
    w_reg = config.get('w_reg', 1e-5)
    use_lbfgs = config.get('use_lbfgs', False)
    lbfgs_epochs = config.get('lbfgs_epochs', 200)
    seed = config.get('seed', 42)
    n_colloc = config.get('n_collocation', 100)
    t_max = config.get('t_max', 1.0)
    verbose = config.get('verbose', False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect all parameters
    all_params = list(model.parameters())
    for p in rate_params.values():
        if isinstance(p, nn.Module):
            all_params.extend(p.parameters())
        elif isinstance(p, nn.Parameter):
            all_params.append(p)

    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Prepare data targets
    obs_names = list(observables_torch.keys())
    n_obs = len(obs_names)
    N = t_data.shape[0]
    target_tensor = torch.stack([measurements_data[name] for name in obs_names], dim=-1)  # (N, n_obs)

    log = {
        'total_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'ic_loss': [],
        'epoch_time': [],
    }

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()

        # --- Data loss ---
        t_data_grad = t_data.clone().requires_grad_(True)
        rho_data = model(t_data_grad)  # (N, d, d)

        # Compute predicted observables
        pred_obs_list = []
        for obs_name in obs_names:
            O = observables_torch[obs_name]  # (d, d)
            # Tr(O @ rho) for each sample
            # (N, d, d) @ (d, d) -> trace over last two dims
            expect = torch.einsum('bij,ji->b', rho_data, O).real
            pred_obs_list.append(expect)
        pred_obs = torch.stack(pred_obs_list, dim=-1)  # (N, n_obs)

        d_loss = data_loss(pred_obs, target_tensor, mask)

        # --- Physics residual loss ---
        t_colloc = torch.linspace(0, t_max, n_colloc, dtype=torch.float32).requires_grad_(True)
        rho_colloc = model(t_colloc)
        p_loss = physics_residual_loss(rho_colloc, t_colloc.unsqueeze(-1),
                                       lambda rho: lindblad_rhs_fn(rho, rate_params))

        # --- Initial condition loss ---
        t_zero = torch.zeros(1, dtype=torch.float32)
        rho_pred_0 = model(t_zero)
        ic_loss_val = initial_condition_loss(rho_pred_0, rho0)

        # --- Regularization ---
        reg_loss = torch.tensor(0.0)
        for name, p in rate_params.items():
            if isinstance(p, PositiveParameter):
                reg_loss = reg_loss + p.raw ** 2

        # --- Total loss ---
        loss = total_loss(d_loss, p_loss, ic_loss_val, reg_loss,
                         w_data, w_physics, w_ic, w_reg)

        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        log['total_loss'].append(float(loss))
        log['data_loss'].append(float(d_loss))
        log['physics_loss'].append(float(p_loss))
        log['ic_loss'].append(float(ic_loss_val))
        log['epoch_time'].append(epoch_time)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d} | loss={loss.item():.4e} "
                  f"data={d_loss.item():.4e} phys={p_loss.item():.4e} "
                  f"ic={ic_loss_val.item():.4e}")

    # Optional L-BFGS refinement
    if use_lbfgs:
        lbfgs_opt = torch.optim.LBFGS(all_params, lr=0.5, max_iter=20,
                                        line_search_fn='strong_wolfe')

        for epoch in range(lbfgs_epochs):
            def closure():
                lbfgs_opt.zero_grad()
                t_data_grad = t_data.clone().requires_grad_(True)
                rho_data = model(t_data_grad)
                pred_obs_list = []
                for obs_name in obs_names:
                    O = observables_torch[obs_name]
                    expect = torch.einsum('bij,ji->b', rho_data, O).real
                    pred_obs_list.append(expect)
                pred_obs = torch.stack(pred_obs_list, dim=-1)
                d_loss = data_loss(pred_obs, target_tensor, mask)

                t_colloc = torch.linspace(0, t_max, n_colloc).requires_grad_(True)
                rho_colloc = model(t_colloc)
                p_loss = physics_residual_loss(rho_colloc, t_colloc.unsqueeze(-1),
                                               lambda rho: lindblad_rhs_fn(rho, rate_params))

                t_zero = torch.zeros(1)
                rho_pred_0 = model(t_zero)
                ic_loss_val = initial_condition_loss(rho_pred_0, rho0)
                reg_loss = torch.tensor(0.0)
                for p in rate_params.values():
                    if isinstance(p, PositiveParameter):
                        reg_loss = reg_loss + p.raw ** 2

                loss = total_loss(d_loss, p_loss, ic_loss_val, reg_loss,
                                 w_data, w_physics, w_ic, w_reg)
                loss.backward()
                return loss

            lbfgs_opt.step(closure)

    total_time = time.time() - start_time
    log['total_time'] = total_time

    return log
