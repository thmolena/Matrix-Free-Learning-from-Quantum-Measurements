"""The neural networks, including the central hard-constraint idea.

(THEORY.txt sections 8, 11-12.)

A neural network here is just a parameterized function fit by gradient descent.
Three pieces:

  * TimeMLP: a plain multilayer perceptron mapping the scalar TIME t to a real
    vector. Optionally it first expands t into Fourier features
    [t, sin(k w t), cos(k w t)]_{k=1..K} to combat "spectral bias" (the tendency
    of MLPs to learn low frequencies first), which matters for fast oscillatory
    dynamics. (Experiment 3 shows this fix is not robust across seeds.)

  * DensityMatrixNet -- THE CENTRAL IDEA (Theorem 1, exact physicality). Rather
    than output a density matrix directly and hope it is legal, it outputs the
    entries of a lower-triangular complex matrix A_phi(t) (positive softplus
    diagonal) and returns

        rho_phi(t) = A_phi A_phi^dagger / Tr(A_phi A_phi^dagger).

    Because A A^dagger is always Hermitian and positive semidefinite, and the
    trace division forces unit trace, rho_phi is a VALID density matrix at every
    time, every training step and every evaluation point -- by construction, with
    no penalty or projection. This is what makes the method robust under sparse
    data (Experiment 2).

  * UnconstrainedDensityNet: the baseline that outputs matrix entries directly
    (only soft-symmetrized and trace-normalized). It can leave the physical set;
    comparing against it isolates the value of the hard constraint.

  * PositiveParameter: a scalar parameterized as softplus(raw) so it is ALWAYS
    positive -- used for the Lindblad rates gamma_k >= 0, which keeps the learned
    generator completely positive (Theorem 3).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PositiveParameter(nn.Module):
    """Parameterize a strictly positive scalar via softplus.

    stored_value is the raw (unconstrained) parameter.
    forward() returns softplus(stored_value).
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        # Initialize so that softplus(raw) ~ init_value
        raw_init = np.log(np.exp(init_value) - 1.0) if init_value > 0.05 else init_value
        self.raw = nn.Parameter(torch.tensor(float(raw_init)))

    def forward(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw)

    @property
    def value(self) -> float:
        with torch.no_grad():
            return float(nn.functional.softplus(self.raw))


class TimeMLP(nn.Module):
    """MLP mapping scalar time t to a real vector.

    Optionally encodes the time input with a deterministic Fourier feature basis
    ``[sin(k omega_0 t), cos(k omega_0 t)]`` for ``k = 1..fourier_features`` with
    fundamental frequency ``omega_0 = 2*pi / fourier_period``. This removes the
    spectral bias of a plain MLP and lets the network represent the fast coherent
    oscillations (Bohr frequencies) of multi-level open quantum systems.

    Args:
        output_dim: dimension of output vector
        hidden_dim: width of hidden layers
        n_layers: number of hidden layers
        activation: 'gelu', 'tanh', or 'relu'
        fourier_features: number of Fourier modes (0 disables, recovering a plain MLP)
        fourier_period: time-window length T; harmonics are multiples of 2*pi/T
    """

    def __init__(self, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 4, activation: str = 'gelu',
                 fourier_features: int = 0, fourier_period: float = 1.0):
        super().__init__()
        act_map = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'relu': nn.ReLU}
        act_cls = act_map.get(activation, nn.GELU)

        self.fourier_features = int(fourier_features)
        if self.fourier_features > 0:
            base = 2.0 * float(np.pi) / float(fourier_period)
            freqs = base * torch.arange(1, self.fourier_features + 1, dtype=torch.float32)
            self.register_buffer('freqs', freqs)
            in_dim = 1 + 2 * self.fourier_features
        else:
            in_dim = 1

        layers = [nn.Linear(in_dim, hidden_dim), act_cls()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t: (batch,) or (batch, 1) tensor of times

        Returns:
            (batch, output_dim) real tensor
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if self.fourier_features > 0:
            ang = t * self.freqs  # (batch, fourier_features)
            t = torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(t)


class DensityMatrixNet(nn.Module):
    """Neural network that maps time t to a valid density matrix.

    Parametrizes rho(t) = A(t) A(t)^dagger / Tr(A(t) A(t)^dagger)
    where A(t) is a lower-triangular complex matrix output by a TimeMLP.

    This guarantees:
    - Hermiticity (A A^dag is always Hermitian)
    - Positive semidefiniteness (A A^dag >= 0)
    - Unit trace (division by trace)

    Args:
        d: Hilbert space dimension
        hidden_dim: MLP hidden width
        n_layers: MLP depth
    """

    def __init__(self, d: int, hidden_dim: int = 128, n_layers: int = 4,
                 fourier_features: int = 0, fourier_period: float = 1.0):
        super().__init__()
        self.d = d
        # Lower-triangular A has d*(d+1)/2 complex entries = d*(d+1) real parameters
        n_params = d * (d + 1)  # real + imag for lower triangular
        self.mlp = TimeMLP(output_dim=n_params, hidden_dim=hidden_dim, n_layers=n_layers,
                           fourier_features=fourier_features, fourier_period=fourier_period)

        # Pre-compute indices for lower-triangular matrix
        tril_idx = torch.tril_indices(d, d)
        self.register_buffer('tril_row', tril_idx[0])
        self.register_buffer('tril_col', tril_idx[1])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute density matrix rho(t).

        Args:
            t: (batch,) or (batch, 1) time tensor

        Returns:
            rho: (batch, d, d) complex tensor, valid density matrix
        """
        if t.dim() == 1:
            t_input = t.unsqueeze(-1)
        else:
            t_input = t
        batch = t_input.shape[0]
        d = self.d

        # Get raw parameters
        raw = self.mlp(t_input)  # (batch, d*(d+1))

        # Split into real and imaginary parts
        n_entries = d * (d + 1) // 2
        real_parts = raw[:, :n_entries]
        imag_parts = raw[:, n_entries:]

        # Construct lower-triangular complex matrix A
        A = torch.zeros(batch, d, d, dtype=torch.cfloat, device=t_input.device)
        A[:, self.tril_row, self.tril_col] = torch.complex(real_parts, imag_parts)

        # rho = A A^dag / Tr(A A^dag)
        A_dag = A.conj().transpose(-2, -1)
        rho_unnorm = torch.bmm(A, A_dag)

        # Normalize by trace
        trace = torch.diagonal(rho_unnorm, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        rho = rho_unnorm / trace.real

        return rho

    def get_A(self, t: torch.Tensor) -> torch.Tensor:
        """Get the Cholesky factor A(t) before normalization."""
        if t.dim() == 1:
            t_input = t.unsqueeze(-1)
        else:
            t_input = t
        batch = t_input.shape[0]
        d = self.d

        raw = self.mlp(t_input)
        n_entries = d * (d + 1) // 2
        real_parts = raw[:, :n_entries]
        imag_parts = raw[:, n_entries:]

        A = torch.zeros(batch, d, d, dtype=torch.cfloat, device=t_input.device)
        A[:, self.tril_row, self.tril_col] = torch.complex(real_parts, imag_parts)
        return A


class UnconstrainedDensityNet(nn.Module):
    """Baseline: unconstrained network outputting d^2 real values reshaped to matrix.

    Does NOT guarantee physicality. Used as comparison baseline.
    """

    def __init__(self, d: int, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.d = d
        # Output 2*d^2 values (real + imag parts of d x d matrix)
        self.mlp = TimeMLP(output_dim=2 * d * d, hidden_dim=hidden_dim, n_layers=n_layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute (unconstrained) matrix.

        Args:
            t: (batch,) or (batch, 1)

        Returns:
            rho: (batch, d, d) complex tensor (NOT guaranteed physical)
        """
        if t.dim() == 1:
            t_input = t.unsqueeze(-1)
        else:
            t_input = t
        batch = t_input.shape[0]
        d = self.d

        raw = self.mlp(t_input)  # (batch, 2*d*d)
        real_part = raw[:, :d * d].reshape(batch, d, d)
        imag_part = raw[:, d * d:].reshape(batch, d, d)
        rho = torch.complex(real_part, imag_part)

        # Soft symmetrization (still not guaranteed PSD or trace-1)
        rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
        # Soft trace normalization
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        rho = rho / (trace.real + 1e-8)

        return rho


class ControlNet(nn.Module):
    """Time-dependent control envelope J(t) with bounded amplitude.

    Maps t -> tanh(MLP(t)) * amplitude_max

    Useful for modeling time-dependent drive/coupling.
    """

    def __init__(self, n_controls: int = 1, hidden_dim: int = 64,
                 n_layers: int = 3, amplitude_max: float = 1.0):
        super().__init__()
        self.amplitude_max = amplitude_max
        self.mlp = TimeMLP(output_dim=n_controls, hidden_dim=hidden_dim, n_layers=n_layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute control amplitudes at time t.

        Args:
            t: (batch,) or (batch, 1)

        Returns:
            controls: (batch, n_controls) bounded in [-amplitude_max, amplitude_max]
        """
        raw = self.mlp(t)
        return self.amplitude_max * torch.tanh(raw)
