"""Test density matrix parameterization produces valid states."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pytest
from ctpcpinn.models import DensityMatrixNet, PositiveParameter


class TestDensityParameterization:
    """Test that DensityMatrixNet outputs valid density matrices."""

    @pytest.fixture(params=[2, 3, 4])
    def dim(self, request):
        return request.param

    def test_hermiticity(self, dim):
        """rho should be Hermitian: rho = rho^dag."""
        torch.manual_seed(42)
        model = DensityMatrixNet(d=dim, hidden_dim=32, n_layers=2)
        t = torch.linspace(0, 1, 10)
        with torch.no_grad():
            rho = model(t)  # (10, d, d)
        rho_np = rho.numpy()
        for i in range(10):
            herm_err = np.max(np.abs(rho_np[i] - rho_np[i].conj().T))
            assert herm_err < 1e-5, f"Hermiticity error {herm_err} at t[{i}]"

    def test_trace_one(self, dim):
        """Tr(rho) should equal 1."""
        torch.manual_seed(42)
        model = DensityMatrixNet(d=dim, hidden_dim=32, n_layers=2)
        t = torch.linspace(0, 1, 10)
        with torch.no_grad():
            rho = model(t)
        rho_np = rho.numpy()
        for i in range(10):
            tr = np.trace(rho_np[i])
            assert abs(tr - 1.0) < 1e-5, f"Trace error {abs(tr-1)} at t[{i}]"

    def test_positive_semidefinite(self, dim):
        """All eigenvalues of rho should be >= 0."""
        torch.manual_seed(42)
        model = DensityMatrixNet(d=dim, hidden_dim=32, n_layers=2)
        t = torch.linspace(0, 1, 10)
        with torch.no_grad():
            rho = model(t)
        rho_np = rho.numpy()
        for i in range(10):
            eigvals = np.linalg.eigvalsh(rho_np[i])
            min_eig = np.min(eigvals)
            assert min_eig >= -1e-6, f"Negative eigenvalue {min_eig} at t[{i}]"

    def test_batch_output_shape(self, dim):
        """Output shape should be (batch, d, d)."""
        torch.manual_seed(42)
        model = DensityMatrixNet(d=dim, hidden_dim=32, n_layers=2)
        t = torch.linspace(0, 1, 7)
        with torch.no_grad():
            rho = model(t)
        assert rho.shape == (7, dim, dim)


class TestPositiveParameter:
    """Test PositiveParameter ensures positive values."""

    def test_always_positive(self):
        for init in [0.01, 0.1, 1.0, 5.0]:
            pp = PositiveParameter(init_value=init)
            val = pp()
            assert val.item() > 0, f"PositiveParameter({init}) gave non-positive {val.item()}"

    def test_gradient_flows(self):
        pp = PositiveParameter(init_value=1.0)
        val = pp()
        loss = (val - 2.0) ** 2
        loss.backward()
        assert pp.raw.grad is not None
        assert pp.raw.grad.item() != 0.0
