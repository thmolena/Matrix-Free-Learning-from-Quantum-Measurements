"""Ground-truth ODE integration and synthetic data.

To test the learned model we need (a) the TRUE trajectory and (b) noisy
measurements of it. This module provides both:

  * solve_lindblad_trajectory(): integrates the master equation
    d rho/dt = L[rho] from an initial state with a high-accuracy adaptive
    Runge-Kutta solver (scipy.integrate.solve_ivp). It vectorizes rho to a
    length-d^2 vector; for time-independent systems it precomputes the dense
    Liouvillian for speed, otherwise it evaluates the structured RHS each step.
    This is the "forward problem" -- the reference the PINN must match.
  * generate_measurements(): turns a trajectory into synthetic data by computing
    expectation values <O_j> = Tr[O_j rho(t_i)] and adding Gaussian measurement
    noise of a fixed standard deviation. A fixed seed makes the data
    reproducible; the experiments vary that seed to get error bars.

These solvers are deliberately separate from the learning code: they are the
trusted "physics oracle" that the neural surrogate (models.py) is graded against.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .lindblad import vectorize_rho, unvectorize_rho, liouvillian_dense


def solve_lindblad_trajectory(H_fn, Ls_fn, rho0: np.ndarray,
                               t_grid: np.ndarray, method: str = 'RK45',
                               rtol: float = 1e-8, atol: float = 1e-10) -> np.ndarray:
    """Solve the Lindblad master equation exactly via scipy's ODE solver.

    For time-independent systems, uses vectorized Liouvillian for efficiency.
    For time-dependent systems, uses direct RHS evaluation.

    Args:
        H_fn: callable(t) -> H (d,d) Hamiltonian
        Ls_fn: callable(t) -> list of (d,d) Lindblad operators
        rho0: (d,d) initial density matrix
        t_grid: (N,) array of time points
        method: ODE method ('RK45', 'DOP853', etc.)
        rtol: relative tolerance
        atol: absolute tolerance

    Returns:
        rhos: (N, d, d) complex array of density matrices
    """
    d = rho0.shape[0]
    vec0 = vectorize_rho(rho0)

    # Check if time-independent by comparing at two times
    H0 = H_fn(0.0)
    H1 = H_fn(0.5)
    Ls0 = Ls_fn(0.0)
    Ls1 = Ls_fn(0.5)

    time_independent = (np.allclose(H0, H1) and
                        len(Ls0) == len(Ls1) and
                        all(np.allclose(L0, L1) for L0, L1 in zip(Ls0, Ls1)))

    if time_independent:
        # Precompute Liouvillian
        L_super = liouvillian_dense(H0, Ls0)

        def rhs(t, vec):
            return L_super @ vec
    else:
        def rhs(t, vec):
            from .lindblad import lindblad_rhs as _lindblad_rhs
            rho = unvectorize_rho(vec, d)
            H = H_fn(t)
            Ls = Ls_fn(t)
            drho = _lindblad_rhs(rho, H, Ls)
            return vectorize_rho(drho)

    sol = solve_ivp(
        rhs, [t_grid[0], t_grid[-1]], vec0,
        t_eval=t_grid, method=method, rtol=rtol, atol=atol,
        dense_output=False
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Reshape to density matrices
    N = len(t_grid)
    rhos = np.zeros((N, d, d), dtype=np.complex128)
    for i in range(N):
        rhos[i] = unvectorize_rho(sol.y[:, i], d)

    return rhos


def generate_measurements(rhos: np.ndarray, observables: dict,
                          noise_std: float = 0.0, seed: int = 42) -> dict:
    """Generate synthetic measurement data from exact trajectories.

    Args:
        rhos: (N, d, d) density matrices over time
        observables: dict of name -> (d,d) observable operators
        noise_std: Gaussian noise standard deviation
        seed: random seed for reproducibility

    Returns:
        measurements: dict of name -> (N,) measured expectation values
    """
    rng = np.random.default_rng(seed)
    N = rhos.shape[0]
    measurements = {}

    for name, O in observables.items():
        vals = np.array([float(np.real(np.trace(O @ rhos[i]))) for i in range(N)])
        if noise_std > 0:
            vals += rng.normal(0, noise_std, size=N)
        measurements[name] = vals

    return measurements
