"""Spectral-frame parameterization and symmetric Fourier approximation.

This module implements an interaction-picture density parameterization and a
separate symmetric temporal-Fourier diagnostic for approximating an open-system
(Lindblad/GKSL) generator. The diagnostic is a linear Fourier partial sum, not a
claim that operator-system truncation preserves the GKSL cone.

1. EIGEN-OPERATOR (INTERACTION-PICTURE) DENSITY PARAMETERIZATION.
   The coherent generator ad_H = -i[H, .] is normal; in the Hamiltonian
   eigenbasis its eigen-operators are the transition operators |a><b| with purely
   imaginary eigenvalues -i*omega_ab, omega_ab = E_a - E_b (the Bohr
   frequencies). These Bohr frequencies are exactly the fast oscillations that
   defeat a smooth-in-time density network. We move to the interaction picture:
   the network learns the SLOW envelope

        rho_tilde_E(t) = U(t)^dag rho(t) U(t),   U(t) = exp(-i H t),

   in the eigenbasis, via the SAME Cholesky map (so rho_tilde is a valid density
   matrix), and the lab state rho(t) = U(t) rho_tilde(t) U(t)^dag is recovered by
   an analytic, exact phase rotation. Conjugation by a unitary preserves
   D(H), so physicality is still architectural; spectral bias is removed because
   the network only has to represent a slowly varying envelope.

   With H diagonal in the eigenbasis the transform is a Hadamard phase product:
        (rho_tilde_E)_ab = exp(i omega_ab t) (rho_E)_ab,
   and the interaction-picture jump operator is
        L_tilde_k(t)_ab = exp(i omega_ab t) (L_k^E)_ab.

2. SYMMETRIC FOURIER TRUNCATION OF THE GENERATOR.
   In the interaction picture the generator L_tilde(t) is almost periodic; on the
   window [0, T] it has a Fourier series L_tilde(t) = sum_k exp(i k w0 t) L_k,
   w0 = 2*pi/T. The level-N Fourier approximation keeps the |k| <= N band -- a
   linear projection onto low temporal frequencies. Retaining the k=0 mode and
   conjugate-symmetric pairs preserves Hermiticity and trace. It does NOT, in
   general, preserve complete positivity or GKSL form: the Dirichlet projection
   is not a positive averaging kernel. The k=0 time average is GKSL because it is
   a convex average of pointwise GKSL generators, but calling it a Davies/secular
   generator requires additional weak-coupling and Bohr-averaging assumptions.
   The reported out-of-band weight is the sum of Frobenius norms of the omitted
   superoperator coefficient matrices; it bounds the F-to-F generator error. A
   GKSL-preserving finite-band model must instead truncate a Hamiltonian and a
   factor B_N(t), then form C_N(t)=B_N(t)B_N(t)^dagger; that nonlinear model is
   not the raw Fourier partial sum and needs its own approximation certificate.

The functions below provide both a NumPy reference (used by the ground-truth
solver and the truncation experiment) and PyTorch operations (used inside the
training residual).
"""

import numpy as np
import torch

from .lindblad import lindblad_rhs, liouvillian_dense, vectorize_rho, unvectorize_rho


# ---------------------------------------------------------------------------
# Hamiltonian spectrum and Bohr frequencies (NumPy reference).
# ---------------------------------------------------------------------------
def eigh_hamiltonian(H: np.ndarray):
    """Eigendecomposition H = V diag(E) V^dag with ascending real eigenvalues.

    Returns:
        E: (d,) real eigenvalues (ascending)
        V: (d, d) complex eigenvectors (columns), unitary
    """
    E, V = np.linalg.eigh(0.5 * (H + H.conj().T))
    return E.real.astype(np.float64), V.astype(np.complex128)


def bohr_frequency_matrix(E: np.ndarray) -> np.ndarray:
    """Bohr-frequency matrix omega_ab = E_a - E_b (shape (d, d))."""
    E = np.asarray(E, dtype=np.float64)
    return E[:, None] - E[None, :]


def to_eigenbasis(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Rotate an operator from the lab basis to the Hamiltonian eigenbasis."""
    return V.conj().T @ M @ V


def from_eigenbasis(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Rotate an operator from the eigenbasis back to the lab basis."""
    return V @ M @ V.conj().T


def interaction_envelope(rho_lab: np.ndarray, t: float, E: np.ndarray,
                         V: np.ndarray) -> np.ndarray:
    """Lab state -> interaction-picture envelope in the eigenbasis.

    rho_tilde_E = diag(e^{iE t}) (V^dag rho V) diag(e^{-iE t}).
    """
    rho_E = to_eigenbasis(rho_lab, V)
    phase = np.exp(1j * bohr_frequency_matrix(E) * t)
    return phase * rho_E


def lab_from_envelope(rho_tilde_E: np.ndarray, t: float, E: np.ndarray,
                      V: np.ndarray) -> np.ndarray:
    """Interaction-picture envelope (eigenbasis) -> lab state.

    rho_lab = V [ diag(e^{-iE t}) rho_tilde_E diag(e^{iE t}) ] V^dag.
    """
    phase = np.exp(-1j * bohr_frequency_matrix(E) * t)
    rho_E = phase * rho_tilde_E
    return from_eigenbasis(rho_E, V)


def interaction_jump(L_lab: np.ndarray, t: float, E: np.ndarray,
                     V: np.ndarray) -> np.ndarray:
    """Interaction-picture jump operator in the eigenbasis at time t.

    L_tilde_E(t) = diag(e^{iE t}) (V^dag L V) diag(e^{-iE t}).
    """
    L_E = to_eigenbasis(L_lab, V)
    phase = np.exp(1j * bohr_frequency_matrix(E) * t)
    return phase * L_E


# ---------------------------------------------------------------------------
# Interaction-picture generator and its symmetric Fourier approximation.
# ---------------------------------------------------------------------------
def interaction_liouvillian(t: float, E: np.ndarray, V: np.ndarray,
                            Hc_lab_fn, Ls_lab_fn) -> np.ndarray:
    """Dense (d^2 x d^2) interaction-picture Liouvillian in the eigenbasis at t.

    The frame is set by the *full* spectrum E (so the H0 = sum_a E_a|a><a|
    coherent part is removed). ``Hc_lab_fn(t)`` is any RESIDUAL coherent term that
    is NOT in the frame Hamiltonian (e.g. a slow control coupling); pass ``None``
    for a pure interaction picture. ``Ls_lab_fn(t)`` returns the lab-frame jump
    operators.

    Returns the superoperator acting on vec(rho_tilde_E) (column-stacking).
    """
    d = len(E)
    Ls_E = [interaction_jump(L, t, E, V) for L in Ls_lab_fn(t)]
    if Hc_lab_fn is not None:
        Hc_E = interaction_jump(Hc_lab_fn(t), t, E, V)  # same phase rotation
    else:
        Hc_E = np.zeros((d, d), dtype=np.complex128)
    return liouvillian_dense(Hc_E, Ls_E)


def generator_fourier_modes(E: np.ndarray, V: np.ndarray, Hc_lab_fn,
                            Ls_lab_fn, T: float, n_grid: int = 4096):
    """Fourier series of the interaction-picture generator on [0, T].

    Samples the dense interaction-picture Liouvillian on a uniform grid and takes
    a temporal FFT, giving modes L_k with
        L_tilde(t) ~ sum_k exp(i k w0 t) L_k,  w0 = 2 pi / T.

    Returns:
        modes: (n_grid, d^2, d^2) complex array indexed by FFT bin k
        norms: (n_grid,) Frobenius norm ||L_k|| per FFT bin (fftshifted ordering
               is NOT applied; use ``fourier_mode_order`` for |k| ordering)
    """
    d = len(E)
    ts = np.arange(n_grid) * (T / n_grid)
    gen = np.empty((n_grid, d * d, d * d), dtype=np.complex128)
    for i, t in enumerate(ts):
        gen[i] = interaction_liouvillian(t, E, V, Hc_lab_fn, Ls_lab_fn)
    modes = np.fft.fft(gen, axis=0) / n_grid
    norms = np.linalg.norm(modes.reshape(n_grid, -1), axis=1)
    return modes, norms


def _truncate_modes(modes: np.ndarray, n_keep: int) -> np.ndarray:
    """Zero out all FFT bins outside the lowest ``n_keep`` frequency magnitudes.

    FFT bin k (0..n-1) corresponds to frequency k for k<=n/2 and k-n (negative)
    otherwise. Keeping |k| <= n_keep retains the k=0 mode and conjugate-symmetric
    pairs, so an exactly conjugate-symmetric input remains Hermiticity- and
    trace-preserving.  This operation is not positivity preserving and does not
    generally keep the generator in GKSL form.
    """
    n = modes.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    for k in range(1, n_keep + 1):
        keep[k % n] = True
        keep[(-k) % n] = True
    out = np.zeros_like(modes)
    out[keep] = modes[keep]
    return out


def truncated_generator_at(modes_trunc: np.ndarray, t: float, T: float) -> np.ndarray:
    """Reconstruct the truncated generator at time t from its FFT modes."""
    n = modes_trunc.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)  # integer bin frequencies
    phase = np.exp(1j * 2.0 * np.pi * k * t / T)  # (n,)
    return np.tensordot(phase, modes_trunc, axes=([0], [0]))


def spectral_truncation_error(norms: np.ndarray, n_keep: int) -> float:
    """Sum omitted Frobenius coefficient norms (an F-to-F error bound).

    For a trace-norm trajectory bound in Hilbert dimension ``d``, the manuscript
    carries the additional norm-conversion factor ``sqrt(d)``. This scalar is
    not itself a bound on state infidelity or a complete-positivity diagnostic.
    """
    n = len(norms)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    for k in range(1, n_keep + 1):
        keep[k % n] = True
        keep[(-k) % n] = True
    return float(np.sum(norms[~keep]))


def propagate_truncated(modes: np.ndarray, n_keep: int, E: np.ndarray, V: np.ndarray,
                        rho0_lab: np.ndarray, t_grid: np.ndarray, T: float) -> np.ndarray:
    """Propagate the level-N truncated interaction-picture generator and return
    the lab-frame trajectory (N, d, d).

    Integrates d vec(rho_tilde)/dt = L^(N)(t) vec(rho_tilde) with a high-accuracy
    solver, then rotates the envelope back to the lab frame.  Raw symmetric
    Fourier truncation is not guaranteed to be GKSL; callers that need a
    physical reduced generator must use a PSD-preserving factorization instead.
    """
    from scipy.integrate import solve_ivp
    d = len(E)
    modes_trunc = _truncate_modes(modes, n_keep)
    rho_tilde0 = interaction_envelope(rho0_lab, 0.0, E, V)
    vec0 = vectorize_rho(rho_tilde0)

    def rhs(t, vec):
        Lt = truncated_generator_at(modes_trunc, t, T)
        return Lt @ vec

    sol = solve_ivp(rhs, [t_grid[0], t_grid[-1]], vec0, t_eval=t_grid,
                    method='RK45', rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"truncated propagation failed: {sol.message}")
    out = np.empty((len(t_grid), d, d), dtype=np.complex128)
    for i, t in enumerate(t_grid):
        rho_tilde = unvectorize_rho(sol.y[:, i], d)
        out[i] = lab_from_envelope(rho_tilde, t, E, V)
    return out


def fit_rates_lstsq(rho_tilde_traj: np.ndarray, t_grid: np.ndarray,
                    jump_ops_eig: list, nonneg: bool = True) -> np.ndarray:
    """Identify nonnegative Lindblad rates from a learned interaction-picture
    envelope by least squares (the two-stage spectral rate readout).

    The interaction-picture generator is LINEAR in the rates: for single-frequency
    jump operators L_k it is time-independent in the eigenbasis, so

        d rho_tilde/dt = sum_k gamma_k D_k[rho_tilde],
        D_k[rho] = L_k rho L_k^dag - 1/2 { L_k^dag L_k, rho }.

    Given the (slow, smooth) learned envelope rho_tilde(t_i) we estimate its time
    derivative by a centered finite difference and solve

        min_{gamma >= 0} sum_i || d rho_tilde_i/dt - sum_k gamma_k D_k[rho_tilde_i] ||_F^2,

    a small nonnegative least-squares problem. Because the spectral frame makes the
    envelope derivative well conditioned (it does not carry the fast Bohr
    oscillations), this readout is accurate -- the operational form of the local
    identifiability bound. ``jump_ops_eig`` are the UNIT-RATE jump operators in the
    Hamiltonian eigenbasis (L_k with rate factored out).

    Returns:
        gamma: (m,) array of fitted rates.
    """
    drho = np.gradient(rho_tilde_traj, t_grid, axis=0)   # (N, d, d)
    N = rho_tilde_traj.shape[0]
    m = len(jump_ops_eig)
    # Build design matrix columns = vec(D_k[rho_tilde_i]) stacked over i.
    cols = []
    for L in jump_ops_eig:
        Ld = L.conj().T
        LdL = Ld @ L
        Dk = np.empty_like(rho_tilde_traj)
        for i in range(N):
            r = rho_tilde_traj[i]
            Dk[i] = L @ r @ Ld - 0.5 * (LdL @ r + r @ LdL)
        cols.append(Dk.reshape(-1))
    A = np.stack(cols, axis=1)                            # (N*d*d, m) complex
    b = drho.reshape(-1)                                  # (N*d*d,) complex
    # Real-valued stacked system (Hermitian structure -> real rates).
    A_r = np.vstack([A.real, A.imag])
    b_r = np.concatenate([b.real, b.imag])
    gamma, *_ = np.linalg.lstsq(A_r, b_r, rcond=None)
    if nonneg:
        gamma = np.clip(gamma, 0.0, None)
    return gamma


def fit_rates_integral(rho_tilde_traj: np.ndarray, t_grid: np.ndarray,
                       jump_ops_eig: list, init=None,
                       rate_upper_bound: float = 5.0) -> np.ndarray:
    """Noise-robust integral (matrix-exponential) rate readout.

    For single-frequency jump operators the interaction-picture generator is
    time-independent and linear in the rates: L(gamma) = sum_k gamma_k D_k, with
    D_k the unit-rate dissipator superoperator. We fit nonnegative rates by
    matching the PROPAGATED envelope to the learned one,

        min_{gamma >= 0} sum_i || rho_tilde(t_i) - exp(t_i L(gamma)) rho_tilde(0) ||_F^2,

    a small nonlinear least-squares problem. Unlike the derivative readout
    (``fit_rates_lstsq``) this never differentiates the learned envelope, so it is
    robust to the residual envelope error left by measurement noise.
    """
    from scipy.linalg import expm
    from scipy.optimize import least_squares
    d = rho_tilde_traj.shape[1]
    Dsup = [liouvillian_dense(np.zeros((d, d), dtype=np.complex128), [L])
            for L in jump_ops_eig]
    vec0 = vectorize_rho(rho_tilde_traj[0])
    if init is None:
        init = np.full(len(jump_ops_eig), 0.1)

    # A physically generous upper bound keeps the fit numerically stable even when
    # the input envelope is a poor (e.g. unreconstructed-baseline) trajectory, for
    # which the unbounded least squares can otherwise diverge.
    ub = float(rate_upper_bound)

    def resid(g):
        L = sum(g[k] * Dsup[k] for k in range(len(g)))
        out = []
        for i, t in enumerate(t_grid):
            pred = unvectorize_rho(expm(L * t) @ vec0, d)
            diff = pred - rho_tilde_traj[i]
            out.append(diff.real.ravel()); out.append(diff.imag.ravel())
        return np.concatenate(out)

    sol = least_squares(resid, np.minimum(init, ub), bounds=(0.0, ub), method='trf')
    return sol.x


# ---------------------------------------------------------------------------
# PyTorch operations for the training residual.
# ---------------------------------------------------------------------------
def torch_bohr_phase(E_torch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched Bohr-phase tensor Phi[b, a, c] = exp(i (E_a - E_c) t_b).

    Args:
        E_torch: (d,) real eigenvalues
        t: (batch,) or (batch, 1) times
    Returns:
        (batch, d, d) complex phase tensor.
    """
    if t.dim() > 1:
        t = t.reshape(-1)
    omega = E_torch[:, None] - E_torch[None, :]          # (d, d)
    ang = t[:, None, None] * omega[None, :, :]            # (batch, d, d)
    return torch.complex(torch.cos(ang), torch.sin(ang))


def torch_lab_from_envelope(rho_tilde: torch.Tensor, t: torch.Tensor,
                            E_torch: torch.Tensor, V_torch: torch.Tensor) -> torch.Tensor:
    """Envelope (eigenbasis) -> lab state for a batch: rho = V (Phi^* . rho_tilde) V^dag."""
    phase = torch_bohr_phase(E_torch, t)                  # (batch, d, d)
    rho_E = phase.conj() * rho_tilde                      # Hadamard
    Vd = V_torch.conj().T
    return torch.matmul(torch.matmul(V_torch, rho_E), Vd)


def torch_interaction_rhs(rho_tilde: torch.Tensor, t: torch.Tensor,
                          E_torch: torch.Tensor, Ls_E: list,
                          Hc_E: torch.Tensor = None) -> torch.Tensor:
    """Interaction-picture master-equation RHS for the envelope (eigenbasis).

    d rho_tilde/dt = -i[Hc_tilde(t), rho_tilde]
                     + sum_k ( Lk_tilde rho_tilde Lk_tilde^dag
                               - 1/2 { Lk_tilde^dag Lk_tilde, rho_tilde } ),
    with Lk_tilde(t) = Phi(t) . Lk_E and Hc_tilde(t) = Phi(t) . Hc_E (Hadamard).
    The frame Hamiltonian H0 = diag(E) is removed analytically, so no fast
    coherent term appears here.

    Args:
        rho_tilde: (batch, d, d) complex envelopes (eigenbasis)
        t: (batch,) or (batch, 1) times
        E_torch: (d,) eigenvalues
        Ls_E: list of (d, d) jump operators already rotated to the eigenbasis
              (lab L rotated by V^dag L V); the per-time Bohr phase is applied here
        Hc_E: optional (d, d) residual coherent term in the eigenbasis (or None)
    """
    phase = torch_bohr_phase(E_torch, t)                  # (batch, d, d)
    batch = rho_tilde.shape[0]
    drho = torch.zeros_like(rho_tilde)

    if Hc_E is not None:
        Hc_t = phase * Hc_E.unsqueeze(0)                  # (batch, d, d)
        drho = drho - 1j * (torch.bmm(Hc_t, rho_tilde) - torch.bmm(rho_tilde, Hc_t))

    for L_E in Ls_E:
        L_t = phase * L_E.unsqueeze(0)                    # (batch, d, d)
        L_td = L_t.conj().transpose(-2, -1)
        LdL = torch.bmm(L_td, L_t)
        drho = drho + (torch.bmm(torch.bmm(L_t, rho_tilde), L_td)
                       - 0.5 * torch.bmm(LdL, rho_tilde)
                       - 0.5 * torch.bmm(rho_tilde, LdL))
    return drho
