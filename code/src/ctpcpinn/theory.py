"""From-principles guide for CPTP-Compiler-PINNs.

Linear algebra
--------------
A d-level quantum state is represented by a d by d complex density matrix rho.
It is physical if it is Hermitian, positive semidefinite and trace one. The
helper modules operators.py, lindblad.py and metrics.py implement these checks
and the matrix operations used in every experiment.

Open-system physics
-------------------
Closed-system dynamics use -i[H, rho]. Markovian open-system dynamics add
Lindblad dissipators:

    d rho / dt = -i[H, rho]
        + sum_k gamma_k (L_k rho L_k^dagger
        - 0.5 {L_k^dagger L_k, rho}).

If the rates gamma_k are nonnegative and H is Hermitian, the generated evolution
is completely positive and trace preserving (CPTP). lindblad.py evaluates this
right-hand side and solvers.py integrates it to create ground-truth trajectories.

Machine learning
----------------
A multilayer perceptron is a parameterized function built from affine maps and
nonlinearities. Here the input is time, optionally expanded by Fourier features,
and the output parameterizes a density matrix and a generator. PyTorch automatic
differentiation supplies d rho_phi / dt for the physics residual.

Physics-informed loss
---------------------
losses.py combines three terms:

* data loss against observed expectation values;
* residual loss ||d rho_phi / dt - L_theta[rho_phi]||_F**2;
* initial-condition loss at t = 0.

The Cholesky hard constraint
----------------------------
The central state parameterization is

    rho_phi(t) = A_phi(t) A_phi(t)^dagger / Tr(A_phi(t) A_phi(t)^dagger).

This makes illegal density matrices unrepresentable: every output is Hermitian,
positive semidefinite and trace one. Rates are parameterized with softplus, and
Hermitian bases keep the Hamiltonian legal, so learned generators stay in the
GKSL form.

The spectral frame and symmetric Fourier approximation
------------------------------------------------------
A hard constraint guarantees a legal state but not an accurate one: a smooth
network is biased toward low frequencies and cannot resolve the fast Bohr-
frequency oscillations of multi-level systems. The spectral frame (spectral.py,
models.py:SpectralDensityNet) learns only the slow envelope

    rho_tilde(t) = U(t)^dagger rho(t) U(t),   U(t) = exp(-i H t),

in the Hamiltonian eigenbasis through the same Cholesky map, and rebuilds the
fast lab state rho = U rho_tilde U^dagger analytically. Conjugation by a unitary
preserves physicality, so this is exactly physical and exactly equivalent to the
lab dynamics, while the trainable target is slow -- removing spectral bias.

In this frame the generator is almost periodic; keeping only its slowest temporal
Fourier modes (the band |k| <= N) gives a symmetric linear partial sum. Pairing
positive and negative modes preserves Hermiticity and trace, but not complete
positivity in general. The k=0 time average is GKSL; identifying it as a Davies
generator requires extra weak-coupling/secular assumptions that the generic FFT
construction does not impose. The reported out-of-band weight
eta_N = sum_{|k|>N} ||mathbb(L)_k||_F is a computable F-to-F
generator-approximation bound; conversion to a trace-norm trajectory bound costs
an additional sqrt(d) factor.
A physical finite-band alternative truncates a factor B_N(t) and forms the
Kossakowski matrix C_N(t)=B_N(t)B_N(t)^dagger; it is not the same linear partial
sum and its approximation error must be bounded separately. The dissipative
rates are read out from the learned envelope by a small linear/integral least
squares (spectral.py:fit_rates_integral), the operational form of the local
identifiability bound.

Compiler view
-------------
compiler.py compares dense Liouvillian evaluation, which stores a d**2 by d**2
superoperator, with a structured residual kernel that applies commutators and
dissipators directly. The manuscript's compiler benchmark reports time and memory
for these two representations.

Statistics and reproduction
---------------------------
Each stochastic experiment uses fixed seeds and reports mean plus 95 percent
confidence intervals. To reproduce the submitted artifacts from code:

    export PYTHONPATH=.
    python scripts/reference_run_all.py
    python -m ctpcpinn.validate
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
