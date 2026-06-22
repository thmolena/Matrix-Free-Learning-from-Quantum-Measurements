"""From-scratch foundations for CPTP-Compiler-PINNs.

The goal is to keep the explanatory material inside importable source code:
linear algebra, open quantum dynamics, machine learning, compiler complexity and
the exact reproduction path are all summarized here with pointers to the modules
that implement them.
"""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Linear algebra",
        "A quantum state is represented by a complex density matrix rho.  A "
        "valid rho is Hermitian, positive semidefinite and trace one.  The "
        "operators.py module defines Pauli matrices, tensor products, projectors "
        "and commutators used throughout the experiments.",
    ),
    (
        "Open-system physics",
        "Noisy Markovian dynamics are described by a GKSL/Lindblad generator: "
        "a Hamiltonian commutator plus dissipative jump terms with non-negative "
        "rates.  lindblad.py evaluates this right-hand side and constructs dense "
        "Liouvillian forms for verification.",
    ),
    (
        "CPTP constraint",
        "Physical evolution must be completely positive and trace preserving.  "
        "The learned rates are parameterized through positive maps and the "
        "Hamiltonian through Hermitian bases, so the learned generator remains "
        "legal by construction.",
    ),
    (
        "Neural state map",
        "models.py uses an MLP of time to produce a Cholesky-like factor A(t), "
        "then sets rho(t)=A A^dagger / Tr(A A^dagger).  This hard constraint "
        "makes illegal density matrices unrepresentable, unlike a soft penalty "
        "that merely discourages violations.",
    ),
    (
        "PINN residual",
        "losses.py combines a data term, an initial-condition term and a "
        "physics residual ||d rho_phi/dt - L_theta[rho_phi]||.  PyTorch autograd "
        "computes the time derivative and optimizer gradients.",
    ),
    (
        "Forward and inverse problems",
        "solvers.py integrates the true Lindblad equation to generate reference "
        "trajectories and measurements.  The inverse experiments recover a "
        "trajectory and generator parameters from sparse noisy observations.",
    ),
    (
        "Compiler view",
        "compiler.py represents open-system residual evaluation both as a dense "
        "d^2 by d^2 Liouvillian and as structured kernels.  The scaling study "
        "checks numerical agreement while measuring the runtime and memory "
        "benefit of avoiding dense materialization.",
    ),
    (
        "Statistics",
        "stats.py aggregates fixed-seed repeated experiments as means and 95 "
        "percent confidence intervals.  Timing numbers are machine dependent; "
        "state fidelities, trace distances and table values are deterministic "
        "for the configured seeds and dependency versions.",
    ),
    (
        "Reproduction path",
        "Run python run_all.py in submission/code to regenerate the in-place "
        "tables and figures used by main.tex.  Use ctpcpinn-reproduce for an "
        "installed console entry point and ctpcpinn-validate for invariant "
        "checks.  Dependencies live in pyproject.toml.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
