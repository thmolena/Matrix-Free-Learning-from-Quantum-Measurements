"""CPTP-Compiler-PINNs: Structure-Preserving Physics-Informed Compilation
for Learning Open Quantum Dynamics.

This package reproduces every table and figure of the manuscript. The simplest
entry points are the console scripts installed with the package::

    ctpcpinn-reproduce            # full run -> ./ctpcpinn_results/{tables,figures}
    ctpcpinn-reproduce --quick    # fast smoke test
    ctpcpinn-validate             # dependency-free invariant checks

See foundations.py for the method from first principles and reproduction
guidance mapped to each module.
"""

__version__ = "1.0.0"

from .operators import pauli_x, pauli_y, pauli_z, identity, destroy, create
from .lindblad import lindblad_rhs, liouvillian_dense, trace_distance, fidelity_qubit_or_general
from .models import (DensityMatrixNet, TimeMLP, PositiveParameter, ControlNet,
                     UnconstrainedDensityNet)
from .compiler import QuantumModelIR, CompiledLindbladModel, benchmark_residual_modes
from .stats import aggregate, fmt_pm

__all__ = [
    "__version__",
    "pauli_x", "pauli_y", "pauli_z", "identity", "destroy", "create",
    "lindblad_rhs", "liouvillian_dense", "trace_distance", "fidelity_qubit_or_general",
    "DensityMatrixNet", "TimeMLP", "PositiveParameter", "ControlNet", "UnconstrainedDensityNet",
    "QuantumModelIR", "CompiledLindbladModel", "benchmark_residual_modes",
    "aggregate", "fmt_pm",
]
