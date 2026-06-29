"""Open-system intermediate representation (IR), re-exported from compiler.py.

The "compiler" view of the manuscript (Introduction; Methods, "Open-system IR,
dispatch, and algorithms") specifies a quantum model symbolically -- Hilbert
dimension, Hamiltonian terms, jump operators, rates and observables -- as a
``QuantumModelIR``, then lowers it to either a dense d^2 x d^2 Liouvillian or a
structured matrix-free residual kernel (``CompiledLindbladModel``). This module is
a thin alias so callers can import the IR types from ``ctpcpinn.ir``.
"""

from .compiler import QuantumModelIR, CompiledLindbladModel, benchmark_residual_modes

__all__ = ["QuantumModelIR", "CompiledLindbladModel", "benchmark_residual_modes"]
