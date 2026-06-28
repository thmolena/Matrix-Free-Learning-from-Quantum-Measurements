# ctpcpinn — Spectral-Frame CPTP-PINNs code artifact

Spectral-frame (interaction-picture eigen-operator) structure-preserving residual
learning, operator-system spectral truncation of the open-system generator, and
matrix-free compilation for Markovian open quantum (Lindblad/GKSL) dynamics. This
package regenerates every table and figure of the manuscript deterministically: a
hard-constrained Cholesky density network, a spectral-frame parameterization that
removes spectral bias (`ctpcpinn/spectral.py`, `models.py:SpectralDensityNet`), an
operator-system spectral truncation of the generator with an error certificate, a
GKSL-preserving generator parameterization, and a matrix-free Lindblad compiler,
with six reproducible simulation studies.

Molena Huynh · North Carolina State University · molena.huynh@jmp.com

## Installation

```bash
cd submission/code
pip install .                 # runtime deps: numpy, scipy, torch, matplotlib, cycler
```

Editable install for development:

```bash
pip install -e ".[dev]"       # adds pytest and build
```

For bit-identical reproduction of the committed numbers, pin the exact
environment used to generate the manuscript artifacts:

```bash
pip install ".[exact]"        # numpy 2.4.2, scipy 1.17.1, torch 2.10.0, matplotlib 3.10.8
```

Python ≥ 3.10 is required. The simulations run on CPU; a GPU is not needed.

## Reproduce the tables and figures

The console entry point installed with the package runs the five experiments
sequentially in a single process and writes the LaTeX tables and PDF figures into
an output directory of your choice:

```bash
ctpcpinn-reproduce                       # full run -> ./ctpcpinn_results/{tables,figures}
ctpcpinn-reproduce --output-dir out      # full run -> out/{tables,figures}
ctpcpinn-reproduce --in-place            # write into submission/{tables,figures}
ctpcpinn-reproduce --quick               # fast smoke test (2 seeds; not publication quality)
ctpcpinn-reproduce --experiment exp1     # a single experiment
```

The source-tree script `run_all.py` is the equivalent canonical entry point and
writes directly into `submission/tables` and `submission/figures`:

```bash
python run_all.py                        # full config; reproduces the paper
python run_all.py --quick                # fast smoke test
python run_one.py --experiment exp4      # a single experiment
```

A dependency-light invariant check is available without pytest:

```bash
ctpcpinn-validate                        # operators, trace preservation, density/CPTP, compiler agreement
```

The full pytest suite covers the same invariants in more detail:

```bash
pytest                                   # ctpcpinn/tests/
```

## Regenerated figures and tables

`ctpcpinn-reproduce` (or `run_all.py`) writes:

| Artifact | Source experiment |
| --- | --- |
| `figures/fig1_schematic.pdf` | Programmatic method-overview schematic |
| `figures/exp1_parameter_recovery.pdf`, `figures/exp1_state_fidelity.pdf` | Single-qubit system identification |
| `figures/exp2_sparse_measurements.pdf` | Sparse-measurement ablation |
| `figures/exp3_qutrit_leakage.pdf` | Fast qutrit reconstruction (spectral frame) |
| `figures/exp4_gate_fidelity.pdf` | Two-qubit dissipative gate, held-out generalization |
| `figures/exp5_compiler_scaling.pdf` | Dense-versus-structured compiler scaling |
| `figures/exp6_spectral_truncation.pdf` | Operator-system spectral truncation hierarchy |
| `tables/exp3_leakage_results.tex` … `exp6_spectral_truncation.tex` | Corresponding LaTeX result tables |

Recompiling `submission/main.tex` after a run picks up the regenerated numbers
automatically.

## Determinism

Each training experiment (Experiments 1–4) is repeated over the fixed seeds
`[0, 1, 2, 3, 4]` and reported as mean ± 95% Student-*t* confidence interval;
Experiments 5 (compiler scaling) and 6 (operator-system spectral truncation) are
deterministic. The full configuration is 3000 Adam epochs at learning rate 10⁻³,
100 time points, and measurement-noise standard deviation 0.02
(`ctpcpinn/config.py`, `FULL_CONFIG`).
The quick configuration uses seeds `[0, 1]`, 200 epochs, and 60 time points.

Reproduction is single-threaded by design. The reproduction scripts pin
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1` and force a sequential
MKL backend before importing NumPy, SciPy, or PyTorch. With the pinned `exact`
dependency set and single-threaded execution, the table values reproduce exactly.
The wall-clock *times* in the compiler-scaling experiment (Experiment 5) are
machine dependent; their ratios and the analytic memory columns are not.

## Dependencies

Runtime: `numpy>=1.24`, `scipy>=1.10`, `torch>=2.0`, `matplotlib>=3.7`,
`cycler>=0.11`. The `exact` extra pins the versions used for the committed
artifacts. The `dev` extra adds `pytest>=7.0` and `build>=1.0`.

## Layout

```text
submission/code/
├── pyproject.toml              # package metadata, entry points, pinned deps
├── README.md                   # this file
├── run_all.py / run_one.py     # source-tree reproduction entry points
├── generate_paper_data.py      # regenerates committed data artifacts
└── ctpcpinn/
    ├── reproduce.py            # ctpcpinn-reproduce console script
    ├── validate.py            # ctpcpinn-validate console script
    ├── foundations.py          # theory-to-code map, from first principles
    ├── operators.py / lindblad.py / solvers.py
    ├── models.py / losses.py / metrics.py / stats.py
    ├── compiler.py / ir.py     # matrix-free open-system compiler and IR
    ├── spectral.py             # spectral frame + operator-system truncation + rate readouts
    ├── theory.py               # statements of the guarantees
    ├── experiments/            # exp1–exp6 (exp6 = operator-system spectral truncation)
    └── tests/                  # invariant tests
```

## Building a distribution

```bash
python -m build                          # builds sdist + wheel into dist/
```

## License

MIT. See `../../LICENSE`.
