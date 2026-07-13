# specops-cptppinn — Structure-preserving physics-informed learning of open quantum dynamics

> Distribution name: **`specops-cptppinn`** (import name: `ctpcpinn`). Part of the
> **spectral-truncation operators** (`specops`) program.
>
> Manuscript: *Structure-preserving physics-informed learning of open quantum
> dynamics with spectral-frame encodings and operator-system truncation.*
> Molena Huynh, North Carolina State University, Raleigh, North Carolina 27695, USA
> (molena.huynh@jmp.com).
> Full PDF: <https://thmolena.github.io/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/submission/main.pdf>

## Summary

This package implements and reproduces a structure-preserving, physics-informed
framework for identifying Markovian open quantum systems from finite, noisy
measurements. It supplies a hard-constrained Cholesky density network that emits a
physically valid quantum state at every point by construction; a spectral-frame
(interaction-picture eigen-operator) parameterization that learns only the slow
envelope of the dynamics and supplies the fast coherent oscillations analytically,
removing spectral bias at the architecture level; an operator-system spectral
truncation of the generator with an a posteriori error certificate; and a
matrix-free compiler that lowers a symbolic open-system specification to dense,
structured, or spectrally truncated kernels. It deterministically regenerates every
table and figure of the manuscript over six reproducible, CPU-scale studies.

## Background and problem setting (from first principles)

A quantum device is, from the outside, a stochastic object: every experimental run
returns a single measurement outcome sampled from a distribution fixed by the
device's state. Consequently, everything one learns about the hardware — estimating
an observable, reconstructing a state, certifying a gate, tracking calibration
drift — is statistical inference from samples.

The state of a `d`-dimensional quantum system is a **density matrix** `ρ`: a
`d × d` complex matrix that is Hermitian (`ρ = ρ†`), positive semidefinite
(`ρ ⪰ 0`, so all eigenvalues, read as probabilities, are nonnegative), and unit
trace (`Tr ρ = 1`). The set of such matrices is the state space `𝒟(ℋ)`. Real
hardware is never closed: it relaxes, dephases, leaks population out of the
computational subspace, and couples to its environment. When this dissipation is
memoryless (Markovian), the most general physically admissible law of motion is the
**Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) master equation**,
`ρ̇ = 𝓛[ρ] = −i[H, ρ] + Σ_k ( L_k ρ L_k† − ½{L_k† L_k, ρ} )`,
where `H` is the Hamiltonian (coherent part) and the jump operators `L_k` encode
the noise channels. Its generator `𝓛` produces a completely positive,
trace-preserving (CPTP) semigroup — the exact condition that keeps `ρ(t)` a valid
state for all time.

**Open-system identification** is the inverse problem: given noisy measurements of a
few observables at a few time points, recover the generator `𝓛` (its Hamiltonian
terms and dissipation rates) and the trajectory it produces. Physics-informed
neural networks (PINNs) are a natural tool — they place the equation of motion
directly in the training objective as a residual — but a plain neural map
`t ↦ ρ_θ(t)` has no reason to output a Hermitian, positive, unit-trace matrix, and
penalizing violations after the fact is not the same as forbidding them. A second,
independent obstacle is **spectral bias**: multilevel systems oscillate at fast Bohr
frequencies `ω_ab = E_a − E_b`, and smooth-in-time networks learn low frequencies
first, resolving the fast oscillations slowly and unreliably. This package addresses
both obstacles by construction rather than by penalty.

## Contributions

1. **A hard-constrained density network.** The Cholesky map
   `ρ_φ(t) = A_φ(t)A_φ(t)† / Tr(A_φ(t)A_φ(t)†)`, with `A_φ` a neural
   lower-triangular factor with nonzero diagonal, is differentiable and lands in
   `𝒟(ℋ)` exactly — Hermitian, positive semidefinite, unit trace — at every time
   point, every training iteration, and every evaluation point. Invalid states are
   unrepresentable, not merely penalized.
2. **A spectral-frame (eigen-operator) parameterization.** The network learns only
   the slow interaction-picture envelope `ρ̃(t) = U(t)† ρ(t) U(t)` in the
   Hamiltonian eigenbasis, through the same hard map, and supplies the fast
   oscillation analytically via the known `U(t) = exp(−iHt)`. Because conjugation by
   a unitary preserves `𝒟(ℋ)`, physicality remains architectural while spectral
   bias is removed. A well-conditioned envelope derivative renders the dissipation
   rates identifiable by a linear least-squares readout.
3. **Operator-system spectral truncation of the generator.** In the interaction
   picture the generator is almost periodic with Fourier series
   `𝓛̃_t = Σ_k e^{ikω₀t} 𝓛_k`; keeping the band `|k| ≤ N` projects onto a finite
   operator system of slow Bohr eigen-operators. This yields a controlled
   multi-resolution hierarchy — `N=0` is the secular (Davies) generator, `N=1` the
   rotating-wave approximation, the full band the exact generator — with an
   a posteriori out-of-band-weight certificate `η_N = Σ_{|k|>N} ‖𝓛_k‖`. It lifts
   the spectral-truncation construction of noncommutative geometry and C\*-algebraic
   kernel machines from static kernels to a completely positive dynamical generator.
4. **A CPTP-by-construction generator parameterization and matrix-free compiler.**
   Softplus-positive rates and positive-semidefinite Kossakowski factorizations
   `C = BB†` keep the learned generator in GKSL form; a symbolic open-system
   intermediate representation is lowered to a dense Liouvillian, a structured
   residual kernel, or a spectrally truncated kernel that agree to floating-point
   precision.
5. **Eight exact guarantees and a fully reproducible study.** These include exact
   density-matrix physicality, universal approximation, GKSL preservation, the
   a posteriori residual certificate, local identifiability, dense-versus-structured
   complexity, spectral-frame physicality-and-equivalence, and the operator-system
   truncation bound. Every figure and number is emitted by this package over six
   experiments (five seeds each for the training studies), reported as mean ± a 95%
   confidence interval.

## Method

Training minimizes a physics-informed objective
`𝒥 = λ_data 𝒥_data + λ_phys 𝒥_phys + λ_0 𝒥_0` combining a masked data term (fit to
the noisy observations), a Lindblad physics residual
`𝒥_phys = (1/N_c) Σ_ℓ ‖ρ̇_φ(τ_ℓ) − 𝓛_Θ[ρ_φ(τ_ℓ)]‖_F²` evaluated at collocation
points, and an initial-condition term. The time derivative `ρ̇_φ` is obtained by
automatic differentiation of the density network; the residual `𝓛_Θ[ρ_φ]` is
produced by the matrix-free compiler, which selects dense or structured evaluation
by dimension and batch size. In the spectral frame the same Cholesky map is applied
to the envelope `ρ̃_φ` and the lab state is the analytic rotation
`ρ_φ(t) = U(t) ρ̃_φ(t) U(t)†`. The learned generator is always a legal Lindbladian,
so it can be propagated to held-out initial states with an exact solver — the test
that matters for device characterization.

## Main results

All values are produced by this package and repeated over five seeds (redrawing both
the network initialization and the measurement-noise realization); confidence
intervals are 95% Student-*t* over seeds. The compiler and truncation benchmarks are
deterministic.

- **Fast qutrit (leakage) reconstruction (Experiment 3).** The spectral frame raises
  mean state fidelity from `0.766 ± 0.346` (plain MLP) and `0.524 ± 0.490`
  (Fourier-feature network) — intervals as wide as their means — to
  **`0.985 ± 0.002`**, more than two orders of magnitude tighter, while remaining a
  valid density matrix by construction. The reliable envelope then identifies the
  dominant decay rates (`γ₁₀, γ₂₁`) to relative error `0.273 ± 0.025`.
- **Operator-system spectral truncation (Experiment 6).** The hierarchy converges
  monotonically to the exact generator: secular `N=0` reaches mean fidelity `0.760`
  (`η_N ≈ 1.66×10¹`), rotating-wave `N=1` reaches `0.955` (`η_N ≈ 8.46`), `N=4`
  reaches `0.999988`, and `N=8` reaches machine precision (`η_N ≈ 2.51×10⁻⁴`), with
  the certificate tracking and bounding the error at every level.
- **Single-qubit identification (Experiment 1).** Among the neural methods the hard
  constraint is decisive (dephasing-rate error 13% versus >140% for the soft-penalty
  and unconstrained baselines) with mean fidelity `1.0000` and **zero** positivity
  violations, against 0.4–0.6% of time points for the baselines. A classical
  least-squares fit that already knows the exact model remains strongest on this
  dense-data task; the architecture's value is guaranteed physicality and
  robustness, not beating a well-specified classical fit on dense data.
- **Sparse measurements (Experiment 2).** The constrained model holds the mean trace
  distance between `0.004` and `0.010` across observation fractions from `1.0` down
  to `0.10` — one to two orders of magnitude below the soft and unconstrained
  baselines (`≈0.37`).
- **Transferable two-qubit gate (Experiment 4).** The learned generator reproduces
  the trained `|00⟩` trajectory at fidelity `0.994 ± 0.001` and transfers to three
  held-out initial states at mean fidelities `0.95`, `0.96`, `0.97` (overall
  `0.97 ± 0.005`).
- **Compiler scaling (Experiment 5).** The genuine, hardware-independent separation
  is memory: dense Liouvillian storage scales as `O(d⁴)`, the structured
  representation as `O(md²)`, and the two residuals agree to floating-point
  precision. No speed advantage is claimed; the benchmark's dense path rebuilds the
  Liouvillian per state, so its timings overstate any intrinsic per-matvec gap.

## Significance

Two design principles are validated here. First, physicality belongs in the
architecture as an invariant rather than in the loss as an afterthought: the hard
map guarantees a valid trajectory at every step and degrades gracefully under sparse
data, precisely the regime where soft penalties compete with the data and residual
terms. Second, the fast coherent structure of open-system dynamics should be carried
by an analytic change of frame rather than learned, so the network represents only
what is genuinely unknown — the slow dissipative envelope — which resolves the
principal open problem of the structure-preserving approach and reduces rate
identification to a measurement-noise question. The operator-system truncation
furnishes a single, certified dial between the cheapest time-independent secular
model and the exact generator, connecting open-system identification to the
spectral-truncation program of noncommutative geometry. The results are deliberately
scoped to controlled, fully reproducible simulations; hardware validation,
superiority over every baseline, and production-scale acceleration are future
targets, not claims made here.

## Installation and reproduction

```bash
pip install specops-cptppinn         # from PyPI
```

or from this source tree:

```bash
cd code
pip install .                 # runtime deps: numpy, scipy, torch, matplotlib, cycler
```

This installs the distribution **`specops-cptppinn`** and its console scripts
`cptppinn-reproduce` / `cptppinn-validate` (legacy aliases `ctpcpinn-reproduce` /
`ctpcpinn-validate` are also installed). The importable module is `ctpcpinn`:

```python
import ctpcpinn
from ctpcpinn import SpectralDensityNet, propagate_truncated, fidelity_qubit_or_general
```

Editable install for development, and the pinned environment for bit-identical
reproduction of the committed numbers:

```bash
pip install -e ".[dev]"       # adds pytest and build
pip install ".[exact]"        # numpy 2.4.2, scipy 1.17.1, torch 2.10.0, matplotlib 3.10.8
```

Python ≥ 3.9 is required. The simulations run on CPU; a GPU is not needed.

### Reproduce the tables and figures

The console entry point runs the six experiments sequentially in a single process
and writes the LaTeX tables and PDF figures into an output directory of your choice:

```bash
cptppinn-reproduce                       # full run -> ./ctpcpinn_results/{tables,figures}
cptppinn-reproduce --output-dir out      # full run -> out/{tables,figures}
cptppinn-reproduce --in-place            # write into submission/{tables,figures}
cptppinn-reproduce --quick               # fast smoke test (2 seeds; not publication quality)
cptppinn-reproduce --experiment exp1     # a single experiment
```

The source-tree script `run_all.py` is the equivalent canonical entry point and
writes directly into `submission/tables` and `submission/figures`:

```bash
python run_all.py                        # full config; reproduces the paper
python run_all.py --quick                # fast smoke test
python run_one.py --experiment exp4      # a single experiment
```

A dependency-light invariant check, and the full pytest suite:

```bash
cptppinn-validate                        # operators, trace preservation, density/CPTP, compiler agreement
pytest                                   # ctpcpinn/tests/
```

Recompiling `submission/main.tex` after a run picks up the regenerated numbers
automatically.

### Regenerated figures and tables

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

### Determinism

Each training experiment (Experiments 1–4) is repeated over the fixed seeds
`[0, 1, 2, 3, 4]` and reported as mean ± 95% Student-*t* confidence interval;
Experiments 5 (compiler scaling) and 6 (operator-system spectral truncation) are
deterministic. The full configuration is 3000 Adam epochs at learning rate 10⁻³,
100 time points, and measurement-noise standard deviation 0.02
(`ctpcpinn/config.py`, `FULL_CONFIG`); the quick configuration uses seeds `[0, 1]`,
200 epochs, and 60 time points.

Reproduction is single-threaded by design. The reproduction scripts pin
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1` and force a sequential
MKL backend before importing NumPy, SciPy, or PyTorch. With the pinned `exact`
dependency set and single-threaded execution, the table values reproduce exactly.
The wall-clock *times* in the compiler-scaling experiment are machine dependent;
their ratios and the analytic memory columns are not.

## Extend / tweak

Everything the experiments read is a plain-dict configuration key or a network
constructor argument, so the study can be retuned or extended without touching the
experiment logic.

### Command-line flags (`cptppinn-reproduce`)

| Flag | Default | Effect |
| --- | --- | --- |
| `--output-dir DIR` | `./ctpcpinn_results` | Write `DIR/{tables,figures}`. |
| `--in-place` | off | Write directly into `submission/{tables,figures}`. |
| `--quick` | off | Use `QUICK_CONFIG` (2 seeds, 200 epochs) — a fast smoke test, **not** publication quality. |
| `--experiment {exp1..exp6}` | all | Run a single experiment. |
| `--threads N` | 1 | BLAS/OMP threads; keep at 1 for bit-identical numbers. |

### Configuration keys (`ctpcpinn/config.py`)

Edit `FULL_CONFIG` / `QUICK_CONFIG`, or build your own dict and pass it to an
experiment's `run(config)`. Recognized keys:

| Key | Meaning | Used by |
| --- | --- | --- |
| `seeds` | list of RNG seeds; results are mean ± 95% CI over them | exp1–exp4 |
| `seed` | single seed (per-seed inner calls) | all |
| `n_epochs` | Adam epochs per training run | exp1–exp4 |
| `n_time_points` | number of observation time points | exp1–exp4 |
| `n_colloc` | collocation points for the physics residual | exp1–exp4 |
| `noise_std` | measurement-noise standard deviation | exp1–exp4 |
| `n_batch` | batch of states in the compiler benchmark | exp5 |
| `n_repeats` / `n_repeats_timing` | timed repeats for the median wall time | exp5 |
| `w_phys` | weight `λ_phys` on the Lindblad residual term | exp1–exp4 |
| `fourier_features` | number `K` of deterministic Fourier time-features | exp3/exp4 |
| `trunc_levels` | list of truncation bands `N` to sweep | exp6 |
| `trunc_fft_grid` | FFT length for extracting the generator Fourier modes | exp6 |
| `verbose` | print per-run progress | all |
| `figures_dir` / `tables_dir` | output directories (set for you by the CLI) | all |

Example — retune the qutrit study and run it programmatically:

```python
from ctpcpinn.config import FULL_CONFIG
from ctpcpinn.experiments import exp3_qutrit_leakage as exp3

cfg = dict(FULL_CONFIG)
cfg.update(seeds=[0, 1, 2], n_epochs=5000, noise_std=0.01, fourier_features=64,
           figures_dir="out/figures", tables_dir="out/tables")
exp3.run(cfg)
```

### Network / model knobs

The architectures accept constructor arguments so capacity or the physical setup can
be changed without editing model code:

- `DensityMatrixNet(d, hidden_dim=128, n_layers=4, ...)` — hard Cholesky density
  network; `d` is the Hilbert dimension, `hidden_dim`/`n_layers` set MLP capacity.
- `SpectralDensityNet(H, hidden_dim=96, n_layers=4, ...)` — spectral-frame network;
  pass your **own frame Hamiltonian** `H` (a `d×d` Hermitian tensor) to change the
  system's eigenbasis and Bohr frequencies.
- `UnconstrainedDensityNet(d, ...)` — soft-penalty / unconstrained baseline.

### Adding inputs (a new system or experiment)

Lindblad systems are described symbolically through the compiler IR
(`ctpcpinn/compiler.py:QuantumModelIR`): supply `H`, the jump operators `L_k`, the
rates, and observables, then call `CompiledLindbladModel` to lower to a dense
Liouvillian or a matrix-free structured kernel. Operator-system truncation of any
such generator is available directly via
`ctpcpinn.spectral.generator_fourier_modes` and `propagate_truncated`.

To add a new experiment:

1. Create `ctpcpinn/experiments/exp7_myname.py` exposing `run(config)` that writes
   its `.tex`/`.pdf` into `config['tables_dir']` / `config['figures_dir']`.
2. Register it in `ctpcpinn/reproduce.py` (`EXPERIMENTS`, `_TITLES`, and the
   `_load` module-name map).
3. Run `cptppinn-reproduce --experiment exp7`.

To plug the building blocks into another project (nothing depends on the experiment
harness):

```python
import torch
from ctpcpinn import (SpectralDensityNet, propagate_truncated,
                      generator_fourier_modes, fidelity_qubit_or_general)

H = torch.diag(torch.tensor([0.0, 2*torch.pi*2.0, 2*torch.pi*3.8]))  # your frame
net = SpectralDensityNet(H)          # learns the slow interaction-picture envelope
# ... train net against your data with the physics residual (see losses.py) ...
```

## Layout

```text
code/
├── pyproject.toml              # package metadata, entry points, pinned deps
├── README.md                   # this file
├── run_all.py / run_one.py     # source-tree reproduction entry points
├── generate_paper_data.py      # regenerates committed data artifacts
└── ctpcpinn/
    ├── reproduce.py            # cptppinn-reproduce console script
    ├── validate.py             # cptppinn-validate console script
    ├── foundations.py          # theory-to-code map, from first principles
    ├── operators.py / lindblad.py / solvers.py
    ├── models.py / losses.py / metrics.py / stats.py
    ├── compiler.py / ir.py     # matrix-free open-system compiler and IR
    ├── spectral.py             # spectral frame + operator-system truncation + rate readouts
    ├── theory.py               # statements of the guarantees
    ├── experiments/            # exp1–exp6 (exp6 = operator-system spectral truncation)
    └── tests/                  # invariant tests
```

## Cite this work

If this package or its method is used, please cite the manuscript:

```bibtex
@article{huynh2026cptppinn,
  author  = {Huynh, Molena},
  title   = {Structure-preserving physics-informed learning of open quantum
             dynamics with spectral-frame encodings and operator-system truncation},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (specops) program}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## Building a distribution

```bash
python -m build                          # builds sdist + wheel into dist/
```

## License

MIT. See `../LICENSE`.
