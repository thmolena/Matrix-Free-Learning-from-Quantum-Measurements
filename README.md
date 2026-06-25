# CPTP-Compiler-PINNs

**Structure-preserving residual learning and matrix-free compilation for Markovian open quantum (Lindblad/GKSL) dynamics.**

Molena Huynh · North Carolina State University · molena.huynh@jmp.com

[Full report and interactive illustration](https://thmolena.github.io/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/)

## Summary

Open quantum systems are the working language of quantum hardware: every
superconducting, trapped-ion, photonic, spin, and neutral-atom processor extends
beyond its ideal Hamiltonian into relaxation, dephasing, leakage, crosstalk, and
control-induced dissipation. Throughout, one object must stay physical at all
times — the density matrix. This repository identifies Markovian (GKSL/Lindblad)
generators from sparse, noisy observations on two design principles: physicality
is an invariant of the architecture rather than a penalty applied after the fact,
and the Lindblad residual is evaluated by a compiler that bypasses the dense
`d²×d²` Liouvillian. A Cholesky density network makes invalid quantum states
unrepresentable, a GKSL-preserving parameterization keeps the learned generator a
completely positive, trace-preserving (CPTP) semigroup, and a matrix-free Lindblad
compiler lowers a symbolic open-system specification to dense or structured
kernels. Six exact mathematical guarantees and five reproducible simulation
studies establish the value of the structural constraints and the compiler path.

The hard constraint is proved free of expressive cost and yields a transferable,
certified generator. The accompanying study is deliberately reproducible and
modest in scale; it constitutes a controlled proof of concept, separate from
quantum-hardware data or state-of-the-art claims, and the limitations (rate
identification in the fast-oscillatory regime, simulation-only scope) are stated
plainly.

The repository also includes a reproducible PINN benchmark suite on canonical
quantum model problems, with every quantitative claim drawn directly from
committed CSV artifacts in `outputs/`.

## Principal contributions

1. **A hard-constrained density network.** The Cholesky map
   `ρ = A·Aᵀ / Tr(A·Aᵀ)` enforces Hermiticity, positive semidefiniteness, and
   unit trace exactly at every time point, every training iteration, and every
   evaluation point.
2. **A CPTP-by-construction generator parameterization.** Softplus-positive
   Lindblad rates and positive semidefinite Kossakowski factorizations `C = BBᵀ`
   keep the learned generator within GKSL form.
3. **A matrix-free open-system compiler.** A symbolic open-system intermediate
   representation lowers to either a dense superoperator or a structured residual
   kernel that stores `H` and the `m` jump operators in place of a `d²×d²`
   superoperator; the two modes agree to floating-point precision.
4. **Six exact mathematical guarantees.** Exact physicality, universal trajectory
   approximation, CPTP semigroup preservation, an a posteriori trace-norm residual
   certificate, a local identifiability bound, and a dense-versus-structured
   complexity separation.
5. **A fully reproducible study.** Every figure and number is emitted by the
   accompanying scripts; each training experiment is repeated over five seeds and
   reported as mean ± 95% confidence interval against a soft-positivity penalty,
   an unconstrained network, and a classical model-based least-squares fit.

## Headline results

All values are transcribed verbatim from the committed tables.

**Single-qubit system identification (Experiment 1).** True values
ω = 3.1416, Ω = 0.6283, γ₁ = 0.3000, γ_φ = 0.1500. Parameter-recovery relative
error, mean state fidelity, and positivity-violation rate (mean ± 95% CI):

| Method | ω err | Ω err | γ₁ err | γ_φ err | Mean fidelity | Pos. viol. |
| --- | --- | --- | --- | --- | --- | --- |
| **CPTP-PINN (ours)** | 0.001 ± 0.001 | 0.026 ± 0.013 | 0.079 ± 0.027 | 0.131 ± 0.050 | 1.0000 ± 0.0000 | 0.000 |
| Soft-penalty PINN | 0.158 ± 0.195 | 0.358 ± 0.287 | 0.266 ± 0.288 | 1.455 ± 1.184 | 0.9507 ± 0.1163 | 0.004 |
| Unconstrained | 0.157 ± 0.195 | 0.362 ± 0.284 | 0.276 ± 0.283 | 1.439 ± 1.185 | 0.9511 ± 0.1165 | 0.006 |
| Classical LSQ | 0.001 ± 0.001 | 0.016 ± 0.012 | 0.019 ± 0.008 | 0.020 ± 0.019 | 1.0000 ± 0.0000 | 0.000 |

**Robustness under sparse measurements (Experiment 2).** Mean trace distance to
the exact single-qubit trajectory versus retained observation fraction:

| Fraction | CPTP-PINN (ours) | Soft-penalty PINN | Unconstrained |
| --- | --- | --- | --- |
| 1.00 | 0.0044 ± 0.0038 | 0.3668 ± 0.3497 | 0.3688 ± 0.3482 |
| 0.50 | 0.0052 ± 0.0037 | 0.3683 ± 0.3469 | 0.3717 ± 0.3436 |
| 0.25 | 0.0050 ± 0.0018 | 0.3682 ± 0.3490 | 0.3709 ± 0.3467 |
| 0.10 | 0.0100 ± 0.0062 | 0.3572 ± 0.3616 | 0.3572 ± 0.3626 |

**Two-qubit dissipative gate: generalization across initial states
(Experiment 4).** State fidelity for the trained |00⟩ trajectory and three
held-out initial states obtained by propagating the *learned generator*:

| Initial state | Mean fidelity | Final fidelity |
| --- | --- | --- |
| \|00⟩ (trained) | 0.9944 ± 0.0014 | 0.9998 ± 0.0001 |
| \|01⟩ (held-out) | 0.9504 ± 0.0065 | 0.9731 ± 0.0092 |
| \|10⟩ (held-out) | 0.9551 ± 0.0087 | 0.9808 ± 0.0086 |
| \|+0⟩ (held-out) | 0.9743 ± 0.0043 | 0.9808 ± 0.0045 |
| Average | 0.9685 ± 0.0047 | — |

**Dense-versus-structured compiler scaling (Experiment 5).** Median
residual-evaluation time (50 repeats, batch of 100 states) and analytic storage;
CPU, single machine. Extended runs reach a 276.65× speedup at d = 24 with
relative error ~10⁻¹⁶.

| d | Dense (s) | Structured (s) | Speedup | Dense (MB) | Structured (MB) |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.0069 | 0.0012 | 5.9× | 0.000 | 0.00024 |
| 3 | 0.0095 | 0.0016 | 6.1× | 0.001 | 0.00069 |
| 4 | 0.0099 | 0.0015 | 6.5× | 0.004 | 0.00122 |
| 6 | 0.0125 | 0.0016 | 7.7× | 0.020 | 0.00275 |
| 8 | 0.0238 | 0.0019 | 12.9× | 0.062 | 0.00488 |

**Residual certificate, verified numerically (Theorem 4).** Physical
perturbations of the exact qubit trajectory: attained trace distance, the
certificate bound computed from the residual alone, and their ratio:

| Perturbation amplitude | Max trace distance | Certificate (Thm 4) | Certificate / error |
| --- | --- | --- | --- |
| 0.010 | 0.0039 | 0.0092 | 2.37 |
| 0.025 | 0.0097 | 0.0228 | 2.35 |
| 0.050 | 0.0193 | 0.0454 | 2.35 |
| 0.100 | 0.0387 | 0.0908 | 2.35 |

**PINN benchmark suite (committed CSV artifacts in `outputs/`).** On the harmonic
oscillator, a physics-constrained loss attains a ground-state relative L2 error of
**0.001569**, a **148x reduction** relative to the unconstrained tanh baseline at
0.2323 on the same task and architecture. The specialist Hamiltonian formulation
reaches **0.001569** against 0.1196 for the shared non-specialist protocol (**76x**
lower); a 5-layer × 64-unit shared model reaches **0.2658** against 1.4193 for the
2-layer × 64-unit baseline (**5.3x** lower); under 20% input noise the error is
**0.2503** against 0.2565 on the clean-input reference (**2.4% lower** under
corruption); and a 100-point collocation run reaches **0.24794** against 0.24773
at 2000 points (within **0.1%** at 20x fewer points).

## Installation

The reproducible code artifact is distributed as the package `ctpcpinn`:

```bash
pip install ctpcpinn
```

From the repository, install the source tree directly:

```bash
cd submission/code
pip install .                 # runtime deps: numpy, scipy, torch, matplotlib, cycler
pip install ".[exact]"        # pinned versions for bit-identical numbers
```

Python ≥ 3.10 is required. The simulations run on CPU.

## Reproduction

```bash
cd submission/code
ctpcpinn-reproduce            # regenerates every table and figure -> ./ctpcpinn_results
ctpcpinn-validate             # dependency-light invariant checks
```

The source-tree script `python run_all.py` is the equivalent canonical entry
point and writes directly into `submission/tables` and `submission/figures`. Each
training experiment runs over the fixed seeds `[0, 1, 2, 3, 4]` and reports
mean ± 95% Student-*t* CI; the compiler benchmark reports the median over repeated
timings. The full configuration is 3000 Adam epochs at learning rate 10⁻³, 100
time points, and noise standard deviation 0.02. Reproduction is single-threaded by
design; see `submission/code/README.md` for the determinism details.

The PINN benchmark notebooks regenerate from committed data with:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/pinn_harmonic_oscillator.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/pinn_schrodinger.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_pinn_combined.ipynb
```

## Repository layout

```text
QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/
├── README.md
├── LICENSE
├── index.html              # research report + interactive illustration (GitHub Pages)
├── data/                   # CSV anchors for the benchmark studies
├── notebooks/              # executed PINN benchmark notebooks
├── outputs/                # committed CSV/SVG benchmark artifacts
├── src/                    # PINN models, physics, training, inference API
├── submission/             # open-quantum manuscript, theory, and code
│   └── code/               # ctpcpinn package, experiments, tests, reproduction scripts
└── website/                # local inference interface
```

## Citation

```bibtex
@misc{huynh2026cptpcompilerpinns,
  author       = {Molena Huynh},
  title        = {{CPTP-Compiler-PINNs}: Structure-Preserving Residual Learning
                  and Matrix-Free Compilation for Open Quantum Dynamics},
  year         = {2026},
  howpublished = {\url{https://github.com/thmolena/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling}},
  note         = {North Carolina State University. Contact: molena.huynh@jmp.com}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
