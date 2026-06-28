# Spectral-Frame CPTP-PINNs

**Eigen-operator structure-preserving learning and operator-system spectral truncation for open quantum dynamics.**

Molena Huynh · North Carolina State University · molena.huynh@jmp.com

**[Read the manuscript (PDF)](submission/main.pdf)** · [Full tutorial](index.html) · [Reproducibility package (`ctpcpinn`)](submission/code) · [Live interactive report](https://thmolena.github.io/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/)

## Most novel contribution

Learning a Markovian (GKSL/Lindblad) open-system generator from sparse, noisy
observations confronts two obstacles: a neural density surrogate can leave the
physical state space during training, and a smooth-in-time network suffers
*spectral bias* on the fast Bohr-frequency oscillations of multi-level systems.
The central contribution of this work is a **spectral-frame (interaction-picture
eigen-operator) parameterization** together with an **operator-system spectral
truncation of the open-system generator**, which together remove the second
obstacle at the level of the representation rather than the loss:

- The density network learns only the **slow envelope** `ρ̃(t) = U(t)† ρ(t) U(t)`
  in the Hamiltonian eigenbasis, through the same hard Cholesky map (so the
  envelope, and hence the lab state `ρ = U ρ̃ U†`, is Hermitian, positive
  semidefinite, and unit-trace by construction). The fast oscillation carried by
  `U(t) = exp(−iHt)` is supplied **analytically**, so the trainable target is slow
  and spectral bias is removed at the architecture level.
- Band-limiting the interaction-picture generator to its slow Bohr eigen-operators
  is the **spectral-truncation construction of noncommutative geometry**
  (Connes–van Suijlekom operator systems), recently used to build noncommutative
  C\*-algebraic kernel machines. Here it is lifted, for the first time, from
  *static kernels* to a **dynamical, completely positive generator**: a controlled
  multi-resolution hierarchy whose coarsest levels are the secular (Davies)
  generator and the rotating-wave approximation, converging to the exact generator,
  with an a posteriori out-of-band-weight error certificate at every level.

This resolves the principal open problem of the prior structure-preserving
approach — reliable reconstruction of fast multi-level (qutrit leakage) dynamics —
and reduces dissipative-rate identification to a measurement-noise question.

## Demo (from the manuscript)

The figures and tables below are emitted verbatim by the `ctpcpinn` package and
appear in [the manuscript](submission/main.pdf). The two results that carry
the novel contribution:

**The spectral frame reconstructs fast qutrit (leakage) dynamics (Experiment 3).**
Mean state fidelity and the recovery error of the dominant decay rates
(`γ₁₀, γ₂₁`), read out identically for every method by an integral least-squares
fit of the linear-in-rates generator to the learned interaction-picture envelope.
Mean ± 95% CI over five seeds.

| Density network | Mean state fidelity | Dominant-rate error |
| --- | --- | --- |
| Plain MLP | `0.7657 ± 0.3463` | — |
| Fourier time-features | `0.5238 ± 0.4897` | — |
| **Spectral frame (ours)** | `0.9847 ± 0.0015` | `0.273 ± 0.025` |

The plain and Fourier baselines are unreliable across seeds; the spectral frame
reconstructs the fast dynamics reliably (a confidence interval more than two orders
of magnitude tighter, near unity) and is the only method whose envelope is accurate
enough to identify the rates (the readout is reported only for a reconstructed
trajectory, mean fidelity ≥ 0.9).

**Operator-system spectral truncation of the generator (Experiment 6).** Mean
state fidelity of the level-*N* truncated interaction-picture generator propagated
against the exact dynamics of a driven dissipative two-qubit system, and the
out-of-band spectral weight `η_N = Σ_{|k|>N} ‖L_k‖` that certifies the
generator-approximation error. Deterministic (no seeds).

| N | Interpretation | Mean fidelity | Out-of-band weight |
| --- | --- | --- | --- |
| 0 | secular (Davies) | `0.7600` | `1.66e+01` |
| 1 | rotating-wave | `0.9553` | `8.46e+00` |
| 2 | partial band | `0.9931` | `3.27e+00` |
| 4 | partial band | `0.999988` | `1.82e-01` |
| 8 | partial band | `1.000000` | `2.51e-04` |

The hierarchy converges monotonically to the exact generator (and the structured
kernel) as the band is widened; the certificate `η_N` tracks and bounds the error.

## Principal contributions

1. **A hard-constrained density network.** The Cholesky map `ρ = A·A† / Tr(A·A†)`
   enforces Hermiticity, positive semidefiniteness, and unit trace exactly at every
   time point, training iteration, and evaluation point.
2. **A spectral-frame (eigen-operator) parameterization.** The network learns the
   slow interaction-picture envelope through the same hard map and supplies the fast
   Bohr-frequency oscillations analytically, removing spectral bias at the
   architecture level and making the dissipative rates identifiable by a linear
   least-squares readout.
3. **Operator-system spectral truncation of the generator.** Band-limiting the
   interaction-picture generator to its slow Bohr modes gives a controlled
   multi-resolution hierarchy — secular (Davies) and rotating-wave as special cases,
   converging to the exact generator — with an a posteriori truncation-error
   certificate, generalizing spectral truncation from static C\*-algebraic kernels
   to a completely positive dynamical generator.
4. **A CPTP generator parameterization and matrix-free compiler.** Softplus-positive
   rates and positive semidefinite Kossakowski factorizations keep the generator in
   GKSL form; a symbolic open-system IR lowers to dense, structured, or spectrally
   truncated kernels.
5. **Eight exact mathematical guarantees**, including spectral-frame
   physicality-and-equivalence and the operator-system truncation bound, and a fully
   reproducible study (six experiments, five seeds each, mean ± 95% CI).

## Supporting results

All values are emitted by the package and transcribed verbatim from the committed
tables.

**Single-qubit system identification (Experiment 1).** True values
ω = 3.1416, Ω = 0.6283, γ₁ = 0.3000, γ_φ = 0.1500. Relative parameter error, mean
state fidelity, and positivity-violation rate (mean ± 95% CI):

| Method | ω err | Ω err | γ₁ err | γ_φ err | Mean fidelity | Pos. viol. |
| --- | --- | --- | --- | --- | --- | --- |
| **CPTP-PINN (ours)** | 0.001 ± 0.001 | 0.026 ± 0.013 | 0.079 ± 0.027 | 0.131 ± 0.050 | 1.0000 ± 0.0000 | 0.000 |
| Soft-penalty PINN | 0.158 ± 0.195 | 0.358 ± 0.287 | 0.266 ± 0.288 | 1.455 ± 1.184 | 0.9507 ± 0.1163 | 0.004 |
| Unconstrained | 0.157 ± 0.195 | 0.362 ± 0.284 | 0.276 ± 0.283 | 1.439 ± 1.185 | 0.9511 ± 0.1165 | 0.006 |
| Classical LSQ | 0.001 ± 0.001 | 0.016 ± 0.012 | 0.019 ± 0.008 | 0.020 ± 0.019 | 1.0000 ± 0.0000 | 0.000 |

**Robustness under sparse measurements (Experiment 2).** Mean trace distance to the
exact single-qubit trajectory versus retained observation fraction:

| Fraction | CPTP-PINN (ours) | Soft-penalty PINN | Unconstrained |
| --- | --- | --- | --- |
| 1.00 | 0.0044 ± 0.0038 | 0.3668 ± 0.3497 | 0.3688 ± 0.3482 |
| 0.50 | 0.0052 ± 0.0037 | 0.3683 ± 0.3469 | 0.3717 ± 0.3436 |
| 0.25 | 0.0050 ± 0.0018 | 0.3682 ± 0.3490 | 0.3709 ± 0.3467 |
| 0.10 | 0.0100 ± 0.0062 | 0.3572 ± 0.3616 | 0.3572 ± 0.3626 |

**Two-qubit dissipative gate: generalization across initial states (Experiment 4).**
State fidelity for the trained |00⟩ trajectory and three held-out initial states
obtained by propagating the *learned generator*:

| Initial state | Mean fidelity | Final fidelity |
| --- | --- | --- |
| \|00⟩ (trained) | 0.9944 ± 0.0014 | 0.9998 ± 0.0001 |
| \|01⟩ (held-out) | 0.9504 ± 0.0065 | 0.9731 ± 0.0092 |
| \|10⟩ (held-out) | 0.9551 ± 0.0087 | 0.9808 ± 0.0086 |
| \|+0⟩ (held-out) | 0.9743 ± 0.0043 | 0.9808 ± 0.0045 |
| Average | 0.9685 ± 0.0047 | — |

**Dense-versus-structured compiler scaling (Experiment 5).** The robust,
hardware-independent separation is the analytic memory scaling (`O(d⁴)` dense versus
`O(md²)` structured); the two residual evaluators agree to floating-point precision.

## Installation

The reproducible code artifact is the package `ctpcpinn`:

```bash
cd submission/code
pip install .                 # runtime deps: numpy, scipy, torch, matplotlib, cycler
pip install ".[exact]"        # pinned versions for bit-identical numbers
```

Python ≥ 3.10 is required. The simulations run on CPU.

## Reproduction

```bash
cd submission/code
ctpcpinn-reproduce            # regenerates every table and figure (6 experiments)
ctpcpinn-validate             # dependency-light invariant checks (incl. spectral frame)
```

`python run_full.py` is the canonical source-tree entry point and writes the
figure PDFs into `submission/code/figures` (the manuscript `\graphicspath`). Each
training experiment runs over the fixed seeds `[0, 1, 2, 3, 4]` and reports
mean ± 95% Student-*t* CI; the compiler and spectral-truncation benchmarks are
deterministic. The full configuration is 3000 Adam epochs at learning rate 10⁻³,
100 time points, and noise standard deviation 0.02. Reproduction is single-threaded
by design; see `submission/code/README.md` for the determinism details.

## Repository layout

```text
QuantumPINNs-.../
├── README.md
├── LICENSE
├── index.html              # research report + interactive illustration (GitHub Pages)
├── submission/             # open-quantum manuscript, theory, and code
│   ├── main.tex            # the manuscript
│   └── code/               # ctpcpinn package, experiments, tests, reproduction scripts
│       └── ctpcpinn/spectral.py   # the spectral frame + operator-system truncation
└── website/                # local inference interface
```

## Citation

```bibtex
@misc{huynh2026spectralcptppinns,
  author       = {Molena Huynh},
  title        = {{Spectral-Frame CPTP-PINNs}: Eigen-Operator Structure-Preserving
                  Learning and Operator-System Spectral Truncation for Open Quantum
                  Dynamics},
  year         = {2026},
  howpublished = {\url{https://github.com/thmolena/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling}},
  note         = {North Carolina State University. Contact: molena.huynh@jmp.com}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
