# Safety-Gated Spectral Reconstruction of Experimental Ramsey Fringes

This repository contains a nested, trace-local reconstruction study on 40,787
measured probabilities from 177 trapped-ion Ramsey fringes. Linear
interpolation is retained unless a regularized spectral candidate improves a
disjoint development slice by at least five percent.

The outer test set contains every fourth phase point. Mean RMSE is 0.09146 for
the gated method and 0.09434 for interpolation. The complete record retains 69
improved traces, 86 interpolation ties, and 22 accepted-model losses.

## Reproduction

```bash
python -m pip install "./code[test,plot]"
ramsey-adaptive-study --output code/results/adaptive_ramsey.json --verify
ramsey-adaptive-figures
python -m pytest -q code/tests
```

The source is Zenodo DOI `10.5281/zenodo.15797402`, released under CC BY 4.0.
The archive and derived CSV digests, worksheet provenance, and download script
are retained under `code/`.

The matrix-free implementation solves independent trace blocks exactly. At 32
traces, the recorded equivalent solve is 17.1 times faster and the stored
design is 30.5 times smaller than the explicit global block design.
