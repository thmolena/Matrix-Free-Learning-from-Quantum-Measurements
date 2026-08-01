# Experimental Ramsey data

`ramsey_zenodo15797402.csv` is a deterministic transcription of all numeric
cells in `Ramsey.zip` from:

A. Kovalenko, M. T. Pham, K. Singh, O. Číp, and L. Slodicka,
“Experimental Data - Quantum non-Gaussian coherences of an oscillating atom,”
Zenodo (2025), DOI: 10.5281/zenodo.15797402.

The record and data are distributed under CC BY 4.0. The associated peer-reviewed
article is A. Kovalenko et al., “Quantum non-Gaussian coherences of an oscillating
atom,” *Physical Review Research* **7**, 033075 (2025),
DOI: 10.1103/5nxx-r97j.

Provenance:

- Official archive URL:
  `https://zenodo.org/api/records/15797402/files/Ramsey.zip/content`
- Zenodo MD5: `c78b673a91a93399b1b4f13699f31ce8`
- Downloaded archive SHA-256:
  `506b83ec63b0934dc813c630625402bd81070ed3529ff3c81af66c0208f0fec7`
- Derived CSV SHA-256:
  `197424d048c612e5ab8605129a45a46108153758af257ae5b1b63c6436291711`
- Derived rows: 40,787 observations from 177 experimental traces and eight
  target superposition labels.

The transformation is lossless for the numeric worksheet cells: it adds parsed
filename metadata (state, delay, phase step, and sequence repetitions) and writes
one CSV row per probability value. Regenerate and verify it with:

```bash
cptppinn-fetch-real-data --output ramsey_zenodo15797402.csv
```

or, without a network request:

```bash
cptppinn-fetch-real-data \
  --archive /path/to/Ramsey.zip \
  --output ramsey_zenodo15797402.csv
```
