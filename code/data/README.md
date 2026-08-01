# Data provenance

**Dataset:** 40,787 measured probabilities from 177 trapped-ion Ramsey traces in Zenodo record 15797402.

**Original source:** https://zenodo.org/records/15797402

**Split:** Every fourth phase point is withheld on the clean eleven-trace subset; heterogeneous traces form the broader scope audit.

**Integrity:** The downloader verifies the original Ramsey archive before deriving the CSV.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
