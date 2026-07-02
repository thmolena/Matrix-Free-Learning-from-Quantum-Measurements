# Website — QuantumPINNs Demo Site

This directory contains the static web interface for the QuantumPINNs project.

## Files

| File | Purpose |
|---|---|
| `index.html` | Local development API demo page |
| `style.css` | Shared stylesheet |
| `demo.js` | API client logic for the prediction interface |

## Running Locally

```bash
# From the repository root:
python -m http.server 8000
# Open http://localhost:8000/website/
```

## GitHub Pages

The root `index.html` (at the repository root) is the GitHub Pages entry point and contains the full self-contained demo with simulated results. It does not require a running API server.

The `website/index.html` is a local developer interface that connects to the Flask API (`src/server.py`) running at `http://localhost:5000`.
