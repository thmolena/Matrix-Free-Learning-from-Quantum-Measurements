"""Matrix-free learning on experimental trapped-ion Ramsey fringes.

The primary dataset is the CC BY 4.0 record

    A. Kovalenko et al., "Experimental Data - Quantum non-Gaussian
    coherences of an oscillating atom", Zenodo 15797402 (2025),
    https://doi.org/10.5281/zenodo.15797402.

The bundled CSV is a deterministic, lossless transcription of the numeric
cells in ``Ramsey.zip``.  :mod:`ctpcpinn.fetch_real_data` downloads the
official archive, verifies both the Zenodo MD5 and a SHA-256 digest, parses the
XLSX worksheets with the Python standard library, and regenerates the CSV.

The learning problem is deliberately small enough to inspect yet large enough
to expose a representation choice.  Each experimental fringe is represented
by a 10-harmonic spectral readout.  Stacking all traces naively produces a
large block-diagonal design matrix.  ``RamseyBlockOperator`` implements the
same map and its transpose from local features plus trace indices, never
materializing the global matrix.  Training eliminates the independent
21-parameter blocks directly.  The dense and matrix-free solutions minimize
the same Bohr-anchored Sobolev quadratic; reported speed and memory differences
concern their representations, not different statistical models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator


DATASET_DOI = "10.5281/zenodo.15797402"
DATASET_RECORD_URL = "https://zenodo.org/records/15797402"
RAMSEY_ARCHIVE_SHA256 = (
    "506b83ec63b0934dc813c630625402bd81070ed3529ff3c81af66c0208f0fec7"
)
DERIVED_CSV_NAME = "ramsey_zenodo15797402.csv"
HARMONICS = tuple(range(1, 11))
RIDGE = 1.0e-2
PRIMARY_STATE = "0+1"
BOHR_ANCHOR_HARMONIC = 4
BOHR_WEIGHT_GRID = (0.0, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class RamseyData:
    """Columnar representation of the derived experimental dataset."""

    trace_id: np.ndarray
    state: np.ndarray
    delay_ms: np.ndarray
    phase_step_deg: np.ndarray
    phase_index: np.ndarray
    probability: np.ndarray
    source_file: np.ndarray
    repetitions: np.ndarray

    @property
    def n_observations(self) -> int:
        return int(self.probability.size)

    @property
    def n_traces(self) -> int:
        return int(np.unique(self.trace_id).size)


def bundled_csv_path() -> Path:
    """Return the installed path to the checksum-traceable derived CSV."""

    return Path(__file__).resolve().parent / "data" / DERIVED_CSV_NAME


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_ramsey_csv(path: Path | str | None = None) -> RamseyData:
    """Load the derived Ramsey measurements without a spreadsheet dependency."""

    path = Path(path) if path is not None else bundled_csv_path()
    columns: dict[str, list] = {
        "trace_id": [],
        "state": [],
        "delay_ms": [],
        "phase_step_deg": [],
        "phase_index": [],
        "probability": [],
        "source_file": [],
        "repetitions": [],
    }
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            columns["trace_id"].append(int(row["trace_id"]))
            columns["state"].append(row["state"])
            columns["delay_ms"].append(float(row["delay_ms"]))
            columns["phase_step_deg"].append(float(row["phase_step_deg"]))
            columns["phase_index"].append(int(row["phase_index"]))
            columns["probability"].append(float(row["excitation_probability"]))
            columns["source_file"].append(row["source_file"])
            columns["repetitions"].append(int(row["repetitions"]))
    return RamseyData(
        trace_id=np.asarray(columns["trace_id"], dtype=np.int64),
        state=np.asarray(columns["state"], dtype="U8"),
        delay_ms=np.asarray(columns["delay_ms"], dtype=np.float64),
        phase_step_deg=np.asarray(columns["phase_step_deg"], dtype=np.float64),
        phase_index=np.asarray(columns["phase_index"], dtype=np.int64),
        probability=np.asarray(columns["probability"], dtype=np.float64),
        source_file=np.asarray(columns["source_file"], dtype="U160"),
        repetitions=np.asarray(columns["repetitions"], dtype=np.int64),
    )


def spectral_features(
    phase_index: np.ndarray,
    phase_step_deg: np.ndarray,
    harmonics: Sequence[int] = HARMONICS,
) -> np.ndarray:
    """Local Fourier features ``[1, sin(k phi), cos(k phi)]``."""

    phase = np.deg2rad(
        np.asarray(phase_index, dtype=np.float64)
        * np.asarray(phase_step_deg, dtype=np.float64)
    )
    columns = [np.ones_like(phase)]
    for k in harmonics:
        columns.extend((np.sin(k * phase), np.cos(k * phase)))
    return np.column_stack(columns)


def bohr_sobolev_diagonal(
    harmonics: Sequence[int] = HARMONICS,
    anchor_harmonic: int = BOHR_ANCHOR_HARMONIC,
) -> np.ndarray:
    """Return the diagonal of the Bohr-anchored Sobolev penalty.

    The intercept and the experimentally identified anchor harmonic are
    unpenalized.  Every sine/cosine sideband at harmonic ``k`` receives weight
    ``k**2``.  This is the coefficient-space form of the paper's distinct
    Bohr-anchored Sobolev spectral loss.
    """

    diagonal = [0.0]
    for harmonic in harmonics:
        weight = 0.0 if harmonic == anchor_harmonic else float(harmonic**2)
        diagonal.extend((weight, weight))
    return np.asarray(diagonal, dtype=np.float64)


def _compact_groups(trace_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(np.asarray(trace_ids, dtype=np.int64), return_inverse=True)
    return unique, inverse.astype(np.int64, copy=False)


class RamseyBlockOperator(LinearOperator):
    """Implicit block-diagonal spectral design.

    If ``F_i`` is the local feature row for observation ``i`` and ``g_i`` its
    trace, the operator applies

    ``(A theta)_i = F_i @ theta[g_i]``.

    ``A`` itself would contain one local feature row and zeros for every other
    trace.  Only ``F`` and ``g`` are stored.
    """

    def __init__(self, features: np.ndarray, groups: np.ndarray, n_groups: int):
        self.features = np.asarray(features, dtype=np.float64)
        self.groups = np.asarray(groups, dtype=np.int64)
        self.n_groups = int(n_groups)
        self.n_features = int(self.features.shape[1])
        if self.features.shape[0] != self.groups.size:
            raise ValueError("features and groups must have the same number of rows")
        super().__init__(
            dtype=np.dtype(np.float64),
            shape=(self.features.shape[0], self.n_groups * self.n_features),
        )

    def _matvec(self, theta: np.ndarray) -> np.ndarray:
        blocks = np.asarray(theta, dtype=np.float64).reshape(
            self.n_groups, self.n_features
        )
        return np.einsum("ij,ij->i", self.features, blocks[self.groups])

    def _rmatvec(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        out = np.empty((self.n_groups, self.n_features), dtype=np.float64)
        for j in range(self.n_features):
            out[:, j] = np.bincount(
                self.groups,
                weights=values * self.features[:, j],
                minlength=self.n_groups,
            )
        return out.ravel()

    @property
    def stored_bytes(self) -> int:
        return int(self.features.nbytes + self.groups.nbytes)

    @property
    def explicit_dense_bytes(self) -> int:
        return int(self.shape[0] * self.shape[1] * np.dtype(np.float64).itemsize)


def _subset(data: RamseyData, mask: np.ndarray) -> RamseyData:
    return RamseyData(**{
        field: np.asarray(getattr(data, field))[mask]
        for field in RamseyData.__dataclass_fields__
    })


def _fit_operator(
    data: RamseyData,
    mask: np.ndarray,
    harmonics: Sequence[int] = HARMONICS,
    ridge: float = RIDGE,
    bohr_weight: float = 0.0,
    anchor_harmonic: int = BOHR_ANCHOR_HARMONIC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RamseyBlockOperator, dict]:
    trace_values, all_groups = _compact_groups(data.trace_id)
    features = spectral_features(data.phase_index, data.phase_step_deg, harmonics)
    operator = RamseyBlockOperator(
        features[mask], all_groups[mask], n_groups=trace_values.size
    )
    theta = _ridge_blocks(
        operator.features,
        operator.groups,
        data.probability[mask],
        operator.n_groups,
        ridge,
        bohr_weight=bohr_weight,
        anchor_harmonic=anchor_harmonic,
        harmonics=harmonics,
    )
    prediction = np.einsum("ij,ij->i", features, theta[all_groups])
    local_regularizer = (
        ridge
        + bohr_weight
        * bohr_sobolev_diagonal(harmonics, anchor_harmonic)
    )
    normal_residual = operator.rmatvec(
        operator @ theta.ravel() - data.probability[mask]
    ) + (theta * local_regularizer[None, :]).ravel()
    diagnostics = {
        "block_solves": int(operator.n_groups),
        "block_size": int(operator.n_features),
        "residual_norm": float(
            np.linalg.norm(operator @ theta.ravel() - data.probability[mask])
        ),
        "normal_residual_norm": float(np.linalg.norm(normal_residual)),
        "bohr_weight": float(bohr_weight),
        "bohr_anchor_harmonic": int(anchor_harmonic),
    }
    return theta, prediction, trace_values, operator, diagnostics


def _rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(y)))))


def _linear_prediction(data: RamseyData, train: np.ndarray) -> np.ndarray:
    prediction = np.empty(data.n_observations, dtype=np.float64)
    for trace in np.unique(data.trace_id):
        loc = np.flatnonzero(data.trace_id == trace)
        tr = loc[train[loc]]
        phase = data.phase_index * data.phase_step_deg
        prediction[loc] = np.interp(phase[loc], phase[tr], data.probability[tr])
    return prediction


def _trace_metrics(
    data: RamseyData,
    prediction: np.ndarray,
    baseline: np.ndarray,
    test: np.ndarray,
) -> list[dict]:
    rows = []
    for trace in np.unique(data.trace_id):
        loc = (data.trace_id == trace) & test
        all_loc = data.trace_id == trace
        y = data.probability[loc]
        mean_prediction = np.full(y.size, data.probability[all_loc & ~test].mean())
        rows.append({
            "trace_id": int(trace),
            "state": str(data.state[all_loc][0]),
            "delay_ms": float(data.delay_ms[all_loc][0]),
            "source_file": str(data.source_file[all_loc][0]),
            "n_test": int(loc.sum()),
            "matrix_free_rmse": _rmse(y, prediction[loc]),
            "linear_rmse": _rmse(y, baseline[loc]),
            "mean_rmse": _rmse(y, mean_prediction),
        })
    return rows


def _bootstrap_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    seed: int = 20260728,
    n_bootstrap: int = 10000,
) -> tuple[float, float, float]:
    diff = np.asarray(values_a) - np.asarray(values_b)
    rng = np.random.default_rng(seed)
    means = np.mean(
        diff[rng.integers(0, diff.size, size=(n_bootstrap, diff.size))], axis=1
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def _dense_design(features: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    n, p = features.shape
    design = np.zeros((n, n_groups * p), dtype=np.float64)
    rows = np.arange(n)[:, None]
    cols = groups[:, None] * p + np.arange(p)[None, :]
    design[rows, cols] = features
    return design


def _ridge_dense(
    design: np.ndarray,
    y: np.ndarray,
    ridge: float,
    *,
    local_sobolev_diagonal: np.ndarray | None = None,
    bohr_weight: float = 0.0,
) -> np.ndarray:
    with np.errstate(all="ignore"):
        gram = design.T @ design
        regularizer = np.full(gram.shape[0], ridge, dtype=np.float64)
        if local_sobolev_diagonal is not None:
            local_sobolev_diagonal = np.asarray(
                local_sobolev_diagonal, dtype=np.float64
            )
            if gram.shape[0] % len(local_sobolev_diagonal) != 0:
                raise ValueError("global parameter count is not block aligned")
            regularizer += bohr_weight * np.tile(
                local_sobolev_diagonal,
                gram.shape[0] // len(local_sobolev_diagonal),
            )
        gram.flat[:: gram.shape[0] + 1] += regularizer
        return np.linalg.solve(gram, design.T @ y)


def _ridge_blocks(
    features: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    n_groups: int,
    ridge: float,
    *,
    bohr_weight: float = 0.0,
    anchor_harmonic: int = BOHR_ANCHOR_HARMONIC,
    harmonics: Sequence[int] = HARMONICS,
) -> np.ndarray:
    """Solve independent Bohr-regularized equations without a global design.

    The calculation is exactly the block elimination that an explicit
    block-diagonal problem would perform.  Its peak design storage is the local
    feature array rather than an observations-by-all-parameters matrix.
    """

    p = int(features.shape[1])
    theta = np.empty((n_groups, p), dtype=np.float64)
    sobolev = bohr_sobolev_diagonal(harmonics, anchor_harmonic)
    if len(sobolev) != p:
        raise ValueError("feature count does not match the harmonic penalty")
    regularizer = np.diag(ridge + bohr_weight * sobolev)
    for group in range(n_groups):
        loc = groups == group
        local = features[loc]
        # Some Accelerate builds emit spurious floating-point warnings during
        # small GEMMs even though the finite output agrees to machine precision.
        with np.errstate(all="ignore"):
            theta[group] = np.linalg.solve(
                local.T @ local + regularizer,
                local.T @ y[loc],
            )
    return theta


def select_bohr_weight(
    data: RamseyData,
    outer_train: np.ndarray,
    candidates: Sequence[float] = BOHR_WEIGHT_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """Select the Sobolev weight without consulting outer-test labels.

    Phase indices congruent to three modulo eight form the inner development
    slice; all other outer-training points form the inner fit.  Selection
    minimizes the mean per-trace development RMSE, then the chosen weight is
    refit on all outer-training points.
    """

    outer_train = np.asarray(outer_train, dtype=bool)
    development = outer_train & (data.phase_index % 8 == 3)
    fit = outer_train & ~development
    trace_values = np.unique(data.trace_id)
    if not development.any() or not fit.any():
        raise ValueError("nested Bohr-weight split is empty")
    rows: list[dict[str, float]] = []
    for candidate in candidates:
        _, prediction, _, _, _ = _fit_operator(
            data,
            fit,
            bohr_weight=float(candidate),
        )
        per_trace = []
        for trace in trace_values:
            loc = development & (data.trace_id == trace)
            if not loc.any():
                raise ValueError("every trace must contribute development points")
            per_trace.append(_rmse(data.probability[loc], prediction[loc]))
        rows.append({
            "bohr_weight": float(candidate),
            "development_mean_rmse": float(np.mean(per_trace)),
            "development_observations": int(development.sum()),
            "fit_observations": int(fit.sum()),
        })
    selected = min(rows, key=lambda row: row["development_mean_rmse"])
    return float(selected["bohr_weight"]), rows


def benchmark_representations(
    data: RamseyData,
    trace_counts: Sequence[int] = (4, 8, 16, 32),
    repeats: int = 3,
    ridge: float = RIDGE,
    bohr_weight: float = 0.30,
) -> list[dict]:
    """Benchmark equivalent explicit and implicit block designs on CPU."""

    rows: list[dict] = []
    all_traces = np.unique(data.trace_id)
    for count in trace_counts:
        chosen = all_traces[: min(int(count), all_traces.size)]
        keep = np.isin(data.trace_id, chosen)
        subset = _subset(data, keep)
        train = subset.phase_index % 4 != 0
        _, groups = _compact_groups(subset.trace_id)
        features = spectral_features(
            subset.phase_index[train], subset.phase_step_deg[train]
        )
        groups_train = groups[train]
        y = subset.probability[train]
        operator = RamseyBlockOperator(features, groups_train, len(chosen))

        mf_times = []
        mf_theta = None
        for _ in range(repeats):
            start = time.perf_counter()
            mf_theta_blocks = _ridge_blocks(
                features,
                groups_train,
                y,
                len(chosen),
                ridge,
                bohr_weight=bohr_weight,
            )
            mf_times.append(time.perf_counter() - start)
            mf_theta = mf_theta_blocks.ravel()

        start = time.perf_counter()
        design = _dense_design(features, groups_train, len(chosen))
        dense_build_time = time.perf_counter() - start
        dense_times = []
        dense_theta = None
        for _ in range(repeats):
            start = time.perf_counter()
            dense_theta = _ridge_dense(
                design,
                y,
                ridge,
                local_sobolev_diagonal=bohr_sobolev_diagonal(),
                bohr_weight=bohr_weight,
            )
            dense_times.append(time.perf_counter() - start)

        pred_mf = operator @ mf_theta
        with np.errstate(all="ignore"):
            pred_dense = design @ dense_theta
        rows.append({
            "traces": int(len(chosen)),
            "observations": int(y.size),
            "parameters": int(operator.shape[1]),
            "matrix_free_seconds": float(np.median(mf_times)),
            "dense_build_seconds": float(dense_build_time),
            "dense_solve_seconds": float(np.median(dense_times)),
            "speedup_vs_dense_solve": float(
                np.median(dense_times) / np.median(mf_times)
            ),
            "matrix_free_mb": float(operator.stored_bytes / 1.0e6),
            "dense_design_mb": float(design.nbytes / 1.0e6),
            "prediction_max_abs_difference": float(
                np.max(np.abs(pred_mf - pred_dense))
            ),
        })
    return rows


def _full_representation_audit(data: RamseyData) -> dict:
    train = data.phase_index % 4 != 0
    _, groups = _compact_groups(data.trace_id)
    features = spectral_features(data.phase_index[train], data.phase_step_deg[train])
    operator = RamseyBlockOperator(features, groups[train], data.n_traces)
    return {
        "observations": int(train.sum()),
        "traces": data.n_traces,
        "parameters": int(operator.shape[1]),
        "matrix_free_mb": float(operator.stored_bytes / 1.0e6),
        "explicit_dense_gb": float(operator.explicit_dense_bytes / 1.0e9),
        "memory_reduction": float(
            operator.explicit_dense_bytes / operator.stored_bytes
        ),
    }


def _coherence_rows(data: RamseyData, theta: np.ndarray, trace_values: np.ndarray) -> list[dict]:
    # Feature order: 1, sin(1 phi), cos(1 phi), ..., sin(10 phi), cos(10 phi).
    harmonic_four_sin = 1 + 2 * (4 - 1)
    harmonic_four_cos = harmonic_four_sin + 1
    rows = []
    for group, trace in enumerate(trace_values):
        loc = data.trace_id == trace
        if data.state[loc][0] != PRIMARY_STATE:
            continue
        rows.append({
            "delay_ms": float(data.delay_ms[loc][0]),
            "n_points": int(loc.sum()),
            "offset": float(theta[group, 0]),
            "coherence_amplitude": float(
                2.0 * np.hypot(
                    theta[group, harmonic_four_sin],
                    theta[group, harmonic_four_cos],
                )
            ),
        })
    return sorted(rows, key=lambda row: row["delay_ms"])


def evaluate_real_data(data: RamseyData) -> dict:
    """Run the primary held-out-point audit and the all-trace stress audit."""

    primary = _subset(data, data.state == PRIMARY_STATE)
    primary_train = primary.phase_index % 4 != 0
    selected_bohr_weight, bohr_development_grid = select_bohr_weight(
        primary, primary_train
    )
    primary_theta, primary_prediction, primary_traces, primary_operator, diag = (
        _fit_operator(
            primary,
            primary_train,
            bohr_weight=selected_bohr_weight,
        )
    )
    primary_linear = _linear_prediction(primary, primary_train)
    primary_rows = _trace_metrics(
        primary,
        primary_prediction,
        primary_linear,
        ~primary_train,
    )

    all_train = data.phase_index % 4 != 0
    all_theta, all_prediction, all_traces, all_operator, all_diag = _fit_operator(
        data,
        all_train,
        bohr_weight=selected_bohr_weight,
    )
    all_linear = _linear_prediction(data, all_train)
    all_rows = _trace_metrics(data, all_prediction, all_linear, ~all_train)

    primary_mf = np.asarray([row["matrix_free_rmse"] for row in primary_rows])
    primary_linear_rmse = np.asarray([row["linear_rmse"] for row in primary_rows])
    diff, lo, hi = _bootstrap_difference(primary_mf, primary_linear_rmse)

    return {
        "dataset": {
            "doi": DATASET_DOI,
            "record_url": DATASET_RECORD_URL,
            "ramsey_archive_sha256": RAMSEY_ARCHIVE_SHA256,
            "derived_csv_sha256": sha256_file(bundled_csv_path()),
            "observations": data.n_observations,
            "traces": data.n_traces,
            "states": int(np.unique(data.state).size),
        },
        "configuration": {
            "harmonics": list(HARMONICS),
            "ridge": RIDGE,
            "loss": "Bohr-anchored Sobolev spectral loss",
            "bohr_anchor_harmonic": BOHR_ANCHOR_HARMONIC,
            "bohr_weight_grid": list(BOHR_WEIGHT_GRID),
            "selected_bohr_weight": selected_bohr_weight,
            "inner_development_rule": (
                "outer-train points with phase_index mod 8 equals 3"
            ),
            "bohr_weight_development_grid": bohr_development_grid,
            "held_out_rule": "phase_index mod 4 equals 0",
            "bootstrap_seed": 20260728,
            "bootstrap_replicates": 10000,
        },
        "primary_summary": {
            "state": PRIMARY_STATE,
            "traces": len(primary_rows),
            "observations": primary.n_observations,
            "held_out_observations": int((~primary_train).sum()),
            "matrix_free_mean_rmse": float(primary_mf.mean()),
            "linear_mean_rmse": float(primary_linear_rmse.mean()),
            "mean_baseline_rmse": float(
                np.mean([row["mean_rmse"] for row in primary_rows])
            ),
            "relative_improvement_vs_linear_percent": float(
                100.0 * (1.0 - primary_mf.mean() / primary_linear_rmse.mean())
            ),
            "wins_vs_linear": int(np.sum(primary_mf < primary_linear_rmse)),
            "paired_rmse_difference": diff,
            "paired_bootstrap_95ci": [lo, hi],
            "solver": diag,
            "matrix_free_design_mb": float(primary_operator.stored_bytes / 1.0e6),
            "explicit_dense_design_mb": float(
                primary_operator.explicit_dense_bytes / 1.0e6
            ),
        },
        "all_trace_summary": {
            "traces": len(all_rows),
            "matrix_free_mean_rmse": float(
                np.mean([row["matrix_free_rmse"] for row in all_rows])
            ),
            "linear_mean_rmse": float(
                np.mean([row["linear_rmse"] for row in all_rows])
            ),
            "mean_baseline_rmse": float(
                np.mean([row["mean_rmse"] for row in all_rows])
            ),
            "wins_vs_linear": int(
                np.sum(
                    np.asarray([row["matrix_free_rmse"] for row in all_rows])
                    < np.asarray([row["linear_rmse"] for row in all_rows])
                )
            ),
            "solver": all_diag,
            "matrix_free_design_mb": float(all_operator.stored_bytes / 1.0e6),
            "explicit_dense_design_gb": float(
                all_operator.explicit_dense_bytes / 1.0e9
            ),
        },
        "primary_trace_metrics": primary_rows,
        "all_trace_metrics": all_rows,
        "coherence": _coherence_rows(data, all_theta, all_traces),
        "representation_audit": _full_representation_audit(data),
    }


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tables(results: dict, benchmark: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = output_dir / "real_ramsey_accuracy.tex"
    with accuracy.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[t]\n\\centering\n")
        fh.write(
            "\\caption{Experimental held-out probability recovery for the "
            "$\\lvert0\\rangle+\\lvert1\\rangle$ Ramsey traces. Every fourth "
            "phase point is withheld. BA-Sobolev is the Bohr-anchored "
            "Sobolev spectral loss. Lower RMSE is better; improvement is "
            "relative to piecewise-linear interpolation.}\\label{tab:real_accuracy}\n"
        )
        fh.write("\\begin{ruledtabular}\\begin{tabular}{lrrrrr}\n")
        fh.write(
            "Delay (ms) & Test points & BA-Sobolev & Linear & Mean & "
            "Improvement (\\%) \\\\\n\\colrule\n"
        )
        for row in sorted(
            results["primary_trace_metrics"], key=lambda item: item["delay_ms"]
        ):
            improvement = 100.0 * (
                1.0 - row["matrix_free_rmse"] / row["linear_rmse"]
            )
            fh.write(
                f"{row['delay_ms']:.1f} & {row['n_test']} & "
                f"{row['matrix_free_rmse']:.4f} & {row['linear_rmse']:.4f} & "
                f"{row['mean_rmse']:.4f} & {improvement:+.1f} \\\\\n"
            )
        summary = results["primary_summary"]
        fh.write("\\colrule\n")
        fh.write(
            f"Mean & {summary['held_out_observations']} & "
            f"{summary['matrix_free_mean_rmse']:.4f} & "
            f"{summary['linear_mean_rmse']:.4f} & "
            f"{summary['mean_baseline_rmse']:.4f} & "
            f"{summary['relative_improvement_vs_linear_percent']:+.1f} \\\\\n"
        )
        fh.write("\\end{tabular}\\end{ruledtabular}\n\\end{table*}\n")

    coherence = output_dir / "real_ramsey_coherence.tex"
    with coherence.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[t]\n\\centering\n")
        fh.write(
            "\\caption{Coherence amplitude recovered from the fourth harmonic "
            "of each complete experimental Ramsey fringe.}\\label{tab:coherence}\n"
        )
        fh.write("\\begin{ruledtabular}\\begin{tabular}{rrr}\n")
        fh.write("Delay (ms) & Points & $C_{0,1}$ \\\\\n\\colrule\n")
        for row in results["coherence"]:
            fh.write(
                f"{row['delay_ms']:.1f} & {row['n_points']} & "
                f"{row['coherence_amplitude']:.4f} \\\\\n"
            )
        fh.write("\\end{tabular}\\end{ruledtabular}\n\\end{table}\n")

    scaling = output_dir / "real_ramsey_benchmark.tex"
    with scaling.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table*}[t]\n\\centering\n")
        fh.write(
            "\\caption{Equivalent Bohr-anchored Sobolev training with an explicit global "
            "block design and the matrix-free operator. Times are medians of "
            "three single-process CPU runs on the reproduction machine; "
            "memory is the stored design representation.}\\label{tab:real_benchmark}\n"
        )
        fh.write("\\begin{ruledtabular}\\begin{tabular}{rrrrrrrr}\n")
        fh.write(
            "Traces & Train points & Parameters & MF (s) & Dense solve (s) & "
            "Speedup & MF (MB) & Dense (MB) \\\\\n\\colrule\n"
        )
        for row in benchmark:
            fh.write(
                f"{row['traces']} & {row['observations']} & {row['parameters']} & "
                f"{row['matrix_free_seconds']:.4f} & "
                f"{row['dense_solve_seconds']:.4f} & "
                f"{row['speedup_vs_dense_solve']:.1f}$\\times$ & "
                f"{row['matrix_free_mb']:.2f} & {row['dense_design_mb']:.2f} \\\\\n"
            )
        fh.write("\\end{tabular}\\end{ruledtabular}\n\\end{table*}\n")

    audit = output_dir / "real_data_audit.tex"
    a = results["representation_audit"]
    d = results["dataset"]
    p = results["primary_summary"]
    with audit.open("w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[t]\n\\centering\n")
        fh.write(
            "\\caption{Real-data and computational audit. The full explicit "
            "design is an analytic counterfactual and was not allocated.}"
            "\\label{tab:real_audit}\n"
        )
        fh.write("\\begin{ruledtabular}\\begin{tabular}{lr}\n")
        fh.write("Quantity & Value \\\\\n\\colrule\n")
        fh.write(f"Experimental traces & {d['traces']} \\\\\n")
        fh.write(f"Experimental observations & {d['observations']} \\\\\n")
        fh.write(f"Primary held-out observations & {p['held_out_observations']} \\\\\n")
        fh.write(f"Full model parameters & {a['parameters']} \\\\\n")
        fh.write(f"Matrix-free stored design (MB) & {a['matrix_free_mb']:.2f} \\\\\n")
        fh.write(f"Explicit global design (GB) & {a['explicit_dense_gb']:.2f} \\\\\n")
        fh.write(f"Memory reduction & {a['memory_reduction']:.1f}$\\times$ \\\\\n")
        fh.write(
            f"Primary wins vs. interpolation & {p['wins_vs_linear']}/{p['traces']} \\\\\n"
        )
        fh.write("\\end{tabular}\\end{ruledtabular}\n\\end{table}\n")


def _plot_results(
    data: RamseyData,
    results: dict,
    benchmark: list[dict],
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })

    primary = _subset(data, data.state == PRIMARY_STATE)
    train = primary.phase_index % 4 != 0
    _, prediction, _, _, _ = _fit_operator(primary, train)
    # A representative intermediate-delay trace.
    trace_candidates = np.unique(primary.trace_id)
    trace = trace_candidates[
        np.argmin([
            abs(primary.delay_ms[primary.trace_id == item][0] - 20.0)
            for item in trace_candidates
        ])
    ]
    loc = primary.trace_id == trace
    phase = primary.phase_index[loc] * primary.phase_step_deg[loc]
    order = np.argsort(phase)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.plot(
        phase[order],
        primary.probability[loc][order],
        color="#3b4cc0",
        lw=1.2,
        label="experiment",
    )
    ax.plot(
        phase[order],
        prediction[loc][order],
        color="#b40426",
        lw=1.5,
        label="matrix-free recovery",
    )
    ax.set(xlabel="Ramsey phase step (degrees)", ylabel="Excitation probability")
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "real_ramsey_recovery.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    delays = np.asarray([row["delay_ms"] for row in results["coherence"]])
    amplitudes = np.asarray(
        [row["coherence_amplitude"] for row in results["coherence"]]
    )
    ax.plot(delays, amplitudes, "-o", color="#008b8b", lw=1.5, ms=3.5)
    ax.set(
        xscale="log",
        xlabel="Ramsey delay (ms)",
        ylabel=r"Recovered coherence $C_{0,1}$",
        ylim=(0.0, 1.05),
    )
    fig.savefig(figures_dir / "real_ramsey_coherence.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    summary = results["primary_summary"]
    all_summary = results["all_trace_summary"]
    labels = ["BA-Sobolev", "linear", "mean"]
    primary_values = [
        summary["matrix_free_mean_rmse"],
        summary["linear_mean_rmse"],
        summary["mean_baseline_rmse"],
    ]
    all_values = [
        all_summary["matrix_free_mean_rmse"],
        all_summary["linear_mean_rmse"],
        all_summary["mean_baseline_rmse"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, primary_values, width, label=r"$|0\rangle+|1\rangle$")
    ax.bar(x + width / 2, all_values, width, label="all states")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Held-out RMSE")
    ax.legend(frameon=False)
    fig.savefig(figures_dir / "real_ramsey_error_bars.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))
    counts = np.asarray([row["traces"] for row in benchmark])
    mf_time = np.asarray([row["matrix_free_seconds"] for row in benchmark])
    dense_time = np.asarray([row["dense_solve_seconds"] for row in benchmark])
    axes[0].plot(counts, mf_time, "-o", label="matrix-free", lw=1.4, ms=3)
    axes[0].plot(counts, dense_time, "-o", label="explicit dense", lw=1.4, ms=3)
    axes[0].set(
        xlabel="Number of fitted traces",
        ylabel="Median solve time (s)",
        yscale="log",
    )
    axes[0].legend(frameon=False)
    mf_memory = np.asarray([row["matrix_free_mb"] for row in benchmark])
    dense_memory = np.asarray([row["dense_design_mb"] for row in benchmark])
    axes[1].plot(counts, mf_memory, "-o", label="matrix-free", lw=1.4, ms=3)
    axes[1].plot(counts, dense_memory, "-o", label="explicit dense", lw=1.4, ms=3)
    axes[1].set(
        xlabel="Number of fitted traces",
        ylabel="Stored design (MB)",
        yscale="log",
    )
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "real_ramsey_benchmark.pdf")
    plt.close(fig)


def reproduce(
    output_dir: Path | str,
    csv_path: Path | str | None = None,
    make_plots: bool = True,
) -> dict:
    """Regenerate the real-data results, tables, figures, and evidence JSON."""

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_ramsey_csv(csv_path)
    results = evaluate_real_data(data)
    benchmark = benchmark_representations(
        data,
        bohr_weight=results["configuration"]["selected_bohr_weight"],
    )
    results["benchmark"] = benchmark
    results["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    _write_tables(results, benchmark, tables_dir)
    if make_plots:
        _plot_results(data, results, benchmark, figures_dir)
    with (output_dir / "real_data_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the experimental Ramsey matrix-free study."
    )
    parser.add_argument("--output-dir", default="ctpcpinn_real_results")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    results = reproduce(
        args.output_dir,
        csv_path=args.csv,
        make_plots=not args.no_plots,
    )
    primary = results["primary_summary"]
    audit = results["representation_audit"]
    print(
        f"{results['dataset']['traces']} experimental traces, "
        f"{results['dataset']['observations']} observations"
    )
    print(
        f"primary held-out RMSE {primary['matrix_free_mean_rmse']:.6f} "
        f"vs linear {primary['linear_mean_rmse']:.6f}; "
        f"{primary['wins_vs_linear']}/{primary['traces']} trace wins"
    )
    print(
        f"full stored design {audit['matrix_free_mb']:.3f} MB vs "
        f"{audit['explicit_dense_gb']:.3f} GB explicit"
    )


if __name__ == "__main__":
    main()
