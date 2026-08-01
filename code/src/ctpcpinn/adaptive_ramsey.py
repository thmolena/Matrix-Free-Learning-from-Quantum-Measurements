"""Safety-gated spectral reconstruction of experimental Ramsey fringes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import real_data as rd


HARMONIC_COUNTS = (4, 6, 8, 10)
WEIGHTS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SAFETY_MARGIN = 0.05


def _evaluate_trace(data: rd.RamseyData, margin: float = SAFETY_MARGIN) -> dict:
    outer_train = data.phase_index % 4 != 0
    outer_test = ~outer_train
    development = outer_train & (data.phase_index % 8 == 3)
    inner_fit = outer_train & ~development

    inner_linear = rd._linear_prediction(data, inner_fit)
    linear_development = rd._rmse(data.probability[development], inner_linear[development])
    candidates = []
    for count in HARMONIC_COUNTS:
        harmonics = tuple(range(1, count + 1))
        for weight in WEIGHTS:
            _, prediction, _, _, _ = rd._fit_operator(
                data, inner_fit, harmonics=harmonics,
                bohr_weight=weight, anchor_harmonic=4,
            )
            candidates.append({
                "harmonics": count, "weight": weight,
                "development_rmse": rd._rmse(data.probability[development], prediction[development]),
            })
    best = min(candidates, key=lambda row: (row["development_rmse"], row["harmonics"], row["weight"]))
    use_spectral = best["development_rmse"] < (1.0 - margin) * linear_development

    linear_prediction = rd._linear_prediction(data, outer_train)
    if use_spectral:
        _, prediction, _, _, _ = rd._fit_operator(
            data, outer_train,
            harmonics=tuple(range(1, best["harmonics"] + 1)),
            bohr_weight=best["weight"], anchor_harmonic=4,
        )
    else:
        prediction = linear_prediction
    loc = outer_test
    return {
        "trace_id": int(data.trace_id[0]), "state": str(data.state[0]),
        "delay_ms": float(data.delay_ms[0]), "source_file": str(data.source_file[0]),
        "test_observations": int(loc.sum()), "selected": "spectral" if use_spectral else "linear",
        "selected_harmonics": int(best["harmonics"]) if use_spectral else 0,
        "selected_weight": float(best["weight"]) if use_spectral else 0.0,
        "linear_development_rmse": linear_development,
        "best_spectral_development_rmse": float(best["development_rmse"]),
        "adaptive_test_rmse": rd._rmse(data.probability[loc], prediction[loc]),
        "linear_test_rmse": rd._rmse(data.probability[loc], linear_prediction[loc]),
    }


def run_study() -> dict:
    data = rd.load_ramsey_csv()
    start = perf_counter()
    rows = [rd._subset(data, data.trace_id == trace) for trace in np.unique(data.trace_id)]
    records = [_evaluate_trace(trace) for trace in rows]
    elapsed = perf_counter() - start

    def summarize(selected: list[dict]) -> dict:
        adaptive = np.asarray([row["adaptive_test_rmse"] for row in selected])
        linear = np.asarray([row["linear_test_rmse"] for row in selected])
        return {
            "traces": len(selected), "adaptive_mean_rmse": float(adaptive.mean()),
            "linear_mean_rmse": float(linear.mean()),
            "adaptive_pooled_rmse": float(np.sqrt(np.mean(adaptive**2))),
            "linear_pooled_rmse": float(np.sqrt(np.mean(linear**2))),
            "relative_mean_improvement_percent": float(100.0 * (1.0 - adaptive.mean() / linear.mean())),
            "paired_wins": int(np.sum(adaptive < linear)),
            "spectral_selected": int(sum(row["selected"] == "spectral" for row in selected)),
        }
    states = sorted(set(row["state"] for row in records))
    source_results = json.loads((Path(__file__).resolve().parents[2] / "manuscript_assets/real_data_results.json").read_text())
    return {
        "schema_version": 1,
        "dataset": {"doi": rd.DATASET_DOI, "license": "CC-BY-4.0", "observations": data.n_observations, "traces": data.n_traces, "states": len(states), "derived_csv_sha256": rd.sha256_file(rd.bundled_csv_path()), "archive_sha256": rd.RAMSEY_ARCHIVE_SHA256},
        "protocol": {"outer_test": "phase_index mod 4 equals 0", "inner_development": "outer-training phase_index mod 8 equals 3", "harmonic_counts": list(HARMONIC_COUNTS), "sobolev_weights": list(WEIGHTS), "spectral_acceptance_margin": SAFETY_MARGIN, "candidates_per_trace": len(HARMONIC_COUNTS) * len(WEIGHTS) + 1},
        "overall": summarize(records),
        "by_state": {state: summarize([row for row in records if row["state"] == state]) for state in states},
        "records": records,
        "runtime": {"full_nested_study_seconds": elapsed, "benchmark": source_results["benchmark"], "representation_audit": source_results["representation_audit"]},
    }


def semantic_view(result: dict) -> dict:
    value = dict(result); value["runtime"] = {"benchmark": result["runtime"]["benchmark"], "representation_audit": result["runtime"]["representation_audit"]}; return value


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("code/results/adaptive_ramsey.json")); parser.add_argument("--verify", action="store_true"); args = parser.parse_args()
    result = run_study()
    if args.verify and args.output.exists() and semantic_view(json.loads(args.output.read_text())) != semantic_view(result):
        raise SystemExit("semantic results differ from locked artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    payload = json.dumps(semantic_view(result), sort_keys=True).encode(); print("semantic_sha256", hashlib.sha256(payload).hexdigest()); print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__": main()
