"""Aggregation helpers for multi-seed experiments.

All experiments are repeated over an independent set of random seeds. These
helpers turn a list of per-seed scalar metrics into a mean and a 95% confidence
interval (Student-t, so the interval is honest for the small seed counts used
here) and format the result for LaTeX tables. No value reported in the
manuscript is hand-entered: the experiment scripts call these helpers and write
the resulting strings directly into the table files.
"""

import numpy as np

try:
    from scipy.stats import t as _student_t
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dependency in practice
    _HAVE_SCIPY = False


def aggregate(values, confidence: float = 0.95) -> dict:
    """Summarise a list of per-seed scalar metrics.

    Args:
        values: 1D iterable of per-seed metric values.
        confidence: confidence level for the (two-sided) interval.

    Returns:
        dict with n, mean, std (sample, ddof=1), sem, the t critical value,
        the CI half-width ``ci``, and interval endpoints ``lo``/``hi``.
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    mean = float(np.mean(v)) if n > 0 else float("nan")
    if n <= 1:
        return {"n": n, "mean": mean, "std": 0.0, "sem": 0.0,
                "tcrit": 0.0, "ci": 0.0, "lo": mean, "hi": mean}
    std = float(np.std(v, ddof=1))
    sem = std / np.sqrt(n)
    if _HAVE_SCIPY:
        tcrit = float(_student_t.ppf(0.5 + confidence / 2.0, df=n - 1))
    else:  # normal-approximation fallback
        tcrit = 1.959963984540054
    ci = tcrit * sem
    return {"n": n, "mean": mean, "std": std, "sem": sem,
            "tcrit": tcrit, "ci": ci, "lo": mean - ci, "hi": mean + ci}


def fmt_pm(values, digits: int = 4, confidence: float = 0.95) -> str:
    """Format a metric as ``mean $\\pm$ ci`` (95% CI half-width) for LaTeX."""
    a = aggregate(values, confidence=confidence)
    if a["n"] <= 1:
        return f"{a['mean']:.{digits}f}"
    return f"{a['mean']:.{digits}f} $\\pm$ {a['ci']:.{digits}f}"


def fmt_mean(values, digits: int = 4) -> str:
    """Format just the mean of a metric for LaTeX."""
    a = aggregate(values)
    return f"{a['mean']:.{digits}f}"
