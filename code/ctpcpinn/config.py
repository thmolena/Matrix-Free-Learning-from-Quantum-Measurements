"""Configuration for experiments.

Every training experiment (1-4) is repeated over the ``seeds`` list and reports
mean +/- 95% CI; the compiler benchmark (5) reports the median over
``n_repeats_timing`` timed repeats. ``FULL_CONFIG`` reproduces the numbers in the
manuscript; ``QUICK_CONFIG`` is a fast smoke test.
"""

# Fast smoke test (CI / development): 2 seeds, few epochs. NOT publication quality.
QUICK_CONFIG = {
    'seeds': [0, 1],
    'n_epochs': 200,
    'n_time_points': 60,
    'noise_std': 0.02,
    'verbose': True,
    'n_batch': 100,
    'n_repeats_timing': 15,
}

# Full configuration that reproduces the manuscript numbers.
FULL_CONFIG = {
    'seeds': [0, 1, 2, 3, 4],
    'n_epochs': 3000,
    'n_time_points': 100,
    'noise_std': 0.02,
    'verbose': True,
    'n_batch': 100,
    'n_repeats_timing': 50,
}

# Default == full: the manuscript is the source of truth and must stay reproducible.
DEFAULT_CONFIG = dict(FULL_CONFIG)
