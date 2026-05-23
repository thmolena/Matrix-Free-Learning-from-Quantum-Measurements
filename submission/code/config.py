"""Configuration for experiments."""

# Default configuration for quick runs (CI/smoke test)
QUICK_CONFIG = {
    'seed': 42,
    'n_epochs': 300,
    'n_time_points': 40,
    'noise_std': 0.02,
    'verbose': True,
    'n_batch': 50,
    'n_repeats': 3,
}

# Full configuration for publication-quality results
FULL_CONFIG = {
    'seed': 42,
    'n_epochs': 3000,
    'n_time_points': 100,
    'noise_std': 0.02,
    'verbose': True,
    'n_batch': 200,
    'n_repeats': 10,
}

# Default (moderate) configuration
DEFAULT_CONFIG = {
    'seed': 42,
    'n_epochs': 1500,
    'n_time_points': 80,
    'noise_std': 0.02,
    'verbose': True,
    'n_batch': 100,
    'n_repeats': 5,
}
