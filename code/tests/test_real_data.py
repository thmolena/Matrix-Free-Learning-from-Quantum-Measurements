import numpy as np

from ctpcpinn.real_data import (
    BOHR_ANCHOR_HARMONIC,
    PRIMARY_STATE,
    RamseyBlockOperator,
    _dense_design,
    _ridge_blocks,
    _ridge_dense,
    bohr_sobolev_diagonal,
    evaluate_real_data,
    load_ramsey_csv,
    spectral_features,
)


def test_block_operator_adjoint_identity():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(23, 5))
    groups = rng.integers(0, 4, size=23)
    operator = RamseyBlockOperator(features, groups, n_groups=4)
    theta = rng.normal(size=20)
    values = rng.normal(size=23)
    assert np.allclose(
        np.dot(operator @ theta, values),
        np.dot(theta, operator.rmatvec(values)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_bundled_real_dataset_and_primary_recovery():
    data = load_ramsey_csv()
    assert data.n_traces == 177
    assert data.n_observations == 40787
    assert np.all((0.0 <= data.probability) & (data.probability <= 1.0))
    assert np.sum(data.state == PRIMARY_STATE) == 1997
    result = evaluate_real_data(data)
    primary = result["primary_summary"]
    assert primary["traces"] == 11
    assert primary["wins_vs_linear"] >= 9
    assert primary["matrix_free_mean_rmse"] < primary["linear_mean_rmse"]
    assert result["configuration"]["selected_bohr_weight"] == 0.3
    assert result["configuration"]["bohr_anchor_harmonic"] == 4


def test_bohr_loss_matrix_free_and_dense_solutions_agree():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(40, 7))
    groups = np.repeat(np.arange(4), 10)
    values = rng.normal(size=40)
    harmonics = (1, 2, BOHR_ANCHOR_HARMONIC)
    bohr_weight = 0.3
    ridge = 1.0e-2
    block_theta = _ridge_blocks(
        features,
        groups,
        values,
        4,
        ridge,
        bohr_weight=bohr_weight,
        harmonics=harmonics,
    ).ravel()
    design = _dense_design(features, groups, 4)
    dense_theta = _ridge_dense(
        design,
        values,
        ridge,
        local_sobolev_diagonal=bohr_sobolev_diagonal(harmonics),
        bohr_weight=bohr_weight,
    )
    assert np.allclose(block_theta, dense_theta, rtol=1e-11, atol=1e-11)


def test_feature_shape_and_finiteness():
    phase_index = np.arange(12)
    phase_step = np.full(12, 2.0)
    features = spectral_features(phase_index, phase_step)
    assert features.shape == (12, 21)
    assert np.all(np.isfinite(features))
