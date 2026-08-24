import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

from mojolime import kernels


@pytest.fixture(scope="module")
def dense():
    return np.random.RandomState(4).normal(size=(257, 13))


def test_exponential_kernel_matches_lime_formula():
    distances = np.linspace(0, 30, 1003)
    expected = np.sqrt(np.exp(-(distances ** 2) / 25 ** 2))
    assert np.allclose(kernels.exponential_kernel(distances, 25), expected, atol=3e-13)


def test_exponential_kernel_parallel_threshold_and_tail():
    distances = np.linspace(0, 30, 1_000_003)
    expected = np.sqrt(np.exp(-(distances ** 2) / 25 ** 2))
    assert np.allclose(kernels.exponential_kernel(distances, 25), expected, atol=3e-13)


def test_affine_samples(dense):
    center = np.linspace(-2, 2, dense.shape[1])
    scale = np.linspace(0.5, 3, dense.shape[1])
    assert np.allclose(
        kernels.affine_samples(dense, center, scale),
        dense * scale + center,
    )


def test_standardize(dense):
    mean = dense.mean(axis=0)
    scale = dense.std(axis=0)
    assert np.allclose(
        kernels.standardize(dense, mean, scale),
        (dense - mean) / scale,
    )


def test_fused_affine_standardize_simd_tail(dense):
    center = np.linspace(-2, 2, dense.shape[1])
    sample_scale = np.linspace(0.5, 3, dense.shape[1])
    mean = np.linspace(-1, 1, dense.shape[1])
    standard_scale = np.linspace(1, 2, dense.shape[1])
    original = dense[11].copy()
    expected_inverse = dense * sample_scale + center
    expected_inverse[0] = original
    actual, inverse = kernels.affine_standardize_samples(
        dense.copy(), original, center, sample_scale, mean, standard_scale
    )
    assert np.allclose(inverse, expected_inverse)
    assert np.allclose(actual, (expected_inverse - mean) / standard_scale)


def test_euclidean_row_distances(dense):
    expected = pairwise_distances(dense, dense[:1], metric="euclidean").ravel()
    assert np.allclose(kernels.row_distances(dense), expected)


def test_euclidean_row_distances_parallel_threshold_and_tail():
    values = np.random.RandomState(17).normal(size=(31_001, 65))
    expected = np.sqrt(np.square(values - values[0]).sum(axis=1))
    assert np.allclose(kernels.row_distances(values), expected)


def test_cosine_row_distances_including_zero_row():
    values = np.random.RandomState(7).randint(0, 2, size=(200, 31)).astype(float)
    values[0] = 1
    values[10] = 0
    expected = pairwise_distances(values, values[:1], metric="cosine").ravel()
    assert np.allclose(kernels.row_distances(values, "cosine"), expected)


def test_weighted_ridge_matches_sklearn(dense):
    rng = np.random.RandomState(8)
    labels = dense @ rng.normal(size=dense.shape[1]) + rng.normal(size=len(dense))
    weights = rng.uniform(0.01, 1, size=len(dense))
    expected = Ridge(alpha=1).fit(dense, labels, sample_weight=weights)
    intercept, coefficients, score, local_prediction = kernels.weighted_ridge(
        dense, labels, weights
    )
    assert intercept == pytest.approx(expected.intercept_, abs=1e-11)
    assert np.allclose(coefficients, expected.coef_, atol=1e-11)
    assert score == pytest.approx(
        expected.score(dense, labels, sample_weight=weights), abs=1e-12
    )
    assert np.allclose(local_prediction, expected.predict(dense[:1]), atol=1e-11)


@pytest.mark.parametrize("shape", [(63, 7), (10_001, 79)])
def test_weighted_ridge_serial_and_parallel_simd_tails(shape):
    rng = np.random.RandomState(sum(shape))
    values = rng.normal(size=shape)
    labels = values @ rng.normal(size=shape[1]) + rng.normal(size=shape[0])
    weights = rng.uniform(0.01, 1, size=shape[0])
    expected = Ridge(alpha=1).fit(values, labels, sample_weight=weights)
    actual = kernels.weighted_ridge(values, labels, weights)
    assert actual[0] == pytest.approx(expected.intercept_, abs=1e-11)
    assert np.allclose(actual[1], expected.coef_, atol=1e-11)
    assert actual[2] == pytest.approx(
        expected.score(values, labels, sample_weight=weights), abs=1e-12
    )
    assert np.allclose(actual[3], expected.predict(values[:1]), atol=1e-11)


def test_image_neighborhood_matches_reference():
    rng = np.random.RandomState(9)
    image = rng.uniform(size=(15, 12, 3))
    fudged = image * 0.25
    segments = np.arange(15 * 12).reshape(15, 12) % 7
    data = rng.randint(0, 2, size=(11, 7))
    actual = kernels.image_neighborhood(image, fudged, segments, data)
    expected = np.empty_like(actual)
    for row in range(len(data)):
        expected[row] = np.where(
            data[row, segments, None].astype(bool), image, fudged
        )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda: kernels.affine_samples_inplace(np.ones((2, 3)), np.ones(2), np.ones(3)),
        lambda: kernels.standardize_inplace(np.ones((2, 3)), np.ones(3), np.ones(2)),
        lambda: kernels.affine_standardize_samples(
            np.ones((2, 3)), np.ones(2), np.ones(3), np.ones(3), np.ones(3), np.ones(3)
        ),
        lambda: kernels.row_distances(np.empty((0, 3))),
        lambda: kernels.weighted_ridge(np.empty((0, 3)), np.empty(0), np.empty(0)),
    ],
)
def test_ffi_wrappers_reject_unsafe_shapes(call):
    with pytest.raises(ValueError):
        call()


def test_empty_kernel_does_not_cross_ffi():
    assert kernels.exponential_kernel(np.empty((0, 2)), 1).shape == (0, 2)


def test_complex_input_is_not_silently_narrowed():
    with pytest.raises(TypeError):
        kernels.exponential_kernel(np.array([1 + 2j]), 1)


@pytest.mark.parametrize("width", [0, -1, np.inf, np.nan])
def test_kernel_rejects_invalid_width(width):
    with pytest.raises(ValueError):
        kernels.exponential_kernel([1.0], width)


@pytest.mark.parametrize(
    "segments,data,error",
    [
        (np.array([[0.5]]), np.ones((1, 1)), TypeError),
        (np.array([[-1]]), np.ones((1, 1)), ValueError),
        (np.array([[1]]), np.ones((1, 1)), ValueError),
        (np.array([[0]]), np.array([[0.5]]), ValueError),
    ],
)
def test_image_neighborhood_validates_indices_and_binary_data(segments, data, error):
    image = np.ones((1, 1, 1))
    with pytest.raises(error):
        kernels.image_neighborhood(image, image, segments, data)


@pytest.mark.parametrize(
    "labels,weights,alpha",
    [
        (np.array([np.nan, 1.0]), np.ones(2), 1.0),
        (np.ones(2), np.array([1.0, -1.0]), 1.0),
        (np.ones(2), np.ones(2), -1.0),
    ],
)
def test_weighted_ridge_rejects_invalid_numeric_inputs(labels, weights, alpha):
    with pytest.raises(ValueError):
        kernels.weighted_ridge(np.ones((2, 1)), labels, weights, alpha)
