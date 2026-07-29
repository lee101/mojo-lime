import inspect

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from lime.lime_base import LimeBase as UpstreamBase
from lime.lime_image import ImageExplanation as UpstreamImageExplanation
from lime.lime_image import LimeImageExplainer as UpstreamImage
from lime.lime_tabular import LimeTabularExplainer as UpstreamTabular
from lime.lime_text import IndexedCharacters as UpstreamCharacters
from lime.lime_text import IndexedString as UpstreamString
from lime.lime_text import LimeTextExplainer as UpstreamText

from mojolime.lime_base import LimeBase
from mojolime.lime_image import ImageExplanation, LimeImageExplainer
from mojolime.lime_tabular import LimeTabularExplainer
from mojolime.lime_text import IndexedCharacters, IndexedString, LimeTextExplainer


def assert_explanation_pairs(actual, expected, atol=2e-11):
    assert [item[0] for item in actual] == [item[0] for item in expected]
    assert np.allclose(
        [item[1] for item in actual],
        [item[1] for item in expected],
        atol=atol,
        rtol=1e-7,
    )


@pytest.fixture(scope="module")
def tabular_data():
    rng = np.random.RandomState(10)
    values = rng.normal(size=(600, 8))
    values[:, 7] = rng.randint(0, 4, size=len(values))
    return values


@pytest.mark.parametrize("discretize", [False, True])
def test_tabular_neighborhood_parity(tabular_data, discretize):
    kwargs = dict(
        categorical_features=[7],
        discretize_continuous=discretize,
        random_state=19,
    )
    upstream = UpstreamTabular(tabular_data, **kwargs)
    mojo = LimeTabularExplainer(tabular_data, **kwargs)
    expected = upstream._LimeTabularExplainer__data_inverse(tabular_data[3], 400)
    actual = mojo._LimeTabularExplainer__data_inverse(tabular_data[3], 400)
    assert np.allclose(actual[0], expected[0], atol=1e-14)
    assert np.allclose(actual[1], expected[1], atol=1e-14)


def test_tabular_sample_around_instance_parity(tabular_data):
    kwargs = dict(
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=23,
    )
    upstream = UpstreamTabular(tabular_data, **kwargs)
    mojo = LimeTabularExplainer(tabular_data, **kwargs)
    expected = upstream._LimeTabularExplainer__data_inverse(tabular_data[8], 250)
    actual = mojo._LimeTabularExplainer__data_inverse(tabular_data[8], 250)
    assert np.allclose(actual[0], expected[0], atol=1e-14)
    assert np.allclose(actual[1], expected[1], atol=1e-14)


def test_tabular_classification_explanation_parity(tabular_data):
    labels = (
        tabular_data[:, 0] - 1.5 * tabular_data[:, 2] + tabular_data[:, 5] > 0
    ).astype(int)
    classifier = LogisticRegression().fit(tabular_data, labels)
    kwargs = dict(
        feature_names=[f"feature {index}" for index in range(tabular_data.shape[1])],
        categorical_features=[7],
        random_state=31,
        feature_selection="none",
    )
    upstream = UpstreamTabular(tabular_data, **kwargs)
    mojo = LimeTabularExplainer(tabular_data, **kwargs)
    call = dict(labels=(1,), num_features=8, num_samples=1200)
    expected = upstream.explain_instance(
        tabular_data[0], classifier.predict_proba, **call
    )
    actual = mojo.explain_instance(
        tabular_data[0], classifier.predict_proba, **call
    )
    assert actual.predict_proba == pytest.approx(expected.predict_proba)
    assert actual.intercept[1] == pytest.approx(expected.intercept[1], abs=2e-11)
    assert_explanation_pairs(actual.local_exp[1], expected.local_exp[1])
    assert actual.score == pytest.approx(expected.score, abs=2e-12)
    assert actual.local_pred == pytest.approx(expected.local_pred, abs=2e-11)
    assert_explanation_pairs(actual.as_list(1), expected.as_list(1))


def test_tabular_regression_explanation_parity(tabular_data):
    target = tabular_data[:, :7] @ np.arange(7) + 0.2
    regressor = LinearRegression().fit(tabular_data, target)
    kwargs = dict(
        mode="regression",
        discretize_continuous=False,
        random_state=37,
        feature_selection="none",
    )
    upstream = UpstreamTabular(tabular_data, **kwargs)
    mojo = LimeTabularExplainer(tabular_data, **kwargs)
    expected = upstream.explain_instance(
        tabular_data[4], regressor.predict, num_features=8, num_samples=1000
    )
    actual = mojo.explain_instance(
        tabular_data[4], regressor.predict, num_features=8, num_samples=1000
    )
    assert actual.predicted_value == pytest.approx(expected.predicted_value)
    assert actual.intercept[1] == pytest.approx(expected.intercept[1], abs=2e-11)
    assert_explanation_pairs(actual.local_exp[1], expected.local_exp[1])
    assert actual.score == pytest.approx(expected.score, abs=2e-12)


@pytest.mark.parametrize("method,num_features", [
    ("forward_selection", 4),
    ("highest_weights", 7),
    ("lasso_path", 5),
])
def test_feature_selection_parity(tabular_data, method, num_features):
    rng = np.random.RandomState(41)
    labels = tabular_data @ rng.normal(size=tabular_data.shape[1])
    weights = rng.uniform(0.1, 1, size=len(tabular_data))
    expected = UpstreamBase(lambda d: d, random_state=12).feature_selection(
        tabular_data, labels, weights, num_features, method
    )
    actual = LimeBase(lambda d: d, random_state=12).feature_selection(
        tabular_data, labels, weights, num_features, method
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("bow", [False, True])
def test_indexed_string_parity(bow):
    text = "A good model is good, but a robust model is better."
    upstream = UpstreamString(text, bow=bow)
    mojo = IndexedString(text, bow=bow)
    assert mojo.num_words() == upstream.num_words()
    assert [
        mojo.word(index) for index in range(mojo.num_words())
    ] == [
        upstream.word(index) for index in range(upstream.num_words())
    ]
    for index in range(mojo.num_words()):
        assert np.array_equal(
            mojo.string_position(index), upstream.string_position(index)
        )
    removed = np.array([0, min(2, mojo.num_words() - 1)])
    assert mojo.inverse_removing(removed) == upstream.inverse_removing(removed)


def test_indexed_characters_parity():
    text = "local model"
    upstream = UpstreamCharacters(text, bow=False)
    mojo = IndexedCharacters(text, bow=False)
    assert mojo.num_words() == upstream.num_words()
    assert mojo.inverse_removing(np.array([1, 4, 7])) == upstream.inverse_removing(
        np.array([1, 4, 7])
    )


def _text_classifier(strings):
    positive = np.asarray([
        (value.count("good") + value.count("robust") + 1) / 10
        for value in strings
    ])
    positive = np.clip(positive, 0.05, 0.95)
    return np.column_stack((1 - positive, positive))


def test_text_neighborhood_parity():
    text = "a good local model is robust and a good explanation is useful"
    upstream = UpstreamText(random_state=43)
    mojo = LimeTextExplainer(random_state=43)
    expected = upstream._LimeTextExplainer__data_labels_distances(
        UpstreamString(text), _text_classifier, 500
    )
    actual = mojo._LimeTextExplainer__data_labels_distances(
        IndexedString(text), _text_classifier, 500
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2], atol=2e-12)


def test_text_explanation_parity():
    text = "a good local model is robust and a good explanation is useful"
    kwargs = dict(random_state=47, feature_selection="none")
    upstream = UpstreamText(**kwargs)
    mojo = LimeTextExplainer(**kwargs)
    expected = upstream.explain_instance(
        text, _text_classifier, labels=(1,), num_features=8, num_samples=700
    )
    actual = mojo.explain_instance(
        text, _text_classifier, labels=(1,), num_features=8, num_samples=700
    )
    assert_explanation_pairs(actual.as_list(1), expected.as_list(1))
    assert actual.intercept[1] == pytest.approx(expected.intercept[1], abs=2e-11)
    assert actual.score == pytest.approx(expected.score, abs=2e-12)


def _image_classifier(images):
    red = (images[..., 0] ** 2).mean(axis=(1, 2))
    column_weight = np.linspace(0.5, 1.5, images.shape[2])
    green = (images[..., 1] * column_weight).mean(axis=(1, 2))
    logits = np.column_stack((red, green, 1 - (red + green) / 2))
    logits = np.exp(logits)
    return logits / logits.sum(axis=1, keepdims=True)


def test_image_neighborhood_parity():
    rng = np.random.RandomState(53)
    image = rng.uniform(size=(20, 18, 3))
    fudged = image * 0.3
    segments = np.arange(20 * 18).reshape(20, 18) % 12
    upstream = UpstreamImage(random_state=59)
    mojo = LimeImageExplainer(random_state=59)
    expected = upstream.data_labels(
        image, fudged, segments, _image_classifier, 80, batch_size=13
    )
    actual = mojo.data_labels(
        image, fudged, segments, _image_classifier, 80, batch_size=13
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1])


def test_image_explanation_parity():
    rng = np.random.RandomState(61)
    image = rng.uniform(size=(16, 16, 3))
    segments = np.repeat(np.arange(8), 32).reshape(16, 16)
    segmentation = lambda _: segments
    kwargs = dict(random_state=67, feature_selection="none")
    upstream = UpstreamImage(**kwargs)
    mojo = LimeImageExplainer(**kwargs)
    call = dict(
        labels=(1,),
        top_labels=None,
        num_features=8,
        num_samples=160,
        batch_size=23,
        segmentation_fn=segmentation,
    )
    expected = upstream.explain_instance(image, _image_classifier, **call)
    actual = mojo.explain_instance(image, _image_classifier, **call)
    assert actual.intercept[1] == pytest.approx(expected.intercept[1], abs=2e-11)
    assert_explanation_pairs(actual.local_exp[1], expected.local_exp[1])
    assert actual.score == pytest.approx(expected.score, abs=2e-12)


@pytest.mark.parametrize(
    "positive_only,negative_only,hide_rest",
    [(True, False, False), (False, True, True), (False, False, False)],
)
def test_image_mask_parity(positive_only, negative_only, hide_rest):
    image = np.arange(8 * 8 * 3, dtype=float).reshape(8, 8, 3)
    segments = np.repeat(np.arange(4), 16).reshape(8, 8)
    local_exp = [(0, 0.8), (1, -0.7), (2, 0.3), (3, -0.1)]
    upstream = UpstreamImageExplanation(image, segments)
    mojo = ImageExplanation(image, segments)
    upstream.local_exp[1] = local_exp
    mojo.local_exp[1] = local_exp
    kwargs = dict(
        positive_only=positive_only,
        negative_only=negative_only,
        hide_rest=hide_rest,
        num_features=3,
    )
    expected = upstream.get_image_and_mask(1, **kwargs)
    actual = mojo.get_image_and_mask(1, **kwargs)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "ours,theirs,methods",
    [
        (LimeTabularExplainer, UpstreamTabular, ["__init__", "explain_instance"]),
        (LimeTextExplainer, UpstreamText, ["__init__", "explain_instance"]),
        (LimeImageExplainer, UpstreamImage, ["__init__", "explain_instance", "data_labels"]),
        (LimeBase, UpstreamBase, ["__init__", "explain_instance_with_data"]),
    ],
)
def test_covered_api_signatures(ours, theirs, methods):
    for method in methods:
        assert inspect.signature(getattr(ours, method)) == inspect.signature(
            getattr(theirs, method)
        )
