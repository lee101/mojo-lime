"""Measured mojo-lime versus upstream LIME sampling and surrogate kernels."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
import lime.lime_image as upstream_image_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lime.lime_image import LimeImageExplainer as UpstreamImage  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer as UpstreamTabular  # noqa: E402
from mojolime import kernels  # noqa: E402
from mojolime.lime_image import LimeImageExplainer  # noqa: E402
from mojolime.lime_tabular import LimeTabularExplainer  # noqa: E402

upstream_image_module.tqdm = lambda iterable: iterable


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


CASES = []


def case(name):
    def register(function):
        CASES.append((name, function))
        return function
    return register


@case("proximity kernel (5M distances)")
def proximity_case():
    distances = np.linspace(0, 50, 5_000_000)
    width = 25.0
    ours = lambda: kernels.exponential_kernel(distances, width)
    upstream = lambda: np.sqrt(np.exp(-(distances ** 2) / width ** 2))
    assert np.allclose(ours(), upstream(), atol=3e-13)
    return ours, upstream


@case("Euclidean distances (100k x 64)")
def euclidean_case():
    values = np.random.RandomState(1).normal(size=(100_000, 64))
    ours = lambda: kernels.row_distances(values, "euclidean")
    upstream = lambda: pairwise_distances(
        values, values[:1], metric="euclidean"
    ).ravel()
    assert np.allclose(ours(), upstream())
    return ours, upstream


@case("text cosine distances (100k x 128)")
def cosine_case():
    values = np.random.RandomState(2).randint(
        0, 2, size=(100_000, 128)
    ).astype(np.float64)
    values[0] = 1
    ours = lambda: kernels.row_distances(values, "cosine", multiplier=100)

    def upstream():
        matrix = sparse.csr_matrix(values)
        return pairwise_distances(
            matrix, matrix[0], metric="cosine"
        ).ravel() * 100

    assert np.allclose(ours(), upstream(), atol=2e-12)
    return ours, upstream


@case("image data_labels (96 x 128 x 128 RGB)")
def image_case():
    rng = np.random.RandomState(3)
    image = rng.uniform(size=(128, 128, 3))
    fudged = image * 0.2
    segments = (
        np.arange(128)[:, None] // 16 * 8
        + np.arange(128)[None, :] // 16
    )

    def classifier(images):
        mean = images.mean(axis=(1, 2, 3))
        return np.column_stack((mean, 1 - mean))

    mojo = LimeImageExplainer(random_state=4)
    lime = UpstreamImage(random_state=4)
    ours = lambda: mojo.data_labels(
        image, fudged, segments, classifier, 96, batch_size=16
    )
    upstream = lambda: lime.data_labels(
        image, fudged, segments, classifier, 96, batch_size=16
    )
    return ours, upstream


@case("weighted ridge surrogate (30k x 32)")
def ridge_case():
    rng = np.random.RandomState(5)
    values = rng.normal(size=(30_000, 32))
    labels = values @ rng.normal(size=32) + rng.normal(size=len(values)) * 0.1
    weights = rng.uniform(0.01, 1, size=len(values))
    ours = lambda: kernels.weighted_ridge(values, labels, weights)
    upstream = lambda: Ridge(alpha=1).fit(
        values, labels, sample_weight=weights
    )
    actual = ours()
    expected = upstream()
    assert np.allclose(actual[1], expected.coef_, atol=1e-10)
    return ours, upstream


@case("tabular explain_instance (5k x 24)")
def tabular_case():
    rng = np.random.RandomState(6)
    training = rng.normal(size=(10_000, 24))
    coefficients = rng.normal(size=24)

    def predict(values):
        logits = values @ coefficients
        positive = 1 / (1 + np.exp(-logits))
        return np.column_stack((1 - positive, positive))

    mojo = LimeTabularExplainer(
        training,
        discretize_continuous=False,
        feature_selection="none",
        random_state=7,
    )
    lime = UpstreamTabular(
        training,
        discretize_continuous=False,
        feature_selection="none",
        random_state=7,
    )
    ours = lambda: mojo.explain_instance(
        training[0], predict, labels=(1,), num_features=24, num_samples=5000
    )
    upstream = lambda: lime.explain_instance(
        training[0], predict, labels=(1,), num_features=24, num_samples=5000
    )
    return ours, upstream


def format_time(seconds):
    milliseconds = seconds * 1000
    return f"{milliseconds:.2f} ms" if milliseconds < 100 else f"{milliseconds:.1f} ms"


def main():
    print(
        f"Machine: {machine_name()}; {platform.system()} {platform.release()}",
        flush=True,
    )
    print(flush=True)
    print("| case | mojo-lime | upstream | result |", flush=True)
    print("| --- | ---: | ---: | ---: |", flush=True)
    for name, prepare in CASES:
        ours, upstream = prepare()
        ours()
        upstream()
        mojo_time = timeit(ours)
        upstream_time = timeit(upstream)
        ratio = upstream_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {format_time(mojo_time)} | "
            f"{format_time(upstream_time)} | {result} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
