"""Image LIME neighborhoods with Mojo perturbation materialization."""

from __future__ import annotations

from functools import partial

import numpy as np
from skimage.color import gray2rgb
from skimage.segmentation import quickshift
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from . import lime_base
from .kernels import exponential_kernel, image_neighborhood, row_distances


class ImageExplanation:
    def __init__(self, image, segments):
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None

    def get_image_and_mask(
        self,
        label,
        positive_only=True,
        negative_only=False,
        hide_rest=False,
        num_features=5,
        min_weight=0.0,
    ):
        if label not in self.local_exp:
            raise KeyError("Label not in explanation")
        if positive_only and negative_only:
            raise ValueError(
                "Positive_only and negative_only cannot be true at the same time."
            )
        mask = np.zeros(self.segments.shape, self.segments.dtype)
        temp = (
            np.zeros(self.image.shape)
            if hide_rest else self.image.copy()
        )
        exp = self.local_exp[label]
        if positive_only:
            features = [
                feature for feature, weight in exp
                if weight > 0 and weight > min_weight
            ][:num_features]
        elif negative_only:
            features = [
                feature for feature, weight in exp
                if weight < 0 and abs(weight) > min_weight
            ][:num_features]
        else:
            features = []
        if positive_only or negative_only:
            for feature in features:
                selected = self.segments == feature
                temp[selected] = self.image[selected].copy()
                mask[selected] = 1
            return temp, mask
        for feature, weight in exp[:num_features]:
            if abs(weight) < min_weight:
                continue
            selected = self.segments == feature
            channel = 0 if weight < 0 else 1
            mask[selected] = -1 if weight < 0 else 1
            temp[selected] = self.image[selected].copy()
            temp[selected, channel] = np.max(self.image)
        return temp, mask


class LimeImageExplainer:
    def __init__(
        self,
        kernel_width=0.25,
        kernel=None,
        verbose=False,
        feature_selection="auto",
        random_state=None,
    ):
        kernel_width = float(kernel_width)
        kernel_fn = (
            partial(kernel, kernel_width=kernel_width)
            if kernel is not None
            else partial(exponential_kernel, kernel_width=kernel_width)
        )
        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )

    def explain_instance(
        self,
        image,
        classifier_fn,
        labels=(1,),
        hide_color=None,
        top_labels=5,
        num_features=100000,
        num_samples=1000,
        batch_size=10,
        segmentation_fn=None,
        distance_metric="cosine",
        model_regressor=None,
        random_seed=None,
    ):
        image = np.asarray(image)
        if image.ndim == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        if segmentation_fn is None:
            segmentation_fn = lambda value: quickshift(
                value,
                kernel_size=4,
                max_dist=200,
                ratio=0.2,
                rng=random_seed,
            )
        segments = np.asarray(segmentation_fn(image))
        fudged_image = image.copy()
        if hide_color is None:
            for feature in np.unique(segments):
                selected = segments == feature
                fudged_image[selected] = tuple(
                    np.mean(image[selected][:, channel])
                    for channel in range(image.shape[2])
                )
        else:
            fudged_image[:] = hide_color
        data, predictions = self.data_labels(
            image,
            fudged_image,
            segments,
            classifier_fn,
            num_samples,
            batch_size=batch_size,
        )
        if distance_metric in {"euclidean", "cosine"}:
            distances = row_distances(data, distance_metric)
        else:
            distances = pairwise_distances(
                data, data[0].reshape(1, -1), metric=distance_metric
            ).ravel()
        result = ImageExplanation(image, segments)
        selected_labels = labels
        if top_labels:
            selected_labels = np.argsort(predictions[0])[-top_labels:]
            result.top_labels = list(selected_labels)[::-1]
        for label in selected_labels:
            (
                result.intercept[label],
                result.local_exp[label],
                result.score,
                result.local_pred,
            ) = self.base.explain_instance_with_data(
                data,
                predictions,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
        return result

    def data_labels(
        self,
        image,
        fudged_image,
        segments,
        classifier_fn,
        num_samples,
        batch_size=10,
    ):
        image = np.asarray(image)
        segments = np.asarray(segments)
        unique = np.unique(segments)
        feature_count = unique.shape[0]
        data = self.random_state.randint(
            0, 2, num_samples * feature_count
        ).reshape((num_samples, feature_count))
        data[0, :] = 1
        # LIME indexes data columns from zero, so normalize arbitrary segment IDs.
        compact_segments = np.searchsorted(unique, segments)
        predictions = []
        for start in range(0, num_samples, batch_size):
            batch_data = data[start:start + batch_size]
            images = image_neighborhood(
                image, fudged_image, compact_segments, batch_data
            )
            if images.dtype != image.dtype:
                images = images.astype(image.dtype)
            predictions.extend(classifier_fn(images))
        return data, np.asarray(predictions)
