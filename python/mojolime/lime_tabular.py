"""Tabular LIME neighborhoods with Mojo dense sampling kernels."""

from __future__ import annotations

import collections
import copy
import warnings
from functools import partial

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from . import explanation, lime_base
from .discretize import (
    BaseDiscretizer,
    DecileDiscretizer,
    EntropyDiscretizer,
    QuartileDiscretizer,
    StatsDiscretizer,
)
from .kernels import (
    affine_samples_inplace,
    affine_standardize_samples,
    exponential_kernel,
    row_distances,
    standardize_inplace,
)


class TableDomainMapper(explanation.DomainMapper):
    def __init__(
        self,
        feature_names,
        feature_values,
        scaled_row,
        categorical_features,
        discretized_feature_names=None,
        feature_indexes=None,
    ):
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        self.all_categorical = (
            False if sparse.issparse(scaled_row)
            else len(categorical_features) == len(scaled_row)
        )
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        names = (
            self.discretized_feature_names
            if self.discretized_feature_names is not None
            else self.exp_feature_names
        )
        return [(names[feature], weight) for feature, weight in exp]


class LimeTabularExplainer:
    """Explain dense tabular predictions with locally sampled surrogates."""

    def __init__(
        self,
        training_data,
        mode="classification",
        training_labels=None,
        feature_names=None,
        categorical_features=None,
        categorical_names=None,
        kernel_width=None,
        kernel=None,
        verbose=False,
        class_names=None,
        feature_selection="auto",
        discretize_continuous=True,
        discretizer="quartile",
        sample_around_instance=False,
        random_state=None,
        training_data_stats=None,
    ):
        if sparse.issparse(training_data):
            raise NotImplementedError("mojolime's tabular port currently supports dense data")
        training_data = np.asarray(training_data)
        if training_data.ndim != 2:
            raise ValueError("training_data must be a two-dimensional array")
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats
        if training_data_stats:
            self.validate_training_data_stats(training_data_stats)
        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(feature) for feature in range(training_data.shape[1])]
        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)
        self.discretizer = None
        discretized_training_data = None
        if discretize_continuous:
            if training_data_stats:
                self.discretizer = StatsDiscretizer(
                    training_data,
                    self.categorical_features,
                    self.feature_names,
                    labels=training_labels,
                    data_stats=training_data_stats,
                    random_state=self.random_state,
                )
            elif discretizer == "quartile":
                self.discretizer = QuartileDiscretizer(
                    training_data,
                    self.categorical_features,
                    self.feature_names,
                    labels=training_labels,
                    random_state=self.random_state,
                )
            elif discretizer == "decile":
                self.discretizer = DecileDiscretizer(
                    training_data,
                    self.categorical_features,
                    self.feature_names,
                    labels=training_labels,
                    random_state=self.random_state,
                )
            elif discretizer == "entropy":
                self.discretizer = EntropyDiscretizer(
                    training_data,
                    self.categorical_features,
                    self.feature_names,
                    labels=training_labels,
                    random_state=self.random_state,
                )
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError(
                    "Discretizer must be 'quartile', 'decile', 'entropy' "
                    "or a BaseDiscretizer instance"
                )
            self.categorical_features = list(range(training_data.shape[1]))
            if training_data_stats is None:
                discretized_training_data = self.discretizer.discretize(training_data)
        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * 0.75
        self.kernel_width = float(kernel_width)
        kernel_fn = (
            partial(kernel, kernel_width=self.kernel_width)
            if kernel is not None
            else partial(exponential_kernel, kernel_width=self.kernel_width)
        )
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )
        self.class_names = class_names
        self.scaler = StandardScaler(with_mean=False).fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}
        for feature in self.categorical_features:
            if training_data_stats is None:
                column = (
                    discretized_training_data[:, feature]
                    if self.discretizer is not None
                    else training_data[:, feature]
                )
                counts = collections.Counter(column)
                values, frequencies = map(list, zip(*sorted(counts.items())))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (
                np.asarray(frequencies) / float(sum(frequencies))
            )
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ["%.2f" % value for value in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):
        required = {
            "means", "mins", "maxs", "stds",
            "feature_values", "feature_frequencies",
        }
        missing = list(required - set(training_data_stats))
        if missing:
            raise Exception(
                "Missing keys in training_data_stats. Details: %s" % missing
            )

    def explain_instance(
        self,
        data_row,
        predict_fn,
        labels=(1,),
        top_labels=None,
        num_features=10,
        num_samples=5000,
        distance_metric="euclidean",
        model_regressor=None,
    ):
        if sparse.issparse(data_row):
            raise NotImplementedError("mojolime's tabular port currently supports dense data")
        data_row = np.asarray(data_row)
        fused_scaling = (
            self.discretizer is None and not self.categorical_features
        )
        data, inverse = self.__data_inverse(
            data_row, num_samples, scale=fused_scaling
        )
        scaled_data = (
            data if fused_scaling else
            standardize_inplace(data, self.scaler.mean_, self.scaler.scale_)
        )
        if distance_metric in {"euclidean", "cosine"}:
            distances = row_distances(scaled_data, distance_metric)
        else:
            distances = pairwise_distances(
                scaled_data, scaled_data[0].reshape(1, -1), metric=distance_metric
            ).ravel()
        predictions = np.asarray(predict_fn(inverse))
        if self.mode == "classification":
            if predictions.ndim == 1:
                raise NotImplementedError(
                    "LIME does not currently support classifier models "
                    "without probability scores."
                )
            if predictions.ndim != 2:
                raise ValueError(
                    f"Your model outputs arrays with {predictions.ndim} dimensions"
                )
            if self.class_names is None:
                self.class_names = [str(index) for index in range(predictions.shape[1])]
            else:
                self.class_names = list(self.class_names)
            if not np.allclose(predictions.sum(axis=1), 1.0):
                warnings.warn(
                    "Prediction probabilities do not sum to 1; check that "
                    "predict_fn returns probabilities.",
                    stacklevel=2,
                )
        else:
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions[:, 0]
            if predictions.ndim != 1:
                raise ValueError(
                    "Your model needs to output single-dimensional numpy arrays, "
                    f"not arrays of {predictions.shape} dimensions"
                )
            predicted_value = predictions[0]
            min_y = min(predictions)
            max_y = max(predictions)
            predictions = predictions[:, np.newaxis]
        feature_names = copy.deepcopy(self.feature_names)
        values = self.convert_and_round(data_row)
        for feature in self.categorical_features:
            if self.discretizer is not None and feature in self.discretizer.lambdas:
                continue
            name = int(data_row[feature])
            if feature in self.categorical_names:
                name = self.categorical_names[feature][name]
            feature_names[feature] = f"{feature_names[feature]}={name}"
            values[feature] = "True"
        categorical_features = self.categorical_features
        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discrete_row = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for feature in self.discretizer.names:
                discretized_feature_names[feature] = self.discretizer.names[feature][
                    int(discrete_row[feature])
                ]
        mapper = TableDomainMapper(
            feature_names,
            values,
            scaled_data[0],
            categorical_features=categorical_features,
            discretized_feature_names=discretized_feature_names,
        )
        result = explanation.Explanation(
            mapper, mode=self.mode, class_names=self.class_names
        )
        if self.mode == "classification":
            result.predict_proba = predictions[0]
            if top_labels:
                labels = np.argsort(predictions[0])[-top_labels:]
                result.top_labels = list(labels)[::-1]
        else:
            result.predicted_value = predicted_value
            result.min_value = min_y
            result.max_value = max_y
            labels = [0]
        for label in labels:
            (
                result.intercept[label],
                result.local_exp[label],
                result.score,
                result.local_pred,
            ) = self.base.explain_instance_with_data(
                scaled_data,
                predictions,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )
        if self.mode == "regression":
            result.intercept[1] = result.intercept[0]
            result.local_exp[1] = list(result.local_exp[0])
            result.local_exp[0] = [
                (feature, -weight) for feature, weight in result.local_exp[1]
            ]
        return result

    def __data_inverse(self, data_row, num_samples, *, scale=False):
        num_columns = data_row.shape[0]
        if self.discretizer is None:
            normals = self.random_state.normal(
                0, 1, num_samples * num_columns
            ).reshape(num_samples, num_columns)
            center = data_row if self.sample_around_instance else self.scaler.mean_
            if scale and not self.categorical_features:
                return affine_standardize_samples(
                    normals,
                    data_row,
                    center,
                    self.scaler.scale_,
                    self.scaler.mean_,
                    self.scaler.scale_,
                )
            data = affine_samples_inplace(normals, center, self.scaler.scale_)
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            data = np.zeros((num_samples, num_columns), dtype=np.float64)
            categorical_features = range(num_columns)
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            inverse_column = self.random_state.choice(
                self.feature_values[column],
                size=num_samples,
                replace=True,
                p=self.feature_frequencies[column],
            )
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse
