"""Locally weighted sparse linear surrogate models."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state

from .kernels import weighted_ridge


class LimeBase:
    """Learn a locally weighted linear model from perturbed observations."""

    def __init__(self, kernel_fn, verbose=False, random_state=None):
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        alphas, _, coefficients = lars_path(
            weighted_data, weighted_labels, method="lasso", verbose=False
        )
        return alphas, coefficients

    def forward_selection(self, data, labels, weights, num_features):
        model = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            best_score = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                selected = used_features + [feature]
                model.fit(data[:, selected], labels, sample_weight=weights)
                score = model.score(data[:, selected], labels, sample_weight=weights)
                if score > best_score:
                    best = feature
                    best_score = score
            used_features.append(best)
        return np.asarray(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        if method == "none":
            return np.arange(data.shape[1])
        if method == "forward_selection":
            return self.forward_selection(data, labels, weights, num_features)
        if method == "highest_weights":
            model = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            model.fit(data, labels, sample_weight=weights)
            if sparse.issparse(data):
                weighted = sparse.csr_matrix(model.coef_).multiply(data[0])
                size = len(weighted.data)
                order = np.abs(weighted.data).argsort()
                if size < num_features:
                    indices = weighted.indices[order[::-1]]
                    pad = num_features - size
                    indices = np.concatenate(
                        (indices, np.zeros(pad, dtype=indices.dtype))
                    )
                    selected = set(indices)
                    cursor = 0
                    for feature in range(data.shape[1]):
                        if feature not in selected:
                            indices[size + cursor] = feature
                            cursor += 1
                            if cursor >= pad:
                                break
                    return indices
                chosen = order[size - num_features:size][::-1]
                return weighted.indices[chosen]
            weighted = model.coef_ * data[0]
            ranked = sorted(
                zip(range(data.shape[1]), weighted),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            return np.asarray([feature for feature, _ in ranked[:num_features]])
        if method == "lasso_path":
            centered_data = (
                data - np.average(data, axis=0, weights=weights)
            ) * np.sqrt(weights[:, np.newaxis])
            centered_labels = (
                labels - np.average(labels, weights=weights)
            ) * np.sqrt(weights)
            nonzero = range(centered_data.shape[1])
            _, coefficients = self.generate_lars_path(centered_data, centered_labels)
            for column in range(len(coefficients.T) - 1, 0, -1):
                nonzero = coefficients.T[column].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            return nonzero
        if method == "auto":
            selected_method = (
                "forward_selection" if num_features <= 6 else "highest_weights"
            )
            return self.feature_selection(
                data, labels, weights, num_features, selected_method
            )
        raise ValueError(f"unknown feature_selection method: {method}")

    def explain_instance_with_data(
        self,
        neighborhood_data,
        neighborhood_labels,
        distances,
        label,
        num_features,
        feature_selection="auto",
        model_regressor=None,
    ):
        """Fit and report the local surrogate using LIME's return contract."""
        weights = np.asarray(self.kernel_fn(distances), dtype=np.float64)
        labels_column = np.asarray(neighborhood_labels[:, label], dtype=np.float64)
        used_features = self.feature_selection(
            neighborhood_data,
            labels_column,
            weights,
            num_features,
            feature_selection,
        )
        selected = neighborhood_data[:, used_features]
        if model_regressor is None and not sparse.issparse(selected):
            intercept, coefficients, score, local_prediction = weighted_ridge(
                selected, labels_column, weights, alpha=1.0
            )
        else:
            model = model_regressor
            if model is None:
                model = Ridge(
                    alpha=1, fit_intercept=True, random_state=self.random_state
                )
            model.fit(selected, labels_column, sample_weight=weights)
            intercept = model.intercept_
            coefficients = model.coef_
            score = model.score(selected, labels_column, sample_weight=weights)
            local_prediction = model.predict(
                neighborhood_data[0, used_features].reshape(1, -1)
            )
        if self.verbose:
            print("Intercept", intercept)
            print("Prediction_local", local_prediction)
            print("Right:", neighborhood_labels[0, label])
        explanation = sorted(
            zip(used_features, coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        return intercept, explanation, score, local_prediction
