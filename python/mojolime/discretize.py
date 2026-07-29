"""Discretizers used by tabular LIME neighborhood sampling."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state


class BaseDiscretizer(metaclass=ABCMeta):
    def __init__(
        self,
        data,
        categorical_features,
        feature_names,
        labels=None,
        random_state=None,
        data_stats=None,
    ):
        self.to_discretize = [
            feature for feature in range(data.shape[1])
            if feature not in categorical_features
        ]
        self.data_stats = data_stats
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.random_state = check_random_state(random_state)
        bins = [np.unique(value) for value in self.bins(data, labels)]
        if data_stats:
            self.means = data_stats.get("means")
            self.stds = data_stats.get("stds")
            self.mins = data_stats.get("mins")
            self.maxs = data_stats.get("maxs")
        for feature, cuts in zip(self.to_discretize, bins):
            low, high = np.min(data[:, feature]), np.max(data[:, feature])
            name = feature_names[feature]
            self.names[feature] = [f"{name} <= {cuts[0]:.2f}"]
            self.names[feature] += [
                f"{cuts[i]:.2f} < {name} <= {cuts[i + 1]:.2f}"
                for i in range(len(cuts) - 1)
            ]
            self.names[feature].append(f"{name} > {cuts[-1]:.2f}")
            self.lambdas[feature] = lambda value, cuts=cuts: np.searchsorted(cuts, value)
            if data_stats:
                continue
            discrete = self.lambdas[feature](data[:, feature])
            self.means[feature] = []
            self.stds[feature] = []
            for bin_id in range(len(cuts) + 1):
                selected = data[discrete == bin_id, feature]
                self.means[feature].append(0 if not len(selected) else np.mean(selected))
                self.stds[feature].append(
                    (0 if not len(selected) else np.std(selected)) + 1e-11
                )
            self.mins[feature] = [low] + cuts.tolist()
            self.maxs[feature] = cuts.tolist() + [high]

    @abstractmethod
    def bins(self, data, labels):
        raise NotImplementedError

    def discretize(self, data):
        result = data.copy()
        for feature, transform in self.lambdas.items():
            if data.ndim == 1:
                result[feature] = int(transform(result[feature]))
            else:
                result[:, feature] = transform(result[:, feature]).astype(int)
        return result

    def get_undiscretize_values(self, feature, values):
        indexes = np.asarray(values, dtype=int)
        mins = np.asarray(self.mins[feature])[indexes]
        maxs = np.asarray(self.maxs[feature])[indexes]
        means = np.asarray(self.means[feature])[indexes]
        stds = np.asarray(self.stds[feature])[indexes]
        low = (mins - means) / stds
        high = (maxs - means) / stds
        unequal = low != high
        result = low.copy()
        result[unequal] = stats.truncnorm.rvs(
            low[unequal],
            high[unequal],
            loc=means[unequal],
            scale=stds[unequal],
            random_state=self.random_state,
        )
        return result

    def undiscretize(self, data):
        result = data.copy()
        for feature in self.means:
            if data.ndim == 1:
                result[feature] = self.get_undiscretize_values(
                    feature, result[feature].astype(int).reshape(-1, 1)
                )
            else:
                result[:, feature] = self.get_undiscretize_values(
                    feature, result[:, feature].astype(int)
                )
        return result


class StatsDiscretizer(BaseDiscretizer):
    def bins(self, data, labels):
        supplied = self.data_stats.get("bins") or {}
        return [
            np.asarray(supplied[feature])
            for feature in self.to_discretize
            if feature in supplied
        ]


class QuartileDiscretizer(BaseDiscretizer):
    def bins(self, data, labels):
        return [
            np.asarray(np.percentile(data[:, feature], [25, 50, 75]))
            for feature in self.to_discretize
        ]


class DecileDiscretizer(BaseDiscretizer):
    def bins(self, data, labels):
        return [
            np.asarray(np.percentile(data[:, feature], np.arange(10, 100, 10)))
            for feature in self.to_discretize
        ]


class EntropyDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if labels is None:
            raise ValueError("Labels must be not None when using EntropyDiscretizer")
        super().__init__(
            data, categorical_features, feature_names,
            labels=labels, random_state=random_state,
        )

    def bins(self, data, labels):
        result = []
        for feature in self.to_discretize:
            tree = DecisionTreeClassifier(
                criterion="entropy", max_depth=3, random_state=self.random_state
            )
            tree.fit(data[:, feature].reshape(-1, 1), labels)
            cuts = tree.tree_.threshold[tree.tree_.children_left > -1]
            result.append(
                np.asarray([np.median(data[:, feature])])
                if not len(cuts) else np.sort(cuts)
            )
        return result
