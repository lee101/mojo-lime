"""Text LIME neighborhoods with Mojo cosine-distance weighting."""

from __future__ import annotations

import itertools
import re
from functools import partial

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from . import explanation, lime_base
from .kernels import exponential_kernel, row_distances


class TextDomainMapper(explanation.DomainMapper):
    def __init__(self, indexed_string):
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        if positions:
            return [
                (
                    f"{self.indexed_string.word(feature)}_"
                    + "-".join(
                        map(str, self.indexed_string.string_position(feature))
                    ),
                    weight,
                )
                for feature, weight in exp
            ]
        return [
            (self.indexed_string.word(feature), weight)
            for feature, weight in exp
        ]


class IndexedString:
    def __init__(
        self,
        raw_string,
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
    ):
        self.raw = raw_string
        self.mask_string = "UNKWORDZ" if mask_string is None else mask_string
        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            token_set = set(tokens)

            def non_word(value):
                return value not in token_set
        else:
            splitter = re.compile(r"(%s)|$" % split_expression)
            self.as_list = [value for value in splitter.split(self.raw) if value]
            non_word = splitter.match
        self.as_np = np.asarray(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(value) for value in self.as_np[:-1]]))
        )
        vocabulary = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocabulary = set()
        for index, word in enumerate(self.as_np):
            if word in non_vocabulary:
                continue
            if non_word(word):
                non_vocabulary.add(word)
                continue
            if bow:
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                self.positions[vocabulary[word]].append(index)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(index)
        if not bow:
            self.positions = np.asarray(self.positions)

    def raw_string(self):
        return self.raw

    def num_words(self):
        return len(self.inverse_vocab)

    def word(self, id_):
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        positions = self.positions[id_] if self.bow else [self.positions[id_]]
        return self.string_start[positions]

    def inverse_removing(self, words_to_remove):
        mask = np.ones(self.as_np.shape[0], dtype=bool)
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return "".join(
                self.as_list[index] if mask[index] else self.mask_string
                for index in range(mask.shape[0])
            )
        return "".join(self.as_list[index] for index in mask.nonzero()[0])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        result = []
        pointer = 0
        for token in tokens:
            between = []
            while not text[pointer:].startswith(token):
                between.append(text[pointer])
                pointer += 1
                if pointer >= len(text):
                    raise ValueError(
                        "Tokenization produced tokens that do not belong in string!"
                    )
            pointer += len(token)
            if between:
                result.append("".join(between))
            result.append(token)
        if pointer < len(text):
            result.append(text[pointer:])
        return result

    def __get_idxs(self, words):
        if self.bow:
            return list(itertools.chain.from_iterable(
                self.positions[word] for word in words
            ))
        return self.positions[words]


class IndexedCharacters:
    def __init__(self, raw_string, bow=True, mask_string=None):
        self.raw = raw_string
        self.as_list = list(raw_string)
        self.as_np = np.asarray(self.as_list)
        self.mask_string = chr(0) if mask_string is None else mask_string
        self.string_start = np.arange(len(raw_string))
        vocabulary = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        for index, character in enumerate(self.as_np):
            if bow:
                if character not in vocabulary:
                    vocabulary[character] = len(vocabulary)
                    self.inverse_vocab.append(character)
                    self.positions.append([])
                self.positions[vocabulary[character]].append(index)
            else:
                self.inverse_vocab.append(character)
                self.positions.append(index)
        if not bow:
            self.positions = np.asarray(self.positions)

    def raw_string(self):
        return self.raw

    def num_words(self):
        return len(self.inverse_vocab)

    def word(self, id_):
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        positions = self.positions[id_] if self.bow else [self.positions[id_]]
        return self.string_start[positions]

    def inverse_removing(self, words_to_remove):
        mask = np.ones(self.as_np.shape[0], dtype=bool)
        mask[self.__get_idxs(words_to_remove)] = False
        if not self.bow:
            return "".join(
                self.as_list[index] if mask[index] else self.mask_string
                for index in range(mask.shape[0])
            )
        return "".join(self.as_list[index] for index in mask.nonzero()[0])

    def __get_idxs(self, words):
        if self.bow:
            return list(itertools.chain.from_iterable(
                self.positions[word] for word in words
            ))
        return self.positions[words]


class LimeTextExplainer:
    def __init__(
        self,
        kernel_width=25,
        kernel=None,
        verbose=False,
        class_names=None,
        feature_selection="auto",
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
        random_state=None,
        char_level=False,
    ):
        kernel_width = float(kernel_width)
        kernel_fn = (
            partial(kernel, kernel_width=kernel_width)
            if kernel is not None
            else partial(exponential_kernel, kernel_width=kernel_width)
        )
        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    def explain_instance(
        self,
        text_instance,
        classifier_fn,
        labels=(1,),
        top_labels=None,
        num_features=10,
        num_samples=5000,
        distance_metric="cosine",
        model_regressor=None,
    ):
        indexed = (
            IndexedCharacters(
                text_instance, bow=self.bow, mask_string=self.mask_string
            )
            if self.char_level
            else IndexedString(
                text_instance,
                bow=self.bow,
                split_expression=self.split_expression,
                mask_string=self.mask_string,
            )
        )
        mapper = TextDomainMapper(indexed)
        data, predictions, distances = self.__data_labels_distances(
            indexed, classifier_fn, num_samples, distance_metric=distance_metric
        )
        if self.class_names is None:
            self.class_names = [str(index) for index in range(predictions.shape[1])]
        result = explanation.Explanation(
            mapper, class_names=self.class_names, random_state=self.random_state
        )
        result.predict_proba = predictions[0]
        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            result.top_labels = list(labels)[::-1]
        for label in labels:
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

    def __data_labels_distances(
        self,
        indexed_string,
        classifier_fn,
        num_samples,
        distance_metric="cosine",
    ):
        document_size = indexed_string.num_words()
        sample_sizes = self.random_state.randint(
            1, document_size + 1, num_samples - 1
        )
        data = np.ones((num_samples, document_size))
        feature_range = range(document_size)
        inverse_data = [indexed_string.raw_string()]
        for row, size in enumerate(sample_sizes, start=1):
            inactive = self.random_state.choice(
                feature_range, size, replace=False
            )
            data[row, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))
        predictions = np.asarray(classifier_fn(inverse_data))
        if distance_metric == "cosine":
            distances = row_distances(data, "cosine", multiplier=100.0)
        elif distance_metric == "euclidean":
            distances = row_distances(data, "euclidean", multiplier=100.0)
        else:
            distances = pairwise_distances(
                sparse.csr_matrix(data),
                sparse.csr_matrix(data[0].reshape(1, -1)),
                metric=distance_metric,
            ).ravel() * 100
        return data, predictions, distances
