"""Core explanation containers compatible with LIME's common inspection API."""

from __future__ import annotations


class DomainMapper:
    def map_exp_ids(self, exp, **kwargs):
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name, **kwargs):
        return ""


class Explanation:
    def __init__(self, domain_mapper, mode="classification", class_names=None, random_state=None):
        self.random_state = random_state
        self.mode = mode
        self.domain_mapper = domain_mapper
        self.local_exp = {}
        self.intercept = {}
        self.score = None
        self.local_pred = None
        if mode == "classification":
            self.class_names = class_names
            self.top_labels = None
            self.predict_proba = None
        elif mode == "regression":
            self.class_names = ["negative", "positive"]
            self.predicted_value = None
            self.min_value = 0.0
            self.max_value = 1.0
            self.dummy_label = 1
        else:
            raise ValueError(
                f'Invalid explanation mode "{mode}". '
                'Should be either "classification" or "regression".'
            )

    def available_labels(self):
        if self.mode != "classification":
            raise NotImplementedError("Not supported for regression explanations.")
        return list(self.top_labels if self.top_labels else self.local_exp.keys())

    def as_list(self, label=1, **kwargs):
        selected = label if self.mode == "classification" else self.dummy_label
        mapped = self.domain_mapper.map_exp_ids(self.local_exp[selected], **kwargs)
        return [(name, float(weight)) for name, weight in mapped]

    def as_map(self):
        return self.local_exp
