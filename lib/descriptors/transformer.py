from __future__ import annotations

import abc
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.pipeline import NotFittedError


class DescriptorTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    def fit(self, X=None, y=None) -> DescriptorTransformer:
        self.is_fitted_ = True
        return self

    @abc.abstractmethod
    def transform_one(self, X) -> Any:
        pass

    def transform(self, X):
        try:
            check_is_fitted(self)
        except NotFittedError:
            self.fit()

        return np.array([self.transform_one(it) for it in X])

    @abc.abstractmethod
    def get_feature_names_out(self, input_features=None) -> list[str]:
        pass

    def __getstate__(self):
        return self.get_params()

    def __setstate__(self, state):
        self.__init__(**state)
