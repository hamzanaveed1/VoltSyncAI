"""
Solar model: gradient boosting (Chapter 3.4.2).

XGBoost. Solar generation is a feature-based prediction problem driven by
measurable external factors - weather and time of day - rather than a sequence
learning problem. Gradient boosting is accurate on tabular features, fast, and
reports feature importance, so the model can be explained rather than treated
as a black box (Chen & Guestrin, 2016; Persson et al., 2017).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C


class SolarGBM:
    def __init__(self, params=None):
        self.params = dict(params or C.XGB_PARAMS)
        self.model = None
        self.features = None

    def fit(self, Xtr, ytr, Xva, yva, verbose=False):
        self.features = list(Xtr.columns)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                       verbose=100 if verbose else False)
        return self

    def predict(self, X):
        # Generation cannot be negative
        return np.clip(self.model.predict(X[self.features]), 0.0, None)

    @property
    def best_iteration(self):
        return getattr(self.model, "best_iteration", None)

    def importance(self, top=15) -> pd.Series:
        return (pd.Series(self.model.feature_importances_, index=self.features)
                .sort_values(ascending=False).head(top))

    def save(self, name="gbm_solar"):
        self.model.save_model(str(C.MODELS / f"{name}.json"))
