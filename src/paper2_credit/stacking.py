"""Heterogeneous stacking on top of a TabNet feature extractor.

Layout follows the arrangement of Wang and Zhang (2024, 2025): a TabNet model as
feature extractor, XGBoost, LightGBM, CatBoost, k-NN and an RBF SVM as level-one
learners, and XGBoost as meta-learner, so the comparison varies the extractor
and holds the ensemble fixed.

Exactly what the learners see:

* level-one input: the ordinal view of the raw columns concatenated with the
  extractor's decision representation (``use_raw_features``); k-NN and the SVM
  read it through a standard scaler and, above 8,000 training rows, a stratified
  8,000-row subsample;
* meta-learner input: the five level-one probabilities plus the extractor's own
  probability (``include_tabnet_prediction``), and nothing else.

Two protocols:

``nested`` (cross-fitting, the default). For each of K inner folds of the outer
training rows, the preprocessing, the extractor and the level-one learners are
fitted on the inner-training rows only, and the held-out inner rows receive
out-of-fold predictions from all six models, the extractor's probability
included. The meta-learner is trained on those out-of-fold predictions alone.
For the test fold, preprocessing, extractor and level-one learners are refitted
on all outer training rows. No row's own label reaches the features the
meta-learner is trained on.

``shared`` (the first submission's protocol, kept to measure what nesting
changes). Preprocessing and the extractor are fitted once on the outer training
rows, the level-one learners are cross-fitted on the resulting features, and the
extractor's probability for each training row is in-sample.

The extractor early-stops on the outer validation block, which belongs to the
outer training partition; the test fold is touched only by ``score``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.splits import inner_folds
from src.paper2_credit import data as credit_data
from src.paper2_credit.baselines import (
    SUBSAMPLE_ABOVE,
    build_baseline,
    build_scaled,
    stratified_subsample,
)
from src.paper2_credit.tabnet_gated import GMATabNetClassifier, TabNetConfig

DEFAULT_BASE_LEARNERS = ("xgboost", "lightgbm", "catboost", "knn", "svm")
SCALED = {"knn", "svm"}


@dataclass
class StackConfig:
    tabnet: TabNetConfig = field(default_factory=TabNetConfig)
    base_learners: tuple[str, ...] = DEFAULT_BASE_LEARNERS
    meta_learner: str = "xgboost"
    n_inner_folds: int = 5
    use_extractor: bool = True
    use_raw_features: bool = True
    include_tabnet_prediction: bool = True
    protocol: str = "nested"          # "nested" or "shared"
    expensive_subsample: int = SUBSAMPLE_ABOVE
    seed: int = 0

    def label(self) -> str:
        parts = [self.protocol]
        if self.use_extractor:
            parts.append(f"tabnet-{self.tabnet.variant}")
        parts.append("stack[" + ",".join(self.base_learners) + "]")
        parts.append(f"meta-{self.meta_learner}")
        return "+".join(parts)


class HeterogeneousStack:
    """Extractor -> level-one learners on out-of-fold predictions -> meta-learner."""

    def __init__(self, config: StackConfig | None = None) -> None:
        self.config = config or StackConfig()
        if self.config.protocol not in ("nested", "shared"):
            raise ValueError(f"unknown protocol {self.config.protocol!r}")
        self.preprocessor_ = None
        self.extractor_: GMATabNetClassifier | None = None
        self.base_models_: dict[str, object] = {}
        self.meta_model_ = None
        self.timings_: dict[str, float] = {}
        self.oof_auc_: dict[str, float] = {}
        self.inner_folds_: list[tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------ #
    def _fit_extractor(self, X, y, X_val, y_val, preprocessor) -> GMATabNetClassifier:
        model = GMATabNetClassifier(self.config.tabnet)
        model.fit(X, y, X_val, y_val, preprocessor.cat_idxs, preprocessor.cat_dims)
        return model

    def _augment(self, X: np.ndarray, extractor: GMATabNetClassifier | None) -> np.ndarray:
        if extractor is None:
            return X
        embedding = extractor.transform(X)
        if self.config.use_raw_features:
            return np.hstack([X, embedding]).astype(np.float32)
        return embedding

    def _fit_base(self, name: str, X: np.ndarray, y: np.ndarray):
        cfg = self.config
        model = build_scaled(name, seed=cfg.seed) if name in SCALED else build_baseline(name, seed=cfg.seed)
        if name in SCALED and len(y) > cfg.expensive_subsample:
            X, y = stratified_subsample(X, y, cfg.expensive_subsample, cfg.seed)
        model.fit(X, y)
        return model

    def _level_one(self, Z_fit, y_fit, Z_out) -> np.ndarray:
        columns = []
        for name in self.config.base_learners:
            model = self._fit_base(name, Z_fit, y_fit)
            columns.append(model.predict_proba(Z_out)[:, 1])
        return np.column_stack(columns).astype(np.float32)

    def _audit(self, kind: str, rows: np.ndarray) -> None:
        if credit_data.FIT_AUDIT is not None:
            credit_data.FIT_AUDIT.append((kind, "stack", np.asarray(rows)))

    # ------------------------------------------------------------------ #
    def fit(self, dataset, train_rows: np.ndarray, val_rows: np.ndarray) -> "HeterogeneousStack":
        """Fit on ``train_rows`` (original row ids); early-stop extractors on ``val_rows``."""
        from sklearn.metrics import roc_auc_score

        cfg = self.config
        y = dataset.y
        train_rows = np.asarray(train_rows)
        n_extra = 1 if (cfg.use_extractor and cfg.include_tabnet_prediction) else 0
        oof = np.zeros((len(train_rows), len(cfg.base_learners) + n_extra), dtype=np.float32)
        position = {row: i for i, row in enumerate(train_rows)}
        self.inner_folds_ = inner_folds(y, train_rows, cfg.n_inner_folds, cfg.seed)

        started = time.perf_counter()
        # The outer fit is needed by both protocols: for the test fold under
        # "nested", for the shared features under "shared".
        self.preprocessor_, (X_train, X_val) = credit_data.ordinal_arrays(dataset, train_rows, val_rows)
        if cfg.use_extractor:
            self._audit("extractor_fit", train_rows)
            self._audit("extractor_early_stop", val_rows)
            self.extractor_ = self._fit_extractor(X_train, y[train_rows], X_val, y[val_rows],
                                                  self.preprocessor_)
        Z_train = self._augment(X_train, self.extractor_)
        self.timings_["outer_fit_s"] = time.perf_counter() - started

        started = time.perf_counter()
        for inner_train, inner_valid in self.inner_folds_:
            slots = np.array([position[r] for r in inner_valid])
            if cfg.protocol == "shared":
                slots_fit = np.array([position[r] for r in inner_train])
                oof[slots, : len(cfg.base_learners)] = self._level_one(
                    Z_train[slots_fit], y[inner_train], Z_train[slots]
                )
                if n_extra:
                    # In-sample: the extractor was fitted on these very rows.
                    oof[slots, -1] = self.extractor_.predict_proba(X_train[slots])[:, 1]
                continue

            preprocessor, (X_fit, X_out, X_es) = credit_data.ordinal_arrays(
                dataset, inner_train, inner_valid, val_rows
            )
            extractor = None
            if cfg.use_extractor:
                self._audit("extractor_fit", inner_train)
                extractor = self._fit_extractor(X_fit, y[inner_train], X_es, y[val_rows], preprocessor)
            oof[slots, : len(cfg.base_learners)] = self._level_one(
                self._augment(X_fit, extractor), y[inner_train], self._augment(X_out, extractor)
            )
            if n_extra:
                oof[slots, -1] = extractor.predict_proba(X_out)[:, 1]
        self.timings_["oof_s"] = time.perf_counter() - started

        names = list(cfg.base_learners) + (["tabnet"] if n_extra else [])
        for column, name in enumerate(names):
            self.oof_auc_[name] = float(roc_auc_score(y[train_rows], oof[:, column]))

        started = time.perf_counter()
        for name in cfg.base_learners:
            self.base_models_[name] = self._fit_base(name, Z_train, y[train_rows])
        self.timings_["refit_s"] = time.perf_counter() - started

        self.meta_model_ = self._build_meta()
        self.meta_model_.fit(oof, y[train_rows])
        return self

    def _build_meta(self):
        if self.config.meta_learner == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", eval_metric="auc", n_jobs=-1,
                random_state=self.config.seed,
            )
        if self.config.meta_learner == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=2000, random_state=self.config.seed)
        raise ValueError(f"unknown meta learner {self.config.meta_learner!r}")

    # ------------------------------------------------------------------ #
    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Default probability for raw rows (a slice of ``dataset.X``)."""
        assert self.meta_model_ is not None, "call fit() first"
        X = self.preprocessor_.transform_ordinal(frame)
        Z = self._augment(X, self.extractor_)
        columns = [self.base_models_[n].predict_proba(Z)[:, 1] for n in self.config.base_learners]
        if self.extractor_ is not None and self.config.include_tabnet_prediction:
            columns.append(self.extractor_.predict_proba(X)[:, 1])
        meta_input = np.column_stack(columns).astype(np.float32)
        return self.meta_model_.predict_proba(meta_input)[:, 1]

    def score(self, dataset, rows: np.ndarray) -> np.ndarray:
        return self.score_frame(dataset.X.iloc[np.asarray(rows)])
