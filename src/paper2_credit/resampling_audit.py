"""How much does resampling before the split inflate a reported ROC-AUC?

Several credit-scoring studies balance the classes as a preprocessing step and
only then split into training and test sets. This stage measures what that does,
on all three datasets, for four common resamplers and three models.

Design (one record per dataset x seed x protocol x resampler x model):

* A stratified 20% **holdout** is set aside first and never resampled or seen
  by any fit. It measures how the trained model really performs on the
  population of applicants.
* The remaining 80% is the study data. Under ``none`` it is split 80/20 into
  training and test rows; under ``train_only`` the same split is made and only
  the training rows are resampled; under ``before_split`` the whole study data
  is preprocessed and resampled first and then split, as the shortcut does.
* Each record carries ``auc_reported`` (on that protocol's own test rows, what
  a paper would print) and ``auc_holdout`` (on the untouched holdout). Their
  difference is the inflation.

Resamplers: random over-sampling, random under-sampling, SMOTE (SMOTENC where
categorical columns exist) and NearMiss-2 (``protocol.nearmiss2``, checked
against imblearn). Models: XGBoost (default baseline configuration) and
logistic regression on the ordinal view, and the token-attention TabNet with
default settings for the three cells that matter most (none, and ROS / NearMiss
before the split).
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.io import results_root, write_json
from src.common.seeds import set_seed
from src.common.torch_utils import release_memory, run_resilient
from src.paper2_credit.baselines import build_baseline
from src.paper2_credit.data import CreditPreprocessor
from src.paper2_credit.protocol import nearmiss2

LOGGER = logging.getLogger("paper2.resampling")

RESAMPLERS = ("ros", "rus", "smote", "nearmiss2")
PROTOCOLS = ("before_split", "train_only")
CPU_MODELS = ("xgboost", "logistic_regression")
TOKEN_CELLS = {("none", "none"), ("before_split", "ros"), ("before_split", "nearmiss2")}
SEEDS = (0, 1, 2, 3, 4)


def resample(method: str, X: np.ndarray, y: np.ndarray, cat_idxs: list[int], seed: int):
    if method == "nearmiss2":
        keep = nearmiss2(X.astype(np.float64), y)
        return X[keep], y[keep]
    from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    if method == "ros":
        sampler = RandomOverSampler(random_state=seed)
    elif method == "rus":
        sampler = RandomUnderSampler(random_state=seed)
    elif method == "smote":
        numeric = X.shape[1] - len(cat_idxs)
        sampler = (SMOTENC(categorical_features=list(cat_idxs), random_state=seed)
                   if cat_idxs and numeric > 0 else SMOTE(random_state=seed))
    else:
        raise ValueError(method)
    X_out, y_out = sampler.fit_resample(X, y)
    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out)


def fit_score(model: str, X_train, y_train, X_eval: list, seed: int, cat_idxs, cat_dims) -> list[float]:
    """Fit ``model`` and return its scores on each array in ``X_eval``."""
    if model == "xgboost":
        clf = build_baseline("xgboost", seed=seed)
        clf.fit(X_train, y_train)
        return [clf.predict_proba(X)[:, 1] for X in X_eval]
    if model == "logistic_regression":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), build_baseline("logistic_regression", seed=seed))
        clf.fit(X_train, y_train)
        return [clf.predict_proba(X)[:, 1] for X in X_eval]
    if model == "token":
        from src.paper2_credit.run_experiments import finalise
        from src.paper2_credit.tabnet_gated import GMATabNetClassifier, TabNetConfig

        fit_rows, stop_rows = train_test_split(np.arange(len(y_train)), test_size=0.15,
                                               stratify=y_train, random_state=seed)
        config = finalise(TabNetConfig(variant="vanilla", use_tokens=True, seed=seed))
        clf = run_resilient(lambda: GMATabNetClassifier(config).fit(
            X_train[fit_rows], y_train[fit_rows], X_train[stop_rows], y_train[stop_rows],
            cat_idxs, cat_dims))
        scores = [clf.predict_proba(X)[:, 1] for X in X_eval]
        del clf
        release_memory()
        return scores
    raise ValueError(model)


def run_cell(dataset, seed: int, protocol: str, method: str, models: list[str]) -> list[dict]:
    y = dataset.y
    rows = np.arange(len(y))
    study, holdout = train_test_split(rows, test_size=0.2, stratify=y, random_state=seed)
    records = []

    if protocol == "before_split":
        prep = CreditPreprocessor(dataset.key).fit(dataset.X.iloc[study])
        X_study = prep.transform_ordinal(dataset.X.iloc[study])
        X_res, y_res = resample(method, X_study, y[study], prep.cat_idxs, seed)
        train, test = train_test_split(np.arange(len(y_res)), test_size=0.2, stratify=y_res,
                                       random_state=seed)
        X_train, y_train, X_test, y_test = X_res[train], y_res[train], X_res[test], y_res[test]
    else:
        train_rows, test_rows = train_test_split(study, test_size=0.2, stratify=y[study], random_state=seed)
        prep = CreditPreprocessor(dataset.key).fit(dataset.X.iloc[train_rows])
        X_train = prep.transform_ordinal(dataset.X.iloc[train_rows])
        y_train = y[train_rows]
        X_test, y_test = prep.transform_ordinal(dataset.X.iloc[test_rows]), y[test_rows]
        if protocol == "train_only":
            X_train, y_train = resample(method, X_train, y_train, prep.cat_idxs, seed)
    X_holdout = prep.transform_ordinal(dataset.X.iloc[holdout])

    for model in models:
        started = time.perf_counter()
        set_seed(seed)
        reported, true = fit_score(model, X_train, y_train, [X_test, X_holdout], seed,
                                   prep.cat_idxs, prep.cat_dims)
        records.append({
            "dataset": dataset.key, "seed": seed, "protocol": protocol, "resampler": method,
            "model": model, "n_train": int(len(y_train)), "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "auc_reported": float(roc_auc_score(y_test, reported)),
            "auc_holdout": float(roc_auc_score(y[holdout], true)),
            "seconds": time.perf_counter() - started,
        })
    return records


def stage_resampling_audit(dataset, args) -> None:
    from src.paper2_credit.run_experiments import SMOKE

    target = results_root() / "paper2" / "protocol" / f"{dataset.key}__resampling_audit.json"
    done = json.loads(target.read_text(encoding="utf-8"))["records"] if target.exists() and not args.force else []
    have = {(r["seed"], r["protocol"], r["resampler"], r["model"]) for r in done}
    cells = [("none", "none")] + [(p, m) for p in PROTOCOLS for m in RESAMPLERS]
    for seed in (SEEDS[:1] if SMOKE else SEEDS):
        for protocol, method in cells:
            models = list(CPU_MODELS) + (["token"] if (protocol, method) in TOKEN_CELLS else [])
            models = [m for m in models if (seed, protocol, method, m) not in have]
            if not models:
                continue
            LOGGER.info("%s | resampling | %s/%s | seed %d | %s", dataset.key, protocol, method,
                        seed, ",".join(models))
            for record in run_cell(dataset, seed, protocol, method, models):
                done.append(record)
                LOGGER.info("  %s reported %.4f holdout %.4f", record["model"],
                            record["auc_reported"], record["auc_holdout"])
            write_json(target, {"dataset": dataset.key, "records": done})
