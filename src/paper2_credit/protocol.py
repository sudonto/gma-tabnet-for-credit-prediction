"""How much of a published Give Me Some Credit score is protocol?

Wang and Zhang (2025) report ROC-AUC 0.941 for their improved TabNet on Give Me
Some Credit. Their Section 4 describes the protocol: two features dropped
(NumberRealEstateLoansOrLines, NumberOfDependents), the dataset balanced from
13.96:1 to 1:1 with NearMiss (version 2) under-sampling as a preprocessing step,
and a 4:1 train/test split. As described, the balancing precedes the split, so
the test set is itself balanced and drawn from the rows NearMiss judged most
representative.

This stage runs that protocol and two others on the same data, models and seeds:

``wang``        impute and NearMiss-2 on the whole dataset, then split 4:1
``train_only``  split 4:1 first, NearMiss-2 on the training part only, test untouched
``none``        split 4:1, no resampling (our protocol, single split)

with XGBoost (default baseline configuration) and GMA-TabNet (default
configuration) and five split seeds. The difference between ``wang`` and
``train_only`` is what the ordering of resampling and splitting does to the
reported number; it replaces the first submission's general remark about
oversampling with a measured, study-specific comparison.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.common.io import results_root, write_json
from src.common.seeds import set_seed
from src.common.torch_utils import release_memory, run_resilient
from src.paper2_credit.baselines import build_baseline
from src.paper2_credit.data import CreditPreprocessor

LOGGER = logging.getLogger("paper2.protocol")

DROPPED = ["NumberRealEstateLoansOrLines", "NumberOfDependents"]
PROTOCOLS = ("wang", "train_only", "none")
MODELS = ("xgboost", "gma")
SEEDS = (0, 1, 2, 3, 4)


def nearmiss2(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3, chunk: int = 2048) -> np.ndarray:
    """Positions NearMiss version 2 keeps, balancing the classes 1:1.

    The same selection as ``imblearn.under_sampling.NearMiss(version=2)``: every
    majority row is scored by the sum of its distances to the ``n_neighbors``
    *farthest* minority rows, and the lowest-scoring rows are kept, ties broken
    by position. imblearn computes this through a full neighbour query with as
    many neighbours as there are minority rows, which on Give Me Some Credit is a
    140,000 x 10,000 index matrix; here it is computed in chunks.
    ``scripts/check_nearmiss.py`` checks the two agree.
    """
    from scipy.spatial.distance import cdist

    minority = np.flatnonzero(y == 1)
    majority = np.flatnonzero(y == 0)
    scores = np.empty(len(majority))
    for start in range(0, len(majority), chunk):
        distances = cdist(X[majority[start : start + chunk]], X[minority])
        farthest = np.partition(distances, -n_neighbors, axis=1)[:, -n_neighbors:]
        scores[start : start + chunk] = farthest.sum(axis=1)
    keep = majority[np.argsort(scores, kind="stable")[: len(minority)]]
    return np.sort(np.concatenate([minority, keep]))


def _nearmiss_rows(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Row ids NearMiss-2 keeps out of ``rows`` (1:1), imputing with their medians."""
    block = frame.iloc[rows]
    filled = block.fillna(block.median(numeric_only=True)).to_numpy(dtype=np.float64)
    return rows[nearmiss2(filled, y[rows])]


def _fit_score(model_name: str, frame, y, train_rows, val_rows, test_rows, seed: int) -> float:
    from src.paper2_credit.tabnet_gated import GMATabNetClassifier, TabNetConfig
    from src.paper2_credit.run_experiments import finalise

    preprocessor = CreditPreprocessor("give_me_some_credit").fit(frame.iloc[train_rows])
    X_train, X_val, X_test = (preprocessor.transform_ordinal(frame.iloc[r])
                              for r in (train_rows, val_rows, test_rows))
    if model_name == "xgboost":
        model = build_baseline("xgboost", seed=seed)
        model.fit(X_train, y[train_rows])
        return float(roc_auc_score(y[test_rows], model.predict_proba(X_test)[:, 1]))
    config = finalise(TabNetConfig(seed=seed))
    model = run_resilient(lambda: GMATabNetClassifier(config).fit(
        X_train, y[train_rows], X_val, y[val_rows], preprocessor.cat_idxs, preprocessor.cat_dims))
    score = float(roc_auc_score(y[test_rows], model.predict_proba(X_test)[:, 1]))
    del model
    release_memory()
    return score


def run_protocol(dataset, protocol: str, seed: int) -> dict[str, float]:
    y = dataset.y
    frame = dataset.X.drop(columns=DROPPED) if protocol == "wang" else dataset.X
    all_rows = np.arange(len(y))

    if protocol == "wang":
        pool = _nearmiss_rows(frame, y, all_rows)
        train_full, test = train_test_split(pool, test_size=0.2, stratify=y[pool], random_state=seed)
    else:
        train_full, test = train_test_split(all_rows, test_size=0.2, stratify=y, random_state=seed)
        if protocol == "train_only":
            train_full = _nearmiss_rows(frame, y, np.sort(train_full))
    train, val = train_test_split(train_full, test_size=0.15, stratify=y[train_full],
                                  random_state=seed)
    result = {"n_train": int(len(train)), "n_test": int(len(test)),
              "test_positive_rate": float(y[test].mean())}
    for model_name in MODELS:
        set_seed(seed)
        result[model_name] = _fit_score(model_name, frame, y, np.sort(train), np.sort(val),
                                        np.sort(test), seed)
    return result


def stage_protocol_sensitivity(dataset, args) -> None:
    from src.paper2_credit.run_experiments import SMOKE

    target = results_root() / "paper2" / "protocol" / f"{dataset.key}__protocol_sensitivity.json"
    if target.exists() and not args.force:
        return
    seeds = SEEDS[:1] if SMOKE else SEEDS
    records = []
    for protocol in PROTOCOLS:
        for seed in seeds:
            started = time.perf_counter()
            LOGGER.info("%s | protocol | %s | seed %d", dataset.key, protocol, seed)
            outcome = run_protocol(dataset, protocol, seed)
            outcome.update(protocol=protocol, seed=seed, seconds=time.perf_counter() - started)
            LOGGER.info("  xgboost %.4f  gma %.4f", outcome["xgboost"], outcome["gma"])
            records.append(outcome)
    write_json(target, {"dataset": dataset.key, "dropped_in_wang": DROPPED, "records": records,
                        "reported_by_wang_2025": {"improved_tabnet_auc": 0.941,
                                                  "xgboost_auc": 0.796}})
