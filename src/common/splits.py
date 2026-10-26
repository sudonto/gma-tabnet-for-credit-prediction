"""Data splitting protocols.

Two regimes are supported:

* Time-ordered splitting for the fraud study, where the test set must consist of
  transactions that occur strictly after everything the model was fitted on.
  Random splitting on that dataset lets a model see the future, and combined
  with oversampling it is the main reason published ULB numbers are optimistic.
* Stratified k-fold for the credit scoring datasets, which carry no usable
  timestamp and are conventionally evaluated that way.

Both regimes return a train / validation / test triple. The validation portion
exists so that early stopping and the decision threshold are never chosen on
test data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class Split:
    """Index arrays for one train / validation / test partition.

    ``repeat`` and ``seed`` are used by repeated cross-validation, where every
    outer fold carries its own model seed; ``search_idx`` is the subset of
    ``train_idx`` a hyperparameter search may fit on.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    fold: int = 0
    name: str = ""
    repeat: int = 0
    seed: int = 0
    search_idx: np.ndarray | None = None

    @property
    def tag(self) -> str:
        return f"r{self.repeat}f{self.fold}"

    @property
    def outer_train_idx(self) -> np.ndarray:
        """Everything outside the test fold: training plus validation rows."""
        return np.sort(np.concatenate([self.train_idx, self.val_idx]))

    def hash(self) -> str:
        """Fingerprint of the partition, stored with every result it produced."""
        digest = hashlib.blake2b(digest_size=8)
        for part in (self.train_idx, self.val_idx, self.test_idx):
            digest.update(np.sort(np.asarray(part, dtype=np.int64)).tobytes())
            digest.update(b"|")
        return digest.hexdigest()

    def sizes(self) -> dict[str, int]:
        return {
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_test": len(self.test_idx),
        }


def time_ordered_split(
    time_values: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
) -> Split:
    """Single chronological cut: oldest -> train, then validation, then test."""
    order = np.argsort(np.asarray(time_values), kind="stable")
    n = len(order)
    n_test = int(round(n * test_size))
    n_remaining = n - n_test
    n_val = int(round(n_remaining * val_size))

    train_idx = order[: n_remaining - n_val]
    val_idx = order[n_remaining - n_val : n_remaining]
    test_idx = order[n_remaining:]
    return Split(train_idx, val_idx, test_idx, fold=0, name="time_ordered")


def rolling_origin_splits(
    time_values: np.ndarray,
    n_folds: int = 5,
    test_size: float = 0.1,
    val_size: float = 0.15,
    expanding: bool = True,
) -> list[Split]:
    """Rolling-origin evaluation: several chronological cuts over the stream.

    Fold i trains on everything before origin i and tests on the window that
    follows it, so each fold answers "how well would this model have worked had
    it been deployed at that moment".
    """
    order = np.argsort(np.asarray(time_values), kind="stable")
    n = len(order)
    window = int(round(n * test_size))
    if window < 1:
        raise ValueError("test_size too small for this dataset")

    # Leave enough history for the earliest fold to have a usable training set.
    first_origin = n - n_folds * window
    if first_origin < window:
        raise ValueError(
            f"cannot build {n_folds} folds of {window} samples from {n} rows"
        )

    splits: list[Split] = []
    for fold in range(n_folds):
        origin = first_origin + fold * window
        history = order[:origin] if expanding else order[max(0, origin - first_origin) : origin]
        n_val = int(round(len(history) * val_size))
        splits.append(
            Split(
                train_idx=history[: len(history) - n_val],
                val_idx=history[len(history) - n_val :],
                test_idx=order[origin : origin + window],
                fold=fold,
                name=f"rolling_origin_fold{fold}",
            )
        )
    return splits


def stratified_splits(
    y: np.ndarray,
    n_folds: int = 5,
    val_size: float = 0.15,
    seed: int = 0,
) -> list[Split]:
    """Stratified k-fold, with a stratified validation slice cut from each train fold."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: list[Split] = []
    for fold, (train_full, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        train_idx, val_idx = train_test_split(
            train_full,
            test_size=val_size,
            stratify=y[train_full],
            random_state=seed,
        )
        splits.append(
            Split(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold=fold,
                name=f"stratified_fold{fold}",
            )
        )
    return splits


def model_seed(repeat: int, fold: int) -> int:
    """Seed for everything trained in outer fold (repeat, fold)."""
    return 1000 * repeat + fold


def repeated_stratified_splits(
    y: np.ndarray,
    n_folds: int = 5,
    n_repeats: int = 3,
    val_size: float = 0.15,
    search_rows: int | None = None,
) -> list[Split]:
    """Repeated stratified k-fold with a nested validation block.

    Repeat ``r`` reshuffles with ``random_state=r``. Inside each outer training
    partition a stratified validation block (``val_size``) is carved off for
    early stopping, threshold choice and hyperparameter selection, so the test
    fold is never used for any decision. ``search_idx`` is a stratified subset
    of the remaining training rows, at most ``search_rows`` of them, that a
    hyperparameter search fits on.
    """
    y = np.asarray(y)
    splits: list[Split] = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat)
        for fold, (train_full, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            train_idx, val_idx = train_test_split(
                train_full, test_size=val_size, stratify=y[train_full],
                random_state=100 * repeat + fold,
            )
            search_idx = train_idx
            if search_rows is not None and len(train_idx) > search_rows:
                search_idx, _ = train_test_split(
                    train_idx, train_size=search_rows, stratify=y[train_idx],
                    random_state=10_000 + 100 * repeat + fold,
                )
            splits.append(
                Split(
                    train_idx=np.sort(train_idx), val_idx=np.sort(val_idx),
                    test_idx=np.sort(test_idx), fold=fold, repeat=repeat,
                    seed=model_seed(repeat, fold), search_idx=np.sort(search_idx),
                    name=f"repeated_stratified_r{repeat}f{fold}",
                )
            )
    return splits


def inner_folds(y: np.ndarray, rows: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified inner folds over ``rows`` (original row ids), for stacking."""
    y = np.asarray(y)
    rows = np.asarray(rows)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(rows[a], rows[b]) for a, b in skf.split(np.zeros(len(rows)), y[rows])]


def leaky_random_split(
    y: np.ndarray, test_size: float = 0.2, val_size: float = 0.15, seed: int = 0
) -> Split:
    """A deliberately naive random split.

    Used only as the control arm of the leakage ablation in Paper 1, to quantify
    how much of the headline performance in the literature comes from ignoring
    the temporal ordering.
    """
    idx = np.arange(len(y))
    train_full, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_full, test_size=val_size, stratify=y[train_full], random_state=seed
    )
    return Split(train_idx, val_idx, test_idx, fold=0, name="leaky_random")
