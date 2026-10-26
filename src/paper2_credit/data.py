"""Preprocessing for the three credit scoring datasets.

The three differ enough that a single transformer will not do: German Credit is
mostly categorical, Give Me Some Credit is numeric with missing income, and the
Taiwan dataset encodes repayment history as small integer codes that are ordinal
in name but categorical in behaviour.

Two views of each dataset are produced from the same fitted state:

``ordinal``
    Categorical columns as integer codes, numerics left on their own scale.
    This is what TabNet and the gradient boosting models consume; TabNet learns
    an embedding per category and trees do not care about scale.
``dense``
    One-hot categoricals and standardised numerics, for the linear, kernel and
    plain neural baselines that need it.

Everything is fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.io import Dataset, load_dataset

CREDIT_DATASETS = ["uci_dccc", "german_credit", "give_me_some_credit"]

# Columns that look numeric but are really unordered codes. Treating PAY_* as
# continuous implies that "payment delayed 8 months" is eight times "delayed one
# month", which is not how the risk actually behaves.
FORCED_CATEGORICAL = {
    "uci_dccc": [
        "x2", "x3", "x4", "x6", "x7", "x8", "x9", "x10", "x11",
        "SEX", "EDUCATION", "MARRIAGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    ],
}

# An integer column with at most this many distinct training values is treated
# as categorical, as is any object or category column.
CATEGORICAL_MAX_UNIQUE = 15

# A categorical level is kept only if it occurs at least MIN_CATEGORY_COUNT
# times in the training rows, and at most MAX_CATEGORY_CARDINALITY levels (the
# most frequent) are kept per column. Everything else, and any level first seen
# at prediction time, shares one bucket.
MIN_CATEGORY_COUNT = 10
MAX_CATEGORY_CARDINALITY = 60

# Set to a list by scripts/check_leakage.py: every fit then records the row ids
# it saw, so the check can prove no test row was ever fitted on.
FIT_AUDIT: list | None = None

# The OpenML copy of the Taiwan dataset ships the columns as x1..x23. The names
# below are the ones the source publication uses, in the same order, and are
# used only for display so that an attribution table means something to a reader.
DISPLAY_NAMES = {
    "uci_dccc": {
        "x1": "LIMIT_BAL", "x2": "SEX", "x3": "EDUCATION", "x4": "MARRIAGE",
        "x5": "AGE", "x6": "PAY_0", "x7": "PAY_2", "x8": "PAY_3", "x9": "PAY_4",
        "x10": "PAY_5", "x11": "PAY_6", "x12": "BILL_AMT1", "x13": "BILL_AMT2",
        "x14": "BILL_AMT3", "x15": "BILL_AMT4", "x16": "BILL_AMT5",
        "x17": "BILL_AMT6", "x18": "PAY_AMT1", "x19": "PAY_AMT2",
        "x20": "PAY_AMT3", "x21": "PAY_AMT4", "x22": "PAY_AMT5",
        "x23": "PAY_AMT6",
    },
}


# Short labels for tables, where the Give Me Some Credit column names are too
# long to sit beside two columns of numbers. Each table that uses them carries
# the legend.
FEATURE_ALIASES = {
    "give_me_some_credit": {
        "RevolvingUtilizationOfUnsecuredLines": "RUUL",
        "age": "AGE",
        "NumberOfTime30-59DaysPastDueNotWorse": "PD30",
        "DebtRatio": "DR",
        "MonthlyIncome": "MI",
        "NumberOfOpenCreditLinesAndLoans": "NOCL",
        "NumberOfTimes90DaysLate": "PD90",
        "NumberRealEstateLoansOrLines": "NREL",
        "NumberOfTime60-89DaysPastDueNotWorse": "PD60",
        "NumberOfDependents": "NDEP",
    },
    "german_credit": {
        "checking_status": "CHK", "duration": "DUR", "credit_history": "HIST",
        "purpose": "PURP", "credit_amount": "AMT", "savings_status": "SAV",
        "employment": "EMP", "installment_commitment": "INST",
        "personal_status": "PERS", "other_parties": "OTHP",
        "residence_since": "RES", "property_magnitude": "PROP", "age": "AGE",
        "other_payment_plans": "OPP", "housing": "HOUS",
        "existing_credits": "NCRED", "job": "JOB", "num_dependents": "NDEP",
        "own_telephone": "TEL", "foreign_worker": "FORW",
    },
}


def display_name(dataset_key: str, feature: str) -> str:
    return DISPLAY_NAMES.get(dataset_key, {}).get(feature, feature)


def short_name(dataset_key: str, feature: str) -> str:
    """Table label: the alias if one exists, otherwise the display name."""
    return FEATURE_ALIASES.get(dataset_key, {}).get(feature, display_name(dataset_key, feature))


@dataclass
class CreditView:
    """One fitted representation of a dataset."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    cat_idxs: list[int] = field(default_factory=list)
    cat_dims: list[int] = field(default_factory=list)


class CreditPreprocessor:
    """Fit on the training rows, then emit both views for any partition."""

    def __init__(self, dataset_key: str) -> None:
        self.dataset_key = dataset_key
        self.categorical_: list[str] = []
        self.numeric_: list[str] = []
        self.categories_: dict[str, list] = {}
        self.medians_: dict[str, float] = {}
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame) -> "CreditPreprocessor":
        if FIT_AUDIT is not None:
            FIT_AUDIT.append(("preprocessor", self.dataset_key, np.asarray(X.index)))
        frame = X.copy()
        frame.columns = [str(c) for c in frame.columns]

        forced = set(FORCED_CATEGORICAL.get(self.dataset_key, []))
        for column in frame.columns:
            series = frame[column]
            is_object = isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object
            few_values = series.nunique(dropna=True) <= CATEGORICAL_MAX_UNIQUE
            if is_object or column in forced or (few_values and series.dtype.kind in "iu"):
                self.categorical_.append(column)
            else:
                self.numeric_.append(column)

        for column in self.categorical_:
            values = frame[column].astype(str).fillna("__missing__")
            counts = values.value_counts()
            # Rare levels are folded into one bucket so the embedding table stays
            # small and no category is represented by a handful of training rows.
            frequent = counts[counts >= MIN_CATEGORY_COUNT]
            self.categories_[column] = list(frequent.index[:MAX_CATEGORY_CARDINALITY])

        for column in self.numeric_:
            self.medians_[column] = float(frame[column].median())

        numeric_matrix = self._numeric_matrix(frame)
        self.means_ = numeric_matrix.mean(axis=0)
        self.stds_ = numeric_matrix.std(axis=0)
        self.stds_[self.stds_ < 1e-8] = 1.0
        return self

    # ------------------------------------------------------------------ #
    def _numeric_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.numeric_:
            return np.zeros((len(frame), 0), dtype=np.float32)
        columns = []
        for column in self.numeric_:
            series = pd.to_numeric(frame[column], errors="coerce")
            columns.append(series.fillna(self.medians_[column]).to_numpy(dtype=np.float32))
        return np.column_stack(columns)

    def _ordinal_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.categorical_:
            return np.zeros((len(frame), 0), dtype=np.float32)
        columns = []
        for column in self.categorical_:
            values = frame[column].astype(str).fillna("__missing__")
            lookup = {level: i for i, level in enumerate(self.categories_[column])}
            # Index len(levels) is the shared bucket for unseen or rare levels.
            codes = values.map(lookup).fillna(len(lookup)).to_numpy(dtype=np.float32)
            columns.append(codes)
        return np.column_stack(columns)

    def _onehot_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.categorical_:
            return np.zeros((len(frame), 0), dtype=np.float32)
        blocks = []
        for column in self.categorical_:
            values = frame[column].astype(str).fillna("__missing__")
            levels = self.categories_[column]
            block = np.zeros((len(frame), len(levels) + 1), dtype=np.float32)
            lookup = {level: i for i, level in enumerate(levels)}
            codes = values.map(lookup).fillna(len(levels)).to_numpy(dtype=int)
            block[np.arange(len(frame)), codes] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    # ------------------------------------------------------------------ #
    def transform_ordinal(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.copy()
        frame.columns = [str(c) for c in frame.columns]
        return np.hstack([self._ordinal_matrix(frame), self._numeric_matrix(frame)]).astype(
            np.float32
        )

    def transform_dense(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.copy()
        frame.columns = [str(c) for c in frame.columns]
        numeric = (self._numeric_matrix(frame) - self.means_) / self.stds_
        return np.hstack([self._onehot_matrix(frame), numeric]).astype(np.float32)

    # ------------------------------------------------------------------ #
    @property
    def ordinal_feature_names(self) -> list[str]:
        return list(self.categorical_) + list(self.numeric_)

    @property
    def cat_idxs(self) -> list[int]:
        return list(range(len(self.categorical_)))

    @property
    def cat_dims(self) -> list[int]:
        # +1 for the rare/unseen bucket.
        return [len(self.categories_[c]) + 1 for c in self.categorical_]


def build_views(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[CreditView, CreditView]:
    """Fit on the training slice and return the ordinal and dense views."""
    preprocessor = CreditPreprocessor(dataset.key).fit(dataset.X.iloc[train_idx])

    ordinal = CreditView(
        X_train=preprocessor.transform_ordinal(dataset.X.iloc[train_idx]),
        X_val=preprocessor.transform_ordinal(dataset.X.iloc[val_idx]),
        X_test=preprocessor.transform_ordinal(dataset.X.iloc[test_idx]),
        feature_names=preprocessor.ordinal_feature_names,
        cat_idxs=preprocessor.cat_idxs,
        cat_dims=preprocessor.cat_dims,
    )
    dense = CreditView(
        X_train=preprocessor.transform_dense(dataset.X.iloc[train_idx]),
        X_val=preprocessor.transform_dense(dataset.X.iloc[val_idx]),
        X_test=preprocessor.transform_dense(dataset.X.iloc[test_idx]),
        feature_names=[f"dense_{i}" for i in range(preprocessor.transform_dense(
            dataset.X.iloc[train_idx][:1]).shape[1])],
    )
    return ordinal, dense


def ordinal_arrays(dataset: Dataset, fit_rows: np.ndarray, *rows: np.ndarray):
    """Fit preprocessing on ``fit_rows`` and return the ordinal view of each set.

    Returns (preprocessor, [X for fit_rows, X for each of ``rows``]). The stack
    calls this inside every inner split so its preprocessing never sees the rows
    it produces out-of-fold predictions for.
    """
    preprocessor = CreditPreprocessor(dataset.key).fit(dataset.X.iloc[fit_rows])
    arrays = [preprocessor.transform_ordinal(dataset.X.iloc[r]) for r in (fit_rows, *rows)]
    return preprocessor, arrays


def load_credit_dataset(key: str) -> Dataset:
    if key not in CREDIT_DATASETS:
        raise KeyError(f"{key!r} is not a Paper 2 dataset; use one of {CREDIT_DATASETS}")
    return load_dataset(key)
