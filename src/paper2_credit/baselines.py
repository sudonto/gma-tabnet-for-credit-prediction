"""Baseline models for the credit scoring study.

Each entry declares which of the two data views it needs. Tree models take the
ordinal view, where categories are integer codes and numerics keep their own
scale; distance and gradient based models take the one-hot, standardised view.
Getting this wrong is a common source of unfairly weak baselines, so the choice
is explicit rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# name -> which view of the data the model should be fed
BASELINE_VIEWS = {
    "logistic_regression": "dense",
    "decision_tree": "ordinal",
    "random_forest": "ordinal",
    "knn": "dense",
    "svm": "dense",
    "xgboost": "ordinal",
    "lightgbm": "ordinal",
    "catboost": "ordinal",
    "mlp": "dense",
    "tabnet_vanilla": "ordinal",
    "tabnet_reference": "ordinal",
}

SKLEARN_LIKE = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
    "svm",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
]

# Kernel SVM and KNN are run on a stratified subsample above this many training
# rows, and the paper says so, rather than being dropped from the comparison.
# Kernel SVM fits in time quadratic in the training rows, and asking it for
# calibrated probabilities multiplies that by an internal five-fold Platt fit.
# At 20k rows one fold took over ten minutes; at 8k the whole sweep is minutes,
# and the learning curve for these datasets is flat well before that. KNN is
# capped at the same size so both instance-based baselines see the same data.
SUBSAMPLE_ABOVE = 8_000


def tabnet_batch_sizes(n_train: int) -> tuple[int, int]:
    """Batch and virtual batch size scaled to the training set.

    A fixed batch of 4096 is fine on the two larger datasets but exceeds the
    entire training partition of German Credit, which has 680 rows. That gives
    one gradient step per epoch, and TabNet converges to nothing: it scored an
    AUC of 0.49 before this was fixed, which would have looked like a crippled
    baseline rather than a misconfigured one. Aiming for roughly ten steps per
    epoch keeps the optimisation comparable across datasets of very different
    sizes, and ghost batch normalisation needs the virtual batch to divide the
    batch and to hold enough rows for its statistics to mean anything.
    """
    batch_size = int(min(4096, max(128, 2 ** int(np.log2(max(n_train // 10, 128))))))
    virtual_batch_size = int(max(32, min(256, batch_size // 4)))
    return batch_size, virtual_batch_size


@dataclass
class BaselineSpec:
    name: str
    view: str
    needs_subsample: bool = False


def baseline_spec(name: str, n_train: int) -> BaselineSpec:
    return BaselineSpec(
        name=name,
        view=BASELINE_VIEWS[name],
        needs_subsample=name in {"svm", "knn"} and n_train > SUBSAMPLE_ABOVE,
    )


def build_baseline(name: str, seed: int = 0, class_weight: bool = True,
                   params: dict | None = None):
    """Instantiate a baseline with settings typical of the credit scoring literature.

    ``params`` overrides the defaults, which is how a tuned configuration is
    applied.
    """
    model = _default_baseline(name, seed, class_weight)
    if params:
        model.set_params(**params)
    return model


def _default_baseline(name: str, seed: int, class_weight: bool):
    weight = "balanced" if class_weight else None

    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=3000, C=1.0, class_weight=weight, random_state=seed)
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight=weight, random_state=seed
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample" if class_weight else None,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    if name == "svm":
        from sklearn.svm import SVC

        return SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True,
            class_weight=weight, random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight=weight, n_jobs=-1, verbose=-1, random_state=seed,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            auto_class_weights="Balanced" if class_weight else None,
            verbose=0, allow_writing_files=False, random_seed=seed,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        # "auto" batching and a generous patience matter on German Credit, where
        # a fixed 256-row batch leaves only three updates per epoch and the
        # network stops before it has learned anything.
        return MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
            batch_size="auto", learning_rate_init=1e-3, max_iter=800,
            early_stopping=True, n_iter_no_change=30, validation_fraction=0.15,
            random_state=seed,
        )
    raise ValueError(f"unknown baseline {name!r}")


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_max: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the class ratio while capping the training size."""
    if len(y) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        take = max(1, int(round(n_max * len(idx) / len(y))))
        keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    selected = np.sort(np.concatenate(keep))
    return X[selected], y[selected]


def build_scaled(name: str, seed: int = 0, params: dict | None = None):
    """A distance- or kernel-based learner behind a standard scaler.

    Inside the stack the level-one learners read the ordinal view plus the
    extractor's embedding, whose columns sit on unrelated scales. Trees do not
    care; k-NN and an RBF kernel do, and fed unscaled inputs they score far below
    their standalone selves. The scaler is fitted with the learner, on the same
    rows.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), build_baseline(name, seed=seed, params=params))


# --------------------------------------------------------------------------- #
# Tuned baselines
# --------------------------------------------------------------------------- #
# Trials per tuned baseline. A fixed count rather than a wall-clock budget: the
# baselines run on the CPU beside the GPU stages, and a timer would make their
# tuning depend on what else the machine was doing.
TUNING_TRIALS = {"xgboost": 60, "lightgbm": 60, "random_forest": 40, "svm": 30}
TUNABLE = tuple(TUNING_TRIALS)


def _suggest(trial, name: str) -> dict:
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    raise ValueError(f"{name!r} is not a tunable baseline")


def tune_baseline(name, X_train, y_train, X_val, y_val, seed: int = 0) -> dict:
    """TPE over a fixed number of trials, scored by validation ROC-AUC.

    The SVM is searched on its decision function without Platt scaling, which
    leaves the ranking unchanged and avoids an internal five-fold refit per
    trial; the selected configuration is refitted with probabilities.
    """
    import optuna
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = _suggest(trial, name)
        if name == "svm":
            model = build_baseline("svm", seed=seed, params={**params, "probability": False})
            model.fit(X_train, y_train)
            return float(roc_auc_score(y_val, model.decision_function(X_val)))
        model = build_baseline(name, seed=seed, params=params)
        model.fit(X_train, y_train)
        return float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=TUNING_TRIALS[name])
    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "history": [{"value": t.value, "params": t.params} for t in study.trials],
    }


# --------------------------------------------------------------------------- #
# Reference TabNet
# --------------------------------------------------------------------------- #
def build_tabnet_reference(seed: int = 0, device: str = "auto", config=None,
                           cat_idxs=None, cat_dims=None):
    """The published pytorch-tabnet implementation.

    With no ``config`` this is the setup of the first submission: library
    defaults for width, steps and relaxation, learning rate 0.02, and no
    categorical embeddings, so category codes enter as numbers. With a
    ``config`` it takes those hyperparameters, and with ``cat_idxs`` it embeds
    categories exactly as our network does; the gap decomposition walks from
    the first setup to the second.
    """
    from pytorch_tabnet.tab_model import TabNetClassifier

    from src.paper2_credit.tabnet_gated import embedding_dims

    params = dict(n_d=16, n_a=16, n_steps=4, gamma=1.4, lambda_sparse=1e-3,
                  optimizer_params={"lr": 2e-2})
    if config is not None:
        params = dict(
            n_d=config.n_d, n_a=config.n_a, n_steps=config.n_steps, gamma=config.gamma,
            n_independent=config.n_independent, n_shared=config.n_shared,
            lambda_sparse=config.lambda_sparse, momentum=config.momentum,
            optimizer_params={"lr": config.learning_rate},
        )
    if cat_idxs:
        width = config.cat_emb_dim if config is not None else 8
        params.update(cat_idxs=list(cat_idxs), cat_dims=list(cat_dims),
                      cat_emb_dim=embedding_dims(list(cat_dims), width))
    return TabNetClassifier(**params, seed=seed, verbose=0, device_name=device)
