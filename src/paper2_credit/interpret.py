"""Attributions for Paper 2, separated by model and by method.

Comparing TreeSHAP on a boosted model with attention masks on GMA-TabNet changes
two things at once, the model and the explanation method, so their agreement or
disagreement cannot be pinned on either. Here every model carries a
model-agnostic reference computed on the same model:

GMA-TabNet (the fold's ``gma_full`` checkpoint, nothing retrained)
    decision-weighted attention masks (TabNet's own explanation), integrated
    gradients of the logit, and permutation importance (drop in ROC-AUC).
XGBoost (the default baseline configuration, refitted on the fold's training rows)
    TreeSHAP and permutation importance.
The full GMA stack (computed in the stacking stage, where the stack is in memory)
    permutation importance of the raw columns on the stacked probability.

Method agreement is measured within a model, model agreement with the same method
(permutation) across models. Each attribution is computed on up to 4,000 test
rows of each repeat-0 fold, so stability across the five folds can be measured,
and again on a perturbed copy of those rows (numeric columns plus Gaussian noise
of 0.05 training standard deviations, 5% of categorical codes replaced by a
random training code). Per-applicant reasons are the three columns with the
largest attribution towards default for applicants in the top decile of
predicted risk.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.common.io import results_root, write_json
from src.common.seeds import set_seed

LOGGER = logging.getLogger("paper2.interpret")

PERMUTATION_REPEATS = 3
TOP_K = 3


def output_dir():
    return results_root() / "paper2" / "interpretability"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def share(values: np.ndarray) -> np.ndarray:
    """Normalise non-negative global importances to shares summing to one."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = values.sum()
    return values / total if total > 0 else np.zeros_like(values)


def explain_rows(split, n_rows: int) -> np.ndarray:
    """Positions within the test fold of the rows to explain (a fixed random subset)."""
    rng = np.random.default_rng(split.seed)
    n = len(split.test_idx)
    return np.sort(rng.choice(n, size=min(n_rows, n), replace=False))


def permutation_importance(score_fn, X, y, seed: int, repeats: int = PERMUTATION_REPEATS,
                           permute=None) -> np.ndarray:
    """Mean drop in ROC-AUC when one column is shuffled among the explained rows."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, score_fn(X))
    n_columns = X.shape[1]
    drops = np.zeros(n_columns)
    for column in range(n_columns):
        total = 0.0
        for _ in range(repeats):
            shuffled = permute(X, column, rng) if permute else _permute_array(X, column, rng)
            total += base - roc_auc_score(y, score_fn(shuffled))
        drops[column] = total / repeats
    return drops


def _permute_array(X, column, rng):
    out = X.copy()
    out[:, column] = rng.permutation(out[:, column])
    return out


def perturb(X: np.ndarray, X_train: np.ndarray, cat_idxs: list[int], seed: int) -> np.ndarray:
    """Small, label-free input noise for the stability check."""
    rng = np.random.default_rng(seed + 7)
    out = X.copy()
    categorical = set(cat_idxs)
    for column in range(X.shape[1]):
        if column in categorical:
            swap = rng.random(len(out)) < 0.05
            out[swap, column] = rng.choice(X_train[:, column], size=int(swap.sum()))
        else:
            sd = float(np.std(X_train[:, column])) or 1.0
            out[:, column] = out[:, column] + rng.normal(0.0, 0.05 * sd, size=len(out))
    return out.astype(np.float32)


def top_k(per_sample: np.ndarray, k: int = TOP_K) -> np.ndarray:
    return np.argsort(-per_sample, axis=1)[:, :k]


def jaccard_rows(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Jaccard similarity between the per-row index sets in ``a`` and ``b``."""
    scores = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]
    return float(np.mean(scores)) if scores else float("nan")


def reference_row(X_train: np.ndarray, cat_idxs: list[int]) -> np.ndarray:
    """Baseline applicant for integrated gradients: medians, and modal categories."""
    row = np.median(X_train, axis=0)
    for column in cat_idxs:
        values, counts = np.unique(X_train[:, column], return_counts=True)
        row[column] = values[np.argmax(counts)]
    return row.astype(np.float32)


def _xgboost(X_train, y_train, seed):
    from src.paper2_credit.baselines import build_baseline

    model = build_baseline("xgboost", seed=seed)
    model.fit(X_train, y_train)
    return model


def _tree_shap(model, X) -> np.ndarray:
    """Per-applicant TreeSHAP values (log-odds), through XGBoost's own pred_contribs."""
    import xgboost as xgb

    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contributions[:, :-1]          # the last column is the base value


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def explain_fold(dataset, split, n_rows: int) -> dict | None:
    from src.paper2_credit.run_experiments import checkpoint_path, prepare_fold
    from src.paper2_credit.tabnet_gated import GMATabNetClassifier

    path = checkpoint_path(dataset.key, "gma_full", split)
    if not path.exists():
        LOGGER.warning("%s %s: no gma_full checkpoint, skipping", dataset.key, split.tag)
        return None
    set_seed(split.seed)
    fold = prepare_fold(dataset, split)
    view = fold["ordinal"]
    positions = explain_rows(split, n_rows)
    X, y = view.X_test[positions], fold["y_test"][positions]
    X_noisy = perturb(X, view.X_train, view.cat_idxs, split.seed)
    features = list(view.feature_names)

    gma = GMATabNetClassifier.load(path)
    gma_score = lambda Z: gma.predict_proba(Z)[:, 1]          # noqa: E731
    mask, mask_mass = gma.explain(X)
    mask_noisy, _ = gma.explain(X_noisy)
    baseline = reference_row(view.X_train, view.cat_idxs)
    ig = gma.integrated_gradients(X, baseline)
    ig_noisy = gma.integrated_gradients(X_noisy, baseline)
    gma_perm = permutation_importance(gma_score, X, y, split.seed)

    xgb_model = _xgboost(view.X_train, fold["y_train"], split.seed)
    shap = _tree_shap(xgb_model, X)
    shap_noisy = _tree_shap(xgb_model, X_noisy)
    xgb_perm = permutation_importance(lambda Z: xgb_model.predict_proba(Z)[:, 1], X, y, split.seed)

    methods = {
        "gma_mask": share(mask.mean(axis=0)),
        "gma_mask_unweighted": share(mask_mass.mean(axis=0)),
        "gma_ig": share(np.abs(ig).mean(axis=0)),
        "gma_permutation": share(gma_perm),
        "xgb_shap": share(np.abs(shap).mean(axis=0)),
        "xgb_permutation": share(xgb_perm),
    }
    noisy = {
        "gma_mask": share(mask_noisy.mean(axis=0)),
        "gma_ig": share(np.abs(ig_noisy).mean(axis=0)),
        "xgb_shap": share(np.abs(shap_noisy).mean(axis=0)),
    }

    # Reasons: the columns pushing hardest towards default, for the riskiest
    # tenth of applicants by each model's own score. Masks carry no sign, so
    # their "reasons" are simply the most attended columns.
    risky_gma = np.argsort(-gma_score(X))[: max(1, len(X) // 10)]
    risky_xgb = np.argsort(-xgb_model.predict_proba(X)[:, 1])[: max(1, len(X) // 10)]
    reasons = {
        "gma_mask": top_k(mask[risky_gma]),
        "gma_ig": top_k(ig[risky_gma]),
        "xgb_shap": top_k(shap[risky_xgb]),
    }
    reasons_noisy = {
        "gma_mask": top_k(mask_noisy[risky_gma]),
        "gma_ig": top_k(ig_noisy[risky_gma]),
        "xgb_shap": top_k(shap_noisy[risky_xgb]),
    }
    shared = np.intersect1d(risky_gma, risky_xgb)

    def global_top(name):
        return int(np.argmax(methods[name]))

    payload = {
        "dataset": dataset.key, "split": split.tag, "seed": split.seed,
        "n_rows": int(len(X)), "features": features,
        "global": {k: v.tolist() for k, v in methods.items()},
        "global_perturbed": {k: v.tolist() for k, v in noisy.items()},
        "perturbation_spearman": {
            k: float(spearmanr(methods[k], noisy[k]).statistic) for k in noisy
        },
        "reason_stability_jaccard": {
            k: jaccard_rows(reasons[k], reasons_noisy[k]) for k in reasons
        },
        "reason_agreement_jaccard": {
            "gma_mask~gma_ig": jaccard_rows(reasons["gma_mask"], reasons["gma_ig"]),
            "gma_ig~xgb_shap": jaccard_rows(
                top_k(ig[shared]), top_k(shap[shared])) if len(shared) else float("nan"),
            "gma_mask~xgb_shap": jaccard_rows(
                top_k(mask[shared]), top_k(shap[shared])) if len(shared) else float("nan"),
        },
        "n_shared_risky": int(len(shared)),
        "top1_matches_global": {
            "gma_mask": float(np.mean(reasons["gma_mask"][:, 0] == global_top("gma_mask"))),
            "gma_ig": float(np.mean(reasons["gma_ig"][:, 0] == global_top("gma_ig"))),
            "xgb_shap": float(np.mean(reasons["xgb_shap"][:, 0] == global_top("xgb_shap"))),
        },
        "explained_auc": {"gma": float(roc_auc_score(y, gma_score(X))),
                          "xgb": float(roc_auc_score(y, xgb_model.predict_proba(X)[:, 1]))},
    }
    write_json(output_dir() / f"{dataset.key}__attribution__{split.tag}.json", payload)
    return payload


def stage_interpret(dataset, splits, args) -> None:
    from src.paper2_credit.run_experiments import EXPLAIN_ROWS, SMOKE

    n_rows = EXPLAIN_ROWS // (8 if SMOKE else 1)
    for split in splits:
        target = output_dir() / f"{dataset.key}__attribution__{split.tag}.json"
        if target.exists() and not args.force:
            continue
        LOGGER.info("%s | interpret | %s", dataset.key, split.tag)
        explain_fold(dataset, split, n_rows)


def stage_interpret_token(dataset, splits, args) -> None:
    """Attributions for the token-attention TabNet on the repeat-0 folds.

    Permutation importance on the same explained rows as ``explain_fold``, so it
    can be compared with XGBoost's and GMA-TabNet's, plus the encoder's own
    decision-weighted masks. Integrated gradients are not used here: the token
    branch reads the raw input, which the embedded-space path does not reach.
    """
    from src.paper2_credit.run_experiments import EXPLAIN_ROWS, SMOKE, checkpoint_path, prepare_fold
    from src.paper2_credit.tabnet_gated import GMATabNetClassifier

    n_rows = EXPLAIN_ROWS // (8 if SMOKE else 1)
    for split in splits:
        target = output_dir() / f"{dataset.key}__attribution_token__{split.tag}.json"
        path = checkpoint_path(dataset.key, "token_full", split)
        if (target.exists() and not args.force) or not path.exists():
            continue
        LOGGER.info("%s | interpret token | %s", dataset.key, split.tag)
        set_seed(split.seed)
        fold = prepare_fold(dataset, split)
        view = fold["ordinal"]
        positions = explain_rows(split, n_rows)
        X, y = view.X_test[positions], fold["y_test"][positions]
        model = GMATabNetClassifier.load(path)
        score = lambda Z: model.predict_proba(Z)[:, 1]          # noqa: E731
        drops = permutation_importance(score, X, y, split.seed)
        noisy = permutation_importance(score, perturb(X, view.X_train, view.cat_idxs, split.seed), y,
                                       split.seed)
        mask, _ = model.explain(X)
        write_json(target, {
            "dataset": dataset.key, "split": split.tag, "features": list(view.feature_names),
            "token_permutation": share(drops).tolist(),
            "token_permutation_perturbed": share(noisy).tolist(),
            "token_mask": share(mask.mean(axis=0)).tolist(),
            "explained_auc": float(roc_auc_score(y, score(X))),
        })


def stack_permutation_importance(dataset, stack, split, n_rows: int) -> None:
    """Permutation importance of each raw column on the stacked probability."""
    positions = explain_rows(split, n_rows)
    rows = split.test_idx[positions]
    frame = dataset.X.iloc[rows].reset_index(drop=True)
    y = dataset.y[rows]
    columns = list(frame.columns)

    def permute(current, column, rng):
        out = current.copy()
        name = columns[column]
        out[name] = rng.permutation(out[name].to_numpy())
        return out

    drops = np.zeros(len(columns))
    rng = np.random.default_rng(split.seed)
    base = roc_auc_score(y, stack.score_frame(frame))
    for column in range(len(columns)):
        total = 0.0
        for _ in range(PERMUTATION_REPEATS):
            total += base - roc_auc_score(y, stack.score_frame(permute(frame, column, rng)))
        drops[column] = total / PERMUTATION_REPEATS
    write_json(output_dir() / f"{dataset.key}__stack_permutation__{split.tag}.json", {
        "dataset": dataset.key, "split": split.tag, "features": columns,
        "auc": float(base), "drop": drops.tolist(), "share": share(drops).tolist(),
    })
