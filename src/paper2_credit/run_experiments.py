"""Experiment driver for Paper 2, nested and repeated.

Protocol
--------
Every dataset is evaluated by repeated stratified cross-validation: three
repeats of five folds, fifteen outer folds in all. In each outer fold a
stratified validation block is carved from the training partition. Every
decision that is not the final measurement is made inside the outer training
partition: preprocessing is fitted on the training rows, early stopping and
the decision threshold use the validation block, and the hyperparameter search
fits on (a subset of) the training rows and scores on the validation block.
The test fold is touched once, to score the finished model. There is no
configuration shared between folds.

Stages
------
export_splits         write every outer, inner and search index set to disk
baselines             the ten untuned baselines and the reference TabNet setup
                      of the first submission, on all fifteen folds
baselines_tuned       XGBoost, LightGBM, random forest and SVM tuned by TPE on
                      the validation block of each fold (CPU)
hpo_select            per outer fold, a TPE search for GMA-TabNet and one for
                      vanilla TabNet under the same wall-clock budget; the
                      winner of each is that fold's configuration
hpo_study             repeat 0 only: random search, PSO and a genetic algorithm
                      for GMA-TabNet under the same budget, each selected
                      configuration retrained and scored on the test fold
architecture          fixed-configuration mechanism ablation, equal-tuning
                      comparison, capacity and reproduction controls, and the
                      reference-to-ours TabNet gap decomposition
stacking              nested and shared stacks, with and without extractor
interpret             attributions on the repeat-0 folds
protocol_sensitivity  Give Me Some Credit under the protocol of Wang and Zhang
                      (2025) against ours

    python -m src.paper2_credit.run_experiments --stage all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.io import load_dataset, load_scores, results_root, safe_name, save_scores, write_json
from src.common.metrics import best_f1_threshold, evaluate
from src.common.seeds import set_seed
from src.common.splits import Split, inner_folds, repeated_stratified_splits
from src.common.torch_utils import release_memory, run_resilient
from src.paper2_credit import baselines as baseline_module
from src.paper2_credit import data as credit_data
from src.paper2_credit.baselines import (
    BASELINE_VIEWS,
    SKLEARN_LIKE,
    SUBSAMPLE_ABOVE,
    TUNABLE,
    build_baseline,
    build_tabnet_reference,
    stratified_subsample,
    tabnet_batch_sizes,
    tune_baseline,
)
from src.paper2_credit.data import CREDIT_DATASETS, build_views
from src.paper2_credit.hpo import BudgetExhausted, optimiser_seed, run_optimiser, tabnet_search_space
from src.paper2_credit.stacking import HeterogeneousStack, StackConfig
from src.paper2_credit.tabnet_gated import (
    GMATabNetClassifier,
    TabNetConfig,
    match_capacity,
)

LOGGER = logging.getLogger("paper2")

N_FOLDS = 5
N_REPEATS = 3
VAL_SIZE = 0.15

# Wall-clock budget per search, identical for every optimiser and for both
# architectures on a dataset. German Credit trains in seconds, so 600 s buys it
# as many evaluations as 1,200 s buys the two larger datasets.
BUDGETS = {"german_credit": 600.0, "uci_dccc": 1200.0, "give_me_some_credit": 1200.0}

# The search fits on at most this many training rows and on a shortened
# schedule; the configuration it selects is retrained on the full training
# rows with the full schedule, so nothing reported is fitted on the subset.
SEARCH_ROWS = 15_000
SEARCH_EPOCHS, SEARCH_PATIENCE = 25, 5

# Inner folds of the stack. Three on Give Me Some Credit, where every inner
# fold refits the extractor on 80,000 rows.
STACK_INNER_FOLDS = {"give_me_some_credit": 3}
DEFAULT_INNER_FOLDS = 5

# Rows of each test fold that attributions are computed on.
EXPLAIN_ROWS = 4_000

ARCHITECTURES = ("gma", "vanilla", "token")
# What each searched architecture is, as TabNetConfig fields. "token" is
# vanilla TabNet with feature-token attention beside the encoder.
ARCH_BASE = {
    "gma": {"variant": "gma"},
    "vanilla": {"variant": "vanilla"},
    "token": {"variant": "vanilla", "use_tokens": True},
}
STUDY_OPTIMISERS = ("random", "pso", "ga")

# Set by --smoke: a few epochs and trials, for a dry run of the whole pipeline.
SMOKE = False


# --------------------------------------------------------------------------- #
# Paths and records
# --------------------------------------------------------------------------- #
def paper_dir() -> Path:
    return results_root() / "paper2"


def runs_dir() -> Path:
    return paper_dir() / "raw_runs"


def tuning_dir() -> Path:
    return paper_dir() / "tuning"


def result_path(dataset_key: str, stage: str, arm: str, split: Split) -> Path:
    return runs_dir() / f"{dataset_key}__{stage}__{safe_name(arm)}__{split.tag}__seed{split.seed}.json"


def checkpoint_path(dataset_key: str, arm: str, split: Split) -> Path:
    return paper_dir() / "checkpoints" / dataset_key / split.tag / f"{safe_name(arm)}.pt"


def already_done(dataset_key: str, stage: str, arm: str, split: Split, force: bool) -> bool:
    """True only when both the metrics and the predictions behind them exist."""
    if force or not result_path(dataset_key, stage, arm, split).exists():
        return False
    return load_scores(2, f"{dataset_key}__{stage}", arm, split.tag, split.seed) is not None


def store(dataset_key, stage, arm, split: Split, metrics, extra, y_true, scores) -> None:
    """Write the summary record and the per-applicant test predictions behind it."""
    payload = {
        "paper": 2, "dataset": dataset_key, "stage": stage, "arm": arm,
        "repeat": split.repeat, "fold": split.fold, "seed": split.seed,
        "split_hash": split.hash(), "n_train": len(split.train_idx),
        "n_val": len(split.val_idx), "n_test": len(split.test_idx),
        "metrics": metrics,
    }
    payload.update(extra or {})
    write_json(result_path(dataset_key, stage, arm, split), payload)
    save_scores(2, f"{dataset_key}__{stage}", arm, split.tag, split.seed, y_true, scores,
                index=split.test_idx)
    release_memory()


def scored(y_val, val_scores, y_test, test_scores) -> tuple[dict, np.ndarray]:
    """Test metrics at the validation-selected threshold, plus the test scores."""
    threshold = best_f1_threshold(y_val, val_scores)
    return evaluate(y_test, test_scores, threshold=threshold), test_scores


def finalise(config: TabNetConfig) -> TabNetConfig:
    """Apply the smoke-run schedule cap, if any."""
    if SMOKE:
        return replace(config, max_epochs=min(config.max_epochs, 4),
                       patience=min(config.patience, 2),
                       pretrain_epochs=min(config.pretrain_epochs, 1))
    return config


def audit(kind: str, dataset_key: str, rows: np.ndarray) -> None:
    if credit_data.FIT_AUDIT is not None:
        credit_data.FIT_AUDIT.append((kind, dataset_key, np.asarray(rows)))


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def make_splits(dataset, args) -> list[Split]:
    splits = repeated_stratified_splits(
        dataset.y, n_folds=args.folds, n_repeats=args.repeats, val_size=VAL_SIZE,
        search_rows=args.search_rows,
    )
    if args.only_repeat is not None:
        splits = [s for s in splits if s.repeat in args.only_repeat]
    if args.only_fold is not None:
        splits = [s for s in splits if s.fold in args.only_fold]
    return splits


def prepare_fold(dataset, split: Split) -> dict:
    audit("preprocess_fit", dataset.key, split.train_idx)
    ordinal, dense = build_views(dataset, split.train_idx, split.val_idx, split.test_idx)
    return {
        "ordinal": ordinal, "dense": dense,
        "y_train": dataset.y[split.train_idx],
        "y_val": dataset.y[split.val_idx],
        "y_test": dataset.y[split.test_idx],
    }


def view_arrays(fold: dict, view: str):
    v = fold[view]
    return v.X_train, v.X_val, v.X_test


def inner_fold_count(dataset_key: str) -> int:
    return 2 if SMOKE else STACK_INNER_FOLDS.get(dataset_key, DEFAULT_INNER_FOLDS)


def stage_export_splits(dataset, splits, args) -> None:
    """Every index set used anywhere, as original row ids, for the release."""
    directory = paper_dir() / "splits" / dataset.key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split in splits:
        arrays = {
            "train_idx": split.train_idx, "val_idx": split.val_idx,
            "test_idx": split.test_idx, "search_idx": split.search_idx,
        }
        k = inner_fold_count(dataset.key)
        for j, (a, b) in enumerate(inner_folds(dataset.y, split.train_idx, k, split.seed)):
            arrays[f"stack_inner{j}_train"] = a
            arrays[f"stack_inner{j}_valid"] = b
        np.savez_compressed(directory / f"{split.tag}.npz", **arrays)
        manifest.append({"tag": split.tag, "repeat": split.repeat, "fold": split.fold,
                         "seed": split.seed, "hash": split.hash(),
                         "sizes": {k: int(len(v)) for k, v in arrays.items()}})
    write_json(directory / "manifest.json", {"dataset": dataset.key, "splits": manifest})


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def stage_baselines(dataset, splits, args) -> None:
    names = list(SKLEARN_LIKE) + ["tabnet_reference"]
    for split in splits:
        fold = None
        for name in names:
            if already_done(dataset.key, "baselines", name, split, args.force):
                continue
            fold = fold or prepare_fold(dataset, split)
            set_seed(split.seed)
            LOGGER.info("%s | baselines | %s | %s", dataset.key, name, split.tag)
            started = time.perf_counter()
            try:
                metrics, extra, scores = _run_baseline(name, fold, split.seed)
            except Exception as exc:  # noqa: BLE001 - a failed baseline must not stop the study
                LOGGER.error("  %s failed: %r", name, exc)
                continue
            extra["fit_seconds"] = time.perf_counter() - started
            store(dataset.key, "baselines", name, split, metrics, extra, fold["y_test"], scores)
            LOGGER.info("  AUC=%.4f", metrics["roc_auc"])


def _fit_capped(name, X, y, seed):
    """k-NN and the SVM are fitted on at most SUBSAMPLE_ABOVE rows (a resource cap)."""
    if name in {"svm", "knn"} and len(y) > SUBSAMPLE_ABOVE:
        return (*stratified_subsample(X, y, SUBSAMPLE_ABOVE, seed), True)
    return X, y, False


def _run_baseline(name: str, fold: dict, seed: int, params: dict | None = None):
    if name == "tabnet_reference":
        return _run_reference_tabnet(fold, seed, config=None, with_categories=False)

    X_train, X_val, X_test = view_arrays(fold, BASELINE_VIEWS[name])
    fit_X, fit_y, subsampled = _fit_capped(name, X_train, fold["y_train"], seed)
    model = build_baseline(name, seed=seed, params=params)
    model.fit(fit_X, fit_y)
    metrics, scores = scored(fold["y_val"], model.predict_proba(X_val)[:, 1],
                             fold["y_test"], model.predict_proba(X_test)[:, 1])
    return metrics, {"view": BASELINE_VIEWS[name], "subsampled": subsampled,
                     "n_fit": int(len(fit_y)), "params": params or {}}, scores


def _run_reference_tabnet(fold: dict, seed: int, config: TabNetConfig | None, with_categories: bool):
    """pytorch-tabnet, either in the first submission's setup or with a tuned config."""
    X_train, X_val, X_test = view_arrays(fold, "ordinal")
    ordinal = fold["ordinal"]
    if config is None:
        batch_size, virtual_batch_size = tabnet_batch_sizes(len(fold["y_train"]))
        max_epochs, patience = 120, 20
    else:
        fitted = GMATabNetClassifier(config)._fit_config(len(fold["y_train"]))
        batch_size, virtual_batch_size = fitted.batch_size, fitted.virtual_batch_size
        max_epochs, patience = config.max_epochs, config.patience
    if SMOKE:
        max_epochs, patience = 4, 2

    def fit():
        model = build_tabnet_reference(
            seed=seed, config=config,
            cat_idxs=ordinal.cat_idxs if with_categories else None,
            cat_dims=ordinal.cat_dims if with_categories else None,
        )
        model.fit(X_train, fold["y_train"], eval_set=[(X_val, fold["y_val"])], eval_metric=["auc"],
                  max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                  virtual_batch_size=virtual_batch_size)
        return model

    model = run_resilient(fit)
    metrics, scores = scored(fold["y_val"], model.predict_proba(X_val)[:, 1],
                             fold["y_test"], model.predict_proba(X_test)[:, 1])
    return metrics, {"view": "ordinal", "batch_size": batch_size,
                     "virtual_batch_size": virtual_batch_size,
                     "categorical_embeddings": with_categories,
                     "config": config.as_dict() if config else None}, scores


def stage_baselines_tuned(dataset, splits, args) -> None:
    if SMOKE:
        for name in TUNABLE:
            baseline_module.TUNING_TRIALS[name] = 2
    for split in splits:
        fold = None
        for name in TUNABLE:
            arm = f"{name}_tuned"
            if already_done(dataset.key, "baselines_tuned", arm, split, args.force):
                continue
            fold = fold or prepare_fold(dataset, split)
            set_seed(split.seed)
            LOGGER.info("%s | baselines_tuned | %s | %s", dataset.key, arm, split.tag)
            started = time.perf_counter()
            X_train, X_val, _ = view_arrays(fold, BASELINE_VIEWS[name])
            fit_X, fit_y, _ = _fit_capped(name, X_train, fold["y_train"], split.seed)
            search = tune_baseline(name, fit_X, fit_y, X_val, fold["y_val"], seed=split.seed)
            write_json(tuning_dir() / f"{dataset.key}__{arm}__{split.tag}.json",
                       {"dataset": dataset.key, "arm": arm, "split": split.tag, **search})
            metrics, extra, scores = _run_baseline(name, fold, split.seed, params=search["best_params"])
            extra.update(fit_seconds=time.perf_counter() - started,
                         search_val_auc=search["best_value"])
            store(dataset.key, "baselines_tuned", arm, split, metrics, extra, fold["y_test"], scores)
            LOGGER.info("  AUC=%.4f", metrics["roc_auc"])


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #
def apply_params(base: TabNetConfig, params: dict, seed: int) -> TabNetConfig:
    """Turn a decoded search point into a concrete configuration (n_a tied to n_d)."""
    updates = dict(params)
    if "n_d" in updates:
        updates["n_a"] = updates["n_d"]
    updates["seed"] = seed
    return replace(base, **updates)


def search_path(dataset_key: str, arch: str, optimiser: str, split: Split) -> Path:
    return tuning_dir() / f"{dataset_key}__{arch}__{optimiser}__{split.tag}.json"


def selected_path(dataset_key: str, arch: str, split: Split, optimiser: str = "tpe") -> Path:
    return tuning_dir() / f"{dataset_key}__{arch}__{optimiser}__selected__{split.tag}.json"


def budget_for(dataset_key: str, args) -> float:
    if args.budget is not None:
        return float(args.budget)
    return BUDGETS[dataset_key]


_WARMED_UP: set[str] = set()


def warm_up(X, y, X_val, y_val, ordinal, arch: str) -> None:
    """Pay CUDA and cuDNN start-up costs before any search clock starts.

    The first network a process trains takes tens of seconds longer than the
    rest. Inside a budget that cost would fall on whichever optimiser happened
    to run first.
    """
    if arch in _WARMED_UP:
        return
    rows = min(len(y), 512)
    config = TabNetConfig(**ARCH_BASE[arch], max_epochs=2, patience=2, seed=0)
    GMATabNetClassifier(config).fit(X[:rows], y[:rows], X_val, y_val,
                                    ordinal.cat_idxs, ordinal.cat_dims)
    release_memory()
    _WARMED_UP.add(arch)


def run_search(dataset, split: Split, fold: dict, arch: str, optimiser: str, args) -> dict:
    """One search for ``arch`` with ``optimiser`` on this fold, or the stored one."""
    path = search_path(dataset.key, arch, optimiser, split)
    budget = budget_for(dataset.key, args)
    if path.exists() and not args.force:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if abs(float(stored.get("budget_seconds", -1)) - budget) < 1.0:
            return stored

    from sklearn.metrics import roc_auc_score

    X_train, X_val, _ = view_arrays(fold, "ordinal")
    ordinal = fold["ordinal"]
    positions = np.searchsorted(split.train_idx, split.search_idx)
    X_search, y_search = X_train[positions], fold["y_train"][positions]
    audit("search_fit", dataset.key, split.search_idx)
    audit("search_score", dataset.key, split.val_idx)
    search_base = finalise(TabNetConfig(seed=split.seed, **ARCH_BASE[arch],
                                        max_epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE))

    def objective(params: dict, deadline: float):
        config = apply_params(search_base, params, split.seed)
        model = GMATabNetClassifier(config)
        run_resilient(lambda: model.fit(X_search, y_search, X_val, fold["y_val"],
                                        ordinal.cat_idxs, ordinal.cat_dims, deadline=deadline))
        value = float(roc_auc_score(fold["y_val"], model.predict_proba(X_val)[:, 1]))
        info = {"epochs": model.epochs_run, "diverged": model.diverged}
        del model
        release_memory()
        return value, info

    warm_up(X_search, y_search, X_val, fold["y_val"], ordinal, arch)
    space = tabnet_search_space(include_heads=arch == "gma", tokens=arch == "token")
    seed = optimiser_seed(split.repeat, split.fold, optimiser, arch)
    set_seed(split.seed)
    LOGGER.info("%s | search | %s/%s | %s | %.0fs", dataset.key, arch, optimiser, split.tag, budget)
    result = run_optimiser(optimiser, objective, space, budget, seed=seed)
    payload = result.summary() | {
        "dataset": dataset.key, "architecture": arch, "split": split.tag,
        "budget_seconds": budget, "optimiser_seed": seed,
        "search_rows": int(len(split.search_idx)), "validation_rows": int(len(split.val_idx)),
        "n_interrupted": sum(1 for h in result.history if h["status"] == "interrupted"),
        "train_hash": split.hash(),
    }
    write_json(path, payload)
    return payload


def select(dataset, split: Split, arch: str, search: dict, optimiser: str) -> TabNetConfig:
    if not search.get("best_params"):
        raise RuntimeError(f"{dataset.key} {arch}/{optimiser} {split.tag}: no completed evaluation")
    config = apply_params(TabNetConfig(seed=split.seed, **ARCH_BASE[arch]), search["best_params"], split.seed)
    write_json(selected_path(dataset.key, arch, split, optimiser), {
        "dataset": dataset.key, "architecture": arch, "optimiser": optimiser,
        "split": split.tag, "split_hash": split.hash(),
        "search_val_auc": search["best_value"], "config": config.as_dict(),
    })
    return config


def load_selected(dataset_key: str, arch: str, split: Split, optimiser: str = "tpe") -> TabNetConfig:
    path = selected_path(dataset_key, arch, split, optimiser)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run --stage hpo_select first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["split_hash"] != split.hash():
        raise RuntimeError(f"{path.name} was selected on a different partition")
    return TabNetConfig.from_dict(payload["config"])


def stage_hpo_select(dataset, splits, args) -> None:
    for split in splits:
        fold = None
        for arch in ARCHITECTURES:
            if selected_path(dataset.key, arch, split).exists() and not args.force:
                continue
            fold = fold or prepare_fold(dataset, split)
            search = run_search(dataset, split, fold, arch, "tpe", args)
            select(dataset, split, arch, search, "tpe")


def stage_hpo_study(dataset, splits, args) -> None:
    """Repeat 0: the other three optimisers, each selection scored on the test fold."""
    for split in (s for s in splits if s.repeat == 0):
        fold = None
        for optimiser in STUDY_OPTIMISERS:
            arm = f"gma_{optimiser}"
            if already_done(dataset.key, "hpo_study", arm, split, args.force):
                continue
            fold = fold or prepare_fold(dataset, split)
            search = run_search(dataset, split, fold, "gma", optimiser, args)
            config = select(dataset, split, "gma", search, optimiser)
            _fit_arm(dataset, split, fold, "hpo_study", arm, finalise(config),
                     {"config_source": f"gma/{optimiser}", "search_val_auc": search["best_value"]})


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def architecture_arms(gma: TabNetConfig, vanilla: TabNetConfig, input_dim, cat_idxs, cat_dims,
                      reduced: bool) -> dict[str, tuple[str, TabNetConfig | None, str]]:
    """arm -> (kind, config, source). ``kind`` is "ours" or "reference".

    Fixed-configuration arms all take this fold's GMA selection and change one
    thing; the equal-tuning arm takes vanilla TabNet's own selection under the
    same budget; the gap-decomposition arms walk from the reference library to
    our vanilla trainer on the vanilla selection.
    """
    matched, matched_count, goal = match_capacity(gma, input_dim, cat_idxs, cat_dims)
    arms = {
        "gma_full": ("ours", gma, "gma"),
        "tabnet_vanilla": ("ours", replace(gma, variant="vanilla"), "gma"),
        "gma_single_head": ("ours", replace(gma, n_heads=1), "gma"),
        "gma_fixed_temperature": ("ours", replace(gma, learn_temperature=False), "gma"),
        "gma_no_interaction_gate": ("ours", replace(gma, use_interaction_gate=False), "gma"),
        "gma_no_focal": ("ours", replace(gma, use_focal=False), "gma"),
        "gma_uniform_mix": ("ours", replace(gma, variant="uniform_mix"), "gma"),
        "tabnet_vanilla_matched": ("ours", matched, f"gma, width matched ({matched_count} vs {goal})"),
        "mh_wang_repro": ("ours", replace(gma, variant="mh_wang"), "gma"),
        "tabnet_vanilla_tuned": ("ours", vanilla, "vanilla"),
    }
    if not reduced:
        arms.update({
            "gma_pretrained": ("ours", replace(gma, pretrain_epochs=30), "gma"),
            "ref_lib_tuned": ("reference", vanilla, "vanilla"),
            "ours_refcompat_tuned": ("ours", replace(vanilla, reference_compat=True), "vanilla"),
            "ours_vanilla_tuned_bce": ("ours", replace(vanilla, use_focal=False, class_weighted=False),
                                       "vanilla"),
        })
    return arms


ARCHITECTURE_ARMS = (
    "gma_full", "tabnet_vanilla", "gma_single_head", "gma_fixed_temperature",
    "gma_no_interaction_gate", "gma_no_focal", "gma_uniform_mix", "tabnet_vanilla_matched",
    "mh_wang_repro", "tabnet_vanilla_tuned",
    "gma_pretrained", "ref_lib_tuned", "ours_refcompat_tuned", "ours_vanilla_tuned_bce",
)
# With --reduced, repeats 1 and 2 run only the first this-many arms.
ARCHITECTURE_REDUCED = 10


def _fit_arm(dataset, split, fold, stage, arm, config, extra, save_checkpoint=True) -> None:
    X_train, X_val, X_test = view_arrays(fold, "ordinal")
    ordinal = fold["ordinal"]
    set_seed(split.seed)
    LOGGER.info("%s | %s | %s | %s", dataset.key, stage, arm, split.tag)
    audit("model_fit", dataset.key, split.train_idx)
    audit("early_stop", dataset.key, split.val_idx)
    started = time.perf_counter()

    def fit():
        model = GMATabNetClassifier(config)
        model.fit(X_train, fold["y_train"], X_val, fold["y_val"], ordinal.cat_idxs, ordinal.cat_dims)
        return model

    model = run_resilient(fit)
    metrics, scores = scored(fold["y_val"], model.predict_proba(X_val)[:, 1],
                             fold["y_test"], model.predict_proba(X_test)[:, 1])
    if save_checkpoint:
        model.save(checkpoint_path(dataset.key, arm, split))
    store(dataset.key, stage, arm, split, metrics, {
        **extra,
        "fit_seconds": time.perf_counter() - started,
        "epochs_run": model.epochs_run,
        "best_val_auc": model.best_val_auc,
        "diverged": model.diverged,
        "parameters": model.parameter_count(),
        "learned_temperatures": model.learned_temperatures(),
        "val_curve": [round(h["val_auc"], 5) for h in model.history],
        "config": model.fitted_config.as_dict(),
    }, fold["y_test"], scores)
    LOGGER.info("  AUC=%.4f", metrics["roc_auc"])


def stage_architecture(dataset, splits, args) -> None:
    for split in splits:
        gma = finalise(load_selected(dataset.key, "gma", split))
        vanilla = finalise(load_selected(dataset.key, "vanilla", split))
        reduced = args.reduced and split.repeat > 0
        names = ARCHITECTURE_ARMS if not reduced else ARCHITECTURE_ARMS[:ARCHITECTURE_REDUCED]
        if all(already_done(dataset.key, "architecture", a, split, args.force) for a in names):
            continue
        fold = prepare_fold(dataset, split)
        ordinal = fold["ordinal"]
        arms = architecture_arms(gma, vanilla, ordinal.X_train.shape[1], ordinal.cat_idxs,
                                 ordinal.cat_dims, reduced)
        assert tuple(arms) == names
        for arm, (kind, config, source) in arms.items():
            if already_done(dataset.key, "architecture", arm, split, args.force):
                continue
            if kind == "reference":
                set_seed(split.seed)
                LOGGER.info("%s | architecture | %s | %s", dataset.key, arm, split.tag)
                started = time.perf_counter()
                metrics, extra, scores = _run_reference_tabnet(fold, split.seed, config,
                                                               with_categories=True)
                extra.update(fit_seconds=time.perf_counter() - started, config_source=source)
                store(dataset.key, "architecture", arm, split, metrics, extra, fold["y_test"], scores)
                LOGGER.info("  AUC=%.4f", metrics["roc_auc"])
            else:
                _fit_arm(dataset, split, fold, "architecture", arm, config, {"config_source": source})


# --------------------------------------------------------------------------- #
# Stacking
# --------------------------------------------------------------------------- #
def stacking_arms(gma: TabNetConfig, vanilla: TabNetConfig, k: int, seed: int,
                  token: TabNetConfig | None = None) -> dict[str, StackConfig]:
    arms = {
        "stack_gma_nested": StackConfig(tabnet=gma, n_inner_folds=k, seed=seed),
        "stack_vanilla_nested": StackConfig(tabnet=vanilla, n_inner_folds=k, seed=seed),
        "stack_no_extractor": StackConfig(tabnet=gma, use_extractor=False, n_inner_folds=k, seed=seed),
        "stack_gma_shared": StackConfig(tabnet=gma, protocol="shared", n_inner_folds=k, seed=seed),
    }
    if token is not None:
        arms["stack_token_nested"] = StackConfig(tabnet=token, n_inner_folds=k, seed=seed)
    return arms


def stage_stacking(dataset, splits, args) -> None:
    from src.paper2_credit.interpret import stack_permutation_importance

    for split in splits:
        gma = finalise(load_selected(dataset.key, "gma", split))
        vanilla = finalise(load_selected(dataset.key, "vanilla", split))
        token = (finalise(load_selected(dataset.key, "token", split))
                 if selected_path(dataset.key, "token", split).exists() else None)
        arms = stacking_arms(gma, vanilla, inner_fold_count(dataset.key), split.seed, token)
        for arm, config in arms.items():
            if already_done(dataset.key, "stacking", arm, split, args.force):
                continue
            set_seed(split.seed)
            LOGGER.info("%s | stacking | %s | %s", dataset.key, arm, split.tag)
            audit("stack_fit", dataset.key, split.train_idx)
            started = time.perf_counter()
            stack = run_resilient(lambda: HeterogeneousStack(config).fit(
                dataset, split.train_idx, split.val_idx))
            y = dataset.y
            metrics, scores = scored(y[split.val_idx], stack.score(dataset, split.val_idx),
                                     y[split.test_idx], stack.score(dataset, split.test_idx))
            extra = {"fit_seconds": time.perf_counter() - started, "timings": stack.timings_,
                     "oof_auc": stack.oof_auc_, "label": config.label(),
                     "n_inner_folds": config.n_inner_folds,
                     "config": config.tabnet.as_dict()}
            if arm == "stack_gma_nested" and split.repeat == 0:
                stack_permutation_importance(dataset, stack, split, EXPLAIN_ROWS // (8 if SMOKE else 1))
            store(dataset.key, "stacking", arm, split, metrics, extra, y[split.test_idx], scores)
            LOGGER.info("  AUC=%.4f", metrics["roc_auc"])


# --------------------------------------------------------------------------- #
# Token attention
# --------------------------------------------------------------------------- #
TOKEN_ARMS = ("token_full", "token_local", "token_global", "token_only", "token_gma",
              "token_matched_vanilla")


def token_arms(token: TabNetConfig, gma: TabNetConfig, input_dim, cat_idxs, cat_dims) -> dict:
    """arm -> (config, source). All arms take this fold's token selection.

    ``token_only`` drops the TabNet encoder, ``token_local``/``token_global`` keep
    one attention module, ``token_gma`` swaps the vanilla attentive transformer
    for GMA-TabNet's mask mixture (with the fold's GMA head count), and
    ``token_matched_vanilla`` is plain TabNet widened to the token model's size.
    """
    matched, matched_count, goal = match_capacity(token, input_dim, cat_idxs, cat_dims)
    return {
        "token_full": (token, "token"),
        "token_local": (replace(token, token_modules="local"), "token"),
        "token_global": (replace(token, token_modules="global"), "token"),
        "token_only": (replace(token, tokens_only=True), "token"),
        "token_gma": (replace(token, variant="gma", n_heads=gma.n_heads), "token + gma heads"),
        "token_matched_vanilla": (matched, f"token, width matched ({matched_count} vs {goal})"),
    }


def stage_token_study(dataset, splits, args) -> None:
    for split in splits:
        if all(already_done(dataset.key, "token_study", a, split, args.force) for a in TOKEN_ARMS):
            continue
        token = finalise(load_selected(dataset.key, "token", split))
        gma = finalise(load_selected(dataset.key, "gma", split))
        fold = prepare_fold(dataset, split)
        ordinal = fold["ordinal"]
        arms = token_arms(token, gma, ordinal.X_train.shape[1], ordinal.cat_idxs, ordinal.cat_dims)
        for arm, (config, source) in arms.items():
            if already_done(dataset.key, "token_study", arm, split, args.force):
                continue
            _fit_arm(dataset, split, fold, "token_study", arm, config, {"config_source": source})


# --------------------------------------------------------------------------- #
# Protocol audit: configuration selected on one fold and reused
# --------------------------------------------------------------------------- #
SELECTION_AUDIT_ARCHS = ("gma", "vanilla", "token")


def stage_selection_audit(dataset, splits, args) -> None:
    """The first submission's shortcut, run with today's code.

    For every repeat, the configuration selected on fold 0 is applied unchanged
    to folds 1-4 of the same repeat. Fold 0's validation rows lie in the test
    folds of folds 1-4, so this is a deliberate leak; the arms are paired with
    the nested arms on the same folds to measure what it does to the estimates.
    """
    every = make_splits(dataset, argparse.Namespace(
        folds=args.folds, repeats=args.repeats, val_size=VAL_SIZE, search_rows=args.search_rows,
        only_repeat=None, only_fold=None))
    fold_zero = {s.repeat: s for s in every if s.fold == 0}
    for split in (s for s in splits if s.fold != 0):
        source = fold_zero[split.repeat]
        fold = None
        for arch in SELECTION_AUDIT_ARCHS:
            arm = f"{arch}_fold0cfg"
            if already_done(dataset.key, "selection_audit", arm, split, args.force):
                continue
            if not selected_path(dataset.key, arch, source).exists():
                continue
            fold = fold or prepare_fold(dataset, split)
            config = replace(finalise(load_selected(dataset.key, arch, source)), seed=split.seed)
            audit("deliberate_selection", dataset.key, source.val_idx)
            _fit_arm(dataset, split, fold, "selection_audit", arm, config,
                     {"config_source": f"{arch} selected on {source.tag}"}, save_checkpoint=False)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
STAGES = ("export_splits", "baselines", "baselines_tuned", "hpo_select", "hpo_study",
          "architecture", "stacking", "interpret", "protocol_sensitivity",
          "token_study", "selection_audit", "interpret_token", "resampling_audit")


def main() -> None:
    global SMOKE

    parser = argparse.ArgumentParser(description="Paper 2 experiments (nested, repeated)")
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--datasets", nargs="*", default=CREDIT_DATASETS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--only-repeat", type=int, nargs="*")
    parser.add_argument("--only-fold", type=int, nargs="*")
    parser.add_argument("--budget", type=float, default=None,
                        help="override the per-dataset search budget (seconds)")
    parser.add_argument("--search-rows", type=int, default=SEARCH_ROWS)
    parser.add_argument("--reduced", action="store_true",
                        help="pretraining and gap-decomposition arms on repeat 0 only")
    parser.add_argument("--smoke", action="store_true", help="tiny schedules for a dry run")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    SMOKE = args.smoke
    # Run as ``python -m``, this file is __main__; the interpretation and
    # protocol modules import it again by name and would see their own copy.
    import src.paper2_credit.run_experiments as canonical
    canonical.SMOKE = args.smoke

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    runs_dir().mkdir(parents=True, exist_ok=True)
    tuning_dir().mkdir(parents=True, exist_ok=True)

    stages = STAGES if args.stage == "all" else (args.stage,)
    for key in args.datasets:
        dataset = load_dataset(key)
        LOGGER.info("%s: %d rows, %d features, %d positives (%.2f%%)", key, dataset.n_samples,
                    dataset.n_features, int(dataset.y.sum()), dataset.minority_pct)
        splits = make_splits(dataset, args)
        for stage in stages:
            if stage == "interpret":
                from src.paper2_credit.interpret import stage_interpret
                stage_interpret(dataset, [s for s in splits if s.repeat == 0], args)
            elif stage == "protocol_sensitivity":
                if key == "give_me_some_credit":
                    from src.paper2_credit.protocol import stage_protocol_sensitivity
                    stage_protocol_sensitivity(dataset, args)
            elif stage == "interpret_token":
                from src.paper2_credit.interpret import stage_interpret_token
                stage_interpret_token(dataset, [s for s in splits if s.repeat == 0], args)
            elif stage == "resampling_audit":
                from src.paper2_credit.resampling_audit import stage_resampling_audit
                stage_resampling_audit(dataset, args)
            else:
                globals()[f"stage_{stage}"](dataset, splits, args)

    LOGGER.info("done; %d result files", len(list(runs_dir().glob("*.json"))))


if __name__ == "__main__":
    main()
