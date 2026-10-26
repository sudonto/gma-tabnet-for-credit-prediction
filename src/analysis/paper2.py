"""Tables, figures and prose macros for Paper 2 (protocol v2, JACOST manuscript).

Everything is computed from the raw run records written by
``src.paper2_credit.run_experiments``: one JSON per (dataset, stage, arm,
outer fold), the tuning histories, and the attribution records.

Inference
---------
The unit of analysis is the outer fold. Every comparison pairs two arms on the
same fifteen outer folds (three repeats of five), checked by the partition hash
stored with each record, and is reported as the mean paired difference with the
corrected resampled t interval of Nadeau and Bengio, which accounts for folds
sharing training rows. p-values are Holm-adjusted within the family named in
each table. Rank tests across datasets and tests on pooled predictions are not
used: three datasets give a rank test no power, and pooled cross-validation
predictions are not independent.

Rounding
--------
A difference printed beside two means is the difference of the *printed*
means, so the arithmetic in the text is exact; intervals and p-values come from
the unrounded paired differences. ``write_macros`` records every such triple in
``macro_triples.json`` and ``scripts/check_macro_arithmetic.py`` verifies them.

    python -m src.analysis.run_analysis --paper 2
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import tables as table_module
from src.analysis.macros import PLACEHOLDER, MacroSet
from src.common.io import PROJECT_ROOT, results_root, write_json
from src.common.stats import (
    corrected_resampled_ttest,
    holm,
    kendall_w,
)

LOGGER = logging.getLogger("analysis.paper2")

DATASETS = ["uci_dccc", "german_credit", "give_me_some_credit"]
PREFIX = {"uci_dccc": "Dccc", "german_credit": "German", "give_me_some_credit": "Gmsc"}
LABEL = {"uci_dccc": "UCI Taiwan", "german_credit": "German Credit",
         "give_me_some_credit": "Give Me Some Credit"}
SHORT = {"uci_dccc": "UCI Taiwan", "german_credit": "German", "give_me_some_credit": "GMSC"}

# JACOST column and text widths, for figures drawn at the size they are placed.
COLUMN_IN = 79.0 / 25.4
TEXT_IN = 160.0 / 25.4


@dataclass(frozen=True)
class Arm:
    stage: str
    arm: str
    name: str        # table label
    macro: str       # macro stem


ARMS: dict[str, Arm] = {a.macro: a for a in [
    Arm("baselines", "logistic_regression", "Logistic Regression", "LogReg"),
    Arm("baselines", "decision_tree", "Decision Tree", "Tree"),
    Arm("baselines", "random_forest", "Random Forest", "Rf"),
    Arm("baselines", "knn", "k-NN$^{a}$", "Knn"),
    Arm("baselines", "svm", "SVM (RBF)$^{a}$", "Svm"),
    Arm("baselines", "xgboost", "XGBoost", "Xgb"),
    Arm("baselines", "lightgbm", "LightGBM", "Lgbm"),
    Arm("baselines", "catboost", "CatBoost", "Cat"),
    Arm("baselines", "mlp", "MLP", "Mlp"),
    Arm("baselines", "tabnet_reference", "TabNet (pytorch-tabnet, setelan awal)", "RefDefault"),
    Arm("baselines_tuned", "random_forest_tuned", "Random Forest (disetel)", "RfTuned"),
    Arm("baselines_tuned", "svm_tuned", "SVM (RBF, disetel)$^{a}$", "SvmTuned"),
    Arm("baselines_tuned", "xgboost_tuned", "XGBoost (disetel)", "XgbTuned"),
    Arm("baselines_tuned", "lightgbm_tuned", "LightGBM (disetel)", "LgbmTuned"),
    Arm("architecture", "gma_full", "GMA-TabNet (usulan)", "Gma"),
    Arm("architecture", "tabnet_vanilla_tuned", "vanilla, disetel sendiri", "VanTuned"),
    Arm("architecture", "tabnet_vanilla", "TabNet vanilla", "Van"),
    Arm("architecture", "gma_single_head", "satu kepala ($H=1$)", "OneHead"),
    Arm("architecture", "gma_fixed_temperature", "suhu tetap ($\\tau_h=1$)", "FixedTemp"),
    Arm("architecture", "gma_uniform_mix", "campuran seragam", "UniformMix"),
    Arm("architecture", "gma_no_interaction_gate", "tanpa gerbang interaksi", "NoGate"),
    Arm("architecture", "gma_no_focal", "tanpa focal loss", "NoFocal"),
    Arm("architecture", "gma_pretrained", "prapelatihan swaawasan", "Pretrained"),
    Arm("architecture", "tabnet_vanilla_matched", "vanilla, kapasitas setara", "Matched"),
    Arm("architecture", "mh_wang_repro", "MHA-TabNet~\\cite{wang2025tabnetrating}$^{b}$", "Wang"),
    Arm("architecture", "ref_lib_tuned", "pytorch-tabnet, disetel, dengan embedding", "RefTuned"),
    Arm("architecture", "ours_refcompat_tuned", "implementasi kami, resep referensi", "RefCompat"),
    Arm("architecture", "ours_vanilla_tuned_bce", "implementasi kami, resep kami, BCE", "OursBce"),
    Arm("stacking", "stack_gma_nested", "Stack + GMA-TabNet (bersarang)", "StackGma"),
    Arm("stacking", "stack_vanilla_nested", "Stack + TabNet vanilla (bersarang)", "StackVan"),
    Arm("stacking", "stack_no_extractor", "Stack tanpa ekstraktor", "StackNone"),
    Arm("stacking", "stack_gma_shared", "Stack + GMA-TabNet (protokol lama)", "StackShared"),
    Arm("hpo_study", "gma_random", "Pencarian acak", "HpoRandom"),
    Arm("hpo_study", "gma_pso", "PSO", "HpoPso"),
    Arm("hpo_study", "gma_ga", "Algoritma genetika", "HpoGa"),
]}

BASELINE_MACROS = ["LogReg", "Tree", "Rf", "Knn", "Svm", "Xgb", "Lgbm", "Cat", "Mlp", "RefDefault",
                   "RfTuned", "SvmTuned", "XgbTuned", "LgbmTuned"]
MECHANISM_MACROS = ["Van", "OneHead", "FixedTemp", "UniformMix", "NoGate", "NoFocal", "Pretrained"]
CONTROL_MACROS = ["Matched", "Wang"]
GAP_CHAIN = ["RefDefault", "RefTuned", "RefCompat", "OursBce", "VanTuned", "Gma"]
STACK_MACROS = ["StackGma", "StackVan", "StackNone", "StackShared"]
OPTIMISERS = [("random", "Pencarian acak", "HpoRandom"), ("tpe", "TPE", "Gma"),
              ("pso", "PSO", "HpoPso"), ("ga", "Algoritma genetika", "HpoGa")]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def paper_dir() -> Path:
    return results_root() / "paper2"


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted((paper_dir() / "raw_runs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "repeat" not in payload:
            continue
        row = {k: payload.get(k) for k in (
            "dataset", "stage", "arm", "repeat", "fold", "seed", "split_hash", "n_train",
            "n_val", "n_test", "parameters", "config_source", "epochs_run", "search_val_auc",
            "fit_seconds")}
        row.update(payload.get("metrics", {}))
        row["temperatures"] = [t for step in (payload.get("learned_temperatures") or {}).values()
                               for t in step]
        rows.append(row)
    return pd.DataFrame(rows)


def load_tuning(dataset: str, arch: str, optimiser: str) -> list[dict]:
    files = sorted((paper_dir() / "tuning").glob(f"{dataset}__{arch}__{optimiser}__r*f*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in files if "selected" not in p.name]


def series(runs: pd.DataFrame, dataset: str, macro: str, metric: str = "roc_auc") -> pd.Series:
    arm = ARMS[macro]
    if runs.empty:
        return pd.Series(dtype=float)
    subset = runs[(runs.dataset == dataset) & (runs.stage == arm.stage) & (runs.arm == arm.arm)]
    if subset.empty or metric not in subset:
        return pd.Series(dtype=float)
    return subset.set_index(["repeat", "fold"])[metric].astype(float).sort_index()


def hashes(runs: pd.DataFrame, dataset: str, macro: str) -> pd.Series:
    arm = ARMS[macro]
    subset = runs[(runs.dataset == dataset) & (runs.stage == arm.stage) & (runs.arm == arm.arm)]
    return subset.set_index(["repeat", "fold"])["split_hash"].sort_index()


def test_train_ratio(runs: pd.DataFrame, dataset: str) -> float:
    subset = runs[runs.dataset == dataset]
    return float((subset.n_test / (subset.n_train + subset.n_val)).median())


def paired(runs: pd.DataFrame, dataset: str, a: str, b: str, repeat0: bool = False) -> dict | None:
    """Arm ``a`` minus arm ``b``, paired by outer fold, with the corrected interval."""
    sa, sb = series(runs, dataset, a), series(runs, dataset, b)
    common = sa.index.intersection(sb.index)
    if repeat0:
        common = common[common.get_level_values(0) == 0]
    if len(common) < 2:
        return None
    ha, hb = hashes(runs, dataset, a), hashes(runs, dataset, b)
    if not (ha.loc[common] == hb.loc[common]).all():
        raise RuntimeError(f"{dataset}: {a} and {b} were scored on different partitions")
    diffs = (sa.loc[common] - sb.loc[common]).to_numpy()
    ratio = test_train_ratio(runs, dataset)
    result = corrected_resampled_ttest(diffs, n_train=1.0, n_test=ratio)
    mean_a, mean_b = float(sa.loc[common].mean()), float(sb.loc[common].mean())
    result.update(mean_a=mean_a, mean_b=mean_b,
                  shown_delta=round(round(mean_a, 4) - round(mean_b, 4), 4),
                  complete=len(common) == len(sa) == len(sb))
    return result


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def f4(value) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.4f}"


def signed(value, decimals: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:+.{decimals}f}".replace("-", "$-$").replace("+", "$+$")


def p_text(p) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def mean_sd(values: pd.Series, bold: bool = False) -> str:
    if values is None or len(values) == 0:
        return "--"
    body = f"{values.mean():.4f} $\\pm$ {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f}"
    return f"\\textbf{{{body}}}" if bold else body


def delta_cell(result: dict | None) -> str:
    if result is None:
        return "--"
    return f"{signed(result['shown_delta'])}$\\pm${result['half_width']:.4f}"


def dataset_groups(width: int) -> list[tuple[str, int]]:
    return [("", 1)] + [(LABEL[d], width) for d in DATASETS]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
PROTOCOL_CAPTION = ("Ukuran partisi per lipatan luar (median atas 15 lipatan) dan anggaran pencarian. "
                    "Pencarian dilatih pada baris pencarian dan dinilai pada blok validasi; "
                    "keduanya berada di dalam partisi latih luar.")


def table_protocol(runs: pd.DataFrame, caption: str = PROTOCOL_CAPTION, notes: str | None = None) -> str:
    from src.common.io import load_dataset

    rows_spec = [
        ("Baris / positif (\\%)", "rows"), ("Partisi latih luar", "outer"),
        ("\\quad baris optimasi", "train"), ("\\quad blok validasi", "val"),
        ("Baris pencarian hiperparameter", "search"), ("Lipatan uji", "test"),
        ("Anggaran per pencarian (detik)", "budget"),
        ("Evaluasi selesai, GMA (median)", "evals_gma"),
        ("Evaluasi selesai, vanilla (median)", "evals_van"),
        ("Evaluasi terpotong (total)", "interrupted"),
    ]
    values: dict[str, dict[str, str]] = {key: {} for _, key in rows_spec}
    for dataset in DATASETS:
        data = load_dataset(dataset)
        values["rows"][dataset] = f"{data.n_samples:,} / {data.minority_pct:.1f}".replace(",", "{,}")
        subset = runs[(runs.dataset == dataset) & (runs.stage == "architecture")]
        if not subset.empty:
            values["outer"][dataset] = f"{int((subset.n_train + subset.n_val).median()):,}".replace(",", "{,}")
            values["train"][dataset] = f"{int(subset.n_train.median()):,}".replace(",", "{,}")
            values["val"][dataset] = f"{int(subset.n_val.median()):,}".replace(",", "{,}")
            values["test"][dataset] = f"{int(subset.n_test.median()):,}".replace(",", "{,}")
        gma, van = load_tuning(dataset, "gma", "tpe"), load_tuning(dataset, "vanilla", "tpe")
        if gma:
            values["search"][dataset] = f"{int(np.median([g['search_rows'] for g in gma])):,}".replace(",", "{,}")
            values["budget"][dataset] = f"{gma[0]['budget_seconds']:.0f}"
            values["evals_gma"][dataset] = f"{np.median([g['n_evaluations'] for g in gma]):.0f}"
            values["interrupted"][dataset] = str(sum(g.get("n_interrupted", 0) for g in gma + van))
        if van:
            values["evals_van"][dataset] = f"{np.median([g['n_evaluations'] for g in van]):.0f}"
    rows = [[label] + [values[key].get(d, "--") for d in DATASETS] for label, key in rows_spec]
    return table_module.latex_table(
        ["Besaran"] + [SHORT[d] for d in DATASETS], rows,
        caption=caption, label="tab:p2-protocol", column_format="lrrr", notes=notes,
    )


def table_baselines(runs: pd.DataFrame) -> str:
    means = {d: {m: series(runs, d, m).mean() for m in BASELINE_MACROS} for d in DATASETS}
    best = {d: max((m for m in BASELINE_MACROS if np.isfinite(means[d][m])),
                   key=lambda m: means[d][m], default=None) for d in DATASETS}
    rows = []
    for macro in BASELINE_MACROS:
        row = [ARMS[macro].name]
        for dataset in DATASETS:
            row.append(mean_sd(series(runs, dataset, macro), bold=macro == best[dataset]))
            ks = series(runs, dataset, macro, "ks")
            row.append(f4(ks.mean()) if len(ks) else "--")
        rows.append(row)
        if macro == "RefDefault":
            rows.append("\\midrule")
    return table_module.latex_table(
        ["Model"] + ["ROC-AUC", "KS"] * len(DATASETS), rows,
        caption="Pembanding, rata-rata $\\pm$ simpangan baku ROC-AUC dan rata-rata KS atas 15 lipatan "
                "luar (3 ulangan $\\times$ 5 lipatan); terbaik per dataset dicetak tebal.",
        label="tab:p2-baselines", column_format="l" + "cc" * len(DATASETS),
        groups=dataset_groups(2),
        notes="$^{a}$Pada UCI Taiwan dan GMSC, k-NN dan SVM dilatih pada subsampel bertingkat 8{,}000 baris "
              "karena batasan komputasi, bukan paparan data yang setara. Baris di bawah garis disetel "
              "dengan TPE pada blok validasi tiap lipatan.",
    )


def architecture_family(runs, dataset, macros) -> dict[str, dict]:
    results = {m: paired(runs, dataset, m, "Gma") for m in macros}
    available = {m: r["p"] for m, r in results.items() if r is not None}
    adjusted = holm(available) if available else {}
    for m, r in results.items():
        if r is not None:
            r["p_holm"] = adjusted[m]
    return results


def table_architecture(runs: pd.DataFrame) -> str:
    panels = [
        ("Penyetelan setara (masing-masing dengan pencarian TPE sendiri)", ["VanTuned"]),
        ("Ablasi mekanisme (konfigurasi GMA lipatan tersebut)", MECHANISM_MACROS),
        ("Kontrol kapasitas dan arsitektur terdekat", CONTROL_MACROS),
    ]
    all_macros = [m for _, ms in panels for m in ms]
    families = {d: architecture_family(runs, d, all_macros) for d in DATASETS}
    rows = [["GMA-TabNet (usulan)"] + sum(
        [[f4(series(runs, d, "Gma").mean()), "--", "--"] for d in DATASETS], [])]
    for title, macros in panels:
        rows.append(f"\\multicolumn{{{1 + 3 * len(DATASETS)}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for macro in macros:
            row = [ARMS[macro].name]
            for dataset in DATASETS:
                result = families[dataset][macro]
                count = "" if result is None or result["n"] >= 15 else f"$^{{c}}$"
                row += [f4(series(runs, dataset, macro).mean()) + count, delta_cell(result),
                        p_text(result["p_holm"]) if result else "--"]
            rows.append(row)
    return table_module.latex_table(
        ["Konfigurasi"] + ["AUC", "$\\Delta$ $\\pm$ SK 95\\%", "$p_{\\mathrm{Holm}}$"] * len(DATASETS),
        rows,
        caption="Ablasi arsitektur. $\\Delta$ adalah selisih berpasangan (baris dikurangi GMA-TabNet) "
                "atas lipatan luar, dengan setengah lebar selang kepercayaan 95\\% dari uji-t "
                "tersampel ulang terkoreksi; $p$ disesuaikan Holm dalam tiap dataset.",
        label="tab:p2-architecture", column_format="l" + "ccc" * len(DATASETS),
        groups=dataset_groups(3),
        notes="$^{b}$Implementasi kami atas deskripsi Wang dan Zhang (2025); lihat Bagian~\\ref{sec:p2-method-arch}. "
              "$^{c}$Hanya ulangan pertama (5 lipatan). $\\Delta$ adalah selisih rata-rata yang dicetak; "
              "selang dan $p$ dihitung dari selisih per lipatan sebelum pembulatan.",
    )


def table_gap(runs: pd.DataFrame) -> str:
    rows = []
    for index, macro in enumerate(GAP_CHAIN):
        row = [ARMS[macro].name if macro != "Gma" else "arsitektur GMA-TabNet (disetel sendiri)"]
        if macro == "VanTuned":
            row[0] = "+ focal loss (= TabNet vanilla disetel)"
        for dataset in DATASETS:
            row.append(f4(series(runs, dataset, macro).mean()))
            row.append("--" if index == 0 else delta_cell(paired(runs, dataset, macro, GAP_CHAIN[index - 1])))
        rows.append(row)
    return table_module.latex_table(
        ["Langkah"] + ["AUC", "$\\Delta$ thd. baris sebelumnya"] * len(DATASETS), rows,
        caption="Dekomposisi selisih antara TabNet referensi dan GMA-TabNet: setiap baris mengubah satu "
                "hal dari baris di atasnya. Baris pertama adalah setelan pengajuan pertama (tanpa embedding "
                "kategori); baris kedua sampai kelima memakai konfigurasi yang dipilih untuk TabNet vanilla.",
        label="tab:p2-gap", column_format="l" + "cc" * len(DATASETS), groups=dataset_groups(2),
    )


def hpo_summary(runs: pd.DataFrame, dataset: str) -> dict[str, dict]:
    out = {}
    for optimiser, name, macro in OPTIMISERS:
        searches = [s for s in load_tuning(dataset, "gma", optimiser) if s["split"].startswith("r0")]
        test = series(runs, dataset, macro)
        test = test[test.index.get_level_values(0) == 0] if len(test) else test
        out[optimiser] = {
            "name": name, "macro": macro,
            "val": [s["best_value"] for s in searches],
            "evals": [s["n_evaluations"] for s in searches],
            "interrupted": sum(s.get("n_interrupted", 0) for s in searches),
            "test": test,
            "delta": None if optimiser == "random" else paired(runs, dataset, macro, "HpoRandom", repeat0=True),
        }
    available = {k: v["delta"]["p"] for k, v in out.items() if v["delta"]}
    for k, p in (holm(available) if available else {}).items():
        out[k]["delta"]["p_holm"] = p
    return out


def table_hpo(runs: pd.DataFrame) -> str:
    rows = []
    for dataset in DATASETS:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{LABEL[dataset]}}}}} \\\\")
        for optimiser, summary in hpo_summary(runs, dataset).items():
            val = pd.Series(summary["val"], dtype=float)
            rows.append([
                summary["name"],
                f"{np.median(summary['evals']):.0f} ({summary['interrupted']})" if summary["evals"] else "--",
                mean_sd(val), mean_sd(summary["test"]),
                delta_cell(summary["delta"]) if summary["delta"] else "--",
                p_text(summary["delta"]["p_holm"]) if summary["delta"] else "--",
            ])
    return table_module.latex_table(
        ["Pengoptimasi", "Evaluasi (terpotong)", "AUC validasi terbaik", "AUC uji", "$\\Delta$ uji thd. acak",
         "$p_{\\mathrm{Holm}}$"], rows,
        caption="Empat pengoptimasi pada ruang pencarian dan anggaran waktu yang sama, diulang secara "
                "independen pada 5 lipatan luar ulangan pertama. Konfigurasi terpilih tiap pencarian dilatih "
                "ulang dan dinilai pada lipatan uji yang tidak tersentuh. Evaluasi adalah median evaluasi "
                "yang selesai dalam anggaran; yang terpotong di batas anggaran tidak dihitung.",
        label="tab:p2-hpo", column_format="lccccc",
    )


def comparator(runs: pd.DataFrame, dataset: str) -> str | None:
    means = {m: series(runs, dataset, m).mean() for m in BASELINE_MACROS}
    means = {m: v for m, v in means.items() if np.isfinite(v)}
    return max(means, key=means.get) if means else None


def table_stacking(runs: pd.DataFrame) -> str:
    comparators = {d: comparator(runs, d) for d in DATASETS}
    rows = []
    for macro in STACK_MACROS + ["Comparator", "Gma"]:
        name = {"Comparator": "Pembanding terkuat$^{d}$", "Gma": "GMA-TabNet tunggal"}.get(
            macro, ARMS.get(macro, Arm("", "", macro, macro)).name)
        row = [name]
        for dataset in DATASETS:
            target = comparators[dataset] if macro == "Comparator" else macro
            if target is None:
                row += ["--", "--"]
                continue
            row.append(f4(series(runs, dataset, target).mean()))
            row.append("--" if target == "StackGma" else delta_cell(paired(runs, dataset, target, "StackGma")))
        rows.append(row)
    names = "; ".join(f"{SHORT[d]}: {re.sub(r'[$][^$]*[$]', '', ARMS[c].name)}"
                      for d, c in comparators.items() if c)
    return table_module.latex_table(
        ["Model"] + ["ROC-AUC", "$\\Delta$ $\\pm$ SK 95\\%"] * len(DATASETS), rows,
        caption="Stacking heterogen (ROC-AUC rata-rata atas 15 lipatan). Pembelajar level satu dan meta-learner tetap; hanya ekstraktor dan "
                "protokol yang berubah. $\\Delta$: baris dikurangi stack GMA-TabNet bersarang, "
                "$\\pm$ setengah lebar SK 95\\% terkoreksi.",
        label="tab:p2-stacking", column_format="l" + "cc" * len(DATASETS), groups=dataset_groups(2),
        notes=f"$^{{d}}$Model pembanding dengan ROC-AUC rata-rata tertinggi di antara 14 pembanding ({names}).",
    )


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
METHODS = [("gma_mask", "Mask"), ("gma_ig", "IG"), ("gma_permutation", "Perm."),
           ("xgb_shap", "SHAP"), ("xgb_permutation", "Perm."), ("stack_permutation", "Perm.")]


def load_attribution(dataset: str) -> tuple[list[str], dict[str, np.ndarray], list[dict]]:
    """Feature names and per-fold share matrices (folds x features) for each method."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in
               sorted((paper_dir() / "interpretability").glob(f"{dataset}__attribution__r0f*.json"))]
    if not records:
        return [], {}, []
    features = records[0]["features"]
    shares = {m: np.array([r["global"][m] for r in records]) for m, _ in METHODS if m in records[0]["global"]}
    stack = [json.loads(p.read_text(encoding="utf-8")) for p in
             sorted((paper_dir() / "interpretability").glob(f"{dataset}__stack_permutation__r0f*.json"))]
    if stack:
        order = [stack[0]["features"].index(f) for f in features]
        shares["stack_permutation"] = np.array([np.asarray(s["share"])[order] for s in stack])
    return features, shares, records


def table_attribution() -> str:
    from src.paper2_credit.data import short_name

    rows, legend = [], []
    for dataset in DATASETS:
        features, shares, _ = load_attribution(dataset)
        if not features:
            continue
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{LABEL[dataset]}}}}} \\\\")
        average = np.mean([shares[m].mean(axis=0) for m in shares], axis=0)
        for index in np.argsort(-average)[:5]:
            label = short_name(dataset, features[index])
            if label != features[index] and dataset != "uci_dccc":
                # Long CamelCase names cannot hyphenate; let them break between words.
                breakable = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", r"\\allowbreak ",
                                   table_module.escape(features[index]))
                legend.append(f"{label} = \\texttt{{{breakable}}}")
            rows.append([table_module.escape(label)] + [
                f"{100 * shares[m][:, index].mean():.1f}" if m in shares else "--" for m, _ in METHODS])
    return table_module.latex_table(
        ["Fitur"] + [name for _, name in METHODS], rows,
        caption="Pangsa atribusi global (\\%, rata-rata atas 5 lipatan ulangan pertama) untuk lima fitur "
                "teratas per dataset. Mask, IG dan Perm. pada GMA-TabNet; SHAP dan Perm. pada XGBoost; "
                "Perm. terakhir pada stack GMA-TabNet bersarang.",
        label="tab:p2-attribution", column_format="lrrrrrr",
        groups=[("", 1), ("GMA-TabNet", 3), ("XGBoost", 2), ("Stack", 1)],
        notes="Singkatan: " + "; ".join(dict.fromkeys(legend)) + "." if legend else None,
    )


def attribution_stats(dataset: str) -> dict:
    from scipy.stats import spearmanr

    features, shares, records = load_attribution(dataset)
    if not features:
        return {}

    def rho(a, b):
        return float(np.mean([spearmanr(x, y).statistic for x, y in zip(shares[a], shares[b])]))

    def mean_of(key, method):
        return float(np.mean([r[key][method] for r in records]))

    out = {"W": {m: kendall_w(shares[m]) for m in shares if len(shares[m]) > 1}}
    pairs = {"mask~ig": ("gma_mask", "gma_ig"), "mask~perm": ("gma_mask", "gma_permutation"),
             "ig~perm": ("gma_ig", "gma_permutation"), "shap~perm": ("xgb_shap", "xgb_permutation"),
             "permgma~permxgb": ("gma_permutation", "xgb_permutation"),
             "permgma~permstack": ("gma_permutation", "stack_permutation"),
             "permxgb~permstack": ("xgb_permutation", "stack_permutation"),
             "mask~shap": ("gma_mask", "xgb_shap")}
    out["rho"] = {k: rho(*v) for k, v in pairs.items() if v[0] in shares and v[1] in shares}
    out["perturb_rho"] = {m: mean_of("perturbation_spearman", m) for m in ("gma_mask", "gma_ig", "xgb_shap")}
    out["perturb_jaccard"] = {m: mean_of("reason_stability_jaccard", m) for m in ("gma_mask", "gma_ig", "xgb_shap")}
    out["reason_jaccard"] = {k: float(np.nanmean([r["reason_agreement_jaccard"][k] for r in records]))
                             for k in records[0]["reason_agreement_jaccard"]}
    out["top1_global"] = {m: mean_of("top1_matches_global", m) for m in ("gma_mask", "gma_ig", "xgb_shap")}
    average = {m: shares[m].mean(axis=0) for m in shares}
    out["top"] = {m: features[int(np.argmax(v))] for m, v in average.items()}
    out["top_share"] = {m: float(np.max(v)) for m, v in average.items()}
    return out


def table_attribution_reliability() -> str:
    stats = {d: attribution_stats(d) for d in DATASETS}

    def cell(dataset, *keys):
        value = stats[dataset]
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        return f"{value:.2f}" if isinstance(value, float) and np.isfinite(value) else "--"

    spec = [
        ("Stabilitas antar lipatan (Kendall $W$)", None),
        ("Mask GMA", ("W", "gma_mask")), ("IG GMA", ("W", "gma_ig")),
        ("Permutasi GMA", ("W", "gma_permutation")), ("TreeSHAP XGBoost", ("W", "xgb_shap")),
        ("Permutasi XGBoost", ("W", "xgb_permutation")), ("Permutasi stack", ("W", "stack_permutation")),
        ("Stabilitas thd. gangguan input (Spearman / Jaccard@3 alasan)", None),
        ("Mask GMA", "mask_perturb"), ("IG GMA", "ig_perturb"), ("TreeSHAP XGBoost", "shap_perturb"),
        ("Kesepakatan peringkat global (Spearman)", None),
        ("Mask -- IG (model sama)", ("rho", "mask~ig")),
        ("Mask -- permutasi (model sama)", ("rho", "mask~perm")),
        ("TreeSHAP -- permutasi (model sama)", ("rho", "shap~perm")),
        ("Permutasi GMA -- XGBoost (antar model)", ("rho", "permgma~permxgb")),
        ("Permutasi GMA -- stack", ("rho", "permgma~permstack")),
        ("Mask GMA -- TreeSHAP XGBoost", ("rho", "mask~shap")),
        ("Alasan per pemohon, desil risiko teratas (Jaccard@3)", None),
        ("Mask -- IG", ("reason_jaccard", "gma_mask~gma_ig")),
        ("IG -- TreeSHAP", ("reason_jaccard", "gma_ig~xgb_shap")),
        ("Alasan utama = fitur teratas global (proporsi)", None),
        ("Mask GMA", ("top1_global", "gma_mask")), ("IG GMA", ("top1_global", "gma_ig")),
        ("TreeSHAP XGBoost", ("top1_global", "xgb_shap")),
    ]
    rows = []
    for label, key in spec:
        if key is None:
            rows.append(f"\\multicolumn{{{1 + len(DATASETS)}}}{{l}}{{\\textit{{{label}}}}} \\\\")
            continue
        row = [label]
        for dataset in DATASETS:
            if isinstance(key, str):
                method = {"mask_perturb": "gma_mask", "ig_perturb": "gma_ig", "shap_perturb": "xgb_shap"}[key]
                a, b = cell(dataset, "perturb_rho", method), cell(dataset, "perturb_jaccard", method)
                row.append(f"{a} / {b}")
            else:
                row.append(cell(dataset, *key))
        rows.append(row)
    return table_module.latex_table(
        ["Ukuran"] + [SHORT[d] for d in DATASETS], rows,
        caption="Keandalan atribusi pada 5 lipatan ulangan pertama: stabilitas, kesepakatan metode dalam "
                "model yang sama dan antar model, serta alasan per pemohon.",
        label="tab:p2-attr-reliability", column_format="lccc",
    )


def load_protocol() -> dict | None:
    path = paper_dir() / "protocol" / "give_me_some_credit__protocol_sensitivity.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table_protocol_sensitivity() -> str:
    payload = load_protocol()
    labels = {"wang": "NearMiss-2 sebelum pembagian~\\cite{wang2025tabnetrating}",
              "train_only": "NearMiss-2 hanya pada data latih", "none": "Tanpa penyeimbangan"}
    rows = []
    if payload:
        frame = pd.DataFrame(payload["records"])
        for protocol in ("wang", "train_only", "none"):
            part = frame[frame.protocol == protocol]
            rows.append([labels[protocol], f"{int(part.n_train.median()):,}".replace(",", "{,}"),
                         f"{100 * part.test_positive_rate.mean():.1f}",
                         mean_sd(part.xgboost), mean_sd(part.gma)])
        rows.append(["Dilaporkan Wang dan Zhang (2025)", "--", "--",
                     f"{payload['reported_by_wang_2025']['xgboost_auc']:.3f}",
                     f"{payload['reported_by_wang_2025']['improved_tabnet_auc']:.3f}$^{{e}}$"])
    return table_module.latex_table(
        ["Protokol", "Baris latih", "Positif uji (\\%)", "XGBoost", "GMA-TabNet"], rows,
        caption="Sensitivitas ROC-AUC GMSC terhadap urutan penyeimbangan dan pembagian data "
                "(5 pembagian 4:1, rata-rata $\\pm$ simpangan baku).",
        label="tab:p2-protocol-sensitivity", column_format="lcccc",
        notes="$^{e}$Model mereka (TabNet dengan atensi multi-kepala), bukan GMA-TabNet.",
    )


TABLE_BUILDERS = {
    "paper2_protocol": table_protocol,
    "paper2_baselines": table_baselines,
    "paper2_architecture": table_architecture,
    "paper2_gap": table_gap,
    "paper2_hpo": table_hpo,
    "paper2_stacking": table_stacking,
    "paper2_attribution": lambda runs: table_attribution(),
    "paper2_attr_reliability": lambda runs: table_attribution_reliability(),
    "paper2_protocol_sensitivity": lambda runs: table_protocol_sensitivity(),
}


def build_tables(output_dir: Path, layout=None) -> dict[str, str]:
    runs = load_runs()
    layout = layout or table_module.JACOST
    written = {}
    with table_module.use_layout(layout):
        for name, builder in TABLE_BUILDERS.items():
            try:
                content = builder(runs)
            except Exception as error:  # noqa: BLE001 - one table must not stop the rest
                LOGGER.error("%s failed: %s: %s", name, type(error).__name__, error)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{name}.tex").write_text(content, encoding="utf-8")
            written[name] = content
    return written


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figure_hpo(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from src.analysis.figures import PALETTE, save

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.3), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        searches = {o: [s for s in load_tuning(dataset, "gma", o) if s["split"].startswith("r0")]
                    for o, _, _ in OPTIMISERS}
        if not any(searches.values()):
            axis.set_visible(False)
            continue
        budget = max(s["budget_seconds"] for v in searches.values() for s in v)
        grid = np.linspace(0, budget, 300)
        best_any = {}
        for runs in searches.values():
            for s in runs:
                best_any[s["split"]] = max(best_any.get(s["split"], -np.inf), s["best_value"])
        for colour, (optimiser, name, _) in zip(PALETTE, OPTIMISERS):
            curves = []
            for s in searches[optimiser]:
                done = [(h["ended"], h["value"]) for h in s["history"] if h["status"] == "ok"]
                curve = np.full(len(grid), np.nan)
                incumbent = -np.inf
                for ended, value in sorted(done):
                    incumbent = max(incumbent, value)
                    curve[grid >= ended] = best_any[s["split"]] - incumbent
                curves.append(curve)
            if not curves:
                continue
            curves = np.array(curves)
            with np.errstate(all="ignore"):
                median = np.nanmedian(curves, axis=0)
                low, high = np.nanpercentile(curves, 25, axis=0), np.nanpercentile(curves, 75, axis=0)
            axis.step(grid, median, where="post", color=colour, label=name, linewidth=1.3)
            axis.fill_between(grid, low, high, step="post", color=colour, alpha=0.12, linewidth=0)
        axis.axvline(budget, color="#7f8c8d", linestyle=":", linewidth=1)
        axis.set_title(SHORT[dataset])
        axis.set_xlabel("waktu (detik)")
        axis.set_xlim(0, budget * 1.02)
    visible = [a for a in axes if a.get_visible()]
    if visible:
        visible[0].set_ylabel("regret validasi")
        visible[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, output_dir, "p2_hpo_convergence")


FOREST = [("Gma", "VanTuned", "vs. TabNet vanilla disetel"),
          ("Gma", "Van", "vs. TabNet vanilla, konfig. GMA"),
          ("Gma", "Matched", "vs. vanilla kapasitas setara"),
          ("Gma", "UniformMix", "vs. campuran seragam"),
          ("Gma", "Wang", "vs. MHA-TabNet (Wang)"),
          ("StackGma", "StackNone", "stack: vs. tanpa ekstraktor"),
          ("StackGma", "Comparator", "stack: vs. pembanding terkuat")]


def figure_forest(output_dir: Path, runs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    from src.analysis.figures import PALETTE, save

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.6), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        best = comparator(runs, dataset)
        for row, (a, b, label) in enumerate(FOREST):
            b = best if b == "Comparator" else b
            result = paired(runs, dataset, a, b) if b else None
            if result is None:
                continue
            axis.errorbar(result["mean"], row, xerr=result["half_width"], fmt="o",
                          color=PALETTE[0 if a == "Gma" else 1], markersize=3.5, capsize=2, linewidth=1)
        axis.axvline(0, color="#7f8c8d", linewidth=0.8)
        axis.set_title(SHORT[dataset])
        axis.set_xlabel("$\\Delta$ ROC-AUC")
    axes[0].set_yticks(range(len(FOREST)))
    axes[0].set_yticklabels([label for _, _, label in FOREST])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, output_dir, "p2_forest")


def figure_temperatures(output_dir: Path, runs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    from src.analysis.figures import PALETTE, save

    fig, axis = plt.subplots(figsize=(COLUMN_IN, 2.1))
    rng = np.random.default_rng(0)
    for position, dataset in enumerate(DATASETS):
        subset = runs[(runs.dataset == dataset) & (runs.stage == "architecture") & (runs.arm == "gma_full")]
        values = [t for temps in subset.temperatures for t in temps]
        if not values:
            continue
        axis.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=5, alpha=0.5,
                     color=PALETTE[position], linewidths=0)
    axis.axhline(1.0, color="#7f8c8d", linestyle="--", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_xticks(range(len(DATASETS)))
    axis.set_xticklabels([SHORT[d] for d in DATASETS])
    axis.set_ylabel("suhu $\\tau_h$ (log)")
    fig.tight_layout()
    save(fig, output_dir, "p2_temperatures")


def build_figures(output_dir: Path) -> None:
    from src.analysis.figures import apply_style, figure_gma_tabnet

    runs = load_runs()
    apply_style(serif=True)
    for name, call in (("schematic", lambda: figure_gma_tabnet(output_dir, "id", COLUMN_IN)),
                       ("hpo", lambda: figure_hpo(output_dir)),
                       ("forest", lambda: figure_forest(output_dir, runs)),
                       ("temperatures", lambda: figure_temperatures(output_dir, runs))):
        try:
            call()
        except Exception as error:  # noqa: BLE001
            LOGGER.error("figure %s failed: %s: %s", name, type(error).__name__, error)


# --------------------------------------------------------------------------- #
# Macros
# --------------------------------------------------------------------------- #
class Macros(MacroSet):
    def __init__(self) -> None:
        super().__init__()
        self.triples: list[tuple[str, str, str]] = []

    def add_paired(self, stem: str, result: dict | None, a: str | None = None, b: str | None = None) -> None:
        if result is None:
            for suffix in ("Delta", "Half", "Low", "High", "P", "PHolm", "PRel", "PHolmRel", "Wins", "N"):
                self.values[stem + suffix] = PLACEHOLDER
            return
        self.add(stem + "Delta", result["shown_delta"], signed=True)
        self.add(stem + "Half", result["half_width"])
        self.add(stem + "Low", result["ci_low"], signed=True)
        self.add(stem + "High", result["ci_high"], signed=True)
        for suffix, p in (("P", result["p"]), ("PHolm", result.get("p_holm", result["p"]))):
            text = p_text(p).replace("$<$", "<")
            self.add_text(stem + suffix, text)
            # With its own relation sign, for "$p\XPRel{}$": "=0.432" or "<0.001".
            self.add_text(stem + suffix + "Rel", text if text.startswith("<") else "=" + text)
        self.add_text(stem + "Wins", str(result["n_positive"]))
        self.add_text(stem + "N", str(result["n"]))
        # A triple is exact only when both printed means are over the paired folds.
        if a and b and result.get("complete", True):
            self.triples.append((a, b, stem + "Delta"))


def write_macros(path: Path, manuscript: Path | None = None) -> Macros:
    from src.common.io import load_dataset

    runs = load_runs()
    m = Macros()
    m.add_text("NRepeats", "3")
    m.add_text("NFoldsPerRepeat", "5")
    m.add_text("NOuterFolds", "15")
    m.add_int("ExplainRows", 4000)

    f1 = {}
    for dataset in DATASETS:
        P = PREFIX[dataset]
        data = load_dataset(dataset)
        m.add_int(P + "Rows", data.n_samples)
        m.add(P + "PosPct", data.minority_pct / 100, decimals=1, percent=True)
        subset = runs[(runs.dataset == dataset) & (runs.stage == "architecture")] if not runs.empty else runs
        if not subset.empty:
            m.add_int(P + "OuterTrain", int((subset.n_train + subset.n_val).median()))
            m.add_int(P + "OptRows", int(subset.n_train.median()))
            m.add_int(P + "ValRows", int(subset.n_val.median()))
            m.add_int(P + "TestRows", int(subset.n_test.median()))
        gma, van = load_tuning(dataset, "gma", "tpe"), load_tuning(dataset, "vanilla", "tpe")
        if gma:
            m.add_int(P + "SearchRows", int(np.median([g["search_rows"] for g in gma])))
            m.add_int(P + "Budget", int(gma[0]["budget_seconds"]))
            m.add_int(P + "EvalsGma", int(np.median([g["n_evaluations"] for g in gma])))
            m.add_int(P + "EvalsGmaMin", int(np.min([g["n_evaluations"] for g in gma])))
            m.add_int(P + "Interrupted", sum(g.get("n_interrupted", 0) for g in gma + van))
        if van:
            m.add_int(P + "EvalsVan", int(np.median([g["n_evaluations"] for g in van])))

        for macro in ARMS:
            values = series(runs, dataset, macro)
            m.add(P + macro + "AUC", values.mean() if len(values) else None)
            m.add(P + macro + "SD", values.std(ddof=1) if len(values) > 1 else None)
            ks = series(runs, dataset, macro, "ks")
            m.add(P + macro + "KS", ks.mean() if len(ks) else None)

        family = architecture_family(runs, dataset, ["VanTuned"] + MECHANISM_MACROS + CONTROL_MACROS)
        for macro, result in family.items():
            m.add_paired(f"{P}{macro}VsGma", result, P + macro + "AUC", P + "GmaAUC")
        f1[dataset] = paired(runs, dataset, "Gma", "VanTuned")
        for index in range(1, len(GAP_CHAIN)):
            a, b = GAP_CHAIN[index], GAP_CHAIN[index - 1]
            m.add_paired(f"{P}Gap{a}Vs{b}", paired(runs, dataset, a, b), P + a + "AUC", P + b + "AUC")

        best = comparator(runs, dataset)
        if best:
            m.add_text(P + "ComparatorName", ARMS[best].name.split("$")[0].strip())
            m.add(P + "ComparatorAUC", series(runs, dataset, best).mean())
            m.add_paired(P + "StackGmaVsComparator", paired(runs, dataset, "StackGma", best),
                         P + "StackGmaAUC", P + "ComparatorAUC")
            m.add_paired(P + "GmaVsComparator", paired(runs, dataset, "Gma", best),
                         P + "GmaAUC", P + "ComparatorAUC")
        for macro in ("StackVan", "StackNone", "StackShared"):
            m.add_paired(f"{P}StackGmaVs{macro}", paired(runs, dataset, "StackGma", macro),
                         P + "StackGmaAUC", P + macro + "AUC")

        hpo = hpo_summary(runs, dataset)
        val_means = []
        for optimiser, summary in hpo.items():
            stem = P + "Hpo" + optimiser.capitalize()
            val_mean = float(np.mean(summary["val"])) if summary["val"] else None
            if val_mean is not None:
                val_means.append(val_mean)
            m.add(stem + "Val", val_mean)
            m.add(stem + "Test", summary["test"].mean() if len(summary["test"]) else None)
            m.add_int(stem + "Evals", int(np.median(summary["evals"])) if summary["evals"] else None)
            if summary["delta"]:
                m.add_paired(stem + "VsRandom", summary["delta"])
        m.add(P + "HpoValSpread", max(val_means) - min(val_means) if len(val_means) > 1 else None)
        tests = [s["test"].mean() for s in hpo.values() if len(s["test"])]
        m.add(P + "HpoTestSpread", max(tests) - min(tests) if len(tests) > 1 else None)

        temps = [t for ts in runs[(runs.dataset == dataset) & (runs.arm == "gma_full")].temperatures
                 for t in ts] if not runs.empty else []
        if temps:
            m.add(P + "TempMin", min(temps), decimals=2)
            m.add(P + "TempMax", max(temps), decimals=2)
            m.add(P + "TempMedian", float(np.median(temps)), decimals=2)
        for macro in ("Gma", "Matched", "VanTuned"):
            arm = ARMS[macro]
            subset = runs[(runs.dataset == dataset) & (runs.arm == arm.arm)] if not runs.empty else runs
            if not subset.empty and subset.parameters.notna().any():
                m.add_int(P + "Params" + macro, int(subset.parameters.median()))

        stats = attribution_stats(dataset)
        if stats:
            from src.paper2_credit.data import short_name

            for method, key in (("gma_mask", "Mask"), ("gma_ig", "Ig"), ("gma_permutation", "PermGma"),
                                ("xgb_shap", "Shap"), ("xgb_permutation", "PermXgb"),
                                ("stack_permutation", "PermStack")):
                if method in stats["top"]:
                    m.add_text(P + "Top" + key, "\\texttt{" + table_module.escape(
                        short_name(dataset, stats["top"][method])) + "}")
                    m.add(P + "Top" + key + "Share", stats["top_share"][method], decimals=1, percent=True)
                if method in stats["W"]:
                    m.add(P + "W" + key, stats["W"][method], decimals=2)
            for key, value in stats["rho"].items():
                m.add(P + "Rho" + "".join(w.capitalize() for w in key.split("~")), value, decimals=2)
            for method, key in (("gma_mask", "Mask"), ("gma_ig", "Ig"), ("xgb_shap", "Shap")):
                m.add(P + "PerturbRho" + key, stats["perturb_rho"][method], decimals=2)
                m.add(P + "PerturbJac" + key, stats["perturb_jaccard"][method], decimals=2)
                m.add(P + "TopOneGlobal" + key, stats["top1_global"][method], decimals=0, percent=True)
            m.add(P + "JacMaskIg", stats["reason_jaccard"].get("gma_mask~gma_ig"), decimals=2)
            m.add(P + "JacIgShap", stats["reason_jaccard"].get("gma_ig~xgb_shap"), decimals=2)

    available = {d: r["p"] for d, r in f1.items() if r}
    for dataset, p in (holm(available) if available else {}).items():
        f1[dataset]["p_holm"] = p
    for dataset, result in f1.items():
        P = PREFIX[dataset]
        m.add_paired(P + "GmaVsVanTuned", result, P + "GmaAUC", P + "VanTunedAUC")

    payload = load_protocol()
    if payload:
        frame = pd.DataFrame(payload["records"])
        for protocol, key in (("wang", "Wang"), ("train_only", "TrainOnly"), ("none", "NoBalance")):
            part = frame[frame.protocol == protocol]
            m.add("Protocol" + key + "Xgb", part.xgboost.mean())
            m.add("Protocol" + key + "Gma", part.gma.mean())
            m.add("Protocol" + key + "TestPos", part.test_positive_rate.mean(), decimals=1, percent=True)
        for model in ("Xgb", "Gma"):
            a = round(m_value(m, f"ProtocolWang{model}"), 4)
            b = round(m_value(m, f"ProtocolTrainOnly{model}"), 4)
            m.add(f"ProtocolGap{model}", a - b, signed=True)
            m.triples.append((f"ProtocolWang{model}", f"ProtocolTrainOnly{model}", f"ProtocolGap{model}"))

    add_v1_macros(m)
    add_missing_placeholders(m, manuscript or MANUSCRIPT)
    m.write(path)
    write_json(path.with_name("macro_triples.json"), {"triples": m.triples})
    return m


MANUSCRIPT = PROJECT_ROOT / "papers" / "paper2_credit"
MACRO_PREFIXES = ("Dccc", "German", "Gmsc", "Protocol", "Prev", "NOuter", "NRepeats",
                  "NFolds", "ExplainRows")


def add_missing_placeholders(macros: MacroSet, manuscript: Path = MANUSCRIPT) -> None:
    """Define every result macro the manuscript uses, as a placeholder if need be.

    A stage that has not run yet leaves its macros undefined, and one undefined
    macro stops the proof from compiling at all. A visible red placeholder keeps
    a partial draft readable and makes the gap obvious.
    """
    import re

    sources = list((manuscript / "sections_id").glob("*.tex"))
    sources += [manuscript / "frontmatter_jacost.tex", manuscript / "response" / "response_letter.md"]
    used = set()
    for path in sources:
        if path.exists():
            used |= set(re.findall(r"\\([A-Z][A-Za-z]+)", path.read_text(encoding="utf-8")))
    for name in used:
        if name.startswith(MACRO_PREFIXES) and name not in macros.values:
            macros.values[name] = PLACEHOLDER


def m_value(macros: MacroSet, name: str) -> float:
    try:
        return float(macros.values[name])
    except (KeyError, ValueError):
        return float("nan")


def add_v1_macros(m: Macros) -> None:
    """Headline numbers of the first submission, for the response letter."""
    directory = PROJECT_ROOT / "results" / "paper2_v1_ijies" / "raw_runs"
    if not directory.exists():
        return
    records = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
    frame = pd.DataFrame([{"dataset": r["dataset"], "arm": r["arm"], "auc": r["metrics"]["roc_auc"]}
                          for r in records])
    for dataset in DATASETS:
        P = PREFIX[dataset]
        for arm, key in (("gma_full", "Gma"), ("tabnet_vanilla", "Van"), ("stack_gma", "StackGma"),
                         ("tabnet_reference", "RefDefault")):
            values = frame[(frame.dataset == dataset) & (frame.arm == arm)].auc
            m.add(f"Prev{P}{key}AUC", values.mean() if len(values) else None)


# --------------------------------------------------------------------------- #
def build(tables_dir: Path, figures_dir: Path) -> dict:
    tables = build_tables(tables_dir)
    build_figures(figures_dir)
    macros = write_macros(tables_dir / "macros.tex")
    missing = sum(1 for v in macros.values.values() if "??" in v)
    LOGGER.info("paper 2: %d tables, %d macros (%d awaiting results)", len(tables),
                len(macros.values), missing)
    return tables
