"""Tables, figures and macros for the second JACOST paper (paper2b).

The paper asks two questions of one controlled testbed (three credit datasets,
nested 3x5 cross-validation, equal tuning budgets):

1. How far do common evaluation shortcuts move the conclusion about a proposed
   deep tabular model (GMA-TabNet against TabNet), compared with the reference
   protocol? Each shortcut gets the estimate a study taking it would report,
   the verdict a naive analysis would draw from it (one 5-fold run, an
   uncorrected paired t-test), and the verdict under the reference protocol.
2. Is feature-token attention the modification of TabNet that helps, and which
   part of it? Equal-budget tuning, module ablations and use as a stack
   extractor.

Everything is built on ``src.analysis.paper2``: the same run loader, fold
pairing (with partition-hash checks), corrected resampled t intervals and Holm
adjustment, and the same macro conventions. Output goes to
``results/paper2/paper2b/{tables,figures}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.analysis import paper2 as base
from src.analysis import tables as table_module
from src.analysis.paper2 import (
    ARMS, DATASETS, LABEL, PREFIX, SHORT, Arm, Macros, comparator, dataset_groups, delta_cell,
    f4, holm, load_runs, p_text, paired, series, signed,
)
from src.common.io import PROJECT_ROOT, results_root, write_json

LOGGER = logging.getLogger("analysis.paper2b")
MANUSCRIPT = PROJECT_ROOT / "papers" / "paper2b_jacost"
COLUMN_IN = base.COLUMN_IN
TEXT_IN = base.TEXT_IN

NEW_ARMS = [
    Arm("token_study", "token_full", "TabNet + atensi token (disetel)", "Tok"),
    Arm("token_study", "token_local", "hanya modul lokal", "TokLocal"),
    Arm("token_study", "token_global", "hanya modul global", "TokGlobal"),
    Arm("token_study", "token_only", "token tanpa enkoder TabNet", "TokOnly"),
    Arm("token_study", "token_gma", "token + masker GMA", "TokGma"),
    Arm("token_study", "token_matched_vanilla", "TabNet vanilla, kapasitas setara", "TokMatched"),
    Arm("selection_audit", "gma_fold0cfg", "GMA-TabNet, konfigurasi lipatan-0", "GmaFzero"),
    Arm("selection_audit", "vanilla_fold0cfg", "TabNet vanilla, konfigurasi lipatan-0", "VanFzero"),
    Arm("selection_audit", "token_fold0cfg", "TabNet + token, konfigurasi lipatan-0", "TokFzero"),
    Arm("stacking", "stack_token_nested", "Stack + TabNet token (bersarang)", "StackTok"),
]
ARMS.update({a.macro: a for a in NEW_ARMS})
TOKEN_ABLATIONS = ["TokLocal", "TokGlobal", "TokOnly", "TokGma", "TokMatched", "Wang"]
RESAMPLERS = [("ros", "Ros", "ROS"), ("rus", "Rus", "RUS"), ("smote", "Smote", "SMOTE"),
              ("nearmiss2", "Nearmiss", "NearMiss-2")]


def out_dir() -> Path:
    return results_root() / "paper2" / "paper2b"


# --------------------------------------------------------------------------- #
# Naive analysis: what a single 5-fold study would conclude
# --------------------------------------------------------------------------- #
def naive_verdict(diffs) -> dict:
    """What a single run of k folds would report.

    The verdict is the ranking by mean, which is how applied studies usually
    declare a winner (the first submission did, with a "several fold standard
    deviations" argument); ``p`` is a plain paired t-test with no correction,
    reported beside it.
    """
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2:
        return {"mean": np.nan, "p": np.nan, "verdict": "--", "significant": False}
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.inf * np.sign(d.mean())
    p = float(2 * sps.t.sf(abs(t), len(d) - 1)) if np.isfinite(t) else 0.0
    verdict = "menang" if d.mean() > 0 else "kalah"
    return {"mean": float(d.mean()), "p": p, "verdict": verdict, "significant": p < 0.05}


def fold_diffs(runs, dataset, a, b, repeat=None, folds=None) -> np.ndarray:
    sa, sb = series(runs, dataset, a), series(runs, dataset, b)
    common = sa.index.intersection(sb.index)
    if repeat is not None:
        common = common[common.get_level_values(0) == repeat]
    if folds is not None:
        common = common[common.get_level_values(1).isin(folds)]
    return (sa.loc[common] - sb.loc[common]).to_numpy()


def reference_verdict(result: dict | None) -> str:
    if result is None:
        return "--"
    if result["ci_low"] > 0:
        return "menang"
    if result["ci_high"] < 0:
        return "kalah"
    return "tak terbedakan"


def shortcuts(runs: pd.DataFrame, dataset: str) -> dict[str, dict]:
    """The proposal (GMA-TabNet) against TabNet under each evaluation shortcut."""
    out = {}
    reference = paired(runs, dataset, "Gma", "VanTuned")
    out["reference"] = {"result": reference, "verdict": reference_verdict(reference),
                        "naive": naive_verdict(fold_diffs(runs, dataset, "Gma", "VanTuned", repeat=0))}
    unequal = paired(runs, dataset, "Gma", "Van")
    out["unequal"] = {"result": unequal,
                      "naive": naive_verdict(fold_diffs(runs, dataset, "Gma", "Van", repeat=0))}
    weak = paired(runs, dataset, "Gma", "RefDefault")
    out["weak"] = {"result": weak,
                   "naive": naive_verdict(fold_diffs(runs, dataset, "Gma", "RefDefault", repeat=0))}
    fzero = paired(runs, dataset, "GmaFzero", "VanFzero")
    out["fold0"] = {"result": fzero,
                    "naive": naive_verdict(fold_diffs(runs, dataset, "GmaFzero", "VanFzero", repeat=0)),
                    "optimism_gma": paired(runs, dataset, "GmaFzero", "Gma"),
                    "optimism_van": paired(runs, dataset, "VanFzero", "VanTuned")}
    single = [naive_verdict(fold_diffs(runs, dataset, "Gma", "VanTuned", repeat=r)) for r in range(3)]
    out["single"] = {"repeats": single}
    out["stack"] = {"result": paired(runs, dataset, "StackShared", "StackGma")}
    hpo = base.hpo_summary(runs, dataset)
    vals = {k: np.mean(v["val"]) for k, v in hpo.items() if v["val"]}
    if vals:
        best = max(vals, key=vals.get)
        test = {k: v["test"].mean() for k, v in hpo.items() if len(v["test"])}
        out["optimiser"] = {"best": best, "val_gain": vals[best] - vals.get("random", np.nan),
                            "test_gain": test.get(best, np.nan) - test.get("random", np.nan)}
    return out


# --------------------------------------------------------------------------- #
# Resampling audit
# --------------------------------------------------------------------------- #
def load_resampling(dataset: str) -> pd.DataFrame:
    path = results_root() / "paper2" / "protocol" / f"{dataset}__resampling_audit.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["records"])


def resampling_cell(frame: pd.DataFrame, protocol: str, method: str, model: str) -> dict | None:
    if frame.empty:
        return None
    part = frame[(frame.protocol == protocol) & (frame.resampler == method) & (frame.model == model)]
    if part.empty:
        return None
    return {"reported": float(part.auc_reported.mean()), "holdout": float(part.auc_holdout.mean()),
            "test_pos": float(part.test_positive_rate.mean()), "n": int(len(part))}


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
MAIN_NAMES = {"VanTuned": "TabNet vanilla (disetel)", "RefDefault": "TabNet referensi (setelan awal)",
              "Gma": "GMA-TabNet (disetel)"}


def table_main(runs: pd.DataFrame) -> str:
    """Who wins under the reference protocol."""
    rows_spec = [("Comparator", None), ("XgbTuned", None), ("Xgb", None), ("RefDefault", None),
                 ("VanTuned", None), ("Gma", "VanTuned"), ("Tok", "VanTuned"), ("StackTok", None)]
    rows = []
    for macro, _ in rows_spec:
        if macro == "Comparator":
            row = ["Pembanding terkuat$^{a}$"]
        else:
            row = [MAIN_NAMES.get(macro, ARMS[macro].name)]
        for dataset in DATASETS:
            target = comparator(runs, dataset) if macro == "Comparator" else macro
            values = series(runs, dataset, target) if target else pd.Series(dtype=float)
            row.append(f4(values.mean()) if len(values) else "--")
            if macro in ("Gma", "Tok"):
                row.append(delta_cell(paired(runs, dataset, macro, "VanTuned")))
            elif macro == "StackTok":
                best = comparator(runs, dataset)
                row.append(delta_cell(paired(runs, dataset, "StackTok", best)) if best else "--")
            else:
                row.append("--")
        rows.append(row)
    names = "; ".join(f"{SHORT[d]}: {base.re.sub(r'[$][^$]*[$]', '', ARMS[c].name)}"
                      for d in DATASETS if (c := comparator(runs, d)))
    return table_module.latex_table(
        ["Model"] + ["ROC-AUC", "$\\Delta$"] * len(DATASETS), rows,
        caption="Perbandingan model di bawah protokol rujukan (ROC-AUC rata-rata atas 15 lipatan luar).",
        label="tab:b-main", column_format="l" + "cc" * len(DATASETS), groups=dataset_groups(2),
        notes="$\\Delta$: selisih berpasangan terhadap TabNet vanilla yang disetel, $\\pm$ setengah lebar "
              "SK 95\\% terkoreksi; untuk \\textit{stack}, terhadap pembanding terkuat. "
              f"$^{{a}}$Model dengan ROC-AUC rata-rata tertinggi di antara 14 pembanding ({names}).",
    )


SHORTCUT_ROWS = [
    ("reference", "Protokol rujukan: penyetelan setara, 3$\\times$5 lipatan"),
    ("unequal", "S1 Baseline meminjam konfigurasi usulan"),
    ("weak", "S2 Baseline TabNet tanpa penyetelan"),
    ("fold0", "S3 Konfigurasi dipilih di lipatan-0, dipakai ulang"),
]


def _naive_cell(naive: dict) -> str:
    """The single-run estimate with its ranking arrow and naive-significance star."""
    if not np.isfinite(naive.get("mean", np.nan)):
        return "--"
    tag = {"menang": "$\\uparrow$", "kalah": "$\\downarrow$"}.get(naive["verdict"], "")
    star = "$^{*}$" if naive.get("significant") else ""
    return f"{signed(naive['mean'])} {tag}{star}"


def table_audit(runs: pd.DataFrame) -> str:
    audits = {d: shortcuts(runs, d) for d in DATASETS}
    rows = []
    for key, label in SHORTCUT_ROWS:
        row = [label]
        for dataset in DATASETS:
            entry = audits[dataset][key]
            row.append(_naive_cell(entry["naive"]))
            row.append(delta_cell(entry["result"]))
        rows.append(row)
    row = ["S4 Satu kali 5 lipatan, tiap ulangan"]
    for dataset in DATASETS:
        reps = audits[dataset]["single"]["repeats"]
        row.append(" / ".join(signed(r["mean"], 3) for r in reps))
        row.append(f"{sum(r['verdict'] == 'menang' for r in reps)}/3 ulangan $\\uparrow$")
    rows.append(row)
    row = ["S6 Fitur stack dalam-sampel (lama $-$ bersarang)"]
    for dataset in DATASETS:
        result = audits[dataset]["stack"]["result"]
        row += ["--", delta_cell(result)]
    rows.append(row)
    row = ["S7 Pengoptimasi terbaik di validasi $-$ acak (val / uji)"]
    for dataset in DATASETS:
        opt = audits[dataset].get("optimiser")
        row += [f"{signed(opt['val_gain'], 3)} / {signed(opt['test_gain'], 3)}" if opt else "--", "--"]
    rows.append(row)
    return table_module.latex_table(
        ["Jalan pintas"] + ["5 lipatan", "3$\\times$5, SK 95\\%"] * len(DATASETS), rows,
        caption="Audit protokol: selisih ROC-AUC GMA-TabNet terhadap TabNet di bawah setiap jalan pintas.",
        label="tab:b-audit", column_format="l" + "cc" * len(DATASETS), groups=dataset_groups(2),
        notes="\\textit{5 lipatan}: estimasi satu studi (ulangan pertama) dengan vonis menurut rata-rata "
              "($\\uparrow$ GMA-TabNet lebih tinggi, $\\downarrow$ lebih rendah; $^{*}$ uji-t berpasangan "
              "tanpa koreksi $p<0.05$). \\textit{3$\\times$5}: jalan pintas yang sama atas semua lipatan "
              "dengan SK 95\\% terkoreksi. S3 dinilai pada lipatan 2--5 setiap ulangan.",
        landscape=True,
    )


def table_resampling() -> str:
    frames = {d: load_resampling(d) for d in DATASETS}
    rows = []
    for protocol, label in (("none", "Tanpa penyeimbangan"), ("before_split", "Sebelum pembagian"),
                            ("train_only", "Hanya data latih")):
        methods = [("none", "", "--")] if protocol == "none" else RESAMPLERS
        if protocol != "none":
            rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for method, _, name in methods:
            row = [label if protocol == "none" else name]
            for dataset in DATASETS:
                cell = resampling_cell(frames[dataset], protocol, method, "xgboost")
                row += [f4(cell["reported"]) if cell else "--", f4(cell["holdout"]) if cell else "--"]
            rows.append(row)
    return table_module.latex_table(
        ["Penyeimbangan"] + ["Dilaporkan", "Holdout"] * len(DATASETS), rows,
        caption="ROC-AUC XGBoost menurut urutan penyeimbangan dan pembagian data.",
        label="tab:b-resampling", column_format="l" + "cc" * len(DATASETS), groups=dataset_groups(2),
        notes="Rata-rata lima pembagian. \\textit{Dilaporkan}: pada data uji milik protokol itu sendiri; "
              "\\textit{Holdout}: pada 20\\% data yang disisihkan sejak awal dan tidak pernah diseimbangkan.",
    )


def token_family(runs, dataset) -> dict:
    results = {m: paired(runs, dataset, m, "Tok") for m in TOKEN_ABLATIONS}
    available = {m: r["p"] for m, r in results.items() if r is not None}
    for m, p in (holm(available) if available else {}).items():
        results[m]["p_holm"] = p
    return results


def table_token(runs: pd.DataFrame) -> str:
    families = {d: token_family(runs, d) for d in DATASETS}
    rows = [["TabNet + atensi token (usulan)"] + sum(
        [[f4(series(runs, d, "Tok").mean()), "--", "--"] for d in DATASETS], [])]
    for macro in TOKEN_ABLATIONS:
        name = "token, konfigurasi GMA (tanpa penyetelan sendiri)" if macro == "Wang" else ARMS[macro].name
        row = [name]
        for dataset in DATASETS:
            result = families[dataset][macro]
            row += [f4(series(runs, dataset, macro).mean()), delta_cell(result),
                    p_text(result.get("p_holm")) if result else "--"]
        rows.append(row)
    return table_module.latex_table(
        ["Konfigurasi"] + ["AUC", "$\\Delta$ $\\pm$ SK 95\\%", "$p_{\\mathrm{Holm}}$"] * len(DATASETS), rows,
        caption="Ablasi TabNet dengan atensi token pada konfigurasi token yang dipilih setiap lipatan.",
        label="tab:b-token", column_format="l" + "ccc" * len(DATASETS), groups=dataset_groups(3),
        notes="$\\Delta$: baris dikurangi model token lengkap, berpasangan atas 15 lipatan luar, $\\pm$ "
              "setengah lebar SK 95\\% terkoreksi; $p_{\\mathrm{Holm}}$ disesuaikan dalam famili ablasi "
              "setiap dataset.",
        landscape=True,
    )


TABLES = {
    "b_protocol": lambda runs: base.table_protocol(
        runs, caption="Ukuran partisi dan anggaran pencarian per lipatan luar.",
        notes="Median atas 15 lipatan luar. Pencarian hiperparameter dilatih pada baris pencarian dan "
              "dinilai pada blok validasi; keduanya berada di dalam partisi latih luar."),
    "b_main": table_main,
    "b_audit": table_audit,
    "b_resampling": lambda runs: table_resampling(),
    "b_token": table_token,
}


def build_tables(output_dir: Path, layout=None) -> dict[str, str]:
    runs = load_runs()
    written = {}
    with table_module.use_layout(layout or table_module.JACOST):
        for name, builder in TABLES.items():
            try:
                content = builder(runs)
            except Exception as error:  # noqa: BLE001
                LOGGER.error("%s failed: %s: %s", name, type(error).__name__, error)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{name}.tex").write_text(content, encoding="utf-8")
            written[name] = content
    return written


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figure_audit(output_dir: Path, runs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    from src.analysis.figures import PALETTE, save

    labels = ["rujukan (3$\\times$5, setara)", "S1 konfigurasi dipinjam", "S2 baseline tak disetel",
              "S3 seleksi lipatan-0", "S4 ulangan 1", "S4 ulangan 2", "S4 ulangan 3"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.5), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        audit = shortcuts(runs, dataset)
        points = []
        ref = audit["reference"]["result"]
        points.append((ref["mean"], ref["half_width"], 0) if ref else None)
        for key in ("unequal", "weak", "fold0"):
            result = audit[key]["result"]
            points.append((result["mean"], result["half_width"], 1) if result else None)
        for rep in range(3):
            d = fold_diffs(runs, dataset, "Gma", "VanTuned", repeat=rep)
            half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            points.append((d.mean(), half, 2) if len(d) else None)
        for row, point in enumerate(points):
            if point is None:
                continue
            mean, half, kind = point
            axis.errorbar(mean, row, xerr=half, fmt="o", color=PALETTE[kind], markersize=3.5, capsize=2,
                          linewidth=1)
        axis.axvline(0, color="#7f8c8d", linewidth=0.8)
        axis.set_title(SHORT[dataset])
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    # One shared label: three copies under narrow panels run into each other.
    fig.supxlabel("$\\Delta$ ROC-AUC (GMA-TabNet $-$ TabNet)", fontsize=8)
    fig.tight_layout()
    save(fig, output_dir, "b_audit", dpi=600)


def figure_resampling(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from src.analysis.figures import PALETTE, save

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.4), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        frame = load_resampling(dataset)
        if frame.empty:
            axis.set_visible(False)
            continue
        none = resampling_cell(frame, "none", "none", "xgboost")
        for i, (method, _, name) in enumerate(RESAMPLERS):
            for offset, protocol, colour in ((-0.15, "before_split", PALETTE[1]), (0.15, "train_only", PALETTE[0])):
                cell = resampling_cell(frame, protocol, method, "xgboost")
                if cell is None:
                    continue
                axis.plot([i + offset, i + offset], [cell["holdout"], cell["reported"]], color=colour,
                          linewidth=1)
                axis.scatter([i + offset], [cell["reported"]], color=colour, s=14, zorder=3,
                             label=("sebelum pembagian" if protocol == "before_split" else "hanya data latih")
                             if i == 0 else None)
                axis.scatter([i + offset], [cell["holdout"]], facecolors="white", edgecolors=colour, s=14,
                             zorder=3)
        if none:
            axis.axhline(none["holdout"], color="#7f8c8d", linestyle="--", linewidth=0.8)
        axis.set_xticks(range(len(RESAMPLERS)))
        axis.set_xticklabels([name.replace("NearMiss-", "Near-\nMiss-") for _, _, name in RESAMPLERS],
                             fontsize=7)
        axis.set_title(SHORT[dataset])
    visible = [a for a in axes if a.get_visible()]
    if visible:
        visible[0].set_ylabel("ROC-AUC XGBoost")
        visible[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, output_dir, "b_resampling", dpi=600)


def figure_token_schematic(output_dir: Path) -> None:
    """TabNet encoder and feature-token attention side by side, into one output layer."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from src.analysis.figures import save

    fig = plt.figure(figsize=(COLUMN_IN - 0.04, 2.7))
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7.4)
    axis.set_axis_off()
    ink, faint = "#2c3e50", "#95a5a6"

    def box(x0, y0, x1, y1, text, size=8.5, colour=ink):
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white", edgecolor=colour, linewidth=0.9))
        axis.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size, color=ink,
                  linespacing=1.15)

    def arrow(a, b):
        axis.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, linewidth=0.85, color=ink))

    # Box widths leave room for the longest label at its point size; the first
    # version let three labels run over their borders.
    box(2.4, 6.5, 7.6, 7.2, "masukan $\\mathbf{x}$ (kolom asli)")
    box(0.05, 4.4, 4.95, 5.8, "enkoder TabNet\n(masker sparsemax,\n$T$ langkah)", 8)
    box(5.4, 5.0, 9.95, 5.8, "satu token per kolom")
    box(5.4, 3.6, 7.55, 4.6, "atensi-diri\n(rata-rata)", 8)
    box(7.8, 3.6, 9.95, 4.6, "kueri global\n(dipelajari)", 8)
    box(0.05, 2.0, 9.95, 2.9, "gabungkan: keputusan TabNet $\\oplus$ lokal $\\oplus$ global", 8)
    box(2.4, 0.4, 7.6, 1.2, "lapisan keluaran (logit)")
    arrow((4.4, 6.5), (2.4, 5.82))
    arrow((5.6, 6.5), (7.6, 5.82))
    arrow((6.45, 5.0), (6.45, 4.62))
    arrow((8.75, 5.0), (8.75, 4.62))
    arrow((2.4, 4.4), (2.4, 2.92))
    arrow((6.45, 3.6), (6.45, 2.92))
    arrow((8.75, 3.6), (8.75, 2.92))
    arrow((5.0, 2.0), (5.0, 1.22))
    axis.text(2.6, 3.3, "ablasi: tanpa enkoder\n(token saja)", fontsize=7, color=faint, va="center")
    save(fig, output_dir, "b_token_schematic", pad=0.02, dpi=600)


def build_figures(output_dir: Path) -> None:
    from src.analysis.figures import apply_style

    runs = load_runs()
    apply_style(serif=True)
    for name, call in (("schematic", lambda: figure_token_schematic(output_dir)),
                       ("audit", lambda: figure_audit(output_dir, runs)),
                       ("resampling", lambda: figure_resampling(output_dir))):
        try:
            call()
        except Exception as error:  # noqa: BLE001
            LOGGER.error("figure %s failed: %s: %s", name, type(error).__name__, error)


# --------------------------------------------------------------------------- #
# Macros
# --------------------------------------------------------------------------- #
def write_macros(path: Path) -> Macros:
    """Paper 2's macro set (with the new arms registered), plus paper2b's own."""
    runs = load_runs()
    m = base.write_macros(path, manuscript=MANUSCRIPT)
    # Primary family of this paper: token TabNet against tuned vanilla, across datasets.
    primary = {d: paired(runs, d, "Tok", "VanTuned") for d in DATASETS}
    available = {d: r["p"] for d, r in primary.items() if r is not None}
    for d, p in (holm(available) if available else {}).items():
        primary[d]["p_holm"] = p
    for dataset in DATASETS:
        P = PREFIX[dataset]
        m.add_paired(P + "TokVsVanTuned", primary[dataset], P + "TokAUC", P + "VanTunedAUC")
        for stem, a, b in (("TokVsGma", "Tok", "Gma"),
                           ("StackTokVsStackNone", "StackTok", "StackNone"),
                           ("StackTokVsStackGma", "StackTok", "StackGma")):
            m.add_paired(P + stem, paired(runs, dataset, a, b), P + a + "AUC", P + b + "AUC")
        best = comparator(runs, dataset)
        if best:
            m.add_paired(P + "TokVsComparator", paired(runs, dataset, "Tok", best), P + "TokAUC",
                         P + "ComparatorAUC")
            m.add_paired(P + "StackTokVsComparator", paired(runs, dataset, "StackTok", best),
                         P + "StackTokAUC", P + "ComparatorAUC")
        for macro, result in token_family(runs, dataset).items():
            m.add_paired(f"{P}{macro}VsTok", result, P + macro + "AUC", P + "TokAUC")

        audit = shortcuts(runs, dataset)
        for key, stem in (("reference", "ShortRef"), ("unequal", "ShortUnequal"), ("weak", "ShortWeak"),
                          ("fold0", "ShortFzero")):
            m.add_paired(P + stem, audit[key]["result"])
            m.add_text(P + stem + "Naive", audit[key]["naive"]["verdict"])
            m.add_text(P + stem + "NaiveP", p_text(audit[key]["naive"]["p"]).replace("$<$", "<"))
            m.add(P + stem + "NaiveMean", audit[key]["naive"]["mean"], signed=True)
        m.add_paired(P + "ShortStack", audit["stack"]["result"])
        m.add_paired(P + "OptimFzeroGma", audit["fold0"]["optimism_gma"])
        m.add_paired(P + "OptimFzeroVan", audit["fold0"]["optimism_van"])
        for r, letter in enumerate("ABC"):
            rep = audit["single"]["repeats"][r]
            m.add(f"{P}SingleRep{letter}", rep["mean"], signed=True)
            m.add_text(f"{P}SingleRep{letter}Naive", rep["verdict"])
        m.add_text(P + "SinglePositive", str(sum(r["verdict"] == "menang" for r in audit["single"]["repeats"])))
        m.add_text(P + "SingleSignificant", str(sum(r["significant"] for r in audit["single"]["repeats"])))
        if "optimiser" in audit:
            m.add(P + "ShortOptimValGain", audit["optimiser"]["val_gain"], signed=True)
            m.add(P + "ShortOptimTestGain", audit["optimiser"]["test_gain"], signed=True)

        frame = load_resampling(dataset)
        none = resampling_cell(frame, "none", "none", "xgboost")
        if none:
            m.add(P + "ResNoneXgbRep", none["reported"])
            m.add(P + "ResNoneXgbHold", none["holdout"])
        for method, key, _ in RESAMPLERS:
            for protocol, pkey in (("before_split", "Before"), ("train_only", "Train")):
                for model, mkey in (("xgboost", "Xgb"), ("logistic_regression", "Lr"), ("token", "Tok")):
                    cell = resampling_cell(frame, protocol, method, model)
                    if not cell:
                        continue
                    stem = f"{P}Res{key}{pkey}{mkey}"
                    m.add(stem + "Rep", cell["reported"])
                    m.add(stem + "Hold", cell["holdout"])
                    shown = round(round(cell["reported"], 4) - round(cell["holdout"], 4), 4)
                    m.add(stem + "Infl", shown, signed=True)
                    m.triples.append((stem + "Rep", stem + "Hold", stem + "Infl"))

        token_attr = token_attribution(dataset)
        if token_attr:
            m.add(P + "RhoTokPermXgbPerm", token_attr["rho_xgb"], decimals=2)
            m.add(P + "RhoTokPermGmaPerm", token_attr["rho_gma"], decimals=2)
            m.add(P + "WTokPerm", token_attr["w"], decimals=2)
            m.add(P + "PerturbRhoTok", token_attr["perturb"], decimals=2)
    base.add_missing_placeholders(m, MANUSCRIPT)
    m.write(path)
    write_json(path.with_name("macro_triples.json"), {"triples": m.triples})
    return m


def token_attribution(dataset: str) -> dict | None:
    """Agreement and stability of permutation importance on the token model."""
    directory = results_root() / "paper2" / "interpretability"
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(directory.glob(f"{dataset}__attribution_token__r0f*.json"))]
    others = {json.loads(p.read_text(encoding="utf-8"))["split"]: json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(directory.glob(f"{dataset}__attribution__r0f*.json"))}
    if not records:
        return None
    token = np.array([r["token_permutation"] for r in records])
    noisy = np.array([r["token_permutation_perturbed"] for r in records])
    rho_xgb, rho_gma = [], []
    for record in records:
        other = others.get(record["split"])
        if other:
            rho_xgb.append(sps.spearmanr(record["token_permutation"], other["global"]["xgb_permutation"]).statistic)
            rho_gma.append(sps.spearmanr(record["token_permutation"], other["global"]["gma_permutation"]).statistic)
    return {"rho_xgb": float(np.mean(rho_xgb)) if rho_xgb else np.nan,
            "rho_gma": float(np.mean(rho_gma)) if rho_gma else np.nan,
            "w": base.kendall_w(token) if len(token) > 1 else np.nan,
            "perturb": float(np.mean([sps.spearmanr(a, b).statistic for a, b in zip(token, noisy)]))}


def build(tables_dir: Path | None = None, figures_dir: Path | None = None) -> dict:
    tables_dir = tables_dir or out_dir() / "tables"
    figures_dir = figures_dir or out_dir() / "figures"
    tables = build_tables(tables_dir)
    build_figures(figures_dir)
    macros = write_macros(tables_dir / "macros.tex")
    missing = sum(1 for v in macros.values.values() if "??" in v)
    LOGGER.info("paper2b: %d tables, %d macros (%d awaiting results)", len(tables), len(macros.values), missing)
    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
