"""Assemble the Paper 2 reproducibility package in build/release_paper2/.

What a reader needs to audit and reproduce every number in the manuscript:

code            the model (attentive transformer, ghost batch norm), the
                experiment driver, stacking, attribution, the analysis that
                writes every table, figure and quoted number, and the
                correctness checks
data_manifest   OpenML ids and versions, the SHA-256 of the cached data, row
                and class counts; the data itself is fetched from OpenML
PREPROCESSING   every preprocessing rule, with its constants
splits/         every outer, validation, search and inner stacking index set
seeds.json      how every seed is derived, and the realised values
tuning/         every search history (each evaluation's parameters, timing and
                status, interrupted ones included) and every selected config
raw_runs/       one JSON record per (dataset, stage, arm, outer fold)
predictions/    per-applicant out-of-fold test probabilities, one parquet per
                dataset: row_id, repeat, fold, label, one column per arm
checkpoints/    trained weights of every TabNet-family arm
interpretability/, protocol/   attribution and protocol-sensitivity records
tables/         the generated tables and macros, for comparison after rerun

    python -m scripts.export_paper2_release
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_code_release import copy_file, copy_tree
from src.common.io import DATASETS, PROJECT_ROOT, RAW_DIR, results_root
from src.common.splits import model_seed
from src.paper2_credit import data as credit_data
from src.paper2_credit.data import CREDIT_DATASETS

DEST = PROJECT_ROOT / "build" / "release_paper2"

CODE = ["src/common", "src/paper2_credit"]
FILES = [
    "src/__init__.py",
    "src/analysis/__init__.py", "src/analysis/aggregate.py", "src/analysis/figures.py",
    "src/analysis/literature.py", "src/analysis/macros.py", "src/analysis/significance.py",
    "src/analysis/tables.py", "src/analysis/paper2.py", "src/analysis/paper2b.py",
    "scripts/fetch_datasets.py", "scripts/verify_ghost_bn.py",
    "scripts/verify_tabnet_equivalence.py", "scripts/check_leakage.py",
    "scripts/check_nearmiss.py", "scripts/check_macro_arithmetic.py",
    "scripts/export_code_release.py", "scripts/export_paper2_release.py",
    "docs/paper2_protocol_v2.md", "requirements.txt",
]
RESULT_DIRS = ["raw_runs", "tuning", "splits", "interpretability", "protocol", "checkpoints", "tables", "paper2b"]

RUN_ANALYSIS = '''"""Regenerate the Paper 2 tables, figures and macros from the released records.

    python -m src.analysis.run_analysis
"""

from src.analysis import paper2, paper2b
from src.common.io import results_root

if __name__ == "__main__":
    root = results_root() / "paper2"
    paper2.build(root / "tables_regenerated", root / "figures_regenerated")
    paper2b.build(root / "paper2b" / "tables_regenerated", root / "paper2b" / "figures_regenerated")
'''

README = """# GMA-TabNet and an evaluation-protocol audit for credit scoring: reproducibility package

Code, data manifest, index sets, seeds, search histories, run records and
per-applicant predictions behind every number in two manuscripts built on the
same testbed (UCI Taiwan, Statlog German Credit and Give Me Some Credit, nested
3x5 cross-validation with equal tuning budgets):

* *GMA-TabNet: Atensi Multi-Kepala Bergerbang untuk Prediksi Gagal Bayar Kredit
  dengan Evaluasi Bersarang* (analysis: `src/analysis/paper2.py`, output
  `results/paper2/tables`);
* *Protokol Evaluasi Menentukan Kesimpulan: Audit TabNet dan Atensi Token untuk
  Penilaian Kredit* (analysis: `src/analysis/paper2b.py`, output
  `results/paper2/paper2b/tables`): the audit of seven evaluation shortcuts,
  the resampling audit with an untouched holdout (`results/paper2/protocol`)
  and the token-attention TabNet study.

The datasets are not redistributed; `scripts.fetch_datasets` downloads them
from OpenML and verifies them against `data_manifest.json`.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.fetch_datasets                    # OpenML; verified against data_manifest.json
python -m scripts.verify_ghost_bn                   # ghost batch norm == pytorch-tabnet GBN
python -m scripts.verify_tabnet_equivalence         # vanilla TabNet == pytorch-tabnet, weight for weight
python -m scripts.check_leakage                     # no test row reaches any fit or decision
python -m src.paper2_credit.run_experiments --stage all
python -m src.analysis.run_analysis
python -m scripts.check_macro_arithmetic results/paper2/tables_regenerated
```

The second manuscript adds the stages `hpo_select` (token architecture),
`selection_audit`, `token_study`, `stacking`, `interpret_token` and
`resampling_audit` (`--stage <name>`); they reuse the first study's searches and
splits.

The released records live under `results/paper2/`. To regenerate the tables from
them without rerunning anything, run only the last two commands; the output
should equal `results/paper2/tables/` and `results/paper2/paper2b/tables/`
(compare `macros.tex`).

The first study's sweep took about four days and the second study's stages
about two more, on one NVIDIA RTX 4090 (see `environment.txt`).
Hyperparameter searches have wall-clock budgets, so on other hardware they will
complete a different number of evaluations; everything downstream of the
released selected configurations (`results/paper2/tuning/*selected*`) is
deterministic up to GPU non-determinism.

See `docs/paper2_protocol_v2.md` for the protocol, `PREPROCESSING.md` for the
preprocessing rules and `seeds.json` for every seed.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest() -> dict:
    from src.common.io import load_dataset

    manifest = {}
    for key in CREDIT_DATASETS:
        spec = DATASETS[key]
        dataset = load_dataset(key)
        entry = {
            "openml_id": spec.openml_id, "positive_label": spec.positive_label,
            "rows": dataset.n_samples, "features": dataset.n_features,
            "positives": int(dataset.y.sum()), "columns": list(dataset.X.columns),
            "dropped_columns": list(spec.drop_columns),
        }
        try:
            import openml

            meta = openml.datasets.get_dataset(spec.openml_id, download_data=False,
                                               download_qualities=False,
                                               download_features_meta_data=False)
            entry.update(openml_name=meta.name, openml_version=meta.version)
        except Exception as error:  # noqa: BLE001 - the manifest is still useful offline
            entry["openml_version"] = f"unavailable offline ({type(error).__name__})"
        for path in sorted(RAW_DIR.glob(f"{key}*.parquet")):
            entry.setdefault("cached_sha256", {})[path.name] = sha256(path)
        manifest[key] = entry
    return manifest


def preprocessing_notes() -> str:
    forced = credit_data.FORCED_CATEGORICAL
    return f"""# Preprocessing rules

Fitted on the training rows of each outer fold (and, inside the stack, of each
inner fold), then applied unchanged to validation and test rows.
Source: `src/paper2_credit/data.py`.

- A column is categorical if it has object/category dtype, if it is an integer
  column with at most {credit_data.CATEGORICAL_MAX_UNIQUE} distinct training values, or if it is
  listed as forced: {json.dumps(forced)}.
- Categorical levels are kept if they occur at least {credit_data.MIN_CATEGORY_COUNT} times in the
  training rows, at most {credit_data.MAX_CATEGORY_CARDINALITY} per column (most frequent first);
  all other levels, and levels first seen at prediction time, share one bucket.
- Numeric missing values are filled with the training median.
- Ordinal view (TabNet family, tree models): categorical codes, then numerics on
  their own scale. Categorical embedding width: min(8, max(2, (cardinality + 1) // 2)).
- Dense view (logistic regression, k-NN, SVM, MLP): one-hot categoricals and
  numerics standardised with the training mean and standard deviation.
- k-NN and the SVM are fitted on a stratified subsample of at most 8,000 rows
  where the training partition is larger (a computational constraint).
"""


def seeds_record() -> dict:
    record = {
        "outer_folds": "StratifiedKFold(5, shuffle=True, random_state=repeat)",
        "validation_block": "train_test_split(test_size=0.15, stratify, random_state=100*repeat + fold)",
        "search_subset": "train_test_split(train_size=15000, stratify, random_state=10000 + 100*repeat + fold)",
        "model_seed": "1000*repeat + fold",
        "optimiser_seed": "SeedSequence([repeat, fold, optimiser_id, sum(ord(c) for c in arch)])",
        "stack_inner_folds": "StratifiedKFold(K, shuffle=True, random_state=model_seed)",
        "realised_model_seeds": {f"r{r}f{k}": model_seed(r, k) for r in range(3) for k in range(5)},
    }
    searches = {}
    for path in sorted((results_root() / "paper2" / "tuning").glob("*__r*f*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "optimiser_seed" in payload:
            searches[path.stem] = payload["optimiser_seed"]
    record["realised_optimiser_seeds"] = searches
    return record


def predictions(destination: Path) -> None:
    scores = results_root() / "paper2" / "scores"
    for dataset in CREDIT_DATASETS:
        frames = []
        for path in sorted(scores.glob(f"{dataset}__*.npz")):
            _, stage, arm, tag, _ = path.stem.split("__")
            with np.load(path) as payload:
                if "index" not in payload:
                    continue
                repeat, fold = int(tag[1:tag.index("f")]), int(tag[tag.index("f") + 1:])
                frames.append(pd.DataFrame({
                    "row_id": payload["index"], "repeat": repeat, "fold": fold,
                    "label": payload["y_true"], "column": f"{stage}__{arm}",
                    "probability": payload["scores"],
                }))
        if not frames:
            continue
        long = pd.concat(frames, ignore_index=True)
        wide = long.pivot_table(index=["row_id", "repeat", "fold", "label"], columns="column",
                                values="probability").reset_index()
        destination.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(destination / f"{dataset}.parquet", index=False)


SERVER_ENVIRONMENT = PROJECT_ROOT / "logs" / "server_env.txt"


def environment() -> str:
    """The environment the experiments ran in, not the one the export runs in.

    The sweeps ran on a separate GPU machine; scripts/remote/freeze_env.ps1
    records its interpreter, torch, CUDA, GPU and packages in
    logs/server_env.txt. Without that record the exporting machine is described
    instead, and the file says so.
    """
    if SERVER_ENVIRONMENT.exists():
        record = SERVER_ENVIRONMENT.read_text(encoding="utf-8")
        record = record.split("\n** WARNING")[0].split("\nexit=")[0].strip() + "\n"
        return "# Environment of the machine that ran every experiment\n" + record
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    lines = [f"python {sys.version}", f"platform {platform.platform()}"]
    try:
        import torch

        lines.append(f"torch {torch.__version__}, cuda {torch.version.cuda}")
        if torch.cuda.is_available():
            lines.append(f"gpu {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    return "\n".join(lines) + "\n\n" + freeze


def export() -> Path:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    for rel in CODE:
        copy_tree(PROJECT_ROOT / rel, DEST / rel)
    for rel in FILES:
        copy_file(PROJECT_ROOT / rel, DEST / rel)
    (DEST / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (DEST / "src" / "analysis" / "run_analysis.py").write_text(RUN_ANALYSIS, encoding="utf-8")

    source = results_root() / "paper2"
    for name in RESULT_DIRS:
        if (source / name).exists():
            copy_tree(source / name, DEST / "results" / "paper2" / name)
    predictions(DEST / "results" / "paper2" / "predictions")
    # The first submission's records, which the paper and the response letter
    # quote when explaining what changed (the Prev* macros).
    first_round = PROJECT_ROOT / "results" / "paper2_v1_ijies" / "raw_runs"
    if first_round.exists():
        copy_tree(first_round, DEST / "results" / "paper2_v1_ijies" / "raw_runs")

    (DEST / "README.md").write_text(README, encoding="utf-8")
    (DEST / "PREPROCESSING.md").write_text(preprocessing_notes(), encoding="utf-8")
    (DEST / "data_manifest.json").write_text(json.dumps(data_manifest(), indent=2), encoding="utf-8")
    (DEST / "seeds.json").write_text(json.dumps(seeds_record(), indent=2), encoding="utf-8")
    (DEST / "environment.txt").write_text(environment(), encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "build" / "release" / "LICENSE", DEST / "LICENSE") \
        if (PROJECT_ROOT / "build" / "release" / "LICENSE").exists() else None

    sums = [f"{sha256(p)}  {p.relative_to(DEST).as_posix()}"
            for p in sorted(DEST.rglob("*")) if p.is_file()]
    (DEST / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return DEST


def main() -> int:
    destination = export()
    files = sum(1 for p in destination.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) / 1e6
    print(f"exported {files} files ({size:.1f} MB) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
