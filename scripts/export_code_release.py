"""Copy the Paper 1 subset into build/release/ for the public GitHub repo.

The workspace also holds an unpublished second paper. Git runs only inside
the export directory, never here.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from src.common.io import PROJECT_ROOT

DEST = PROJECT_ROOT / "build" / "release"

SRC_DIRS = [
    "src/common",
    "src/paper1_fraud",
]
ANALYSIS_FILES = [
    "src/analysis/__init__.py",
    "src/analysis/aggregate.py",
    "src/analysis/feedback_report.py",
    "src/analysis/figures.py",
    "src/analysis/literature.py",
    "src/analysis/macros.py",
    "src/analysis/significance.py",
    "src/analysis/tables.py",
]
SCRIPTS = [
    "scripts/fetch_datasets.py",
    "scripts/verify_modules.py",
    "scripts/smoke_feedback.py",
    "scripts/smoke_paper1.py",
    "scripts/status.py",
]
ROOT_FILES = [
    "requirements.txt",
    ".gitignore",
]

README = """# Deployed fraud detection: the label feedback loop

Simulation code for the paper *The label feedback loop in deployed fraud
detection and how to close it*. A deployed detector decides which transactions
get investigated, and the investigations become its next training set;
everything unreviewed is filed as legitimate. The study replays three public
transaction streams through that loop and compares bookkeeping rules.

Every number in the manuscript is generated from the raw run files. Nothing is
transcribed by hand.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Unix
pip install -r requirements.txt
```

Download the Paper 1 streams from OpenML and check the published shapes:

```bash
python -m scripts.fetch_datasets baf_base baf_variant3 ieee_cis ulb_creditcard
python -m scripts.verify_modules
```

## Running the study

```bash
# Structural checks on the review policies and the bookkeeping rules
python -m scripts.smoke_feedback

# Headline study: the loop, its capacity sweep, the detector sensitivity
# sweep, and the robustness stream
python -m src.paper1_fraud.run_feedback --stage all

# Negative-result track: baselines, resampling, encoder ablation, leakage
python -m src.paper1_fraud.run_experiments --stage all
```

Both runners skip an arm whose JSON already exists. Pass `--force` to recompute.

```bash
python -m src.analysis.run_analysis --paper 1
```

regenerates the tables, figures and LaTeX macros from the raw run files.

## License

MIT. See `LICENSE`.
"""

LICENSE = """MIT License

Copyright (c) 2026 Yuri Yudhaswana Joefrie, Josua Crishan Mintamanis,
Michail Sidorov, Dessy Santi, Chairunnisa A.R. Lamasitudju

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

RUN_ANALYSIS = '''"""Regenerate Paper 1 tables, figures and macros from the stored runs.

    python -m src.analysis.run_analysis --paper 1
"""

from __future__ import annotations

import argparse
import json
import logging

from src.analysis.aggregate import PAPER1_HEADLINE, load_runs, summarise
from src.analysis.feedback_report import build as build_feedback_report
from src.analysis.figures import build_paper1_figures
from src.analysis.macros import build_paper1_macros
from src.analysis.tables import BEEI, build_paper1_tables, use_layout
from src.common.io import PROJECT_ROOT, write_json

LOGGER = logging.getLogger("analysis")


def attempt(description, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        LOGGER.error("%s failed: %s: %s", description, type(error).__name__, error)
        return None


def paths_for(paper: int):
    root = PROJECT_ROOT / "results" / f"paper{paper}"
    return root / "tables", root / "figures", root / "summary"


def summarise_paper(paper: int) -> dict:
    runs = load_runs(paper)
    if runs.empty:
        return {}
    digest = {"n_runs": int(len(runs)), "stages": {}}
    for stage, frame in runs.groupby("stage"):
        summary = summarise(frame, PAPER1_HEADLINE, group=["arm"])
        digest["stages"][str(stage)] = json.loads(summary.to_json(orient="records"))
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Paper 1 artefacts")
    parser.add_argument("--paper", type=int, choices=[1], default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tables_dir, figures_dir, summary_dir = paths_for(args.paper)
    with use_layout(BEEI):
        tables = attempt("paper 1 tables", build_paper1_tables, tables_dir) or {}
        feedback = attempt(
            "paper 1 feedback report", build_feedback_report, tables_dir, figures_dir,
        ) or {}
        tables.update(feedback)
    written = [name for name, content in tables.items() if content]
    LOGGER.info("paper 1: %d tables -> %s", len(written), tables_dir)
    attempt("paper 1 figures", build_paper1_figures, figures_dir)
    attempt("paper 1 macros", build_paper1_macros, tables_dir / "macros.tex")
    digest = attempt("paper 1 digest", summarise_paper, args.paper)
    if digest:
        write_json(summary_dir / "summary.json", digest)


if __name__ == "__main__":
    main()
'''


def copy_tree(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src, dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def export() -> Path:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    for rel in SRC_DIRS:
        copy_tree(PROJECT_ROOT / rel, DEST / rel)
    copy_file(PROJECT_ROOT / "src" / "__init__.py", DEST / "src" / "__init__.py")
    for rel in ANALYSIS_FILES:
        copy_file(PROJECT_ROOT / rel, DEST / rel)
    (DEST / "src" / "analysis" / "run_analysis.py").write_text(RUN_ANALYSIS, encoding="utf-8")
    for rel in SCRIPTS:
        copy_file(PROJECT_ROOT / rel, DEST / rel)
    (DEST / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    for rel in ROOT_FILES:
        copy_file(PROJECT_ROOT / rel, DEST / rel)
    (DEST / "README.md").write_text(README, encoding="utf-8")
    (DEST / "LICENSE").write_text(LICENSE, encoding="utf-8")
    return DEST


def verify_imports(root: Path) -> None:
    sys.path.insert(0, str(root))
    modules = [
        "src.common.io",
        "src.common.seeds",
        "src.paper1_fraud.feedback",
        "src.paper1_fraud.run_feedback",
        "src.paper1_fraud.run_experiments",
        "src.analysis.aggregate",
        "src.analysis.feedback_report",
        "src.analysis.run_analysis",
    ]
    for name in modules:
        __import__(name)


def main() -> int:
    dest = export()
    verify_imports(dest)
    print(f"exported Paper 1 code to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
