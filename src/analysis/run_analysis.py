"""Regenerate the Paper 2 tables, figures and macros from the released records.

    python -m src.analysis.run_analysis
"""

from src.analysis import paper2, paper2b
from src.common.io import results_root

if __name__ == "__main__":
    root = results_root() / "paper2"
    paper2.build(root / "tables_regenerated", root / "figures_regenerated")
    paper2b.build(root / "paper2b" / "tables_regenerated", root / "paper2b" / "figures_regenerated")
