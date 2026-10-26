"""Check that every printed difference equals the difference of the printed values.

``src/analysis/paper2.py`` registers each (a, b, a - b) macro triple it writes in
``macro_triples.json``; this reads them back from ``macros.tex`` and fails if any
printed difference is not exactly the printed a minus the printed b. That is the
class of inconsistency the first round of review found (a quoted -0.0035 against
values whose printed difference was -0.0034).

    python -m scripts.check_macro_arithmetic [tables_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.common.io import results_root

DEFINITION = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}$")


def number(text: str) -> float | None:
    cleaned = text.replace("\\%", "").replace("{,}", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else results_root() / "paper2" / "tables"
    values = {}
    for line in (directory / "macros.tex").read_text(encoding="utf-8").splitlines():
        match = DEFINITION.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2)
    triples = json.loads((directory / "macro_triples.json").read_text(encoding="utf-8"))["triples"]

    checked, failures, pending = 0, 0, 0
    for a, b, d in triples:
        va, vb, vd = (number(values.get(n, "")) for n in (a, b, d))
        if None in (va, vb, vd):
            pending += 1
            continue
        checked += 1
        if abs((va - vb) - vd) > 1e-9:
            failures += 1
            print(f"  FAIL \\{d} = {vd:+.4f}, but \\{a} - \\{b} = {va:.4f} - {vb:.4f} = {va - vb:+.4f}")
    print(f"{checked} triples checked, {failures} inconsistent, {pending} awaiting results")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
