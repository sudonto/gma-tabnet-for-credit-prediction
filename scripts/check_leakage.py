"""Prove that no test row reaches any fit, search or stopping decision.

Two checks, on German Credit with smoke schedules and a scratch results root:

1. Audit. With ``data.FIT_AUDIT`` enabled, every preprocessing fit, model fit,
   early-stopping set, search set and stack fit records the original row ids it
   used. Each stage is run one outer fold at a time and every recorded set must
   be disjoint from that fold's test rows. Selected configurations must carry
   the hash of the partition they are applied to.

2. Canary. A column equal to the label on the test rows and to noise elsewhere is
   added. A model that never fits on test rows cannot learn to read it, so its
   test ROC-AUC must not rise by more than 0.01. As a positive control the same
   column carrying the label on *every* row, so that a model can learn it and
   then read it at test time, must raise it to near 1, which shows the check
   can fail.

    python -m scripts.check_leakage
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np


def main() -> int:
    scratch = tempfile.mkdtemp(prefix="paper2_leakage_")
    os.environ["RESEARCH_RESULTS_ROOT"] = scratch

    from src.common.io import load_dataset
    from src.paper2_credit import data as credit_data
    from src.paper2_credit import run_experiments as rx

    rx.SMOKE = True
    args = argparse.Namespace(folds=5, repeats=1, only_repeat=None, only_fold=[0, 1],
                              search_rows=rx.SEARCH_ROWS, budget=8.0, force=False, reduced=True)
    dataset = load_dataset("german_credit")
    splits = rx.make_splits(dataset, args)
    failures = 0

    print("audit")
    for split in splits:
        credit_data.FIT_AUDIT = []
        for stage in ("baselines", "hpo_select", "architecture", "token_study", "stacking"):
            getattr(rx, f"stage_{stage}")(dataset, [split], args)
        test = set(split.test_idx.tolist())
        leaks = [(kind, len(test & set(rows.tolist()))) for kind, _, rows in credit_data.FIT_AUDIT
                 if test & set(rows.tolist())]
        kinds = sorted({kind for kind, _, _ in credit_data.FIT_AUDIT})
        status = "FAIL" if leaks else "OK  "
        failures += bool(leaks)
        print(f"  {status} {split.tag}: {len(credit_data.FIT_AUDIT)} fits audited ({', '.join(kinds)})"
              + (f"  leaks: {leaks}" if leaks else ""))
        for arch in rx.ARCHITECTURES:
            rx.load_selected(dataset.key, arch, split)        # raises on a hash mismatch

    # The selection audit leaks on purpose: fold 0's configuration, chosen on
    # rows that sit in fold 1's test set, is reused on fold 1. The audit must see
    # that overlap, or the shortcut it measures is not the one it claims to be.
    credit_data.FIT_AUDIT = []
    second = [s for s in splits if s.fold == 1]
    rx.stage_selection_audit(dataset, second, args)
    test = set(second[0].test_idx.tolist())
    deliberate = [len(test & set(rows.tolist())) for kind, _, rows in credit_data.FIT_AUDIT
                  if kind == "deliberate_selection"]
    other = [kind for kind, _, rows in credit_data.FIT_AUDIT
             if kind != "deliberate_selection" and test & set(rows.tolist())]
    ok = bool(deliberate) and all(n > 0 for n in deliberate) and not other
    failures += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} selection audit leaks deliberately: "
          f"{deliberate} fold-0 validation rows in fold 1's test set; no other leak {other or ''}")
    credit_data.FIT_AUDIT = None

    print("canary")
    from sklearn.metrics import roc_auc_score

    split = splits[0]
    rng = np.random.default_rng(0)

    def auc_with(canary_rows):
        frame = dataset.X.copy()
        column = rng.normal(size=len(frame))
        column[canary_rows] = dataset.y[canary_rows]
        frame["canary"] = column
        probe = type(dataset)(key=dataset.key, X=frame, y=dataset.y,
                              categorical_columns=dataset.categorical_columns,
                              numeric_columns=dataset.numeric_columns + ["canary"])
        fold = rx.prepare_fold(probe, split)
        metrics, _, _ = rx._run_baseline("xgboost", fold, split.seed)
        return metrics["roc_auc"]

    clean = auc_with(np.array([], dtype=int))
    on_test = auc_with(split.test_idx)
    everywhere = auc_with(np.arange(len(dataset.y)))
    leak_ok = on_test - clean <= 0.01
    control_ok = everywhere >= 0.95
    failures += (not leak_ok) + (not control_ok)
    print(f"  {'OK  ' if leak_ok else 'FAIL'} label on test rows: AUC {clean:.4f} -> {on_test:.4f}")
    print(f"  {'OK  ' if control_ok else 'FAIL'} positive control, label on every row: "
          f"AUC {clean:.4f} -> {everywhere:.4f}")

    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
