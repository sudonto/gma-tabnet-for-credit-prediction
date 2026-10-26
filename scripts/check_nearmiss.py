"""Check the chunked NearMiss-2 against imblearn's on subsamples of Give Me Some Credit.

imblearn cannot run on the whole dataset (it needs a 140,000 x 10,000 neighbour
matrix), so the comparison is made on random stratified subsamples where it can.

    python -m scripts.check_nearmiss
"""

from __future__ import annotations

import sys

import numpy as np
from imblearn.under_sampling import NearMiss

from src.common.io import load_dataset
from src.paper2_credit.protocol import DROPPED, nearmiss2


def main() -> int:
    dataset = load_dataset("give_me_some_credit")
    frame = dataset.X.drop(columns=DROPPED)
    failures = 0
    for seed, size in ((0, 3000), (1, 8000), (2, 15000)):
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(frame), size=size, replace=False))
        block = frame.iloc[rows]
        X = block.fillna(block.median(numeric_only=True)).to_numpy(dtype=np.float64)
        y = dataset.y[rows]
        ours = nearmiss2(X, y, chunk=997)
        sampler = NearMiss(version=2)
        sampler.fit_resample(X, y)
        theirs = np.sort(sampler.sample_indices_)
        same = np.array_equal(ours, theirs)
        overlap = len(np.intersect1d(ours, theirs)) / len(theirs)
        failures += not same
        print(f"  {'OK  ' if same else 'FAIL'} n={size:<6} kept={len(ours):<6} overlap={overlap:.4f}")
    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
