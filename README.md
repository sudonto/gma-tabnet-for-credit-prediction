# GMA-TabNet and an evaluation-protocol audit for credit scoring: reproducibility package

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

## Not in this repository

The trained weights of every TabNet-family arm (`results/paper2/checkpoints`,
about 0.8 GB) are left out of the Git repository because of their size; they
are available from the authors on request. Everything else in the full release,
including all run records, searches, splits and per-applicant predictions, is
here, and `SHA256SUMS` lists the checksum of every file in this repository.
