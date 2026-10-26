# Preprocessing rules

Fitted on the training rows of each outer fold (and, inside the stack, of each
inner fold), then applied unchanged to validation and test rows.
Source: `src/paper2_credit/data.py`.

- A column is categorical if it has object/category dtype, if it is an integer
  column with at most 15 distinct training values, or if it is
  listed as forced: {"uci_dccc": ["x2", "x3", "x4", "x6", "x7", "x8", "x9", "x10", "x11", "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]}.
- Categorical levels are kept if they occur at least 10 times in the
  training rows, at most 60 per column (most frequent first);
  all other levels, and levels first seen at prediction time, share one bucket.
- Numeric missing values are filled with the training median.
- Ordinal view (TabNet family, tree models): categorical codes, then numerics on
  their own scale. Categorical embedding width: min(8, max(2, (cardinality + 1) // 2)).
- Dense view (logistic regression, k-NN, SVM, MLP): one-hot categoricals and
  numerics standardised with the training mean and standard deviation.
- k-NN and the SVM are fitted on a stratified subsample of at most 8,000 rows
  where the training partition is larger (a computational constraint).
