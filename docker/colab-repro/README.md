# Colab feature-selection reproduction

This image uses Python 3.13.18, pandas 2.2.3, NumPy 2.4.6,
scikit-learn 1.6.1, and XGBoost 3.4.1. The runner follows notebooks 2 and 3,
fits on the earlier 80% of transactions, and reports selected lists and hashes.
It reads the raw data from a read-only bind mount; the CSVs are not copied into
the image. V and ID columns are loaded in separate passes to fit Docker
Desktop's memory allocation. Each pass uses the same rows, values, and
notebook downcast rules as the wide merge.

From the repository root:

```bash
docker build --platform linux/amd64 -t ieee-colab-repro docker/colab-repro
docker run --rm --platform linux/amd64 --cpus=2 --memory=7g \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  ieee-colab-repro --group both --n-jobs=-1
```

The Colab run produced V list SHA-256
`76a200aa310238b41f3b9e6de163efa8f5334905e787b117ceb227eb2a23435a`
(229 features), ID list SHA-256
`938d48a18eb14258a0dd28493ad2547bdf1a709750ec484edb9f86479581c02f`
(28 features), and prepared ID matrix SHA-256
`b3f8c0440a80946e774a784e7a711d8ec5269b7c4db951fe72aadd1e2d6dc5b7`.

The completed Docker run is saved in `results.json`. It reproduced both Colab
list hashes and the prepared ID-matrix hash exactly. The first full-width
container attempt exceeded Docker Desktop's memory allocation; the saved run
uses separate V and ID passes with only their required columns.

The investigation, including discarded hypotheses and the remaining limit, is
documented in [DEBUGGING.md](DEBUGGING.md).
