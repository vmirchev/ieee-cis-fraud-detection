# Debugging the Colab and local feature-list mismatch

## What differed

The Kedro run on macOS selected **230 V features and 31 ID features**. The
executed Colab notebooks selected **229 V features and 28 ID features**. Both
used a stable sort by `TransactionDT`, fitted XGBoost rankers on the earlier
80% of transactions, and kept the smallest gain-ranked prefix reaching 95% of
the summed feature-importance scores. The latest 20% was reserved for model
evaluation. A different selected count alone does not imply holdout leakage.

The ID lists shared 26 names. Five appeared only in the local list:
`id_03_is_missing`, `id_21`, `id_24`, `id_36`, and `id_13`. Two appeared only
in the Colab list: `id_13_is_missing` and `id_26`. Several shared features
also had different gain ranks, so this was more than a one-feature difference
at the 95% cutoff.

## Investigation

1. **Checked the seed and split.** Both final rankers set `random_state=42`.
   They use the same 472,432 earlier-period rows and exclude 118,108 later
   rows. The training period ends at `TransactionDT=12192842`; the holdout
   begins at `12192900`. No missing seed or different time boundary was found.
2. **Pinned XGBoost.** The local run used XGBoost 3.4.1, so notebooks 2 and 3
   were changed to install that version. Colab still produced 229/28. An
   initial check incorrectly required the Colab lists to equal the local
   lists byte for byte; it raised an error. That check was replaced with
   metadata recording each run's own list hash, package versions, input-file
   hashes, and split boundary. Notebooks 4 and 5 verify those artifacts before
   modeling.
3. **Verified the raw files.** Colab and the local workspace had identical
   SHA-256 hashes for both CSVs:

   | File | SHA-256 |
   | --- | --- |
   | `train_transaction.csv` | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
   | `train_identity.csv` | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` |

4. **Tested pandas and thread count on the Mac.** An isolated Python 3.13
   environment with pandas 2.2.3 and XGBoost 3.4.1 selected the same 31 ID
   features as the local pandas 3.0.3 environment when `n_jobs=1`. The Mac
   also selected 31 with `n_jobs=-1`. A preliminary comparison script had
   built category levels before limiting data to the training period and
   produced an invalid 30-feature result; that script was corrected before
   drawing conclusions. The corrected full-width and narrow CSV paths agreed.
5. **Compared the prepared ID matrix.** The final diagnostic cell in
   [notebook 3](../../notebooks/03.%20IEEE-CIS%20Fraud%20Detection%20-%20ID%20features%20exploration.ipynb)
   printed the same prepared-matrix SHA-256 on Colab and the Mac:
   `b3f8c0440a80946e774a784e7a711d8ec5269b7c4db951fe72aadd1e2d6dc5b7`.
   Colab's one-thread ranker still selected 28 features, exactly matching its
   saved list. The Mac selected 31.
6. **Matched the remaining reported Python packages.** A Mac run with
   pandas 2.2.3, NumPy 2.4.6, scikit-learn 1.6.1, XGBoost 3.4.1, and one
   thread still selected 31 ID features from that same matrix fingerprint.
7. **Ran Linux Docker.** The first container loaded the full merged table and
   exceeded Docker Desktop's memory allocation (exit 137). The runner was
   changed to load V and ID columns in separate passes. The successful Linux
   run used Python 3.13.18 and the same package versions reported by Colab.
   It reproduced both Colab feature-list hashes exactly.

## Final comparison

| Environment | V count and list SHA-256 | ID count and list SHA-256 |
| --- | --- | --- |
| Colab Linux | 229, `76a200aa310238b41f3b9e6de163efa8f5334905e787b117ceb227eb2a23435a` | 28, `938d48a18eb14258a0dd28493ad2547bdf1a709750ec484edb9f86479581c02f` |
| Reproduction Docker Linux | 229, same hash | 28, same hash |
| Kedro/macOS | 230, `5a079f90d50c808d035bee8a9fc6bf71a7f5fb9cfc7ba1014a02ea16b376c6b9` | 31, `1017cf3d78fc60d8478caee0e3dbda9a7919fa6846929fc0f0fc7800a888fcc1` |

The full Docker output, including selected names, package and native-build
details, source hashes, and split boundary, is in [results.json](results.json).

## Conclusion and limit

The mismatch is **platform or native-build sensitive**: a Linux container
reproduced Colab exactly, while controlled macOS runs did not. The tests rule
out a missing seed, different raw files, a different prepared ID matrix, and
the tested pandas, NumPy, scikit-learn, and thread-count differences as sole
causes. They do not identify the exact XGBoost native operation responsible.
The Colab/Linux and Mac lists are separate reproducible artifacts; model
scores must be reported with the list used for that run. This investigation
reproduced feature selection, not notebook 4 or 5 model scores.

## Reproduce the Linux result

From the repository root, with Docker Desktop running and the Kaggle CSVs in
`data/01_raw/`:

```bash
docker build --platform linux/amd64 -t ieee-colab-repro docker/colab-repro
docker run --rm --platform linux/amd64 --cpus=2 --memory=7g \
  --mount type=bind,src="$PWD",dst=/workspace,readonly \
  ieee-colab-repro --group both --n-jobs=-1 \
  > docker/colab-repro/results.json
```

The CSVs are mounted read-only and are never copied into the image. The runner
uses the downcast function from notebook 3, verifies the time split, and emits
the exact selected lists and hashes as JSON.
