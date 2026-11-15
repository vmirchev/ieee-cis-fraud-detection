# Notebook experiments and findings

These five Google Colab notebooks document the experiments behind the Kedro fraud-detection pipeline. This report records the purpose of each experiment, the choices that carried into the pipeline, and the results saved in the 12 November 2026 notebook downloads. The task is binary classification: estimating whether a transaction's `isFraud` label is 1.

## Evaluation boundary

The transaction data has 590,540 rows. `TransactionDT` measures elapsed seconds from an unspecified reference point; it is useful for ordering transactions, but it is not a calendar timestamp. The notebooks sort by this column, use the **earlier 80% (472,432 rows)** for exploration, feature selection, training, and inner validation, and reserve the **latest 20% (118,108 rows)** for model evaluation.

Within the earlier 80%, some experiments make another chronological split: the first 377,945 rows for fitting and the next 94,487 for inner validation. In terms of the original dataset, these are approximately the first 64% and the next 16%. A random split would mix early and late transactions, making it easier for the model-development process to learn patterns from the period it is meant to predict.

The rule is broader than keeping holdout labels out of `fit()`. Any operation that *learns* from data, such as supervised feature selection, choosing category levels, imputing medians, or tuning a threshold, needs an appropriate training period. The final holdout can be transformed with the fitted rules and scored, but it should not determine those rules.

## Notebook 1: explore the training period

Open [01. EDA](../notebooks/01.%20IEEE-CIS%20Fraud%20Detection%20-%20EDA.ipynb).

The notebook reads the time column first to establish the boundary, then loads only earlier-period transaction rows for label-aware exploration. It downcasts numeric columns once after reading each CSV, then joins transaction and identity data by `TransactionID`. This keeps the large table manageable in Colab. The run reported 472,432 EDA rows, including 16,599 frauds: about **3.51%** of those rows. There were no duplicate `TransactionID` values in this subset.

The low fraud rate changes how to read every later metric. A classifier that predicts “not fraud” for every row would be accurate most of the time but would find no fraud. Fraud *rate* within a category is also more informative than raw fraud count: a common category can contain many frauds simply because it contains many transactions.

The EDA examines missingness, amount, relative time, product, card, address, distance, email, device, and anonymized engineered columns. Several lessons carry into modeling:

- **Missingness can be informative.** A blank identity or distance field may describe how a transaction was collected. Dropping all missing rows would discard both transactions and a possible signal. Later pipelines impute values and add missingness indicators where useful.
- **Skewed values benefit from inspection and transformation.** `TransactionAmt` has a long right tail. `log1p(amount)` compresses large values while remaining defined at zero. It does not, by itself, make fraud and legitimate transactions separable.
- **Numeric storage does not guarantee numeric meaning.** Fields such as `card1`, `card2`, `card3`, `card5`, `addr1`, and `addr2` behave like codes or identifiers. Treating their numerical gaps as meaningful distances can mislead a linear model.
- **Relative time patterns are descriptive.** Day and hour features can expose changing rates, but `TransactionDT` cannot identify actual weekdays or holidays without a known reference date.

The notebook ends with a modeling plan: select features within the earlier period, compare a simple baseline with tree models, and use an inner time slice for tuning.

## Notebook 2: select Vesta `V*` features

Open [02. V features](../notebooks/02.%20IEEE-CIS%20Fraud%20Detection%20-%20Vesta%20Features%20Exploration.ipynb).

There are 339 anonymized `V*` columns. Their names give little semantic guidance, so the notebook measures their predictive contribution. It loads only time, label, and V columns, then fits an XGBoost model on the inner-training period. The inner-validation PR-AUC was **0.4606** for this V-only diagnostic model.

Three ways of looking at feature relevance appear here:

1. **XGBoost gain importance** measures how much a feature's tree splits improve the model's objective. It is fast to obtain from a fitted tree model, but correlated features can divide or exchange credit.
2. **Permutation importance** shuffles one feature on the inner-validation data and measures the drop in average precision. It asks how much the fitted model relies on that feature in that validation slice. It is slower, so the notebook applies it only to a top-ranked subset.
3. **Correlation and top-k experiments** show overlap among V columns and how smaller subsets perform on inner validation. These are diagnostics, not the final rule for choosing the exported count.

For the exported list, the notebook refits the XGBoost ranker on **all 472,432 earlier-period rows**. It sorts features by gain and takes the shortest prefix whose cumulative share reaches 95%:

```text
cumulative share at k = sum(gain of the first k features) / sum(gain of all features)
choose the smallest k with cumulative share >= 0.95
```

This is a share of the model's *reported feature importance*, not a claim that the retained features explain 95% of fraud or preserve 95% of PR-AUC. The current Colab run selected **229 V features**. The old fixed top-75 choice is not used because it came from experiments that crossed the later evaluation period.

The notebook saves the ordered feature list and metadata, including XGBoost version, raw CSV hashes, list hash, row count, time boundary, and the 0.95 rule. The list hash is a compact check that downstream notebooks read the exact ordered list produced by that run. XGBoost is pinned to 3.4.1 because native builds on different platforms have produced slightly different gain rankings even with the same input data.

## Notebook 3: select identity features

Open [03. ID features](../notebooks/03.%20IEEE-CIS%20Fraud%20Detection%20-%20ID%20features%20exploration.ipynb).

Identity information is present for only some transactions. The notebook joins the identity CSV to the transaction key, time, and label. It evaluates both the original `id_*` values and binary indicators such as `id_02_is_missing`. For categorical ID values, it learns category levels from the fitting rows and applies those levels to inner validation, rather than learning new levels from validation.

The inner-validation experiment makes the value of missingness concrete. A model using identity values plus indicators reached **0.1834 PR-AUC**; a model using only missingness indicators reached **0.1215 PR-AUC**. Missingness has signal, but the observed identity values contain additional signal. These scores describe the identity-only diagnostic task and should not be compared directly with the full model's holdout score as if they were evaluated on the same split.

The top-k sweep explores how performance changes as more ranked features are used. The exported count again comes from the predeclared **95% cumulative-gain rule**, now fitted on the whole earlier 80%. The Colab run selected **28 ID features**, including any useful derived indicators. It saves the ordered list and the same provenance fields as notebook 2. A local Mac run can select a different count; the downstream Colab notebooks use the lists and hashes from their own run.

## Notebook 4: preprocess, compare models, choose a threshold

Open [04. Modeling](../notebooks/04.%20IEEE-CIS%20Fraud%20Detection%20-%20Preprocess%20Model.ipynb).

This notebook checks that both feature lists match their metadata, the raw CSV hashes, and the chronological boundary. It then reads only the selected V and ID source columns, adds selected ID missingness indicators, and splits the merged table into the same earlier 80% and later 20%. The run validated **229 V and 28 ID features**. After feature engineering, it assigned 280 columns to numeric preprocessing and 34 to categorical preprocessing.

Preprocessing is placed inside scikit-learn `Pipeline` and `ColumnTransformer` objects. Each model's `fit()` learns medians and frequent categories from its training rows; the fitted transformations are then applied to validation rows. This prevents validation statistics from entering the training transformations.

The **logistic regression baseline** uses median imputation with missing indicators and scaling for numeric columns. Categorical columns have rare values grouped, missing values filled, and one-hot encoding applied. It uses balanced class weights. This baseline checks how much performance is possible with a simpler linear decision rule.

The **LightGBM and XGBoost models** use median imputation with indicators for numeric columns. Categorical values are grouped, filled, and ordinal-encoded; unknown values receive a reserved code. Trees do not require the numeric scaling used by logistic regression. Imbalance is handled differently in the two models: LightGBM uses `class_weight="balanced"`, while XGBoost sets `scale_pos_weight` to the training count of nonfrauds divided by the training count of frauds. Neither setting changes the observed fraud rate; both increase the training penalty for missing frauds.

The later-period comparison was:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic regression | 0.8446 | 0.2242 |
| LightGBM | 0.9047 | **0.5358** |
| XGBoost | **0.9096** | 0.5343 |

**PR-AUC is the selection metric** because fraud is rare and the project cares about the precision–recall tradeoff. ROC-AUC answers a different ranking question and can remain high when precision on the minority class is less useful. LightGBM had the highest PR-AUC in this run, but its lead over XGBoost was only about **0.0015**; that small difference is not strong evidence of general superiority.

A probability score still needs an operating threshold to produce a yes/no alert. The notebook fits separate tree-model copies on the first 64% of transactions and chooses the threshold with the best F1 on the next 16%, entirely inside the earlier 80%. F1 combines precision and recall; a lower threshold generally catches more fraud at the cost of more false alerts. For LightGBM the selected threshold was **0.79**. When applied to the later holdout, it achieved **61.4% precision** and **44.9% recall**. The notebook also displays higher-recall alternatives for cases where missing fraud is more costly than reviewing false alerts.

The selected model, threshold, and feature-list metadata are saved together. The holdout chose the model family, so its score is an internal comparison result, not an untouched final estimate.

## Notebook 5: tune a LightGBM candidate

Open [05. Optuna](../notebooks/05.%20IEEE-CIS%20Fraud%20Detection%20-%20Optuna%20params%20exploration.ipynb).

This notebook repeats the verified feature loading and time split, then checks that notebook 4's saved model was built with the same feature lists and selection metadata. It uses Optuna to try **10** LightGBM hyperparameter combinations on the inner time split. The search varies learning rate, tree size and depth, minimum leaf size, row and column sampling, and regularization. A fixed sampler seed makes the proposed trials reproducible within a matching environment.

Each trial fits on the first 64% of transactions and returns PR-AUC on the next 16%. The best recorded inner PR-AUC was **0.6015**. After choosing parameters, the notebook refits the candidate on the whole earlier 80% and evaluates it on the later 20%:

| Model | Holdout PR-AUC | Holdout ROC-AUC |
|---|---:|---:|
| Saved LightGBM from notebook 4 | 0.5358 | 0.9047 |
| Tuned LightGBM candidate | **0.5397** | **0.9144** |

The PR-AUC gain is about **0.0039**. The tuned model is therefore saved as a *candidate* in a separate file; the notebook does not silently replace notebook 4's selected artifact. This comparison also uses the previously examined holdout, so the small observed gain should be treated as exploratory.

## What the evaluation can and cannot establish

The notebooks enforce an important boundary: feature selection and model fitting use earlier transactions, and the later-period labels are used for scoring. There are still two limits to the reported numbers:

1. The V and ID feature lists were selected on the whole earlier 80%, which includes the 16% inner validation slice. Inner tuning and threshold scores can therefore be optimistic. To obtain an independent inner score, feature selection would have to be refitted inside each inner training fold.
2. The later 20% has been viewed repeatedly during this project, including model-family and tuned-candidate comparisons. It remains useful for comparing this project's experiments, but a fresh later labeled period is needed for a truly untouched final performance estimate.

The practical result is a reproducible training-period feature-selection workflow, a documented preprocessing and imbalance strategy, and two strong tree-model candidates. The result does **not** establish that a 0.0039 PR-AUC difference will persist on future fraud data.
