# CODSOFT TASK

> Note: Notebooks use relative dataset paths (e.g. `datasets/fraud.csv`) so they run locally or in Colab.

---

## Task 1 — Fraud Detection

Objective: Detect fraudulent transactions.

Dataset: `datasets/fraud.csv` (contains features + `Fraud` target).

Approach: data cleaning, categorical encoding, stratified train/test split, model training and evaluation. Preferred models: Logistic Regression (regularized) for interpretability and Random Forest for non-linear patterns.

Algorithms (brief):
- Logistic Regression: linear model optimizing cross-entropy; fast, interpretable; scale numeric features and use class_weight or resampling for imbalance.
- Random Forest: ensemble of decision trees (bagging + feature randomness); captures non-linear interactions and provides feature importances.

Metrics: confusion matrix, precision, recall, F1, ROC-AUC / PR-AUC. Prioritize recall for the fraud class.

---

## Task 2 — Spam Detection

Objective: Classify SMS as spam or ham.

Dataset: `datasets/spam.csv` (`v1` = label, `v2` = message).

Approach: text cleaning, TF–IDF vectorization, train/test split, train Multinomial Naive Bayes, evaluate with classification metrics.

Algorithm (brief):
- TF–IDF: converts text to weighted term vectors (controls vocabulary with n-grams, min_df, max_features).
- Multinomial Naive Bayes: fast probabilistic model for token counts; uses Laplace smoothing.

Metrics: accuracy, precision, recall, F1, confusion matrix.

---

## Task 3 — Churn Prediction

Objective: Predict customer churn.

Dataset: `datasets/churn.csv` (e.g., `customerID`, `gender`, `tenure`, `Churn`).

Approach: drop identifiers, encode categoricals, impute missing values, scale if required, train Random Forest or comparable classifier. Use stratified validation and business-aware metrics.

Algorithm (brief):
- Random Forest: default robust choice for mixed data; tune n_estimators, max_depth, and class weights.

Metrics and notes: precision/recall for churn class, ROC-AUC/PR-AUC, feature importance (permutation/SHAP for interpretation).

---

## General notes
- Use stratified k-fold for tuning.
- Address imbalance with class weights, resampling (SMOTE) or threshold tuning.
- Calibrate probabilities when used for business decisions.
- Notebooks include runnable code and example outputs.

## How to run
- Open notebooks in Colab or run locally (Python 3).
- Ensure required packages (scikit-learn, pandas, numpy, imbalanced-learn).
- Keep datasets in the `datasets/` folder.

---

## Author
**Shravani Satarkar** — https://github.com/shravanisatarkar08 — shravani.satarkar08@gmail.com

(Updated: 2025)