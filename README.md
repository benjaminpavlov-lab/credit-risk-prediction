# Credit Risk Prediction — German Credit Data

End-to-end machine learning project on the UCI Statlog (German Credit) dataset: SQL-based exploration, EDA, a six-model comparison with two class-imbalance strategies, cost-based model selection with decision-threshold tuning, and model interpretation.

## Business question

Can we predict whether a loan applicant is a good or bad credit risk, and which factors drive that risk?

The dataset ships with an official cost matrix: classifying a bad risk as good costs 5 units, the reverse costs 1. "Best model" is therefore defined as the one that minimises expected misclassification cost at its optimal decision threshold — not the one with the highest accuracy.

## Data

1,000 loan applications with 20 attributes (account status, credit history, employment, loan parameters, demographics). Target: good vs. bad credit risk, imbalanced 70/30.

Source: [Statlog (German Credit Data), UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

## Approach

1. **Data preparation** — decoding categorical codes into readable labels, storage in SQLite, exploratory SQL queries.
2. **EDA** — distributions, risk rates per category, outlier and correlation analysis.
3. **Preprocessing** — two pipelines: one-hot encoding only for tree-based models; log transform + standardisation added for linear and distance-based models.
4. **Modelling** — Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM and KNN, each under two imbalance strategies (class weighting and SMOTE). 5-fold stratified cross-validation, hyperparameters tuned with grid search on ROC AUC.
5. **Model selection** — expected cost computed on out-of-fold predictions, sweeping the decision threshold against the cost matrix; the model with the lowest cost wins.
6. **Final evaluation** — a single evaluation on a held-out test set (20%), untouched until this step.
7. **Interpretation** — SHAP values, decision-tree rules and logistic regression odds ratios, cross-checked against the EDA.

## Results

- Final model: class-weighted **Random Forest** operating at a cost-optimal threshold of **0.29** (instead of the default 0.5).
- Test set: ROC AUC **0.789**, catching **83% of bad risks** (50 of 60), with total misclassification cost roughly **30% lower** than the same model at the default threshold.
- The strongest risk drivers — checking-account status, loan duration, savings and credit history — were identified consistently by EDA, SHAP and logistic regression.
- Two counterintuitive patterns held up across all checks: clients with *no checking account* and clients with a *"critical" credit history* are safer than the labels suggest, plausibly because both groups are already known quantities to the lender.

## How to run

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `german.data` from the [UCI repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) into the project root, then open and run `credit_risk_analysis.ipynb` top to bottom. The grid searches could take several minutes on a regular laptop.

## Tech stack

Python (pandas, NumPy), scikit-learn, imbalanced-learn, XGBoost, SHAP, SQLite, seaborn/matplotlib, Jupyter.

## Limitations

- The cost matrix is the dataset's official one; a real lender would calibrate it to its own margins, which would shift the optimal threshold.
- The data describes 1970s German applicants — the method transfers, the specific coefficients do not.
- Probabilities from a class-weighted model are not calibrated; uses that need true default probabilities would require a calibration step.
