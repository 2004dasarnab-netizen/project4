# Type-II Diabetes Prediction Using Logistic Regression — Python

This project is a Python reproduction of the uploaded dissertation **“TYPE-II DIABETES PREDICTION USING LOGISTIC REGRESSION.”**

## Workflow reproduced

1. Remove pre-diabetic records.
2. Remove ID / patient-number columns and missing observations.
3. Encode `CLASS`: N = 0, Y = 1.
4. Exploratory Data Analysis using contingency tables and stacked bar charts.
5. 80:20 train-test split with seed 123.
6. Full logistic regression using Gender, Urea, HbA1c, Cr, Chol, TG, HDL, LDL, VLDL, BMI and AGE.
7. Bidirectional Stepwise AIC.
8. Final logistic regression using HbA1c, Cholesterol, Triglyceride, LDL and BMI.
9. Confusion matrix, Accuracy, Precision, Recall, F1-score and ROC-AUC.
10. ROC curve and output tables.

## Important note about the source report

The dissertation text says 80:20, while one appendix line contains `0.7*nrow(...)`. The reported null/residual degrees of freedom and the 189 observations in the final confusion matrix are consistent with an 80:20 split, so the Python implementation uses **80:20**.

The report's final model reports:
- Accuracy = 0.978836
- Precision = 0.9090909
- Recall = 0.9090909
- F1 score = 0.9090909
- ROC plot labels AUC ≈ 0.95

Exact numerical coefficients/metrics are data-dependent, so the Python script calculates them from the actual CSV rather than hard-coding the dissertation's displayed values.

## Dataset

The dissertation cites the Aravind P. Diabetes Dataset on Kaggle. Download the CSV and place it beside the script as:

`diabetes.csv`

The script can also attempt to download the dataset through `kagglehub` when internet access is available.

## Run

```bash
pip install -r requirements.txt
python diabetes_logistic_regression.py
```

Or specify the CSV:

```bash
python diabetes_logistic_regression.py --data "path/to/diabetes.csv"
```

## Output

The script creates an `output/` directory containing:

- `cleaned_diabetes.csv`
- `train_data.csv`
- `test_data.csv`
- contingency tables for every factor
- stacked bar charts
- full logistic regression summary
- Step-AIC result
- final logistic regression summary
- confusion matrix
- model metrics
- test predictions
- ROC curve

## Source

The uploaded dissertation identifies the data source as:
Aravind, P. Diabetes Dataset, 2023, Kaggle.
