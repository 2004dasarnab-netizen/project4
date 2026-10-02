"""
TYPE-II DIABETES PREDICTION USING LOGISTIC REGRESSION
Python reproduction of the uploaded project dissertation.

Source workflow reproduced:
1. Load Iraqi diabetes dataset.
2. Remove pre-diabetic observations, ID and patient-number columns, and missing rows.
3. Convert CLASS: N -> 0, Y -> 1.
4. EDA: contingency tables and stacked bar charts for Gender, Age, HbA1c,
   Cholesterol, TG, HDL, LDL, Creatinine, Urea, BMI and VLDL.
5. 80:20 train/test split with random_state=123.
6. Full logistic regression.
7. Stepwise AIC model selection.
8. Final logistic regression using HbA1c, Cholesterol, TG, LDL and BMI.
9. Confusion matrix, accuracy, precision, recall, F1-score and ROC-AUC.
10. Save tables, plots and model summaries to the output/ directory.

The dissertation's appendix contains a 70% sampling line, while the report text,
degrees of freedom and 189-row test confusion matrix correspond to an 80:20 split.
This script uses 80:20 so that it follows the reported analysis.
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score
)

warnings.filterwarnings("ignore")

FEATURES = [
    "Gender", "Urea", "HbA1c", "Cr", "Chol", "TG",
    "HDL", "LDL", "VLDL", "BMI", "AGE"
]
FINAL_FEATURES = ["HbA1c", "Chol", "TG", "LDL", "BMI"]


def find_dataset(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        Path("diabetes.csv"),
        Path("Dataset of Diabetes.csv"),
        Path("Dataset of Diabetes (1).csv"),
        Path("diabetes_dataset.csv"),
        Path("diabetes_data.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Optional KaggleHub download. This keeps the project self-contained for
    # users who have internet access; local CSV is always preferred.
    try:
        import kagglehub
        folder = Path(kagglehub.dataset_download("aravindpcoder/diabetes-dataset"))
        csvs = list(folder.rglob("*.csv"))
        if csvs:
            return csvs[0]
    except Exception:
        pass

    raise FileNotFoundError(
        "Dataset CSV not found. Put the Kaggle 'Diabetes Dataset' CSV in the "
        "same folder as this script (recommended filename: diabetes.csv), "
        "or install kagglehub and run with internet access."
    )


def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize common column-name variants.
    rename = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "").replace("_", "")
        mapping = {
            "id": "ID",
            "nopation": "No_Pation",
            "nopatient": "No_Pation",
            "gender": "Gender",
            "age": "AGE",
            "urea": "Urea",
            "cr": "Cr",
            "creatinine": "Cr",
            "hba1c": "HbA1c",
            "hba1": "HbA1c",
            "chol": "Chol",
            "cholesterol": "Chol",
            "tg": "TG",
            "triglyceride": "TG",
            "hdl": "HDL",
            "ldl": "LDL",
            "vldl": "VLDL",
            "bmi": "BMI",
            "class": "CLASS",
        }
        if key in mapping:
            rename[c] = mapping[key]
    df = df.rename(columns=rename)

    required = ["Gender", "AGE", "Urea", "Cr", "HbA1c", "Chol",
                "TG", "HDL", "LDL", "VLDL", "BMI", "CLASS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove pre-diabetic class and unused identifier columns.
    cls = df["CLASS"].astype(str).str.strip().str.upper()
    df = df.loc[~cls.isin(["P", "PRE-DIABETIC", "PREDIABETIC", "PRE DIABETIC"])].copy()
    df["CLASS"] = cls.loc[df.index].map({"N": 0, "Y": 1, "NO": 0, "YES": 1})
    df = df.dropna(subset=["CLASS"])

    for c in ["AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Gender"] = (
        df["Gender"].astype(str).str.strip().str.upper()
        .map({"F": 0, "FEMALE": 0, "M": 1, "MALE": 1})
    )
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def save_contingency(df, variable, bins, outdir):
    cat = pd.cut(df[variable], bins=bins, right=False, include_lowest=True)
    tab = pd.crosstab(cat, df["CLASS"])
    tab.columns = ["0", "1"] if len(tab.columns) == 2 else [str(x) for x in tab.columns]
    tab.to_csv(outdir / f"contingency_{variable}.csv")
    return tab


def plot_stacked(df, variable, bins, title, outdir):
    cat = pd.cut(df[variable], bins=bins, right=False, include_lowest=True)
    tab = pd.crosstab(cat, df["CLASS"]).reindex(columns=[0, 1], fill_value=0)
    labels = [str(x) for x in tab.index]

    ax = tab.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    ax.legend(["0 = Non-Diabetic", "1 = Diabetic"], title="CLASS")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / f"{variable}_stacked_bar.png", dpi=180)
    plt.close()


def design_matrix(df, features):
    X = df[features].copy()
    # Gender is already encoded as 0/1.
    return sm.add_constant(X.astype(float), has_constant="add")


def fit_logit(train, features):
    X = design_matrix(train, features)
    y = train["CLASS"].astype(int)
    return sm.GLM(y, X, family=sm.families.Binomial()).fit()


def stepwise_aic(train, candidates):
    """Bidirectional stepwise AIC, matching the role of MASS::stepAIC."""
    current = list(candidates)
    best_model = fit_logit(train, current)
    best_aic = best_model.aic
    changed = True

    while changed:
        changed = False
        trials = []

        if len(current) > 1:
            for feature in current:
                trial = [x for x in current if x != feature]
                try:
                    m = fit_logit(train, trial)
                    trials.append((m.aic, trial, m))
                except Exception:
                    pass

        remaining = [x for x in candidates if x not in current]
        for feature in remaining:
            trial = current + [feature]
            try:
                m = fit_logit(train, trial)
                trials.append((m.aic, trial, m))
            except Exception:
                pass

        if trials:
            aic, variables, model = min(trials, key=lambda z: z[0])
            if aic + 1e-10 < best_aic:
                current, best_aic, best_model = variables, aic, model
                changed = True

    return current, best_model


def save_model_summary(model, path, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\nCoefficients:\n")
        f.write(model.params.to_string())
        f.write("\n\nP-values:\n")
        f.write(model.pvalues.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to the diabetes CSV file")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    outdir = Path("output")
    outdir.mkdir(exist_ok=True)

    dataset = find_dataset(args.data)
    df = load_and_clean(dataset)

    print(f"Dataset: {dataset}")
    print(f"Rows after preprocessing: {len(df)}")
    print(f"Class counts:\n{df['CLASS'].value_counts().sort_index()}\n")

    df.to_csv(outdir / "cleaned_diabetes.csv", index=False)

    # EDA contingency tables and charts using the project's stated intervals.
    intervals = {
        "AGE": [20, 30, 40, 50, 55, 60, 70, 80],
        "HbA1c": [0, 5.6, 6.5, 14.01],
        "Chol": [0, 5.2, 6.2, 8.81],
        "TG": [0, 1.69, 2.25, 6.01],
        "HDL": [0, 1.03, 1.55, 2.51],
        "LDL": [0, 1.03, 1.55, 2.51],
        "Cr": [6, 53, 100, 800.01],
        "Urea": [0, 2.5, 7, 40.01],
        "BMI": [18.5, 25, 30, 40.01],
        "VLDL": [0, 30, 35.01],
    }

    for variable, bins in intervals.items():
        save_contingency(df, variable, bins, outdir)
        plot_stacked(df, variable, bins, f"Stacked Bar Chart for {variable}", outdir)

    # Gender contingency table / chart.
    gender_tab = pd.crosstab(
        df["Gender"].map({0: "Female", 1: "Male"}), df["CLASS"]
    ).reindex(columns=[0, 1], fill_value=0)
    gender_tab.to_csv(outdir / "contingency_Gender.csv")
    ax = gender_tab.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title("Stacked Bar Chart Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Frequency")
    ax.legend(["0 = Non-Diabetic", "1 = Diabetic"], title="CLASS")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "Gender_stacked_bar.png", dpi=180)
    plt.close()

    # 80:20 split: this agrees with the dissertation's reported 189 test rows.
    train, test = train_test_split(
        df, test_size=0.20, random_state=args.seed, stratify=df["CLASS"]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(outdir / "train_data.csv", index=False)
    test.to_csv(outdir / "test_data.csv", index=False)

    # Full model.
    full_model = fit_logit(train, FEATURES)
    save_model_summary(full_model, outdir / "full_logistic_regression.txt",
                       "Full Logistic Regression Model")

    # STEP-AIC.
    selected, step_model = stepwise_aic(train, FEATURES)
    with open(outdir / "step_aic_result.txt", "w", encoding="utf-8") as f:
        f.write("Stepwise AIC selected predictors\n")
        f.write(", ".join(selected) + "\n")
        f.write(f"AIC = {step_model.aic:.6f}\n")

    # The dissertation's final model explicitly uses these five factors.
    final_features = FINAL_FEATURES
    final_model = fit_logit(train, final_features)
    save_model_summary(final_model, outdir / "final_logistic_regression.txt",
                       "Final Logistic Regression: HbA1c + Chol + TG + LDL + BMI")

    # Predictions.
    X_test = design_matrix(test, final_features)
    probabilities = np.asarray(final_model.predict(X_test))
    predictions = (probabilities > 0.5).astype(int)

    cm = confusion_matrix(test["CLASS"], predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm, index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    cm_df.to_csv(outdir / "confusion_matrix.csv")

    accuracy = accuracy_score(test["CLASS"], predictions)
    precision = precision_score(test["CLASS"], predictions, zero_division=0)
    recall = recall_score(test["CLASS"], predictions, zero_division=0)
    f1 = f1_score(test["CLASS"], predictions, zero_division=0)
    auc = roc_auc_score(test["CLASS"], probabilities)

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Value": [accuracy, precision, recall, f1, auc]
    })
    metrics.to_csv(outdir / "model_metrics.csv", index=False)

    pd.DataFrame({
        "Actual": test["CLASS"].values,
        "Probability": probabilities,
        "Predicted": predictions
    }).to_csv(outdir / "test_predictions.csv", index=False)

    # ROC curve.
    fpr, tpr, _ = roc_curve(test["CLASS"], probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close()

    print("Selected by STEP-AIC:", selected)
    print("\nFinal model predictors:", final_features)
    print("\nConfusion Matrix:\n", cm_df)
    print("\nMetrics:")
    for _, row in metrics.iterrows():
        print(f"{row['Metric']}: {row['Value']:.6f}")
    print(f"\nAll outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
