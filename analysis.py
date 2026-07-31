"""
MovieIQ - Predictive Analytics on Film Success
Stages 0-4: Data Prep, EDA, Statistical Testing, Modeling
Run this once to generate charts (assets/) and print all the answers
required by the project brief. The Streamlit app (MovieIQ.py) reuses
the same load/clean/train functions.
"""

import ast
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
ASSETS = "assets"


# ---------------------------------------------------------------------
# STAGE 1 - DATA PREPARATION
# ---------------------------------------------------------------------
def load_and_clean(path="movies.csv"):
    df = pd.read_csv(path)

    # Drop rows where budget or revenue is missing or zero.
    # A budget/revenue of 0 almost always means the value was never
    # recorded (not that the movie truly cost/earned nothing), so it
    # would badly distort ratios and the success label.
    before = len(df)
    df = df.dropna(subset=["budget", "revenue", "popularity", "runtime", "vote_average"])
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    removed = before - len(df)

    # Target: success = 1 if revenue > budget else 0
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    # Parse genres (stored as a stringified list of dicts) into a
    # clean list of genre names, plus a single "primary_genre" column
    # for simple filtering/analysis.
    def parse_genres(cell):
        try:
            items = ast.literal_eval(cell)
            return [d["name"] for d in items]
        except (ValueError, SyntaxError):
            return []

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda g: g[0] if g else "Unknown")

    return df, removed


# ---------------------------------------------------------------------
# STAGE 4 - MODEL TRAINING (shared with the Streamlit app)
# ---------------------------------------------------------------------
FEATURES = ["budget", "popularity", "runtime", "vote_average"]


def train_model(df, test_size=0.2, random_state=42):
    X = df[FEATURES]
    y = df["success"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, metrics, importances, (X_test, y_test, y_pred)


# ---------------------------------------------------------------------
# MAIN - runs every stage and prints the write-up answers
# ---------------------------------------------------------------------
def main():
    import os
    os.makedirs(ASSETS, exist_ok=True)

    df, removed = load_and_clean()

    print("=" * 70)
    print("STAGE 1 - DATA PREPARATION")
    print("=" * 70)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]} (after removing {removed} bad rows)")
    print("\nSummary statistics:\n", df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())

    success_rate = df["success"].mean()
    print(f"\nProportion successful: {success_rate:.2%}")
    print("Balanced?" , "Roughly balanced" if 0.4 < success_rate < 0.6 else "Imbalanced")

    # ------------------------------------------------------------------
    # STAGE 2 - EDA
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # 1. Budget vs Revenue scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="budget", y="revenue", hue="success", palette={0: "crimson", 1: "seagreen"}, alpha=0.6)
    plt.title("Budget vs. Revenue")
    plt.xlabel("Budget ($)")
    plt.ylabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/budget_vs_revenue.png", dpi=150)
    plt.close()
    corr_br = df["budget"].corr(df["revenue"])
    print(f"Budget-Revenue correlation: {corr_br:.3f} -> higher budgets do tend to earn higher revenue")

    # 2. Genre trends
    genre_exploded = df.explode("genre_list")
    genre_counts = genre_exploded["genre_list"].value_counts().head(10)
    genre_success = genre_exploded.groupby("genre_list")["success"].mean().sort_values(ascending=False).head(10)

    plt.figure(figsize=(9, 5))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis")
    plt.title("Most Common Genres")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/genre_counts.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(x=genre_success.values, y=genre_success.index, palette="magma")
    plt.title("Success Rate by Genre")
    plt.xlabel("Success Rate")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/genre_success_rate.png", dpi=150)
    plt.close()
    print("Most common genre:", genre_counts.index[0])
    print("Highest success-rate genre:", genre_success.index[0])

    # 3. popularity/runtime/vote_average vs success
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(["popularity", "runtime", "vote_average"], 1):
        plt.subplot(1, 3, i)
        sns.boxplot(data=df, x="success", y=col, hue="success",
                    palette={0: "crimson", 1: "seagreen"}, legend=False)
        plt.title(f"{col} vs success")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/feature_vs_success_boxplots.png", dpi=150)
    plt.close()

    means_by_success = df.groupby("success")[["popularity", "runtime", "vote_average"]].mean()
    print("\nMean feature values by success:\n", means_by_success)

    # 4. Correlation heatmap
    plt.figure(figsize=(7, 6))
    num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/correlation_heatmap.png", dpi=150)
    plt.close()
    print("\nCharts saved to assets/: budget_vs_revenue.png, genre_counts.png, "
          "genre_success_rate.png, feature_vs_success_boxplots.png, correlation_heatmap.png")

    # ------------------------------------------------------------------
    # STAGE 3 - STATISTICAL TESTING
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3 - STATISTICAL TESTING")
    print("=" * 70)

    # 1. T-test: vote_average, successful vs unsuccessful
    successful = df.loc[df["success"] == 1, "vote_average"]
    unsuccessful = df.loc[df["success"] == 0, "vote_average"]
    t_stat, p_val_t = stats.ttest_ind(successful, unsuccessful, equal_var=False)
    print("T-Test on vote_average (successful vs unsuccessful)")
    print("H0: mean vote_average is the same for successful and unsuccessful movies")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_val_t:.5f}")
    print("Conclusion:", "Reject H0 (significant difference)" if p_val_t < 0.05 else "Fail to reject H0 (no significant difference)")

    # 2. Chi-Square: primary_genre vs success
    contingency = pd.crosstab(df["primary_genre"], df["success"])
    chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency)
    print("\nChi-Square Test on primary_genre vs success")
    print("H0: genre and success are independent (no association)")
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_val_chi:.5f}")
    print("Conclusion:", "Reject H0 (genre is associated with success)" if p_val_chi < 0.05 else "Fail to reject H0 (no association found)")

    print("\np-value meaning: the probability of seeing a difference this large (or larger) "
          "by random chance if H0 were actually true. We use a significance threshold "
          "(alpha) of 0.05, the standard convention balancing false positives vs false negatives.")

    # ------------------------------------------------------------------
    # STAGE 4 - MODELING
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 4 - PREDICTIVE MODELING (RANDOM FOREST)")
    print("=" * 70)
    print(f"Features used: {FEATURES}")
    print("Excluded: 'title' (identifier, not predictive), 'revenue' (this IS how the "
          "target is defined, so including it would leak the answer), 'genres' (kept "
          "optional/simple here; already tested separately via Chi-Square)")

    model, metrics, importances, _ = train_model(df)
    print(f"\nTrain/test split: 80/20 (stratified) - a held-out test set is needed "
          f"to check the model on data it never saw during training, so its score "
          f"reflects real generalization instead of memorization.")

    print(f"\nAccuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Fail", "Pred Success"],
                yticklabels=["Actual Fail", "Actual Success"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/confusion_matrix.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(x=importances.values, y=importances.index, palette="crest")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{ASSETS}/feature_importance.png", dpi=150)
    plt.close()
    print("\nFeature importances:\n", importances)

    print("\nDone. All charts saved under assets/.")


if __name__ == "__main__":
    main()
