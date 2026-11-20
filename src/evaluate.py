"""
Evaluate the saved best model on the test set.

Run from the project folder:
    python -m src.evaluate

Prints metrics, the worst mistakes, and saves three plots to assets/.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from src.split import RANDOM_STATE, get_split


def plot_predicted_vs_actual(y_true, y_pred):
    """Scatter plot: the closer the points are to the red line, the better."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true * 100_000, y_pred * 100_000, s=5, alpha=0.3, color="#4c72b0")
    limit = [0, 5.2 * 100_000]
    plt.plot(limit, limit, color="red", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual median house value, $")
    plt.ylabel("Predicted median house value, $")
    plt.title("Predicted vs actual (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/predicted_vs_actual.png", dpi=150)
    plt.close()


def plot_error_distribution(y_true, y_pred):
    """Histogram of errors (predicted - actual). Centered at 0 = no bias."""
    errors = (y_pred - y_true) * 100_000
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=60, color="#4c72b0", edgecolor="white")
    plt.axvline(0, color="red", linewidth=1.5)
    plt.xlabel("Error (predicted - actual), $")
    plt.ylabel("Number of districts")
    plt.title("Distribution of prediction errors")
    plt.tight_layout()
    plt.savefig("assets/error_distribution.png", dpi=150)
    plt.close()


def plot_feature_importance(model, X_test, y_test):
    """
    Permutation importance: shuffle one feature at a time and measure
    how much the score drops. A big drop = an important feature.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    order = np.argsort(result.importances_mean)

    plt.figure(figsize=(8, 5))
    plt.barh(X_test.columns[order], result.importances_mean[order], color="#5cb85c")
    plt.xlabel("Drop in R2 when the feature is shuffled")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png", dpi=150)
    plt.close()

    print("\n=== Feature importance (higher = more important) ===")
    for i in order[::-1]:
        print(f"  {X_test.columns[i]:12s} {result.importances_mean[i]:.3f}")


def main():
    os.makedirs("assets", exist_ok=True)
    _, X_test, _, y_test = get_split()

    model = joblib.load("outputs/best_model.joblib")
    y_pred = model.predict(X_test)

    print(f"MAE: ${mean_absolute_error(y_test, y_pred) * 100_000:,.0f}")
    print(f"R2:  {r2_score(y_test, y_pred):.3f}")

    # Share of districts predicted within $50,000 of the real value
    within_50k = (np.abs(y_pred - y_test) * 100_000 <= 50_000).mean()
    print(f"Predictions within $50,000 of the truth: {within_50k * 100:.1f}%")

    # The five biggest mistakes
    errors = np.abs(y_pred - y_test) * 100_000
    worst = errors.sort_values(ascending=False).head(5).index
    print("\n=== Five biggest mistakes ===")
    for idx in worst:
        actual = y_test.loc[idx] * 100_000
        predicted = y_pred[y_test.index.get_loc(idx)] * 100_000
        print(f"  actual ${actual:9,.0f} | predicted ${predicted:9,.0f}")

    plot_predicted_vs_actual(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)
    plot_feature_importance(model, X_test, y_test)
    print("\nPlots saved to assets/")


if __name__ == "__main__":
    main()