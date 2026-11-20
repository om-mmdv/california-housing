"""
Train stronger models and save the best one.

Run from the project folder:
    python -m src.train

Compares the baseline with Random Forest and Gradient Boosting
on the same train/test split, then saves the winner to outputs/.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split import RANDOM_STATE, get_split


def get_models():
    """Return a dictionary: model name -> untrained model."""
    return {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        # n_jobs=-1 means "use all CPU cores"
        "Random Forest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        # Histogram-based gradient boosting: fast and accurate on tables
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred):
    """Return MAE and RMSE (in dollars) and R2."""
    mae = mean_absolute_error(y_true, y_pred) * 100_000
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100_000
    return mae, rmse, r2_score(y_true, y_pred)


def main():
    X_train, X_test, y_train, y_test = get_split()
    os.makedirs("outputs", exist_ok=True)

    best_name, best_r2, best_model = None, -np.inf, None

    for name, model in get_models().items():
        model.fit(X_train, y_train)          # learn from the train part
        predictions = model.predict(X_test)  # predict the unseen test part
        mae, rmse, r2 = evaluate(y_test, predictions)
        print(f"{name:18s} | MAE ${mae:9,.0f} | RMSE ${rmse:9,.0f} | R2 {r2:6.3f}")

        if r2 > best_r2:
            best_name, best_r2, best_model = name, r2, model

    # Save the best model so that other scripts can reuse it
    joblib.dump(best_model, "outputs/best_model.joblib")
    print(f"\nBest model: {best_name} (R2 {best_r2:.3f})")
    print("Saved to outputs/best_model.joblib")


if __name__ == "__main__":
    main()