"""
Baseline models: the simplest possible predictions.

Run from the project folder:
    python -m src.baseline

Any better model must beat these numbers to be worth using.
"""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split import get_split


def score(model_name, y_true, y_pred):
    """Print the three metrics. Prices are converted to dollars."""
    mae = mean_absolute_error(y_true, y_pred) * 100_000
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100_000
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name:22s} | MAE ${mae:9,.0f} | RMSE ${rmse:9,.0f} | R2 {r2:6.3f}")


def main():
    X_train, X_test, y_train, y_test = get_split()

    # Baseline 1: always predict the average training price
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    score("Always the mean", y_test, dummy.predict(X_test))

    # Baseline 2: linear regression.
    # StandardScaler puts all features on a similar scale (mean 0, spread 1).
    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(X_train, y_train)  # "fit" = the model learns from the train part
    score("Linear regression", y_test, linear.predict(X_test))


if __name__ == "__main__":
    main()