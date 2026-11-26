"""
Export the best model and test predictions for the dashboard.

Run from the project folder:
    python -m src.export
"""

import joblib
import pandas as pd

from src.split import get_split


def main():
    _, X_test, _, y_test = get_split()
    model = joblib.load("outputs/best_model.joblib")

    joblib.dump(model, "models/model.joblib", compress=3)

    out = X_test.copy()
    out["actual"] = y_test.to_numpy()
    out["predicted"] = model.predict(X_test)
    out.round(4).to_csv("models/test_predictions.csv", index=False)
    print(f"Saved model and {len(out)} test predictions to models/")


if __name__ == "__main__":
    main()