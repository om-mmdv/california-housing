"""
Data loading for the California Housing project.

Every other script gets its data from load_data(),
so the loading logic lives in exactly one place.
"""

from sklearn.datasets import fetch_california_housing

# Name of the column we want to predict (median house value, in $100,000)
TARGET = "MedHouseVal"


def load_data():
    """
    Download the California Housing dataset (once, then it is cached in data/)
    and return it as a pandas DataFrame.

    Returns a table with 20,640 rows: 8 feature columns + the target column.
    """
    dataset = fetch_california_housing(as_frame=True, data_home="data")
    return dataset.frame


if __name__ == "__main__":
    # Quick self-check: run `python -m src.data`
    df = load_data()
    print(f"Shape: {df.shape}")
    print(df.head())