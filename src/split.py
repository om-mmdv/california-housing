"""
Train/test split for California Housing.

Every model uses the same split (same random_state),
so their scores can be compared fairly.
"""

from sklearn.model_selection import train_test_split

from src.data import TARGET, load_data

RANDOM_STATE = 42  # fixed seed: the split is the same on every run
TEST_SIZE = 0.2    # 20% of districts are kept for the final test


def get_split():
    """
    Load the data and split it into train and test parts.

    Returns: X_train, X_test, y_train, y_test
      X = feature columns (what the model sees)
      y = target column (what the model must predict)
    """
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split()
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Features: {list(X_train.columns)}")