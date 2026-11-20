"""
Exploratory data analysis (EDA) for California Housing.

Run from the project folder:
    python -m src.explore

Prints basic statistics and saves four plots to assets/.
"""

import os

import matplotlib.pyplot as plt

from src.data import TARGET, load_data


def print_summary(df):
    """Print the basic facts about the table."""
    print("=== Shape (rows, columns) ===")
    print(df.shape)

    print("\n=== Missing values per column ===")
    print(df.isna().sum())

    print("\n=== Statistics (mean, min, max, ...) ===")
    print(df.describe().round(2).T)

    # The target is capped: all houses above $500,000 are recorded as 5.0
    capped = (df[TARGET] >= 5.0).sum()
    print(f"\nDistricts with the capped price (>= 5.0): {capped} "
          f"({capped / len(df) * 100:.1f}%)")


def plot_price_distribution(df):
    """Histogram: how are house prices distributed?"""
    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET] * 100_000, bins=50, color="#4c72b0", edgecolor="white")
    plt.xlabel("Median house value, $")
    plt.ylabel("Number of districts")
    plt.title("Distribution of house prices")
    plt.tight_layout()
    plt.savefig("assets/price_distribution.png", dpi=150)
    plt.close()


def plot_map(df):
    """Scatter plot of latitude/longitude, colored by price: draws California."""
    plt.figure(figsize=(8, 7))
    points = plt.scatter(
        df["Longitude"], df["Latitude"],
        c=df[TARGET], cmap="viridis", s=6, alpha=0.5,
    )
    plt.colorbar(points, label="Median house value ($100,000)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("House prices across California")
    plt.tight_layout()
    plt.savefig("assets/price_map.png", dpi=150)
    plt.close()


def plot_correlations(df):
    """Bar chart: which features are linearly related to the price?"""
    # Correlation: +1 = grows together, -1 = opposite, 0 = no linear link
    correlations = df.corr()[TARGET].drop(TARGET).sort_values()

    plt.figure(figsize=(8, 5))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in correlations]
    plt.barh(correlations.index, correlations.values, color=colors)
    plt.xlabel("Correlation with house value")
    plt.title("Which features relate to the price?")
    plt.tight_layout()
    plt.savefig("assets/correlations.png", dpi=150)
    plt.close()

    print("\n=== Correlation with the target ===")
    print(correlations.round(3).sort_values(ascending=False))


def plot_income_vs_price(df):
    """Scatter plot: income vs price (the strongest relationship)."""
    plt.figure(figsize=(8, 5))
    plt.scatter(df["MedInc"], df[TARGET], s=4, alpha=0.3, color="#4c72b0")
    plt.xlabel("Median income (in $10,000)")
    plt.ylabel("Median house value ($100,000)")
    plt.title("Income vs house price")
    plt.tight_layout()
    plt.savefig("assets/income_vs_price.png", dpi=150)
    plt.close()


def main():
    os.makedirs("assets", exist_ok=True)
    df = load_data()

    print_summary(df)
    plot_price_distribution(df)
    plot_map(df)
    plot_correlations(df)
    plot_income_vs_price(df)

    print("\nPlots saved to assets/")


if __name__ == "__main__":
    main()