from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = Path("data")
EDA_DIR = Path("eda") / "outputs"
EDA_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# Missingness
# ============================================================
def plot_missingness(matches):
    """Compute and plot missingness by year."""
    missing_by_year = matches.groupby("year").agg(lambda x: x.isna().mean()).round(2).reset_index()
    missing_by_year.to_csv(EDA_DIR / "missingness_by_year.csv", index=False)

    key_cols = [
        "red_autoPoints", "blue_autoPoints",
        "red_teleopPoints", "blue_teleopPoints",
        "red_foulPoints", "blue_foulPoints",
    ]
    present_cols = [c for c in key_cols if c in missing_by_year.columns]

    if present_cols:
        long_missing = missing_by_year.melt(
            id_vars="year", value_vars=present_cols,
            var_name="column", value_name="missing_fraction"
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=long_missing, x="year", y="missing_fraction", hue="column", marker="o")
        plt.title("Missingness Trend by Year (Selected Match Fields)")
        plt.ylabel("Fraction Missing")
        plt.xlabel("Year")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "missingness_by_year.png")
        plt.close()