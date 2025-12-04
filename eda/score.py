import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
DATA_DIR = Path("data")
EDA_DIR = Path("eda") / "outputs"
EDA_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
# ============================================================
# Auto Phase Analysis
# ============================================================
def plot_auto_analysis(matches):
    """Run all Auto-related plots - identical to original eda.py output."""
    print("\n=== Auto Phase Analysis ===")

    # ---------------------------------------------------------
    # Reproduce matches_clean filtering from eda.py
    # ---------------------------------------------------------
    matches_clean = matches.copy()
    matches_clean = matches_clean.dropna(subset=[
        "red_score", "blue_score",
        "red_autoPoints", "blue_autoPoints",
        "red_teleopPoints", "blue_teleopPoints"
    ])
    matches_clean = matches_clean[matches_clean["winning_alliance"].isin(["red", "blue"])]

    # ---------------------------------------------------------
    # Compute ratios safely
    # ---------------------------------------------------------
    matches_clean["red_auto_ratio"] = matches_clean["red_auto_ratio_norm"]
    matches_clean["blue_auto_ratio"] = matches_clean["blue_auto_ratio_norm"]

    # Keep only valid, realistic ratios (0-1 range)
    ratio_df = matches_clean.dropna(subset=["red_auto_ratio", "blue_auto_ratio"])
    ratio_df = ratio_df[
        (ratio_df["red_auto_ratio"].between(0, 1)) &
        (ratio_df["blue_auto_ratio"].between(0, 1))
    ]

    ratio_df["red_win"] = (ratio_df["winning_alliance"] == "red").astype(int)
    ratio_df["auto_diff"] = ratio_df["red_auto_ratio"] - ratio_df["blue_auto_ratio"]

    # ---------------------------------------------------------
    # Plot 1: Auto Ratio Difference vs Match Outcome
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    sns.histplot(
        data=ratio_df,
        x="auto_diff",
        hue="red_win",
        multiple="stack",
        bins=50,
        palette=["blue", "red"]
    )
    plt.title("Auto Ratio Difference vs Match Outcome")
    plt.xlabel("Red Auto Ratio - Blue Auto Ratio")
    plt.ylabel("Match Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "auto_diff_winrate_relationship.png")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Win Probability vs Auto Ratio Advantage
    # ---------------------------------------------------------
    ratio_df["auto_bin"] = pd.cut(ratio_df["auto_diff"], bins=np.linspace(-0.5, 0.5, 21))
    win_prob = ratio_df.groupby("auto_bin")["red_win"].mean().reset_index()
    win_prob["mid"] = win_prob["auto_bin"].apply(lambda x: x.mid)
    
    # Recompute using a smoother function with confidence intervals
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=win_prob,
        x="mid",
        y="red_win",
        color="red",
        marker="o",
        errorbar=('ci', 95)
    )

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=win_prob, x="mid", y="red_win", marker="o", color="red")
    plt.title("Win Probability vs Auto Ratio Advantage")
    plt.xlabel("Red Auto Ratio - Blue Auto Ratio")
    plt.ylabel("Probability Red Wins")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "auto_win_probability_curve.png")
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Auto vs Teleop Tradeoff
    # ---------------------------------------------------------
    matches_clean["auto_share"] = (
        (matches_clean["red_autoPoints"] + matches_clean["blue_autoPoints"]) /
        (matches_clean["red_score"] + matches_clean["blue_score"])
    )
    matches_clean["teleop_share"] = (
        (matches_clean["red_teleopPoints"] + matches_clean["blue_teleopPoints"]) /
        (matches_clean["red_score"] + matches_clean["blue_score"])
    )

    # Filter invalid shares
    matches_clean = matches_clean[
        (matches_clean["auto_share"].between(0, 1.5)) &
        (matches_clean["teleop_share"].between(0, 3))
    ]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=matches_clean, x="auto_share", y="teleop_share", alpha=0.3)
    sns.regplot(data=matches_clean, x="auto_share", y="teleop_share", scatter=False, color="red")
    plt.title("Relationship Between Auto and Teleop Contributions")
    plt.xlabel("Auto Share of Total Points")
    plt.ylabel("Teleop Share of Total Points")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "auto_vs_teleop_tradeoff.png")
    plt.close()

    print("Auto analysis plots saved.")


# ============================================================
# Teleop Analysis
# ============================================================
def plot_teleop_analysis(matches):
    """Analyze Teleop contribution vs. Win Probability."""
    print("\n=== Teleop Phase Analysis ===")

    # Clean relevant fields
    matches_clean = matches.dropna(subset=[
        "red_score", "blue_score",
        "red_teleopPoints", "blue_teleopPoints",
        "red_autoPoints", "blue_autoPoints"
    ])
    matches_clean = matches_clean[matches_clean["winning_alliance"].isin(["red", "blue"])]

    # Compute ratios
    matches_clean["red_teleop_ratio"] = matches_clean["red_teleop_ratio_norm"]
    matches_clean["blue_teleop_ratio"] = matches_clean["blue_teleop_ratio_norm"]

    ratio_df = matches_clean.dropna(subset=["red_teleop_ratio", "blue_teleop_ratio"])
    ratio_df = ratio_df[
        (ratio_df["red_teleop_ratio"].between(0, 1)) &
        (ratio_df["blue_teleop_ratio"].between(0, 1))
    ]

    ratio_df["red_win"] = (ratio_df["winning_alliance"] == "red").astype(int)
    ratio_df["teleop_diff"] = ratio_df["red_teleop_ratio"] - ratio_df["blue_teleop_ratio"]

    # Win Probability vs Teleop Difference
    ratio_df["teleop_bin"] = pd.cut(ratio_df["teleop_diff"], bins=np.linspace(-0.5, 0.5, 21))
    win_prob = ratio_df.groupby("teleop_bin")["red_win"].mean().reset_index()
    win_prob["mid"] = win_prob["teleop_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=win_prob, x="mid", y="red_win", marker="o", color="blue")
    plt.title("Win Probability vs Teleop Ratio Advantage")
    plt.xlabel("Red Teleop Ratio - Blue Teleop Ratio")
    plt.ylabel("Probability Red Wins")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "teleop_win_probability_curve.png")
    plt.close()
    
    # Plot: Teleop Ratio Difference vs Match Outcome
    plt.figure(figsize=(7, 5))
    sns.histplot(
        data=ratio_df,
        x="teleop_diff",
        hue="red_win",
        multiple="stack",
        bins=50,
        palette=["blue", "red"]
    )
    plt.title("Teleop Ratio Difference vs Match Outcome")
    plt.xlabel("Red Teleop Ratio - Blue Teleop Ratio")
    plt.ylabel("Match Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "teleop_diff_winrate_relationship.png")
    plt.close()


    print("Teleop analysis plot saved.")

# ============================================================
# Endgame Analysis
# ============================================================
def plot_endgame_analysis(matches):
    """Estimate Endgame points and analyze their effect on win probability."""
    print("\n=== Endgame Phase Analysis ===")

    matches_clean = matches.dropna(subset=[
        "red_score", "blue_score",
        "red_autoPoints", "blue_autoPoints",
        "red_teleopPoints", "blue_teleopPoints",
        "red_foulPoints", "blue_foulPoints"
    ])
    matches_clean = matches_clean[matches_clean["winning_alliance"].isin(["red", "blue"])]

    # Derive estimated endgame points
    matches_clean["red_endgamePoints"] = (
        matches_clean["red_score"]
        - (matches_clean["red_autoPoints"] + matches_clean["red_teleopPoints"] + matches_clean["red_foulPoints"])
    )
    matches_clean["blue_endgamePoints"] = (
        matches_clean["blue_score"]
        - (matches_clean["blue_autoPoints"] + matches_clean["blue_teleopPoints"] + matches_clean["blue_foulPoints"])
    )

    # Compute ratios
    matches_clean["red_endgame_ratio"] = matches_clean["red_endgame_ratio_norm"]
    matches_clean["blue_endgame_ratio"] = matches_clean["blue_endgame_ratio_norm"]

    ratio_df = matches_clean.dropna(subset=["red_endgame_ratio", "blue_endgame_ratio"])
    ratio_df = ratio_df[
        (ratio_df["red_endgame_ratio"].between(-0.2, 1.5)) &
        (ratio_df["blue_endgame_ratio"].between(-0.2, 1.5))
    ]

    ratio_df["red_win"] = (ratio_df["winning_alliance"] == "red").astype(int)
    ratio_df["endgame_diff"] = ratio_df["red_endgame_ratio"] - ratio_df["blue_endgame_ratio"]

    # Win Probability vs Endgame Difference
    ratio_df["endgame_bin"] = pd.cut(ratio_df["endgame_diff"], bins=np.linspace(-0.5, 0.5, 21))
    win_prob = ratio_df.groupby("endgame_bin")["red_win"].mean().reset_index()
    win_prob["mid"] = win_prob["endgame_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(7, 5))
    sns.lineplot(data=win_prob, x="mid", y="red_win", marker="o", color="green")
    plt.title("Win Probability vs Endgame Ratio Advantage")
    plt.xlabel("Red Endgame Ratio - Blue Endgame Ratio")
    plt.ylabel("Probability Red Wins")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "endgame_win_probability_curve.png")
    plt.close()
    
    # Plot: Endgame Ratio Difference vs Match Outcome
    plt.figure(figsize=(7, 5))
    sns.histplot(
        data=ratio_df,
        x="endgame_diff",
        hue="red_win",
        multiple="stack",
        bins=50,
        palette=["blue", "red"]
    )
    plt.title("Endgame Ratio Difference vs Match Outcome")
    plt.xlabel("Red Endgame Ratio - Blue Endgame Ratio")
    plt.ylabel("Match Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "endgame_diff_winrate_relationship.png")
    plt.close()


    print("Endgame analysis plot saved.")