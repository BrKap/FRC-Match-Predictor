from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = Path("data")
EDA_DIR = Path("eda") / "outputs"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
# ============================================================
# Core Match Plots
# ============================================================
def plot_basic_distributions(matches, teams):
    """Plots win distribution, yearly average score, and correlations."""
    print("\n=== Basic Match Plots ===")

    # Win distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=matches,
        x="winning_alliance",
        palette={"red": "red", "blue": "blue"}
    )
    plt.title("Win Distribution (Red vs Blue)")
    plt.xlabel("Winning Alliance")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "win_distribution.png")
    plt.close()


    # Average score by year
    matches["avg_score"] = matches[["red_score", "blue_score"]].mean(axis=1)
    yearly_scores = matches.groupby("year")["avg_score"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=yearly_scores, x="year", y="avg_score", marker="o")
    plt.title("Average Match Score by Year")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "yearly_average_score.png")
    plt.close()

    # Correlation heatmap
    num_cols = [
        "red_score", "blue_score",
        "red_autoPoints", "blue_autoPoints",
        "red_teleopPoints", "blue_teleopPoints",
        "red_foulPoints", "blue_foulPoints",
    ]
    corr = matches[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Scoring Components")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close()


# ============================================================
# Team Win Rates and Leaderboards
# ============================================================
def plot_team_winrates(matches):
    """Compute and plot per-team win rates."""
    print("\n=== Team Win Rate Analysis ===")
    matches = matches[matches["winning_alliance"].isin(["red", "blue"])]

    # Build (team, win)
    team_results = []
    for _, row in matches.iterrows():
        red_teams = row["red_teams"].split(",")
        blue_teams = row["blue_teams"].split(",")
        winner = row["winning_alliance"]
        for t in red_teams:
            team_results.append({"team_id": t.strip(), "win": 1 if winner == "red" else 0})
        for t in blue_teams:
            team_results.append({"team_id": t.strip(), "win": 1 if winner == "blue" else 0})

    team_df = pd.DataFrame(team_results)
    team_winrates = (
        team_df.groupby("team_id")["win"]
        .agg(["count", "sum"])
        .rename(columns={"count": "matches_played", "sum": "wins"})
        .assign(win_rate=lambda df: df["wins"] / df["matches_played"])
        .reset_index()
    )

    # Save
    team_winrates.to_csv(EDA_DIR / "team_winrates.csv", index=False)

    # Distribution plot (1% bins)
    plt.figure(figsize=(8, 5))
    sns.histplot(team_winrates["win_rate"], bins=100, kde=True, color="purple")
    plt.title("Distribution of Team Win Rates (1% Bins)")
    plt.xlabel("Win Rate")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "team_winrate_distribution.png")
    plt.close()

    # Top 20 leaderboard
    top20 = (
        team_winrates[team_winrates["matches_played"] >= 20]
        .sort_values(["win_rate", "matches_played"], ascending=[False, False])
        .head(20)
    )
    top20.to_csv(EDA_DIR / "team_winrate_top20.csv", index=False)
    print(top20[["team_id", "matches_played", "wins", "win_rate"]])
    return team_winrates


# ============================================================
# Yearly Trends
# ============================================================
def plot_yearly_winrate_trends(matches):
    """Plot average and distribution of team win rates by year."""
    print("\n=== Yearly Win Rate Trends ===")
    matches = matches[matches["winning_alliance"].isin(["red", "blue"])]

    yearly_results = []
    for _, row in matches.iterrows():
        year = int(row["year"])
        red_teams = row["red_teams"].split(",")
        blue_teams = row["blue_teams"].split(",")
        winner = row["winning_alliance"]
        for t in red_teams:
            yearly_results.append({"year": year, "team_id": t.strip(), "win": 1 if winner == "red" else 0})
        for t in blue_teams:
            yearly_results.append({"year": year, "team_id": t.strip(), "win": 1 if winner == "blue" else 0})

    yearly_df = pd.DataFrame(yearly_results)
    yearly_team_winrate = (
        yearly_df.groupby(["year", "team_id"])["win"]
        .agg(["count", "sum"])
        .rename(columns={"count": "matches_played", "sum": "wins"})
        .assign(win_rate=lambda df: df["wins"] / df["matches_played"])
        .reset_index()
    )

    # Average by year
    yearly_avg = yearly_team_winrate.groupby("year")["win_rate"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=yearly_avg, x="year", y="mean", marker="o", color="teal")
    plt.fill_between(yearly_avg["year"], yearly_avg["mean"] - yearly_avg["std"], yearly_avg["mean"] + yearly_avg["std"], alpha=0.2)
    plt.title("Average Team Win Rate by Year (±1 SD)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "yearly_avg_team_winrate.png")
    plt.close()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=yearly_team_winrate, x="year", y="win_rate", inner="quartile", color="skyblue")
    plt.title("Distribution of Team Win Rates by Year")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "yearly_team_winrate_distribution.png")
    plt.close()