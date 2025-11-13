from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
DATA_DIR = Path("data")
EDA_DIR = Path("eda") / "outputs"
EDA_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
# ============================================================
# Event Competitiveness Analysis (Mean / Median / Mode)
# ============================================================

def analyze_event_competitiveness(matches):
    """
    Compute event-level mean/median/mode competitiveness compared to
    yearly averages for Auto, Teleop, Endgame, and TotalMinusFouls.
    Outputs CSV + distributions.
    """
    print("\n=== Event Competitiveness Analysis ===")
    m = matches.copy()

    # --------------------------------------------------------
    # Add base_event_key (strip 4-digit year prefix)
    # --------------------------------------------------------
    m["base_event_key"] = m["event_key"].str[4:]

    # --------------------------------------------------------
    # Compute Endgame & TotalMinusFouls
    # --------------------------------------------------------
    m["red_endgamePoints"] = (
        m["red_score"]
        - (m["red_autoPoints"] + m["red_teleopPoints"] + m["red_foulPoints"])
    )
    m["blue_endgamePoints"] = (
        m["blue_score"]
        - (m["blue_autoPoints"] + m["blue_teleopPoints"] + m["blue_foulPoints"])
    )

    m["red_totalMinusFouls"] = m["red_score"] - m["red_foulPoints"]
    m["blue_totalMinusFouls"] = m["blue_score"] - m["blue_foulPoints"]

    # Clip negatives that can arise from schema quirks
    for col in ["red_endgamePoints", "blue_endgamePoints"]:
        m[col] = m[col].clip(lower=0)

    # --------------------------------------------------------
    # Melt long format
    # --------------------------------------------------------
    long_scores = m.melt(
        id_vars=["year", "event_key", "base_event_key"],
        value_vars=[
            "red_autoPoints", "blue_autoPoints",
            "red_teleopPoints", "blue_teleopPoints",
            "red_endgamePoints", "blue_endgamePoints",
            "red_totalMinusFouls", "blue_totalMinusFouls"
        ],
        var_name="metric",
        value_name="points"
    ).dropna(subset=["points"])

    long_scores["phase"] = long_scores["metric"].str.extract(
        r"(auto|teleop|endgame|totalMinusFouls)", expand=False
    )

    # --------------------------------------------------------
    # Per-event stats
    # --------------------------------------------------------
    def safe_mode(x):
        mode_vals = x.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else np.nan

    event_stats = (
        long_scores.groupby(["year", "base_event_key", "phase"])["points"]
        .agg(mean="mean", median="median", mode=safe_mode)
        .reset_index()
    )

    # Year-level baselines
    yearly_stats = (
        long_scores.groupby(["year", "phase"])["points"]
        .agg(mean_year="mean", median_year="median", mode_year=safe_mode)
        .reset_index()
    )

    merged = event_stats.merge(yearly_stats, on=["year", "phase"], how="left")

    # --------------------------------------------------------
    # Safe ratio calc: 0/0 = 0, x/0 = NaN
    # --------------------------------------------------------
    def safe_ratio(num, den):
        if pd.isna(num) or pd.isna(den):
            return np.nan
        if den == 0:          # Avoid division by zero
            return 0.0 if num == 0 else np.nan
        return num / den

    for stat in ["mean", "median", "mode"]:
        num_col = stat
        den_col = f"{stat}_year"
        merged[f"{stat}_ratio"] = [
            safe_ratio(n, d) for n, d in zip(merged[num_col], merged[den_col])
        ]

    # Save full CSV
    merged.to_csv(EDA_DIR / "event_competitiveness_full.csv", index=False)

    # --------------------------------------------------------
    # Distribution plots for mean/median/mode
    # --------------------------------------------------------
    bins = np.arange(0.5, 1.6, 0.05)

    for stat in ["mean", "median", "mode"]:
        df = merged.copy()
        df["bucket"] = pd.cut(df[f"{stat}_ratio"], bins=bins)

        counts = (
            df.groupby(["phase", "bucket"], observed=False)
            .size()
            .reset_index(name="event_count")
        )
        counts["bucket_mid"] = counts["bucket"].apply(lambda x: x.mid)

        counts.to_csv(
            EDA_DIR / f"event_competitiveness_{stat}_distribution.csv",
            index=False
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=counts,
            x="bucket_mid",
            y="event_count",
            hue="phase",
            marker="o",
            palette="Set2"
        )
        plt.title(f"Event Competitiveness ({stat.capitalize()} Ratio Distribution)")
        plt.xlabel(f"{stat.capitalize()} Ratio (Event / Year)")
        plt.ylabel("Number of Events")
        plt.axvline(1.0, color="black", linestyle="--")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"event_competitiveness_{stat}_distribution.png")
        plt.close()

    print("Event competitiveness analysis complete.")


# ============================================================
# Event Stability Analysis (Variance / CV / Autocorrelation)
# ============================================================

def compute_event_stability():
    print("\n=== Computing Event Stability ===")

    df = pd.read_csv(EDA_DIR / "event_competitiveness_full.csv")

    # Use base_event_key (added above)
    if "base_event_key" not in df.columns:
        df["base_event_key"] = df["event_key"].str[4:]

    stability_rows = []

    for phase in df["phase"].unique():
        pdf = df[df["phase"] == phase]

        # Group by base_event_key (cross-year event identity)
        for event, g in pdf.groupby("base_event_key"):

            # Need at least 3 years to measure stability
            if g["year"].nunique() < 3:
                continue

            g = g.sort_values("year")

            def safe_var(x): return float(np.nanvar(x))
            def safe_std(x): return float(np.nanstd(x))
            def safe_cv(x):
                mean = np.nanmean(x)
                std = np.nanstd(x)
                return float(std / mean) if mean != 0 else np.nan
            def lag1(x):
                x = pd.Series(x).dropna()
                return x.autocorr(lag=1) if len(x) >= 3 else np.nan

            metrics = {
                "base_event_key": event,
                "phase": phase,
                "years_count": g["year"].nunique(),

                "mean_ratio_var": safe_var(g["mean_ratio"]),
                "mean_ratio_std": safe_std(g["mean_ratio"]),
                "mean_ratio_cv": safe_cv(g["mean_ratio"]),
                "mean_ratio_autocorr": lag1(g["mean_ratio"]),

                "median_ratio_var": safe_var(g["median_ratio"]),
                "median_ratio_std": safe_std(g["median_ratio"]),
                "median_ratio_cv": safe_cv(g["median_ratio"]),
                "median_ratio_autocorr": lag1(g["median_ratio"]),

                "mode_ratio_var": safe_var(g["mode_ratio"]),
                "mode_ratio_std": safe_std(g["mode_ratio"]),
                "mode_ratio_cv": safe_cv(g["mode_ratio"]),
                "mode_ratio_autocorr": lag1(g["mode_ratio"]),
            }

            stability_rows.append(metrics)

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(EDA_DIR / "event_stability.csv", index=False)

    print("Saved event stability to event_stability.csv")
    return stability_df


# ============================================================
# Event Stability Line Plots
# ============================================================

def plot_event_stability_lines():
    print("\n=== Plotting Event Stability Over Time ===")

    df = pd.read_csv(EDA_DIR / "event_competitiveness_full.csv")

    if "base_event_key" not in df.columns:
        df["base_event_key"] = df["event_key"].str[4:]

    # Only keep events with 4+ years of history
    eligible = (
        df.groupby("base_event_key")["year"].nunique()
        .reset_index(name="years")
    )
    eligible = eligible[eligible["years"] >= 4]["base_event_key"].tolist()

    df = df[df["base_event_key"].isin(eligible)]

    for phase in ["auto", "teleop", "endgame", "totalMinusFouls"]:
        pdf = df[df["phase"] == phase]

        plt.figure(figsize=(10, 6))
        for event, g in pdf.groupby("base_event_key"):
            g = g.sort_values("year")
            plt.plot(g["year"], g["mean_ratio"], alpha=0.25)

        plt.title(f"Event Competitiveness Over Time — {phase.capitalize()}")
        plt.xlabel("Year")
        plt.ylabel("Mean Competitiveness Ratio")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"event_stability_{phase}.png")
        plt.close()

    print("Event stability line plots saved.")
    
# ============================================================
# Rank Events by Stability
# ============================================================

def rank_event_stability():
    """
    Load event_stability.csv, compute an overall stability score,
    rank events, and output a list of highly stable competitions.
    """
    print("\n=== Ranking Event Stability ===")

    path = EDA_DIR / "event_stability.csv"
    if not path.exists():
        print("event_stability.csv not found. Run run_event_stability_analysis() first.")
        return None

    df = pd.read_csv(path)

    # We compute a stability score for each event-phase pair.
    # Later we aggregate per event across phases.
    def normalize(series):
        s = series.copy()
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return 1 - s  # smaller var/std/CV = more stable → higher score

    # Normalize each metric so they all go 0..1 where 1 = best
    df["score_var"] = normalize(df["mean_ratio_var"])
    df["score_std"] = normalize(df["mean_ratio_std"])
    df["score_cv"] = normalize(df["mean_ratio_cv"])

    # Autocorrelation: positive = more stable
    ac = df["mean_ratio_autocorr"].fillna(0)
    df["score_autocorr"] = (ac - ac.min()) / (ac.max() - ac.min() + 1e-9)

    # Final stability score (equal-weight average)
    df["stability_score"] = (
        df["score_var"] +
        df["score_std"] +
        df["score_cv"] +
        df["score_autocorr"]
    ) / 4.0

    # Aggregate by event across all phases
    event_scores = (
        df.groupby("base_event_key")["stability_score"]
        .mean()
        .reset_index()
        .sort_values("stability_score", ascending=False)
    )

    # Output ranked list
    ranked_path = EDA_DIR / "event_stability_ranked.csv"
    event_scores.to_csv(ranked_path, index=False)
    print(f"Saved ranked stability list to {ranked_path}")

    # Define stable events: high score
    stable_events = event_scores[event_scores["stability_score"] > 0]

    stable_path = EDA_DIR / "stable_events.csv"
    stable_events.to_csv(stable_path, index=False)
    print(f"Saved stable event list to {stable_path}")

    return event_scores, stable_events

# ============================================================
# Rank Events by Competitiveness (Single CSV Only)
# ============================================================

def rank_event_competitiveness():
    """
    Computes an overall competitiveness score for each event across all years,
    based on mean/median/mode competitiveness ratios.
    Produces ONE ranked CSV:
        event_competitiveness_ranked.csv
    """
    print("\n=== Ranking Event Competitiveness ===")

    path = EDA_DIR / "event_competitiveness_full.csv"
    if not path.exists():
        print("event_competitiveness_full.csv not found. Run analyze_event_competitiveness() first.")
        return None

    df = pd.read_csv(path)

    # Ensure base_event_key exists
    if "base_event_key" not in df.columns:
        df["base_event_key"] = df["event_key"].str[4:]

    # --------------------------------------------------------
    # Aggregate competitiveness values across all years
    # --------------------------------------------------------
    agg = (
        df.groupby("base_event_key")[["mean_ratio", "median_ratio", "mode_ratio"]]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # Normalize each metric (0..1)
    # --------------------------------------------------------
    def normalize(series):
        s = series.copy()
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    agg["mean_score"] = normalize(agg["mean_ratio"])
    agg["median_score"] = normalize(agg["median_ratio"])
    agg["mode_score"] = normalize(agg["mode_ratio"])

    # --------------------------------------------------------
    # Final Composite Competitiveness Score
    # Weighted: mean 60%, median 30%, mode 10%
    # --------------------------------------------------------
    agg["competitiveness_score"] = (
        0.6 * agg["mean_score"] +
        0.3 * agg["median_score"] +
        0.1 * agg["mode_score"]
    )

    # --------------------------------------------------------
    # Sort events by competitiveness
    # --------------------------------------------------------
    ranked = agg.sort_values("competitiveness_score", ascending=False)

    # --------------------------------------------------------
    # Save one ranked CSV
    # --------------------------------------------------------
    out_path = EDA_DIR / "event_competitiveness_ranked.csv"
    ranked.to_csv(out_path, index=False)

    print(f"Saved ranked competitiveness list to {out_path}")

    return ranked




# ============================================================
# Wrapper
# ============================================================

def run_event_stability_analysis():
    compute_event_stability()
    plot_event_stability_lines()
    rank_event_stability()
    rank_event_competitiveness()
    print("\nEvent stability analysis complete.")


