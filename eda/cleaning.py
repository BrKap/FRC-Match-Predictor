from pathlib import Path
import pandas as pd
import seaborn as sns
import re
import numpy as np
DATA_DIR = Path("data")
EDA_DIR = Path("eda") / "outputs"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ============================================================
# Data Loading and Summaries
# ============================================================
def load_data():
    """Load matches and teams, print summaries."""
    matches = pd.read_csv(DATA_DIR / "matches.csv", low_memory=False) # year,event_key,match_key,comp_level,match_number,winning_alliance,actual_time,red_teams,blue_teams,red_score,blue_score,red_autoPoints,blue_autoPoints,red_teleopPoints,blue_teleopPoints,red_foulPoints,blue_foulPoints
    teams = pd.read_csv(DATA_DIR / "teams.csv", low_memory=False) # team_id,city,country,postal_code,rookie_year

    print(f"Matches: {len(matches):,} rows, {matches.shape[1]} cols")
    print(f"Teams:   {len(teams):,} rows, {teams.shape[1]} cols")

    summary = {
        "matches_rows": len(matches),
        "matches_columns": matches.shape[1],
        "teams_rows": len(teams),
        "teams_columns": teams.shape[1],
        "years_range": f"{matches['year'].min()}–{matches['year'].max()}",
        "unique_teams": teams['team_id'].nunique(),
        "unique_events": matches['event_key'].nunique(),
    }
    pd.DataFrame([summary]).to_csv(EDA_DIR / "eda_summary.csv", index=False)
    return matches, teams

# ============================================================
# Filtering and cleaning match dataset
# ============================================================
def filter_matches(matches):
    """
    Clean the raw matches table:
    - remove years < 2016 and year == 2021
    - merge event metadata from events.csv
    - remove offseason events using event_type
    - keep ability to manually ban events
    - remove malformed/missing team keys
    - remove matches missing timestamps
    - remove matches missing scoring data
    """

    print("\n=== Performing additional match-level cleaning... ===")

    m = matches.copy()

    # ------------------------------------------------------------
    # 0. Load the event metadata from events.csv
    # ------------------------------------------------------------
    events_path = DATA_DIR / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError("events.csv not found. You must run TBADownload.py first.")

    events = pd.read_csv(events_path, low_memory=False)

    # Ensure key columns are clean
    events = events.rename(columns={"key": "event_key"})

    # We only need year + key + event_type
    events_meta = events[["event_key", "event_type", "event_type_string", "name"]]

    print(f"Merging event metadata into matches ({len(events_meta)} events)...")

    # Merge events.csv → matches
    m = m.merge(events_meta, on="event_key", how="left")

    # ------------------------------------------------------------
    # 1. Keep official in-season years only
    # ------------------------------------------------------------
    m = m[(m["year"] >= 2016) & (m["year"] != 2021)]

    # ------------------------------------------------------------
    # 2. Remove OFFSEASON EVENTS using event_type
    # ------------------------------------------------------------
    # TBA definitions:
    #   6  = Offseason
    #   99 = Offseason (legacy flag)
    OFFSEASON_TYPES = {6, 99}

    before = len(m)
    m = m[~m["event_type"].isin(OFFSEASON_TYPES)]
    after = len(m)

    print(f"Removed {before - after} matches from official offseason events (event_type=6 or 99).")

    # ------------------------------------------------------------
    # 3. Remove malformed team keys (must match frc####)
    # ------------------------------------------------------------
    def valid_team_list(tlist):
        if not isinstance(tlist, str):
            return False
        parts = tlist.split(",")
        return all(re.fullmatch(r"frc\d{1,5}", x.strip()) for x in parts)

    m = m[m["red_teams"].apply(valid_team_list)]
    m = m[m["blue_teams"].apply(valid_team_list)]

    # ------------------------------------------------------------
    # 4. Remove matches missing actual_time
    # ------------------------------------------------------------
    m = m[m["actual_time"].notna()]

    # ------------------------------------------------------------
    # 5. Remove matches missing ALL key scoring fields
    # ------------------------------------------------------------
    score_cols = [
        "red_autoPoints", "blue_autoPoints",
        "red_teleopPoints", "blue_teleopPoints",
        "red_foulPoints", "blue_foulPoints"
    ]
    m = m.dropna(subset=score_cols, how="all")
    
    # ------------------------------------------------------------
    # 6. Save cleaned matches to matches_cleaned.csv
    # ------------------------------------------------------------
    cleaned_path = DATA_DIR / "matches_cleaned.csv"
    m.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned matches to: {cleaned_path}")


    print(f"Filtered matches remaining: {len(m):,}")
    return m



# ============================================================
# Phase Score Normalization
# ============================================================
def normalize_phase_scores(matches):
    """
    Normalize Auto, Teleop, and Endgame points per year
    relative to the observed maximum (95th percentile) in that year.
    This makes comparisons fair across different FRC games.
    """
    print("\n=== Normalizing Phase Scores by Year ===")
    matches = matches.copy()

    # Derive estimated Endgame points first
    matches["red_endgamePoints"] = (
        matches["red_score"]
        - (matches["red_autoPoints"] + matches["red_teleopPoints"] + matches["red_foulPoints"])
    )
    matches["blue_endgamePoints"] = (
        matches["blue_score"]
        - (matches["blue_autoPoints"] + matches["blue_teleopPoints"] + matches["blue_foulPoints"])
    )

    phases = {
        "auto": ("red_autoPoints", "blue_autoPoints"),
        "teleop": ("red_teleopPoints", "blue_teleopPoints"),
        "endgame": ("red_endgamePoints", "blue_endgamePoints"),
        "total": ("red_score", "blue_score"),
    }

    for phase, (red_col, blue_col) in phases.items():

        if phase == "endgame":
            # Endgame: use MAX per year (100%) because distribution is weird
            max_per_year = (
                matches.groupby("year")[[red_col, blue_col]]
                .max()
                .max(axis=1)
                .rename("max_val")
            )
        else:
            # Auto, teleop, total: use 95th percentile
            max_per_year = (
                matches.groupby("year")[[red_col, blue_col]]
                .quantile(0.95)
                .max(axis=1)
                .rename("max_val")
            )

        # Merge per-year max into matches
        matches = matches.merge(max_per_year, on="year", how="left")

        # Avoid division by zero
        matches["max_val"] = matches["max_val"].replace(0, 1)

        # Compute ratios
        red_ratio = matches[red_col] / matches["max_val"]
        blue_ratio = matches[blue_col] / matches["max_val"]

        # Clean up infinities and NaNs -> treat as 0
        red_ratio = red_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        blue_ratio = blue_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        matches[f"red_{phase}_ratio_norm"] = red_ratio
        matches[f"blue_{phase}_ratio_norm"] = blue_ratio

        matches.drop(columns=["max_val"], inplace=True)


    print("Phase normalization complete.")
    return matches