# preprocessing.py

import pandas as pd
from pathlib import Path
from collections import deque

# Matches_processed columns: year,event_key,match_key,comp_level,match_number,winning_alliance,actual_time,red_teams,blue_teams,red_score,blue_score,red_autoPoints,blue_autoPoints,red_teleopPoints,blue_teleopPoints,red_foulPoints,blue_foulPoints,event_type,event_type_string,name,red_endgamePoints,blue_endgamePoints,red_auto_ratio_norm,blue_auto_ratio_norm,red_teleop_ratio_norm,blue_teleop_ratio_norm,red_endgame_ratio_norm,blue_endgame_ratio_norm,red_total_ratio_norm,blue_total_ratio_norm
# Rolling_Team_Stats columns: match_key,team_id,matches_played_before,avg_auto_before,avg_teleop_before,avg_endgame_before,avg_foul_before,avg_total_before,winrate_before,disp_auto,disp_teleop,disp_endgame,disp_foul,disp_total
# Teams columns: team_id,city,country,postal_code,rookie_year
# Events columns: address,city,country,district,division_keys,end_date,,event_type,event_type_string,first_event_code,first_event_id,gmaps_place_id,gmaps_url,key,lat,lng,location_name,name,parent_event_key,playoff_type,playoff_type_string,postal_code,remap_teams,short_name,start_date,state_prov,timezone,webcasts,website,week,year

# ============================================================
# TEAM-LEVEL FEATURE ENGINEERING
# ============================================================

def generate_rolling_team_stats(matches: pd.DataFrame, out_path: Path):
    """
    Compute rolling (per-match chronological) team statistics.
    Saves to CSV for fast reuse.
    """

    print("Generating rolling per-match team statistics...")

    # Sort matches chronologically
    matches = matches.sort_values("actual_time").reset_index(drop=True)

    # Dictionary to store cumulative stats per team
    team_hist = {}

    rows = []

    for idx, row in matches.iterrows():
        red_teams = row["red_teams"].split(",")
        blue_teams = row["blue_teams"].split(",")

        # compute match stats
        match_stats = {
            "red": {
                "auto": row["red_auto_ratio_norm"],
                "teleop": row["red_teleop_ratio_norm"],
                "endgame": row["red_endgame_ratio_norm"],
                "foul": row["blue_foulPoints"], # inverse foul points
                "total": row["red_total_ratio_norm"],
                "win": 1 if row["winning_alliance"] == "red" else 0,
            },
            "blue": {
                "auto": row["blue_auto_ratio_norm"],
                "teleop": row["blue_teleop_ratio_norm"],
                "endgame": row["blue_endgame_ratio_norm"],
                "foul": row["red_foulPoints"],
                "total": row["blue_total_ratio_norm"],
                "win": 1 if row["winning_alliance"] == "blue" else 0,
            },
        }
        
        # sanitize NaN / inf in match_stats so rolling sums stay numeric
        for alliance in ["red", "blue"]:
            for key in ["auto", "teleop", "endgame", "foul", "total"]:
                val = match_stats[alliance][key]
                if pd.isna(val) or val == float("inf") or val == -float("inf"):
                    match_stats[alliance][key] = 0.0



        # update rolling stats for each team
        for alliance, team_list in [("red", red_teams), ("blue", blue_teams)]:
            for team in team_list:

                if team not in team_hist:
                    team_hist[team] = {
                        "cnt": 0,
                        "sum_auto": 0,
                        "sum_teleop": 0,
                        "sum_end": 0,
                        "sum_foul": 0,
                        "sum_total": 0,
                        "wins": 0,
                        
                        "last5_auto": deque(maxlen=5),
                        "last5_teleop": deque(maxlen=5),
                        "last5_end": deque(maxlen=5),
                        "last5_foul": deque(maxlen=5),
                        "last5_total": deque(maxlen=5),
                        "last5_win": deque(maxlen=5),
                    }

                # pre-match stats (this is what matters!)
                hist = team_hist[team]

                # Pre-match rolling stats
                avg_auto    = hist["sum_auto"] / hist["cnt"] if hist["cnt"] > 0 else 0
                avg_teleop  = hist["sum_teleop"] / hist["cnt"] if hist["cnt"] > 0 else 0
                avg_end     = hist["sum_end"] / hist["cnt"] if hist["cnt"] > 0 else 0
                avg_foul    = hist["sum_foul"] / hist["cnt"] if hist["cnt"] > 0 else 0
                avg_total   = hist["sum_total"] / hist["cnt"] if hist["cnt"] > 0 else 0
                winrate     = hist["wins"] / hist["cnt"] if hist["cnt"] > 0 else 0

                # Compute disparity (actual - expected)
                disp_auto   = match_stats[alliance]["auto"]    - avg_auto
                disp_teleop = match_stats[alliance]["teleop"]  - avg_teleop
                disp_end    = match_stats[alliance]["endgame"] - avg_end
                disp_foul   = match_stats[alliance]["foul"]    - avg_foul
                disp_total  = match_stats[alliance]["total"]   - avg_total
                
                if len(hist["last5_auto"]) > 0:
                    avg_auto_5    = sum(hist["last5_auto"]) / len(hist["last5_auto"])
                    avg_teleop_5  = sum(hist["last5_teleop"]) / len(hist["last5_teleop"])
                    avg_end_5     = sum(hist["last5_end"]) / len(hist["last5_end"])
                    avg_foul_5    = sum(hist["last5_foul"]) / len(hist["last5_foul"])
                    avg_total_5   = sum(hist["last5_total"]) / len(hist["last5_total"])
                    winrate_5     = sum(hist["last5_win"]) / len(hist["last5_win"])
                else:
                    avg_auto_5 = avg_teleop_5 = avg_end_5 = avg_foul_5 = avg_total_5 = winrate_5 = 0.0

                rows.append({
                    "match_key": row["match_key"],
                    "team_id": team,
                    "matches_played_before": hist["cnt"],

                    # Rolling averages
                    "avg_auto_before": avg_auto,
                    "avg_teleop_before": avg_teleop,
                    "avg_endgame_before": avg_end,
                    "avg_foul_before": avg_foul,
                    "avg_total_before": avg_total,
                    "winrate_before": winrate,

                    # Disparity features
                    "disp_auto": disp_auto,
                    "disp_teleop": disp_teleop,
                    "disp_endgame": disp_end,
                    "disp_foul": disp_foul,
                    "disp_total": disp_total,
                    
                    # Last 5 matches averages
                    "avg_auto_before_5": avg_auto_5,
                    "avg_teleop_before_5": avg_teleop_5,
                    "avg_endgame_before_5": avg_end_5,
                    "avg_foul_before_5": avg_foul_5,
                    "avg_total_before_5": avg_total_5,
                    "winrate_before_5": winrate_5,
                })


                # update AFTER collecting stats
                hist["cnt"] += 1
                hist["sum_auto"] += match_stats[alliance]["auto"]
                hist["sum_teleop"] += match_stats[alliance]["teleop"]
                hist["sum_end"] += match_stats[alliance]["endgame"]
                hist["sum_foul"] += match_stats[alliance]["foul"]
                hist["sum_total"] += match_stats[alliance]["total"]
                hist["wins"] += match_stats[alliance]["win"]
                hist["last5_auto"].append(match_stats[alliance]["auto"])
                hist["last5_teleop"].append(match_stats[alliance]["teleop"])
                hist["last5_end"].append(match_stats[alliance]["endgame"])
                hist["last5_foul"].append(match_stats[alliance]["foul"])
                hist["last5_total"].append(match_stats[alliance]["total"])
                hist["last5_win"].append(match_stats[alliance]["win"])

    # Save rolling stats
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved rolling stats to {out_path}")
    return df


def compute_team_stats(matches: pd.DataFrame):
    """
    Compute per-team averages and winrates based on the cleaned match dataset.
    Returns a DataFrame indexed by team_id with:
        avg_auto, avg_teleop, avg_endgame,
        avg_foul, avg_total, win_rate, matches_played
    """

    print("Computing per-team statistics...")

    # Only keep real matches
    m = matches[matches["winning_alliance"].isin(["red", "blue"])].copy()

    team_rows = []

    for _, row in m.iterrows():
        red_teams = row["red_teams"].split(",")
        blue_teams = row["blue_teams"].split(",")

        # Extract scoring values
        red_auto = row["red_auto_ratio_norm"]
        blue_auto = row["blue_auto_ratio_norm"]

        red_tele = row["red_teleop_ratio_norm"]
        blue_tele = row["blue_teleop_ratio_norm"]

        red_end = row["red_endgame_ratio_norm"]
        blue_end = row["blue_endgame_ratio_norm"]

        red_foul = row["red_foulPoints"]
        blue_foul = row["blue_foulPoints"]

        red_total = row["red_score"]
        blue_total = row["blue_score"]

        winner = row["winning_alliance"]

        # Add RED teams
        for t in red_teams:
            team_rows.append({
                "team_id": t.strip(),
                "auto": red_auto,
                "teleop": red_tele,
                "endgame": red_end,
                "foul": red_foul,
                "total": red_total,
                "win": 1 if winner == "red" else 0
            })

        # Add BLUE teams
        for t in blue_teams:
            team_rows.append({
                "team_id": t.strip(),
                "auto": blue_auto,
                "teleop": blue_tele,
                "endgame": blue_end,
                "foul": blue_foul,
                "total": blue_total,
                "win": 1 if winner == "blue" else 0
            })

    df = pd.DataFrame(team_rows)

    # Aggregate per-team averages
    team_stats = (
        df.groupby("team_id")
        .agg({
            "auto": "mean",
            "teleop": "mean",
            "endgame": "mean",
            "foul": "mean",
            "total": "mean",
            "win": ["sum", "count"]
        })
    )

    team_stats.columns = [
        "avg_auto", "avg_teleop", "avg_endgame",
        "avg_foul", "avg_total", "wins", "matches_played"
    ]

    team_stats["win_rate"] = team_stats["wins"] / team_stats["matches_played"]

    return team_stats.reset_index()



# ============================================================
# MATCH-LEVEL FEATURE ENGINEERING
# ============================================================

def attach_event_weeks(matches: pd.DataFrame, data_dir=Path("data")):
    """
    Attach event 'week' from events.csv onto matches.
    """

    events = pd.read_csv(data_dir / "events.csv", low_memory=False)

    # Normalize event key naming
    if "key" in events.columns:
        events = events.rename(columns={"key": "event_key"})

    # Keep only needed columns
    if "week" not in events.columns:
        raise ValueError("events.csv does not contain a 'week' column.")

    events = events[["event_key", "week"]].drop_duplicates()

    # Merge onto matches
    merged = matches.merge(events, on="event_key", how="left")

    # Fill missing weeks
    merged["week"] = merged["week"].fillna(-1).astype(int)



    # 1) Regular competition weeks shift up by +1
    #    TBA 0–5 -> Week 1–6
    normal_mask = (merged["week"] >= 0) & (merged["week"] <= 5)
    merged.loc[normal_mask, "week"] = merged.loc[normal_mask, "week"] + 1
    
    # 2) Special "week0" event -> week 0
    special_mask = (merged["week"] == -1) & merged["event_key"].str.contains("week0")
    merged.loc[special_mask, "week"] = 0

    # 3) Championship events -> week 7
    champs_mask = (merged["week"] == -1) & ~special_mask
    merged.loc[champs_mask, "week"] = 7

    return merged



def attach_team_features(matches: pd.DataFrame, team_stats: pd.DataFrame):
    """
    For each match:
        - look up stats for each team
        - compute aggregated Red features
        - compute aggregated Blue features

    Creates ML-ready per-match features.
    """

    print("Merging team features into match dataset...")

    stats = team_stats.set_index("team_id")

    feature_rows = []

    for _, row in matches.iterrows():
        red_list = row["red_teams"].split(",")
        blue_list = row["blue_teams"].split(",")

        # Pull team stats safely (existing)
        def safe_lookup_agg(team_list):
            existing = [t for t in team_list if t in stats.index]
            if len(existing) == 0:
                # return zeros for all aggregated features
                return pd.DataFrame([{
                # Long-term
                "avg_auto": 0, "avg_teleop": 0, "avg_endgame": 0,
                "avg_foul": 0, "avg_total": 0, "win_rate": 0,

                # Short-term (last 5)
                "avg_auto_5": 0, "avg_teleop_5": 0, "avg_endgame_5": 0,
                "avg_foul_5": 0, "avg_total_5": 0, "win_rate_5": 0,

                # Disparity
                "disp_auto": 0, "disp_teleop": 0, "disp_endgame": 0,
                "disp_foul": 0, "disp_total": 0
            }])
            return stats.loc[existing]

        # Raw per-team stats
        red_df = safe_lookup_agg(red_list)
        blue_df = safe_lookup_agg(blue_list)

        # Aggregated means
        red_stats = red_df.mean()
        blue_stats = blue_df.mean()

        def agg_synergy(df):
            result = {}

            phases = [
                # long-term rolling
                "avg_auto", "avg_teleop", "avg_endgame",
                "avg_foul", "avg_total", "win_rate",

                # last 5 matches
                "avg_auto_5", "avg_teleop_5", "avg_endgame_5",
                "avg_foul_5", "avg_total_5", "win_rate_5",

                # disparities
                "disp_auto", "disp_teleop", "disp_endgame",
                "disp_foul", "disp_total"
            ]

            for p in phases:
                vals = df[p].values

                result[f"{p}_max"] = vals.max()
                result[f"{p}_min"] = vals.min()
                result[f"{p}_std"] = vals.std()

                # top2 mean
                top2 = sorted(vals, reverse=True)[:2]
                result[f"{p}_top2"] = sum(top2) / len(top2)

            return result

        # Compute synergy stats for red and blue
        red_syn = agg_synergy(red_df)
        blue_syn = agg_synergy(blue_df)

        
        comp = row["comp_level"]
        if comp == "qf":
            comp = "sf"

        # One-hot encode match type
        is_qm = 1 if comp == "qm" else 0
        is_sf = 1 if comp == "sf" else 0
        is_f  = 1 if comp == "f"  else 0
        
        feature_rows.append({
            "match_key": row["match_key"],
            "event_key": row["event_key"],
            "comp_level": row["comp_level"],
            "year": row["year"],
            "year_rescaled": row["year"] - 2016,
            "week": row["week"],
            
            # One-hot match type
            "is_qm": is_qm,
            "is_sf": is_sf,
            "is_f": is_f,  

            # Red aggregated
            "red_avg_auto": red_stats["avg_auto"],
            "red_avg_teleop": red_stats["avg_teleop"],
            "red_avg_endgame": red_stats["avg_endgame"],
            "red_avg_foul": red_stats["avg_foul"],
            "red_avg_total": red_stats["avg_total"],
            "red_avg_winrate": red_stats["win_rate"],
            
            "red_max_auto": red_syn["avg_auto_max"],
            "red_min_auto": red_syn["avg_auto_min"],
            "red_std_auto": red_syn["avg_auto_std"],
            "red_top2_auto": red_syn["avg_auto_top2"],

            "red_max_teleop": red_syn["avg_teleop_max"],
            "red_min_teleop": red_syn["avg_teleop_min"],
            "red_std_teleop": red_syn["avg_teleop_std"],
            "red_top2_teleop": red_syn["avg_teleop_top2"],

            "red_max_endgame": red_syn["avg_endgame_max"],
            "red_min_endgame": red_syn["avg_endgame_min"],
            "red_std_endgame": red_syn["avg_endgame_std"],
            "red_top2_endgame": red_syn["avg_endgame_top2"],

            "red_max_foul": red_syn["avg_foul_max"],
            "red_min_foul": red_syn["avg_foul_min"],
            "red_std_foul": red_syn["avg_foul_std"],
            "red_top2_foul": red_syn["avg_foul_top2"],

            "red_max_total": red_syn["avg_total_max"],
            "red_min_total": red_syn["avg_total_min"],
            "red_std_total": red_syn["avg_total_std"],
            "red_top2_total": red_syn["avg_total_top2"],

            # Blue aggregated
            "blue_avg_auto": blue_stats["avg_auto"],
            "blue_avg_teleop": blue_stats["avg_teleop"],
            "blue_avg_endgame": blue_stats["avg_endgame"],
            "blue_avg_foul": blue_stats["avg_foul"],
            "blue_avg_total": blue_stats["avg_total"],
            "blue_avg_winrate": blue_stats["win_rate"],
            
            "blue_max_auto": blue_syn["avg_auto_max"],
            "blue_min_auto": blue_syn["avg_auto_min"],
            "blue_std_auto": blue_syn["avg_auto_std"],
            "blue_top2_auto": blue_syn["avg_auto_top2"],

            "blue_max_teleop": blue_syn["avg_teleop_max"],
            "blue_min_teleop": blue_syn["avg_teleop_min"],
            "blue_std_teleop": blue_syn["avg_teleop_std"],
            "blue_top2_teleop": blue_syn["avg_teleop_top2"],

            "blue_max_endgame": blue_syn["avg_endgame_max"],
            "blue_min_endgame": blue_syn["avg_endgame_min"],
            "blue_std_endgame": blue_syn["avg_endgame_std"],
            "blue_top2_endgame": blue_syn["avg_endgame_top2"],

            "blue_max_foul": blue_syn["avg_foul_max"],
            "blue_min_foul": blue_syn["avg_foul_min"],
            "blue_std_foul": blue_syn["avg_foul_std"],
            "blue_top2_foul": blue_syn["avg_foul_top2"],

            "blue_max_total": blue_syn["avg_total_max"],
            "blue_min_total": blue_syn["avg_total_min"],
            "blue_std_total": blue_syn["avg_total_std"],
            "blue_top2_total": blue_syn["avg_total_top2"],
            
            "red_disp_auto": red_stats["disp_auto"],
            "blue_disp_auto": blue_stats["disp_auto"],

            "red_disp_teleop": red_stats["disp_teleop"],
            "blue_disp_teleop": blue_stats["disp_teleop"],

            "red_disp_endgame": red_stats["disp_endgame"],
            "blue_disp_endgame": blue_stats["disp_endgame"],

            "red_disp_foul": red_stats["disp_foul"],
            "blue_disp_foul": blue_stats["disp_foul"],

            "red_disp_total": red_stats["disp_total"],
            "blue_disp_total": blue_stats["disp_total"],
            
            # Last-5 Rolling Averages
            "red_avg_auto_5": red_stats["avg_auto_5"],
            "red_avg_teleop_5": red_stats["avg_teleop_5"],
            "red_avg_endgame_5": red_stats["avg_endgame_5"],
            "red_avg_foul_5": red_stats["avg_foul_5"],
            "red_avg_total_5": red_stats["avg_total_5"],
            "red_avg_winrate_5": red_stats["win_rate_5"],
            
            "blue_avg_auto_5": blue_stats["avg_auto_5"],
            "blue_avg_teleop_5": blue_stats["avg_teleop_5"],
            "blue_avg_endgame_5": blue_stats["avg_endgame_5"],
            "blue_avg_foul_5": blue_stats["avg_foul_5"],
            "blue_avg_total_5": blue_stats["avg_total_5"],
            "blue_avg_winrate_5": blue_stats["win_rate_5"],


            # Target variable
            "red_win": 1 if row["winning_alliance"] == "red" else 0
        })

    return pd.DataFrame(feature_rows)

# ============================================================
# DATA SPLITTING
# ============================================================
    
def temporal_split_by_year_week(matches, year, week):
    """
    Train = all matches BEFORE the first match of (year, week)
    Test  = all matches ON or AFTER that time.
    """

    # Find earliest timestamp in the target year+week
    mask = (matches["year"] == year) & (matches["week"] == week)

    if not mask.any():
        raise ValueError(f"No matches found for year={year}, week={week}")

    cutoff_time = matches.loc[mask, "actual_time"].min()

    if pd.isna(cutoff_time):
        raise ValueError(f"actual_time missing for year={year}, week={week}")

    # Split
    train_matches = matches[matches["actual_time"] < cutoff_time].copy()
    test_matches  = matches[matches["actual_time"] >= cutoff_time].copy()

    print(f"Temporal Split -> Train: {len(train_matches)}, Test: {len(test_matches)}")

    return train_matches, test_matches




# ============================================================
# MAIN PUBLIC API — PREPARE FEATURES FOR ML
# ============================================================

def prepare_base_matches(matches: pd.DataFrame):
    """
    Clean preprocessing step (NO feature engineering):
        - attaches event weeks
        - ensures matches are ready for temporal splitting
    """

    matches = attach_event_weeks(matches)
    return matches

def load_and_truncate_team_stats(rolling_stats, train_matches, test_matches):
    """
    Keep only rolling stats for matches BEFORE split cutoff.
    Ensures no leakage from test period.
    """

    cutoff = test_matches["actual_time"].min()

    # Only keep rolling rows where the match is part of the train set
    train_keys = set(train_matches["match_key"].unique())

    truncated = rolling_stats[rolling_stats["match_key"].isin(train_keys)].copy()

    return truncated


def aggregate_team_stats_from_rolling(rolling_stats_train: pd.DataFrame):
    """
    Aggregates rolling stats *up to the temporal cutoff*.
    Produces per-team averages that simulate the knowledge available
    at the time the model is trained.
    """

    agg = (
        rolling_stats_train.groupby("team_id")
        .agg({
            "matches_played_before": "max",

            # Rolling averages
            "avg_auto_before": "mean",
            "avg_teleop_before": "mean",
            "avg_endgame_before": "mean",
            "avg_foul_before": "mean",
            "avg_total_before": "mean",
            "winrate_before": "mean",

            # Disparity means
            "disp_auto": "mean",
            "disp_teleop": "mean",
            "disp_endgame": "mean",
            "disp_foul": "mean",
            "disp_total": "mean",
            
            # Last 5 matches averages
            "avg_auto_before_5": "mean",
            "avg_teleop_before_5": "mean",
            "avg_endgame_before_5": "mean",
            "avg_foul_before_5": "mean",
            "avg_total_before_5": "mean",
            "winrate_before_5": "mean",
        })
        .reset_index()
    )

    # Rename rolling columns
    agg = agg.rename(columns={
        "avg_auto_before": "avg_auto",
        "avg_teleop_before": "avg_teleop",
        "avg_endgame_before": "avg_endgame",
        "avg_foul_before": "avg_foul",
        "avg_total_before": "avg_total",
        "winrate_before": "win_rate",
        "matches_played_before": "matches_played",
        
        "avg_auto_before_5": "avg_auto_5",
        "avg_teleop_before_5": "avg_teleop_5",
        "avg_endgame_before_5": "avg_endgame_5",
        "avg_foul_before_5": "avg_foul_5",
        "avg_total_before_5": "avg_total_5",
        "winrate_before_5": "win_rate_5",
    })
    
    # Final safety: no NaNs in team-level stats
    agg = agg.fillna(0.0)

    return agg

def precompute_all_features(cleaned_matches, rolling_stats, out_path):
    """
    Precompute ALL match-level features (team features + synergy + aggregates)
    exactly once. Saves a fully-featured table keyed by match_key.

    This eliminates the attach_team_features bottleneck during rolling evaluation.
    """

    print("=== Precomputing ALL match features (one-time heavy pass) ===")

    # Aggregate rolling stats for ALL matches (as if everything is train)
    print("Aggregating all-team rolling stats...")
    team_stats_full = aggregate_team_stats_from_rolling(rolling_stats)

    print("Attaching team features to ALL matches...")
    all_features = attach_team_features(cleaned_matches.copy(), team_stats_full)

    print(f"Saving precomputed ML features to: {out_path}")
    all_features.to_csv(out_path, index=False)

    return all_features





def build_features(train_matches, test_matches, precomputed):
    """
    Fast build_features using precomputed match-level features.
    No recomputation, just lookups by match_key, and selecting the
    exact feature set we decided to keep.
    """

    # Restrict to the rows we need
    train_feat = precomputed[precomputed["match_key"].isin(train_matches["match_key"])]
    test_feat  = precomputed[precomputed["match_key"].isin(test_matches["match_key"])]

    feature_cols = [
        # Match Info
        # "year",
        # "week",
        "is_qm",
        "is_sf",
        "is_f",
        
        # Team Stats
        "red_avg_auto", "blue_avg_auto",
        "red_avg_teleop", "blue_avg_teleop",
        # "red_avg_endgame", "blue_avg_endgame",
        # "red_avg_foul", "blue_avg_foul",
        "red_avg_total", "blue_avg_total",
        
        "red_max_auto", "red_top2_auto",  "red_min_auto", "red_std_auto",
        "red_max_teleop", "red_top2_teleop",  "red_min_teleop", "red_std_teleop", 
        # "red_max_endgame", "red_top2_endgame", "red_min_endgame", "red_std_endgame", 
        # "red_max_foul", "red_min_foul", "red_std_foul", "red_top2_foul",
        "red_max_total", "red_top2_total", "red_min_total", "red_std_total", 
        
        "blue_max_auto", "blue_top2_auto",  "blue_min_auto", "blue_std_auto", 
        "blue_max_teleop", "blue_top2_teleop",  "blue_min_teleop", "blue_std_teleop", 
        # "blue_max_endgame", "blue_min_endgame", "blue_std_endgame", "blue_top2_endgame",
        # "blue_max_foul", "blue_min_foul", "blue_std_foul", "blue_top2_foul",
        "blue_max_total", "blue_top2_total", "blue_min_total", "blue_std_total", 
        
        # Disparity
        "red_disp_auto", "blue_disp_auto",
        "red_disp_teleop", "blue_disp_teleop",
        # "red_disp_endgame", "blue_disp_endgame",
        # "red_disp_foul", "blue_disp_foul",
        "red_disp_total", "blue_disp_total",
        
        # Last 5 Matches
        "red_avg_auto_5", "blue_avg_auto_5",
        "red_avg_teleop_5", "blue_avg_teleop_5",
        # "red_avg_endgame_5", "blue_avg_endgame_5",
        # "red_avg_foul_5", "blue_avg_foul_5",
        "red_avg_total_5", "blue_avg_total_5",
        "red_avg_winrate_5", "blue_avg_winrate_5",
        
        # Winrate
        "red_avg_winrate", "blue_avg_winrate",
    ]

    X_train = train_feat[feature_cols]
    y_train = train_feat["red_win"].astype(int)

    X_test  = test_feat[feature_cols]
    y_test  = test_feat["red_win"].astype(int)

    return X_train, y_train, X_test, y_test, train_feat, test_feat



