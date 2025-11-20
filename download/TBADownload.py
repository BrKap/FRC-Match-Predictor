import os
import time
import csv
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------
# CONFIGURATION
# ----------------------------
load_dotenv()
TBA_API_KEY = os.getenv("TBA_API_KEY")
BASE_URL = "https://www.thebluealliance.com/api/v3"
HEADERS = {"X-TBA-Auth-Key": TBA_API_KEY}
START_YEAR = 1992
END_YEAR = 2025  # adjust as needed
SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True)
RATE_LIMIT_DELAY = 0.5  # seconds

MATCHES_CSV = SAVE_DIR / "matches.csv"
TEAMS_CSV = SAVE_DIR / "teams.csv"

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def fetch(endpoint):
    """Fetch JSON from a TBA API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {url}")
        return None

def extract_match(m, year):
    """Extract standardized match fields."""
    alliances = m.get("alliances", {})
    blue = alliances.get("blue", {})
    red = alliances.get("red", {})
    score_breakdown = m.get("score_breakdown")
    if not isinstance(score_breakdown, dict):
        score_breakdown = {}

    blue_break = score_breakdown.get("blue", {}) or {}
    red_break = score_breakdown.get("red", {}) or {}

    def ts_to_dt(ts):
        if ts is None:
            return None
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return None

    return {
        "year": year,
        "event_key": m.get("event_key"),
        "match_key": m.get("key"),
        "comp_level": m.get("comp_level"),
        "match_number": m.get("match_number"),
        "winning_alliance": m.get("winning_alliance"),
        "actual_time": ts_to_dt(m.get("actual_time")),
        "red_teams": ",".join(red.get("team_keys", [])),
        "blue_teams": ",".join(blue.get("team_keys", [])),
        "red_score": red.get("score"),
        "blue_score": blue.get("score"),
        "red_autoPoints": red_break.get("autoPoints"),
        "blue_autoPoints": blue_break.get("autoPoints"),
        "red_teleopPoints": red_break.get("teleopPoints"),
        "blue_teleopPoints": blue_break.get("teleopPoints"),
        "red_foulPoints": red_break.get("foulPoints"),
        "blue_foulPoints": blue_break.get("foulPoints"),
    }

def extract_team(t):
    """Extract relevant team fields."""
    return {
        "team_id": t.get("key"),
        "city": t.get("city"),
        "country": t.get("country"),
        "postal_code": t.get("postal_code"),
        "rookie_year": t.get("rookie_year"),
    }
    
def download_all_events(session, headers, start_year, end_year):
    """Download all events across all years and save a single events.csv file."""
    print("\n=== Downloading ALL event metadata ===")

    all_events = []

    for year in range(start_year, end_year + 1):
        print(f"Downloading events for {year}...")
        url = f"{BASE_URL}/events/{year}"
        resp = session.get(url, headers=headers)

        if resp.status_code != 200:
            print(f"  Failed to download events for {year} ({resp.status_code})")
            continue

        events = resp.json()
        print(f"  Found {len(events)} events in {year}")

        # Attach year explicitly
        for e in events:
            e["year"] = year
            all_events.append(e)

        time.sleep(RATE_LIMIT_DELAY)

    # Save to events.csv
    events_df = pd.DataFrame(all_events)
    out_path = SAVE_DIR / "events.csv"
    events_df.to_csv(out_path, index=False)

    print(f"\nSaved ALL events to: {out_path}")
    print(f"Total events stored: {len(events_df)}\n")

    return events_df


def append_to_csv(path, records, fieldnames):
    """Append records to CSV (create if not exists)."""
    if not records:
        return
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(records)

# ----------------------------
# MAIN LOGIC
# ----------------------------
def main():
    all_matches = []
    all_teams = {}
    
    print("\nCollecting event metadata across all years...")
    session = requests.Session()
    download_all_events(session, HEADERS, START_YEAR, END_YEAR)


    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\nDownloading matches for {year} ...")
        events = fetch(f"/events/{year}")
        if not events:
            print(f"No events for {year}")
            continue

        for event in events:
            event_key = event["key"]
            matches = fetch(f"/event/{event_key}/matches")
            if not matches:
                continue

            for m in matches:
                flat = extract_match(m, year)
                all_matches.append(flat)

            time.sleep(RATE_LIMIT_DELAY)

    # ----------------------------
    # Write matches to CSV
    # ----------------------------
    if all_matches:
        print(f"\nSaving {len(all_matches)} matches to {MATCHES_CSV}")
        matches_df = pd.DataFrame(all_matches)
        matches_df.to_csv(MATCHES_CSV, index=False)

    # ----------------------------
    # Download all teams
    # ----------------------------
    print("\nDownloading all teams...")
    page = 0
    while True:
        teams = fetch(f"/teams/{page}")
        if not teams:
            break
        for t in teams:
            tid = t.get("key")
            if tid not in all_teams:
                all_teams[tid] = extract_team(t)
        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    # ----------------------------
    # Write teams to CSV
    # ----------------------------
    print(f"Saving {len(all_teams)} teams to {TEAMS_CSV}")
    teams_df = pd.DataFrame(all_teams.values())
    teams_df.to_csv(TEAMS_CSV, index=False)

    print("\nAll data collection complete!")

# ----------------------------
if __name__ == "__main__":
    main()
