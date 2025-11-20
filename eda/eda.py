"""
eda.py
EDA main for FRC Match Predictor
"""

from pathlib import Path
import seaborn as sns
from cleaning import load_data, filter_matches, normalize_phase_scores
from missingness import plot_missingness
from winrate import plot_basic_distributions, plot_team_winrates, plot_yearly_winrate_trends
from score import plot_auto_analysis, plot_teleop_analysis, plot_endgame_analysis
from competitiveness import analyze_event_competitiveness, run_event_stability_analysis
# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path("data")
EDA_DIR = Path("eda") / "outputs"
EDA_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# ============================================================
# Main
# ============================================================
def main():
    matches, teams = load_data()
    # plot_missingness(matches)
    filtered = filter_matches(matches)
    normalized = normalize_phase_scores(filtered)
    # plot_basic_distributions(normalized, teams)
    # plot_team_winrates(normalized)
    # plot_yearly_winrate_trends(normalized)
    plot_auto_analysis(normalized)
    plot_teleop_analysis(normalized)
    plot_endgame_analysis(normalized)
    analyze_event_competitiveness(normalized)
    run_event_stability_analysis()
    print("\nAll EDA outputs complete and saved.")


if __name__ == "__main__":
    main()
