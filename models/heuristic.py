import numpy as np
import pandas as pd


class HeuristicModel:
    """
    Baseline heuristic predictors for FRC match outcomes.
    
    Supported modes:
        - "random"       : picks red or blue uniformly at random
        - "auto"         : red_win = (red_avg_auto > blue_avg_auto)
        - "teleop"       : red_win = (red_avg_teleop > blue_avg_teleop)
        - "endgame"      : red_win = (red_avg_endgame > blue_avg_endgame)
        - "total"        : red_win = (red_avg_total > blue_avg_total)
        - "winrate"      : red_win = (red_avg_winrate > blue_avg_winrate)
        - "combined"     : votes from the above 5 categories
    """

    def __init__(self, mode="combined"):
        self.mode = mode

    def train(self, X, y):
        """
        Heuristic models do not learn parameters,
        but train() is implemented for compatibility.
        """
        return

    # ---------- Individual Heuristics ----------

    def random_choice(self, X):
        return np.random.randint(0, 2, size=len(X))

    def auto_heuristic(self, X):
        return (X["red_avg_auto"] > X["blue_avg_auto"]).astype(int)

    def teleop_heuristic(self, X):
        return (X["red_avg_teleop"] > X["blue_avg_teleop"]).astype(int)

    def endgame_heuristic(self, X):
        return (X["red_avg_endgame"] > X["blue_avg_endgame"]).astype(int)

    def total_heuristic(self, X):
        return (X["red_avg_total"] > X["blue_avg_total"]).astype(int)

    def winrate_heuristic(self, X):
        return (X["red_avg_winrate"] > X["blue_avg_winrate"]).astype(int)

    def combined_heuristic(self, X):
        """
        A voting system:
        Each phase contributes 1 vote toward red or blue.
        Red wins = majority votes > 2.
        """
        votes = np.zeros(len(X))

        votes += (X["red_avg_auto"] > X["blue_avg_auto"]).astype(int)
        votes += (X["red_avg_teleop"] > X["blue_avg_teleop"]).astype(int)
        votes += (X["red_avg_endgame"] > X["blue_avg_endgame"]).astype(int)
        votes += (X["red_avg_total"] > X["blue_avg_total"]).astype(int)
        votes += (X["red_avg_winrate"] > X["blue_avg_winrate"]).astype(int)

        # Majority vote -> red wins
        return (votes >= 3).astype(int)

    # ---------- Prediction Dispatcher ----------

    def predict(self, X):
        if self.mode == "random":
            return self.random_choice(X)

        elif self.mode == "auto":
            return self.auto_heuristic(X)

        elif self.mode == "teleop":
            return self.teleop_heuristic(X)

        elif self.mode == "endgame":
            return self.endgame_heuristic(X)

        elif self.mode == "total":
            return self.total_heuristic(X)

        elif self.mode == "winrate":
            return self.winrate_heuristic(X)

        elif self.mode == "combined":
            return self.combined_heuristic(X)

        else:
            raise ValueError(f"Unknown heuristic mode: {self.mode}")
