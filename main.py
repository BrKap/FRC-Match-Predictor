# main.py
from pathlib import Path
from xml.parsers.expat import model
import pandas as pd
import time

# --- Import cleaning (from EDA) ---
from eda.cleaning import load_data, filter_matches, normalize_phase_scores

# --- Preprocessing
from preprocessing import build_features, generate_rolling_team_stats, load_and_truncate_team_stats, prepare_base_matches, temporal_split_by_year_week

# --- Evaluator
from evaluator import evaluate_model, evaluate_with_curves, get_knn_importance, get_logreg_importance, get_random_forest_importance, plot_all_roc, plot_all_pr, plot_model_accuracy_over_weeks

# --- Models
from models.knn import KNNModel
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticRegressionModel
from models.heuristic import HeuristicModel



DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Configuration
# ============================================================
REGENERATE_STATS = True  # If True, recompute all csv files from raw data
SPLIT_YEAR = 2025   # Choose the year whose week boundary you want to split at
SPLIT_WEEK = 3      # Choose the competition week for temporal cutoff

# ============================================================
# Rolling Weekly Evaluation
# ============================================================

def rolling_weekly_eval(cleaned_matches, rolling_stats, target_year=2025):
    start_total = time.time()
    results = []
    event_results = []

    year_matches = cleaned_matches[cleaned_matches["year"] == target_year]
    weeks = sorted(year_matches["week"].unique())
    print(f"Rolling weekly evaluation over weeks: {weeks}")

    # Start with all training = everything before target year
    train_matches = cleaned_matches[cleaned_matches["year"] < target_year].copy()

    # Running totals for cumulative accuracy
    running_correct = 0
    running_total = 0

    for wk in weeks:
        if wk == 7:
            print(f"\n===== CHAMPIONSHIP (week {wk}) =====")
        else:
            print(f"\n===== WEEK {wk} =====")
        week_start = time.time()


        # Create per-week folder
        week_dir = OUTPUT_DIR / "rolling" / f"week_{wk}"
        week_dir.mkdir(parents=True, exist_ok=True)

        # Test = current week only
        test_matches = year_matches[year_matches["week"] == wk].copy()

        # Rolling stats truncated to PRE-week
        rolling_stats_train = load_and_truncate_team_stats(
            rolling_stats, train_matches, test_matches
        )

        # Build features
        X_train, y_train, X_test, y_test, train_feat, test_feat = build_features(
            train_matches,
            test_matches,
            rolling_stats_train
        )


        # Train + Evaluate models
        week_result = {"week": wk}
        all_probas = {}  # for combined ROC / PR

        for name, model in {
            "knn": KNNModel(),
            "random_forest": RandomForestModel(),
            "log_reg": LogisticRegressionModel(),
        }.items():

            print(f"\n-- Model: {name} --")

            # Make model-specific directory
            model_out = week_dir / name
            model_out.mkdir(exist_ok=True)

            model.train(X_train, y_train)
            preds = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)

                metrics = evaluate_with_curves(
                    y_test,
                    preds,
                    proba,
                    model_name=f"{name}_week{wk}",
                    out_dir=model_out
                )

                acc = metrics["best_accuracy"]
                week_result[f"{name}_acc"] = acc

                # store for combined multi-model ROC/PR
                all_probas[name] = (y_test, proba)

                # running accuracy
                best_thresh = metrics["best_threshold"]
                final_preds = (proba >= best_thresh).astype(int)
                running_correct += (final_preds == y_test).sum()
                running_total += len(y_test)

            else:
                # KNN has proba, heuristics may not
                metrics = evaluate_model(y_test, preds, model_name=f"{name}_week{wk}")
                acc = metrics["f1"]
                week_result[f"{name}_acc"] = acc
                running_correct += (preds == y_test).sum()
                running_total += len(y_test)
                
            # Per-event accuracy tracking
            temp = test_feat.copy()
            temp = temp.reset_index(drop=True)

            if hasattr(model, "predict_proba"):
                temp["pred"] = final_preds
            else:
                temp["pred"] = preds

            for event, group in temp.groupby("event_key"):
                evt_correct = (group["pred"] == group["red_win"].astype(int)).sum()
                evt_total = len(group)
                evt_acc = evt_correct / evt_total

                event_results.append({
                    "week": wk,
                    "event_key": event,
                    "model": name,
                    "accuracy": evt_acc,
                    "num_matches": evt_total
                })


        # create combined ROC/PR for this week
        if len(all_probas) > 0:
            plot_all_roc(all_probas, week_dir / f"roc_all_models_week{wk}.png")
            plot_all_pr(all_probas, week_dir / f"pr_all_models_week{wk}.png")

        # running cumulative accuracy
        cumulative_acc = running_correct / running_total
        week_result["cumulative_accuracy"] = cumulative_acc

        # time taken this week
        week_time = time.time() - week_start
        week_result["runtime_sec"] = week_time

        print(f"Week {wk} runtime: {week_time:.2f} sec")
        print(f"Cumulative accuracy through week {wk}: {cumulative_acc:.3f}")

        results.append(week_result)

        # Add this week's matches into training for next iteration
        train_matches = pd.concat([train_matches, test_matches], ignore_index=True)

    # Write overall total runtime
    total_time = time.time() - start_total
    print(f"\nTotal rolling evaluation time: {total_time:.2f} sec")
    
    # Save per-event accuracy
    event_df = pd.DataFrame(event_results)
    event_out = OUTPUT_DIR / "rolling" / "event_accuracy.csv"
    event_df.to_csv(event_out, index=False)
    print(f"Saved event accuracy to {event_out}")


    return pd.DataFrame(results)




def main():
    print("=== FRC Match Classifier Pipeline ===")

    # ------------------------------------------------------------
    # 1. Load Raw Data
    # ------------------------------------------------------------
    print("\n[1] Loading raw matches.csv and teams.csv...")
    raw_matches, raw_teams = load_data()

    # ------------------------------------------------------------
    # 2. Clean Matches (using your EDA cleaning pipeline)
    # ------------------------------------------------------------
    print("\n[2] Cleaning match dataset (filtering years, removing offseason, malformed rows)...")
    cleaned_matches = filter_matches(raw_matches)

    # Normalize the scoring phases (auto, teleop, endgame)
    print("\n[3] Normalizing phase scores...")
    cleaned_matches = normalize_phase_scores(cleaned_matches)
    cleaned_matches = prepare_base_matches(cleaned_matches)

    # Save cleaned+normalized dataset for reference
    cleaned_matches.to_csv(DATA_DIR / "matches_processed.csv", index=False)
    
    rolling_stats_path = DATA_DIR / "rolling_team_stats.csv"

    if REGENERATE_STATS or not rolling_stats_path.exists():
        rolling_stats = generate_rolling_team_stats(cleaned_matches, rolling_stats_path)
    else:
        rolling_stats = pd.read_csv(rolling_stats_path)


    # # ------------------------------------------------------------
    # # 3. Preprocessing — Create ML Features
    # # ------------------------------------------------------------
    # print("\n[4] Attaching weeks and preparing match table...")
    # base_matches = prepare_base_matches(cleaned_matches)

    # # Temporal split
    # print("\n[5] Performing temporal split...")
    # train_matches, test_matches = temporal_split_by_year_week(
    #     base_matches,
    #     year=SPLIT_YEAR,
    #     week=SPLIT_WEEK
    # )
    
    # rolling_stats_train = load_and_truncate_team_stats(
    #     rolling_stats,
    #     train_matches,
    #     test_matches    
    # )


    # # Feature generation
    # print("\n[6] Building ML features...")
    # X_train, y_train, X_test, y_test = build_features(
    #     train_matches,
    #     test_matches,
    #     rolling_stats_train
    # )

    # # ------------------------------------------------------------
    # # 4. Initialize Models
    # # ------------------------------------------------------------
    # print("\n[5] Initializing ML models...")

    # models = {
    #     # Heuristic Baselines
    #     # "heuristic_combined": HeuristicModel(mode="combined"),
    #     # "heuristic_random": HeuristicModel(mode="random"),
    #     # "heuristic_auto": HeuristicModel(mode="auto"),
    #     # "heuristic_teleop": HeuristicModel(mode="teleop"),
    #     # "heuristic_endgame": HeuristicModel(mode="endgame"),
    #     # "heuristic_total": HeuristicModel(mode="total"),
    #     # "heuristic_winrate": HeuristicModel(mode="winrate"),

    #     # ML Models
    #     "knn": KNNModel(),
    #     "random_forest": RandomForestModel(),
    #     "log_reg": LogisticRegressionModel(),
    # }


    # results = []

    # # ------------------------------------------------------------
    # # 5. Train & Evaluate Each Model
    # # ------------------------------------------------------------
    # print("\n[6] Training & Evaluating Models...")
    
    
    # all_probas = {}   # model_name → (y_test, y_proba)
    # for name, model in models.items():
    #     print(f"\n--- Running Model: {name} ---")

    #     model.train(X_train, y_train)
    #     preds = model.predict(X_test)

    #     # If the model supports predict_proba -> use probability curves
    #     if hasattr(model, "predict_proba"):
    #         proba = model.predict_proba(X_test)
    #         all_probas[name] = (y_test, proba)

    #         metrics = evaluate_with_curves(
    #             y_test,
    #             preds,
    #             proba,
    #             model_name=name,
    #             out_dir=OUTPUT_DIR
    #         )
            
    #         best_thresh = metrics["best_threshold"]
    #         best_acc = metrics["best_accuracy"]

    #         print(f"Optimal threshold: {best_thresh:.3f}")
    #         print(f"Accuracy at optimal threshold: {best_acc:.3f}")

    #         # Use the threshold for final prediction, if desired:
    #         final_preds = model.predict_with_threshold(X_test, best_thresh)
    #     else:
    #         # Heuristics and some simple models do not output probabilities
    #         metrics = evaluate_model(y_test, preds, model_name=name)
            
    #     # Feature Importance
    #     # if name == "random_forest":
    #     #     fi = get_random_forest_importance(model, X_train.columns)
    #     #     fi.to_csv(OUTPUT_DIR / "feature_importance_random_forest.csv", index=False)

    #     # elif name == "log_reg":
    #     #     fi = get_logreg_importance(model, X_train.columns)
    #     #     fi.to_csv(OUTPUT_DIR / "feature_importance_logreg.csv", index=False)

    #     # elif name == "knn":
    #     #     fi = get_knn_importance(model, X_test, y_test, X_train.columns)
    #     #     fi.to_csv(OUTPUT_DIR / "feature_importance_knn.csv", index=False)


    #     results.append(metrics)
        
    # plot_all_roc(all_probas, OUTPUT_DIR / "all_models_roc.png")
    # plot_all_pr(all_probas, OUTPUT_DIR / "all_models_pr.png")
    
    

    # # ------------------------------------------------------------
    # # 6. Save All Evaluation Metrics
    # # ------------------------------------------------------------
    # results_df = pd.DataFrame(results)
    # results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    
    # print("\nPipeline complete! Results saved to model_results.csv")
    
    # ------------------------------------------------------------
    # 3. Rolling Weekly Evaluation
    # ------------------------------------------------------------
    print("\n[4] Starting rolling weekly evaluation...\n")

    # Save weekly results
    weekly_results = rolling_weekly_eval(cleaned_matches, rolling_stats, target_year=2025)
    out_path = OUTPUT_DIR / "rolling" / "weekly_results.csv"
    weekly_results.to_csv(out_path, index=False)

    print("\nRolling weekly evaluation complete!")
    print(f"Saved results to {out_path}")

    # ---------------------------------------------
    # Plot accuracy comparison across weeks
    # ---------------------------------------------

    plot_out = OUTPUT_DIR / "rolling" / "model_comparison_accuracy.png"
    plot_model_accuracy_over_weeks(weekly_results, plot_out)

    print(f"Saved model comparison plot to {plot_out}")



if __name__ == "__main__":
    main()
