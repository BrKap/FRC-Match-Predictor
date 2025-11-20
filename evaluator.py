from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import numpy as np


def evaluate_model(y_true, y_pred, model_name="model"):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{model_name}:  P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------
def plot_roc_curve(y_true, y_proba, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')  # random chance line

    plt.title(f"ROC Curve — {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_all_roc(all_results, save_path):
    plt.figure(figsize=(7,6))

    for name, (y_true, y_proba) in all_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curves — All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# ------------------------------------------------------------
# Precision-Recall Curve
# ------------------------------------------------------------
def plot_pr_curve(y_true, y_proba, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")

    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_all_pr(all_results, save_path):
    plt.figure(figsize=(7,6))

    for name, (y_true, y_proba) in all_results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    plt.title("Precision-Recall Curves — All Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# ------------------------------------------------------------
# Combined evaluator
# ------------------------------------------------------------
def evaluate_with_curves(y_true, y_pred, y_proba, model_name, out_dir):

    metrics = evaluate_model(y_true, y_pred, model_name=model_name)

    # Curves
    plot_roc_curve(y_true, y_proba, model_name, out_dir / f"{model_name}_roc.png")
    plot_pr_curve(y_true, y_proba, model_name, out_dir / f"{model_name}_pr.png")
    plot_calibration_curve(y_true, y_proba, model_name, out_dir / f"{model_name}_calibration.png")

    # Confusion matrix
    plot_conf_matrix(y_true, y_pred, model_name, out_dir / f"{model_name}_confusion.png")

    # Best threshold
    best_thresh, best_f1 = find_best_threshold(y_true, y_proba)
    metrics["best_threshold"] = best_thresh
    metrics["best_f1"] = best_f1
    
    best_preds = (y_proba >= best_thresh).astype(int)
    best_accuracy = (best_preds == y_true).mean()
    metrics["best_accuracy"] = best_accuracy

    return metrics



def evaluate_model(y_true, y_pred, model_name="model"):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"{model_name}:  P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
# ------------------------------------------------------------
# Calibration Curve
# ------------------------------------------------------------
def plot_calibration_curve(y_true, y_proba, model_name, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=15)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
    plt.plot([0,1],[0,1],'k--', label="Perfectly calibrated")

    plt.title(f"Calibration Curve — {model_name}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
# ------------------------------------------------------------
# Threshold Curve
# ------------------------------------------------------------   
def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 200)

    best_thresh = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------   
def plot_conf_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["Blue", "Red"],
                yticklabels=["Blue", "Red"])

    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------   

def get_random_forest_importance(model, feature_names):
    importance = model.model.feature_importances_
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

def get_logreg_importance(model, feature_names):
    coefs = model.model.named_steps["logreg"].coef_[0]
    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coeff": np.abs(coefs)
    }).sort_values(by="abs_coeff", ascending=False)

def get_knn_importance(model, X_test, y_test, feature_names):
    r = permutation_importance(model.model, X_test, y_test, n_repeats=10, random_state=42)
    return pd.DataFrame({
        "feature": feature_names,
        "importance": r.importances_mean
    }).sort_values(by="importance", ascending=False)

# ------------------------------------------------------------
# Plot Model Accuracy Over Weeks
# ------------------------------------------------------------

def plot_model_accuracy_over_weeks(df, out_path):
    """
    df: weekly_results.csv loaded into a DataFrame
    out_path: filepath to save the combined plot
    """

    import matplotlib.pyplot as plt

    weeks = df["week"]

    plt.figure(figsize=(10, 6))

    # Plot each model
    if "knn_acc" in df.columns:
        plt.plot(weeks, df["knn_acc"], marker="o", label="KNN")

    if "random_forest_acc" in df.columns:
        plt.plot(weeks, df["random_forest_acc"], marker="o", label="Random Forest")

    if "log_reg_acc" in df.columns:
        plt.plot(weeks, df["log_reg_acc"], marker="o", label="Logistic Regression")

    # Optional: cumulative accuracy
    if "cumulative_accuracy" in df.columns:
        plt.plot(weeks, df["cumulative_accuracy"], marker="o",
                 linestyle="--", label="Cumulative Accuracy", color="black")

    plt.xlabel("Week")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Across Weeks (Rolling Evaluation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()
