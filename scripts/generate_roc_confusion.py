"""
Generate ROC curve and confusion matrix for Step 1 (Predictive Modeling).
Uses the same RF + treatment_arm setup as 04_run_experiment.py.
Saves to outputs/plots/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_PATH = "data/synthetic_trial_with_subgroups.csv"
OUT_DIR   = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy":      "#1A2E4A",
    "teal":      "#2E86AB",
    "green":     "#3BB273",
    "red":       "#E84855",
    "gold":      "#F4A261",
    "light_bg":  "#F8F9FA",
    "grid":      "#E0E4EA",
}


def load_and_train():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["SEQN"], errors="ignore")

    y = df["Responded"]
    T = df["treatment_arm"].astype(int)
    X = df.drop(columns=["Responded", "treatment_arm"])
    X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_tr, X_te, y_tr, y_te, T_tr, T_te = train_test_split(
        X_scaled, y, T, test_size=0.20, random_state=42, stratify=y
    )

    # Append treatment_arm as feature (same as 04_run_experiment.py)
    X_tr_aug = X_tr.copy(); X_tr_aug["treatment_arm"] = T_tr.values
    X_te_aug = X_te.copy(); X_te_aug["treatment_arm"] = T_te.values

    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=4,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr_aug, y_tr)

    y_prob = rf.predict_proba(X_te_aug)[:, 1]
    y_pred = rf.predict(X_te_aug)

    return y_te.values, y_pred, y_prob


def plot_roc(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Find point closest to (0,1) — "optimal" threshold
    dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    opt_idx = np.argmin(dist)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor(PALETTE["light_bg"])
    ax.set_facecolor(PALETTE["light_bg"])

    # Shaded area under curve
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["teal"])

    # ROC curve
    ax.plot(fpr, tpr, color=PALETTE["teal"], lw=2.8, label=f"Random Forest  (AUC = {auc:.3f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color=PALETTE["navy"], lw=1.4, linestyle="--",
            alpha=0.5, label="Random classifier  (AUC = 0.500)")

    # Optimal operating point
    ax.scatter(fpr[opt_idx], tpr[opt_idx], color=PALETTE["gold"], s=100, zorder=5,
               label=f"Optimal threshold  (FPR={fpr[opt_idx]:.2f}, TPR={tpr[opt_idx]:.2f})")
    ax.annotate(f"Optimal\nthreshold",
                xy=(fpr[opt_idx], tpr[opt_idx]),
                xytext=(fpr[opt_idx] + 0.08, tpr[opt_idx] - 0.10),
                fontsize=9, color=PALETTE["gold"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["gold"], lw=1.2))

    # AUC annotation box
    ax.text(0.58, 0.18,
            f"AUC = {auc:.3f}\nF1 = 0.754\nAcc = 93.2%",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=PALETTE["navy"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=PALETTE["teal"], linewidth=1.5))

    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12, color=PALETTE["navy"])
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12, color=PALETTE["navy"])
    ax.set_title("ROC Curve — Predictive Model (Random Forest)",
                 fontsize=14, fontweight="bold", color=PALETTE["navy"], pad=14)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85,
              edgecolor=PALETTE["grid"])

    ax.tick_params(colors=PALETTE["navy"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    out = OUT_DIR / "step1_roc_curve.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["light_bg"])
    plt.close()
    print(f"Saved: {out}")


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    labels = [["True Negative\n(TN)", "False Positive\n(FP)"],
              ["False Negative\n(FN)", "True Positive\n(TP)"]]

    colors = [
        [PALETTE["green"],  PALETTE["red"]],
        [PALETTE["red"],    PALETTE["green"]],
    ]

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    fig.patch.set_facecolor(PALETTE["light_bg"])
    ax.set_facecolor(PALETTE["light_bg"])

    cell_vals = [[tn, fp], [fn, tp]]

    for row in range(2):
        for col in range(2):
            val  = cell_vals[row][col]
            pct  = val / total * 100
            color = colors[row][col]

            rect = plt.Rectangle([col, 1 - row], 1, 1,
                                  facecolor=color, alpha=0.20,
                                  edgecolor="white", linewidth=2)
            ax.add_patch(rect)

            ax.text(col + 0.5, 1 - row + 0.62,
                    labels[row][col],
                    ha="center", va="center", fontsize=9.5,
                    color=color, fontweight="bold")

            ax.text(col + 0.5, 1 - row + 0.38,
                    f"{val:,}",
                    ha="center", va="center", fontsize=20,
                    color=PALETTE["navy"], fontweight="bold")

            ax.text(col + 0.5, 1 - row + 0.18,
                    f"{pct:.1f}% of total",
                    ha="center", va="center", fontsize=9,
                    color=PALETTE["navy"], alpha=0.7)

    # Axes labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nNon-Responder", "Predicted\nResponder"],
                       fontsize=11, color=PALETTE["navy"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual\nResponder", "Actual\nNon-Responder"],
                       fontsize=11, color=PALETTE["navy"])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_title("Confusion Matrix — Predictive Model (Random Forest)",
                 fontsize=13, fontweight="bold", color=PALETTE["navy"],
                 pad=38)

    # Summary stats strip
    f1  = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0

    summary = (f"Accuracy: {acc:.1%}   |   Precision: {prec:.1%}   |   "
               f"Recall: {rec:.1%}   |   F1: {f1:.3f}")
    fig.text(0.5, 0.02, summary, ha="center", fontsize=9.5,
             color=PALETTE["navy"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=PALETTE["grid"], linewidth=1))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "step1_confusion_matrix.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["light_bg"])
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Training RF model...")
    y_true, y_pred, y_prob = load_and_train()
    print(f"  AUC = {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  F1  = {f1_score(y_true, y_pred):.3f}")
    print(f"  Acc = {accuracy_score(y_true, y_pred):.3f}")
    print("\nGenerating plots...")
    plot_roc(y_true, y_prob)
    plot_confusion(y_true, y_pred)
    print("\nDone.")
