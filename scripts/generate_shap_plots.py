"""
Generate Step 2 explainability figures:
  1. SHAP beeswarm (feature impact distribution)
  2. Horizontal bar chart of mean |SHAP| feature importance

Uses RF + treatment_arm as feature, matching 04_run_experiment.py pipeline.
Saves to outputs/plots/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_PATH = "data/synthetic_trial_with_subgroups.csv"
OUT_DIR   = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = {
    "RIDAGEYR":     "Age (years)",
    "RIAGENDR":     "Gender",
    "RIDRETH3":     "Race / Ethnicity",
    "LBDHDD":       "HDL Cholesterol",
    "LBXGLU":       "Glucose",
    "LBXTC":        "Total Cholesterol",
    "LBDLDL":       "LDL Cholesterol",
    "LBXTLG":       "Triglycerides",
    "LBXALT":       "ALT (Liver / CYP3A4)",
    "BMXBMI":       "BMI",
    "BPXSY1":       "Systolic BP",
    "BPXDI1":       "Diastolic BP",
    "treatment_arm":"Treatment Arm",
}

PALETTE = {
    "navy":     "#1A2E4A",
    "teal":     "#2E86AB",
    "green":    "#3BB273",
    "red":      "#E84855",
    "gold":     "#F4A261",
    "light_bg": "#F8F9FA",
    "grid":     "#E0E4EA",
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

    X_tr_aug = X_tr.copy(); X_tr_aug["treatment_arm"] = T_tr.values
    X_te_aug = X_te.copy(); X_te_aug["treatment_arm"] = T_te.values

    rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=4,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr_aug, y_tr)

    return rf, X_tr_aug, X_te_aug, y_te


def rename_columns(df):
    return df.rename(columns=FEATURE_NAMES)


def plot_shap_beeswarm(rf, X_train, X_test):
    print("Computing SHAP values (this may take ~30s)...")
    background = shap.sample(X_train, 200, random_state=42)
    explainer  = shap.TreeExplainer(rf, background)
    sv = explainer.shap_values(X_test)

    # shap_values may be list [neg_class, pos_class] or 3D array
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    # Rename for display
    X_display = rename_columns(X_test.copy())
    sv_df = pd.DataFrame(sv, columns=X_test.columns)
    sv_df = sv_df.rename(columns=FEATURE_NAMES)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(PALETTE["light_bg"])

    plt.sca(ax)
    shap.summary_plot(
        sv_df.values,
        X_display,
        feature_names=X_display.columns.tolist(),
        max_display=13,
        show=False,
        plot_size=None,
        color_bar=True,
    )

    ax.set_title("SHAP Feature Impact Distribution",
                 fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=12)
    ax.set_xlabel("SHAP Value  (impact on model output)", fontsize=11, color=PALETTE["navy"])
    ax.tick_params(colors=PALETTE["navy"], labelsize=10)
    ax.set_facecolor(PALETTE["light_bg"])
    fig.patch.set_facecolor(PALETTE["light_bg"])

    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    plt.tight_layout()
    out = OUT_DIR / "step2_shap_beeswarm.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["light_bg"])
    plt.close()
    print(f"Saved: {out}")
    return sv, sv_df.columns.tolist()


def plot_shap_bar(sv, feature_names_display):
    mean_abs = np.abs(sv).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names_display,
        "importance": mean_abs
    }).sort_values("importance", ascending=True)

    # Color: top 4 highlighted, rest muted
    colors = []
    for i, feat in enumerate(imp_df["feature"]):
        rank = len(imp_df) - i  # 1 = most important
        if rank <= 4:
            colors.append("#1E88E5")   # matches SHAP beeswarm blue
        else:
            colors.append("#B0BEC5")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor(PALETTE["light_bg"])
    ax.set_facecolor(PALETTE["light_bg"])

    bars = ax.barh(imp_df["feature"], imp_df["importance"],
                   color=colors, height=0.6, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, imp_df["importance"]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=PALETTE["navy"])

    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)",
                  fontsize=11, color=PALETTE["navy"])
    ax.set_title("SHAP Feature Importance\n(mean absolute SHAP — Random Forest, RF + Treatment Arm)",
                 fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=12)

    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=PALETTE["navy"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    legend_patches = [
        mpatches.Patch(color="#1E88E5",         label="Top 4 features"),
        mpatches.Patch(color="#B0BEC5",         label="Secondary / noise features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9.5,
              framealpha=0.85, edgecolor=PALETTE["grid"])

    plt.tight_layout()
    out = OUT_DIR / "step2_shap_importance_bar.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["light_bg"])
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading data and training RF...")
    rf, X_train, X_test, y_test = load_and_train()
    print("Done training.\n")

    sv, feat_names = plot_shap_beeswarm(rf, X_train, X_test)
    plot_shap_bar(sv, feat_names)
    print("\nAll Step 2 figures saved.")
