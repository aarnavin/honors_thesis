"""
Generate all presentation-ready images from Sublytics analysis results.
Updated for atorvastatin synthetic trial — best model: RF + X-Learner
(AUC=0.868, F1=0.754, PEHE=0.071).
"""

import math
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path("outputs/plots")
DATA_PATH  = Path("data/synthetic_trial_with_subgroups.csv")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Friendly feature name mapping ─────────────────────────────────────────────
FEATURE_NAMES = {
    "treatment_arm": "Treatment Arm",
    "LBDLDL":        "LDL Cholesterol",
    "LBXALT":        "ALT (Liver / CYP3A4)",
    "LBXGLU":        "Glucose",
    "RIDAGEYR":      "Age (years)",
    "RIAGENDR":      "Sex",
    "LBDHDD":        "HDL Cholesterol",
    "LBXTC":         "Total Cholesterol",
    "LBXTLG":        "Triglycerides",
    "BMXBMI":        "BMI",
    "BPXSY1":        "Systolic BP",
    "BPXDI1":        "Diastolic BP",
    "RIDRETH3":      "Ethnicity",
}

# ── Shared palette / style ────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1e3a8a",
    "accent":    "#667eea",
    "highlight": "#f59e0b",
    "danger":    "#ef4444",
    "success":   "#22c55e",
    "gray":      "#94a3b8",
    "light":     "#f1f5f9",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Shared data helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rf_feature_importance():
    """
    Train a Random Forest on the full dataset and return feature importances.
    Mirrors 04_run_experiment.py setup (treatment_arm included as a feature).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["SEQN"], errors="ignore")
    y = df["Responded"]
    X = df.drop(columns=["Responded"])
    X_num = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_scaled, y)

    imp = pd.DataFrame({
        "feature":    X_num.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp["label"] = imp["feature"].map(lambda x: FEATURE_NAMES.get(x, x))
    return imp


def get_subgroup_data():
    """
    Atorvastatin ground-truth subgroup hierarchy (drug arm only).
    Response rates verified from synthetic data generation output.
    Overall drug arm RR = 23.7%; placebo = 3.9%.
    """
    overall_rr = 0.237

    raw = [
        ("LDL>=145 + ALT<30 + Glucose<100\n(Triple positive)",  0.887, 231),
        ("LDL>=145 + ALT<30\n(Efficient CYP3A4)",               0.800, 205),
        ("LDL>=145 + Glucose<100\n(Normoglycemic)",             0.777, 112),
        ("LDL>=145 + Female Age 50-75\n(Post-menopausal)",      0.500,  12),
        ("LDL>=145 only\n(Moderate benefit)",                   0.525,  80),
        ("LDL<145\n(No indication)",                            0.042, 1812),
    ]

    rows = []
    for label, rr, n in raw:
        z = 1.96
        denom   = 1 + z**2 / n
        centre  = (rr + z**2 / (2 * n)) / denom
        margin  = z * math.sqrt(rr * (1 - rr) / n + z**2 / (4 * n**2)) / denom
        ci_lo   = max(0.0, centre - margin)
        ci_hi   = min(1.0, centre + margin)
        ri_pp   = rr - overall_rr
        ri_str  = f"+{ri_pp:.1%}" if ri_pp >= 0 else f"{ri_pp:.1%}"
        rows.append({
            "Subgroup": label, "rr": rr, "n": n,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "ri": ri_str,
        })
    return pd.DataFrame(rows), overall_rr


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model Performance Dashboard  (RF + X-Learner best model)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_model_performance():
    metrics = {"AUC-ROC": 0.868, "F1 Score": 0.754, "Accuracy": 0.932}
    thresholds = {
        "AUC-ROC":  (0.5, 0.7, 0.8),
        "F1 Score": (0.0, 0.4, 0.7),
        "Accuracy": (0.5, 0.7, 0.85),
    }
    interpretations = {
        "AUC-ROC":  "Good\n(strong separation)",
        "F1 Score": "High\n(balanced precision/recall)",
        "Accuracy": "Excellent\n(well-calibrated model)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Summary  (RF + X-Learner)",
                 fontsize=18, fontweight="bold", color=PALETTE["primary"], y=1.01)

    for ax, (metric, value) in zip(axes, metrics.items()):
        lo, mid, hi = thresholds[metric]
        color = (PALETTE["danger"]    if value < mid else
                 PALETTE["highlight"] if value < hi  else
                 PALETTE["success"])

        ax.barh([0], [1],     color=PALETTE["light"], height=0.4, zorder=1)
        ax.barh([0], [value], color=color,            height=0.4, zorder=2)

        for t, label in zip([lo, mid, hi], ["", "Fair", "Good"]):
            ax.axvline(t, color="white", lw=2, zorder=3)
            if label:
                ax.text(t, 0.28, label, ha="center", fontsize=7, color="gray")

        ax.text(0.5, -0.45, f"{value:.3f}", ha="center", va="center",
                fontsize=28, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, -0.75, metric, ha="center", va="center",
                fontsize=12, color=PALETTE["primary"], transform=ax.transAxes)
        ax.text(0.5, -1.05, interpretations[metric], ha="center", va="center",
                fontsize=9, color="gray", transform=ax.transAxes, style="italic")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, 0.3)
        ax.axis("off")

    plt.tight_layout()
    path = PLOTS_DIR / "01_model_performance_dashboard.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature Importance (horizontal bar, RF built-in importances)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance():
    imp = get_rf_feature_importance()
    top = imp.head(10).sort_values("importance")  # ascending for barh

    colors = [PALETTE["accent"] if i >= len(top) - 3 else PALETTE["gray"]
              for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["label"], top["importance"] * 100,
                   color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=9, color="#333")

    ax.set_xlabel("Feature Importance (%)", fontsize=11)
    ax.set_title("Top Predictive Features\n(Random Forest Feature Importance)",
                 fontsize=14, fontweight="bold", color=PALETTE["primary"])
    ax.tick_params(axis="y", labelsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["accent"], label="Top 3 features"),
        mpatches.Patch(facecolor=PALETTE["gray"],   label="Other features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlim(0, top["importance"].max() * 100 * 1.25)

    plt.tight_layout()
    path = PLOTS_DIR / "02_feature_importance_bar.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Subgroup Forest Plot
# ═══════════════════════════════════════════════════════════════════════════════
def plot_subgroup_forest():
    df, overall_rr = get_subgroup_data()
    df = df.sort_values("rr", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, max(5, len(df) * 1.1 + 1.5)))

    y_pos   = range(len(df))
    colors  = [PALETTE["success"] if r > overall_rr else PALETTE["danger"]
               for r in df["rr"]]

    # CI whiskers
    for i, row in df.iterrows():
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                color=PALETTE["gray"], lw=1.5, zorder=1)

    # Point estimates
    ax.scatter(df["rr"], list(y_pos), color=colors, s=90, zorder=3,
               edgecolors="white", linewidths=0.8)

    # Overall reference line
    ax.axvline(overall_rr, color=PALETTE["primary"], linestyle="--", lw=1.5,
               label=f"Overall drug arm ({overall_rr:.1%})", zorder=2)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["Subgroup"].tolist(), fontsize=9)

    # Right-side annotations
    x_ann = 0.96
    for i, row in df.iterrows():
        ax.text(x_ann, i,
                f" {row['rr']:.1%}  (N={row['n']:,})  {row['ri']}",
                va="center", fontsize=8, color="#333",
                transform=ax.get_yaxis_transform())

    ax.set_xlabel("Response Rate", fontsize=11)
    ax.set_title("Atorvastatin Responder Subgroups\n(Ground-Truth Hierarchy — Drug Arm Only)",
                 fontsize=14, fontweight="bold", color=PALETTE["primary"])
    ax.set_xlim(0.0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["success"], label="Above overall rate"),
        mpatches.Patch(facecolor=PALETTE["danger"],  label="Below overall rate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "03_subgroup_forest_plot.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Subgroup Response Rate Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_subgroup_bars():
    df, overall_rr = get_subgroup_data()
    df = df.sort_values("rr", ascending=False).reset_index(drop=True)

    def shorten(s, max_len=40):
        s_flat = s.replace("\n", "  ")
        return s_flat if len(s_flat) <= max_len else s_flat[:max_len - 1] + "..."

    labels = [shorten(s) for s in df["Subgroup"]]
    colors = [PALETTE["accent"] if r > overall_rr else PALETTE["gray"]
              for r in df["rr"]]

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.9 + 1.5)))
    bars = ax.barh(labels, df["rr"], color=colors, edgecolor="white", height=0.6)

    for bar, rr, ri, n in zip(bars, df["rr"], df["ri"], df["n"]):
        ax.text(bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{rr:.1%}  {ri}  (N={n:,})",
                va="center", fontsize=9)

    ax.axvline(overall_rr, color=PALETTE["primary"], linestyle="--", lw=1.5,
               label=f"Overall drug arm: {overall_rr:.1%}")
    ax.set_xlabel("Response Rate", fontsize=11)
    ax.set_title("Subgroup Response Rates vs. Overall Drug Arm",
                 fontsize=14, fontweight="bold", color=PALETTE["primary"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, 1.05)

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["accent"], label="Above overall rate"),
        mpatches.Patch(facecolor=PALETTE["gray"],   label="Below overall rate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "04_subgroup_response_bars.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Key Finding Summary Slide
# ═══════════════════════════════════════════════════════════════════════════════
def plot_summary_slide():
    fig = plt.figure(figsize=(14, 8), facecolor=PALETTE["primary"])

    fig.text(0.5, 0.93, "Sublytics - Atorvastatin Subgroup Discovery Summary",
             ha="center", va="center", fontsize=20, fontweight="bold", color="white")
    fig.text(0.5, 0.87,
             "Synthetic Trial  |  N = 5,000  |  Best Model: RF + X-Learner",
             ha="center", va="center", fontsize=12, color="#94a3b8")

    def add_card(fig, left, bottom, width, height, title, value, subtitle, color):
        ax = fig.add_axes([left, bottom, width, height], facecolor=color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.78, title,    ha="center", va="center", fontsize=10,
                color="white", alpha=0.85)
        ax.text(0.5, 0.48, value,    ha="center", va="center", fontsize=28,
                fontweight="bold", color="white")
        ax.text(0.5, 0.18, subtitle, ha="center", va="center", fontsize=9,
                color="white", alpha=0.75, style="italic")

    card_w, card_h = 0.19, 0.22
    gap = 0.015
    starts = [0.04 + i * (card_w + gap) for i in range(5)]
    top = 0.60

    cards = [
        ("AUC-ROC",       "0.868", "Good discrimination",    "#22c55e"),
        ("F1 Score",      "0.754", "High precision/recall",  "#3b82f6"),
        ("PEHE",          "0.071", "Best causal accuracy",   "#7c3aed"),
        ("ITT Resp. Rate","23.7%", "Appears as failed trial","#ef4444"),
        ("Best Subgroup", "88.7%", "LDL+ALT+Glucose triple", "#f59e0b"),
    ]
    for x, (title, value, subtitle, color) in zip(starts, cards):
        add_card(fig, x, top, card_w, card_h, title, value, subtitle, color)

    # Key findings
    ax_text = fig.add_axes([0.04, 0.08, 0.56, 0.48], facecolor="#1e40af")
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    ax_text.text(0.5, 0.93, "Key Findings", ha="center", fontsize=13,
                 fontweight="bold", color="white")
    findings = [
        "1.  RF + X-Learner: AUC=0.868, PEHE=0.071 (best causal accuracy)",
        "2.  ITT analysis: 23.7% drug vs 3.9% placebo -> 'failed trial'",
        "3.  Triple-positive subgroup: LDL>=145, ALT<30, Glucose<100 -> 88.7%",
        "4.  LDL is the primary gating biomarker for all responder subgroups",
        "5.  5 of 7 model configs found statistically significant subgroups",
    ]
    for i, line in enumerate(findings):
        ax_text.text(0.05, 0.77 - i * 0.155, line, va="center", fontsize=10,
                     color="white")

    # Recommendations
    ax_rec = fig.add_axes([0.63, 0.08, 0.34, 0.48], facecolor="#7c3aed")
    ax_rec.set_xlim(0, 1); ax_rec.set_ylim(0, 1)
    ax_rec.axis("off")
    ax_rec.text(0.5, 0.93, "Recommendation", ha="center", fontsize=13,
                fontweight="bold", color="white")
    rec_lines = [
        "Strong evidence of a",
        "rescuable subgroup.",
        "",
        "Proposed enriched trial:",
        "  Screen for LDL >= 145",
        "  Confirm ALT < 30",
        "  Confirm Glucose < 100",
        "",
        "Expected enriched RR: ~88%",
    ]
    for i, line in enumerate(rec_lines):
        ax_rec.text(0.06, 0.80 - i * 0.085, line, va="center", fontsize=10,
                    color="white")

    path = PLOTS_DIR / "05_summary_slide.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["primary"])
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Top Features - Lollipop Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_feature_lollipop():
    imp = get_rf_feature_importance()
    top = imp.head(10).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hlines(top["label"], 0, top["importance"] * 100,
              color=PALETTE["gray"], linewidth=1.5, linestyle="--")
    ax.scatter(top["importance"] * 100, top["label"],
               color=PALETTE["accent"], s=120, zorder=5,
               edgecolors="white", linewidths=0.8)

    for _, row in top.iterrows():
        ax.text(row["importance"] * 100 + 0.2, row["label"],
                f'{row["importance"]*100:.1f}%', va="center", fontsize=9, color="#333")

    ax.set_xlabel("Feature Importance (%)", fontsize=11)
    ax.set_title("Top 10 Predictive Biomarkers\n(Random Forest Feature Importance)",
                 fontsize=14, fontweight="bold", color=PALETTE["primary"])
    ax.set_xlim(0, top["importance"].max() * 100 * 1.3)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    path = PLOTS_DIR / "06_feature_lollipop.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. AUC Interpretation Gauge  (best model: 0.868)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_auc_gauge():
    from matplotlib.patches import Wedge

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"aspect": "equal"})

    auc = 0.868
    zones = [
        (0.50, 0.60, "#ef4444", "Fail\n(0.5-0.6)"),
        (0.60, 0.70, "#f97316", "Weak\n(0.6-0.7)"),
        (0.70, 0.80, "#eab308", "Fair\n(0.7-0.8)"),
        (0.80, 0.90, "#22c55e", "Good\n(0.8-0.9)"),
        (0.90, 1.00, "#3b82f6", "Excellent\n(0.9-1.0)"),
    ]

    for lo, hi, color, label in zones:
        theta1 = 180 - (lo - 0.5) * 360
        theta2 = 180 - (hi - 0.5) * 360
        wedge = Wedge((0, 0), 1.0, theta2, theta1, width=0.3,
                      facecolor=color, alpha=0.85)
        ax.add_patch(wedge)
        mid_theta = np.radians((theta1 + theta2) / 2)
        ax.text(0.83 * np.cos(mid_theta), 0.83 * np.sin(mid_theta),
                label, ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    needle_angle = np.radians(180 - (auc - 0.5) * 360)
    ax.annotate("",
                xy=(0.65 * np.cos(needle_angle), 0.65 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["primary"], lw=2.5))
    ax.add_patch(plt.Circle((0, 0), 0.08, color=PALETTE["primary"], zorder=5))

    ax.text(0, -0.18, f"AUC = {auc}", ha="center", fontsize=16,
            fontweight="bold", color=PALETTE["primary"])
    ax.text(0, -0.35, "Good — strong treatment-response discrimination",
            ha="center", fontsize=10, color=PALETTE["success"], style="italic")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.55, 1.15)
    ax.axis("off")
    ax.set_title("Model Discrimination (AUC-ROC)  |  RF + X-Learner",
                 fontsize=14, fontweight="bold", color=PALETTE["primary"], pad=10)

    plt.tight_layout()
    path = PLOTS_DIR / "07_auc_gauge.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating presentation images (RF + X-Learner best model)...")
    print()

    print("[1/7] Model performance dashboard...")
    plot_model_performance()

    print("[2/7] Feature importance bar chart (training RF)...")
    plot_feature_importance()

    print("[3/7] Subgroup forest plot...")
    plot_subgroup_forest()

    print("[4/7] Subgroup response bars...")
    plot_subgroup_bars()

    print("[5/7] Summary slide...")
    plot_summary_slide()

    print("[6/7] Feature lollipop chart (training RF)...")
    plot_feature_lollipop()

    print("[7/7] AUC gauge...")
    plot_auc_gauge()

    print()
    print("Done! Images saved to outputs/plots/:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")
