"""
Generate all presentation-ready images from Sublytics analysis results.
Reads existing CSV outputs and existing PNGs, then creates additional charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Friendly feature name mapping ────────────────────────────────────────────
FEATURE_NAMES = {
    "LBXGLU":   "Glucose",
    "LBXSATSI": "AST (Liver Enzyme)",
    "INDFMPIR": "Income-to-Poverty Ratio",
    "RIDAGEYR": "Age (years)",
    "LBXSAL":   "Albumin",
    "LBDLDL":   "LDL Cholesterol",
    "LBXTST":   "Testosterone",
    "LBXSAPSI": "Alkaline Phosphatase",
    "LBXTLG":   "Triglycerides",
    "LBXSASSI": "AST (Alt.)",
    "LBXSUA":   "Uric Acid",
    "LBXSCR":   "Creatinine",
    "DMDEDUC2": "Education Level",
    "DMDHHSIZ": "Household Size",
    "RIDEXPRG": "Pregnancy Status",
}

# ── Shared style ─────────────────────────────────────────────────────────────
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
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ═══════════════════════════════════════════════════════════════════════════
# 1. Model Performance Dashboard
# ═══════════════════════════════════════════════════════════════════════════
def plot_model_performance():
    metrics = {"AUC-ROC": 0.568, "F1 Score": 0.173, "Accuracy": 0.689}
    thresholds = {"AUC-ROC": (0.5, 0.7, 0.8), "F1 Score": (0, 0.4, 0.7), "Accuracy": (0.5, 0.7, 0.85)}
    interpretations = {
        "AUC-ROC":  "Weak\n(barely above random)",
        "F1 Score": "Poor\n(low precision/recall)",
        "Accuracy": "Moderate\n(imbalanced classes)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Summary", fontsize=18, fontweight="bold",
                 color=PALETTE["primary"], y=1.01)

    for ax, (metric, value) in zip(axes, metrics.items()):
        lo, mid, hi = thresholds[metric]
        color = PALETTE["danger"] if value < mid else (PALETTE["highlight"] if value < hi else PALETTE["success"])

        # Gauge bar
        ax.barh([0], [1], color=PALETTE["light"], height=0.4, zorder=1)
        ax.barh([0], [value], color=color, height=0.4, zorder=2)

        # Zone markers
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


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature Importance (horizontal bar, with friendly names)
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_importance():
    df = pd.read_csv(PLOTS_DIR / "top_features.csv")
    df["label"] = df["feature"].map(lambda x: FEATURE_NAMES.get(x, x))
    df = df.sort_values("abs_mean")

    colors = [PALETTE["accent"] if i >= len(df) - 3 else PALETTE["gray"]
              for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["label"], df["abs_mean"] * 1000, color=colors, edgecolor="white", height=0.65)

    # Value labels
    for bar, val in zip(bars, df["abs_mean"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val*1000:.2f}", va="center", fontsize=9, color="#333")

    ax.set_xlabel("Mean |SHAP| value (×10⁻³)", fontsize=11)
    ax.set_title("Top Predictive Features\n(SHAP Feature Importance)", fontsize=14,
                 fontweight="bold", color=PALETTE["primary"])
    ax.tick_params(axis="y", labelsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["accent"], label="Top 3 features"),
        mpatches.Patch(facecolor=PALETTE["gray"],   label="Other features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlim(0, df["abs_mean"].max() * 1000 * 1.2)

    plt.tight_layout()
    path = PLOTS_DIR / "02_feature_importance_bar.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Subgroup Forest Plot
# ═══════════════════════════════════════════════════════════════════════════
def plot_subgroup_forest():
    df = pd.read_csv(PLOTS_DIR / "subgroups.csv")

    # Parse numeric response rate and CI
    def parse_pct(s):
        return float(str(s).replace("%", "").replace("+", "")) / 100

    df["rr_num"] = df["Response_Rate"].apply(parse_pct)

    def parse_ci(s):
        s = str(s).replace("[", "").replace("]", "").replace("%", "")
        lo, hi = s.split(",")
        return float(lo.strip()) / 100, float(hi.strip()) / 100

    df[["ci_lo", "ci_hi"]] = pd.DataFrame(df["CI_95"].apply(parse_ci).tolist())

    overall_rr = 0.413
    df = df.sort_values("rr_num", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.8 + 1.5)))

    y_positions = range(len(df))
    colors = [PALETTE["success"] if r > overall_rr else PALETTE["danger"]
              for r in df["rr_num"]]

    # CI lines
    for i, row in df.iterrows():
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=PALETTE["gray"], lw=1.5, zorder=1)

    # Point estimates
    ax.scatter(df["rr_num"], y_positions, color=colors, s=80, zorder=3, edgecolors="white", linewidths=0.8)

    # Overall reference line
    ax.axvline(overall_rr, color=PALETTE["primary"], linestyle="--", lw=1.5,
               label=f"Overall rate ({overall_rr:.1%})", zorder=2)

    # Labels
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["Subgroup"].tolist(), fontsize=9)

    # Right-side annotations
    for i, row in df.iterrows():
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0.65 else 0.65,
                i, f" {row['Response_Rate']}  {row['Risk_Increase']}",
                va="center", fontsize=8, color="#333")

    ax.set_xlabel("Response Rate", fontsize=11)
    ax.set_title("Subgroup Response Rates\n(Forest Plot Style)", fontsize=14,
                 fontweight="bold", color=PALETTE["primary"])
    ax.set_xlim(0.2, 0.72)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=9)

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


# ═══════════════════════════════════════════════════════════════════════════
# 4. Subgroup Response Rate Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_subgroup_bars():
    df = pd.read_csv(PLOTS_DIR / "subgroups.csv")

    def parse_pct(s):
        return float(str(s).replace("%", "").replace("+", "")) / 100

    df["rr_num"] = df["Response_Rate"].apply(parse_pct)
    df = df.sort_values("rr_num", ascending=False).reset_index(drop=True)

    overall_rr = 0.413
    colors = [PALETTE["accent"] if r > overall_rr else PALETTE["gray"] for r in df["rr_num"]]

    # Shorten labels
    def shorten(label, max_len=38):
        return label if len(label) <= max_len else label[:max_len - 1] + "…"

    labels = [shorten(s) for s in df["Subgroup"]]

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.75 + 1.5)))
    bars = ax.barh(labels, df["rr_num"], color=colors, edgecolor="white", height=0.6)

    # Value labels
    for bar, rr, ri in zip(bars, df["rr_num"], df["Risk_Increase"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{rr:.1%}  ({ri})", va="center", fontsize=9)

    ax.axvline(overall_rr, color=PALETTE["primary"], linestyle="--", lw=1.5,
               label=f"Overall: {overall_rr:.1%}")
    ax.set_xlabel("Response Rate", fontsize=11)
    ax.set_title("Subgroup Response Rates vs. Overall", fontsize=14,
                 fontweight="bold", color=PALETTE["primary"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, 0.65)
    ax.legend(fontsize=9)

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


# ═══════════════════════════════════════════════════════════════════════════
# 5. Key Finding Summary Slide
# ═══════════════════════════════════════════════════════════════════════════
def plot_summary_slide():
    fig = plt.figure(figsize=(14, 8), facecolor=PALETTE["primary"])

    # Title band
    fig.text(0.5, 0.93, "Sublytics – Clinical Trial Analysis Summary",
             ha="center", va="center", fontsize=20, fontweight="bold", color="white")
    fig.text(0.5, 0.87, "NHANES Simulated Trial  |  N = 3,996  |  Random Forest + SHAP",
             ha="center", va="center", fontsize=12, color="#94a3b8")

    # ── Card helper ──────────────────────────────────────────────────────
    def add_card(fig, left, bottom, width, height, title, value, subtitle, color):
        ax = fig.add_axes([left, bottom, width, height], facecolor=color)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.78, title,  ha="center", va="center", fontsize=10, color="white", alpha=0.85)
        ax.text(0.5, 0.48, value,  ha="center", va="center", fontsize=28, fontweight="bold", color="white")
        ax.text(0.5, 0.18, subtitle, ha="center", va="center", fontsize=9, color="white", alpha=0.75, style="italic")

    card_w, card_h = 0.19, 0.22
    gap = 0.015
    starts = [0.04 + i * (card_w + gap) for i in range(5)]
    top = 0.60

    cards = [
        ("AUC-ROC",          "0.568", "Weak signal",             "#ef4444"),
        ("F1 Score",         "0.173", "Low precision/recall",    "#f97316"),
        ("Accuracy",         "68.9%", "Modest",                  "#eab308"),
        ("Response Rate",    "41.3%", "Of test arm patients",    "#3b82f6"),
        ("Best Subgroup",    "+5.6%", "Age ≤31  (N=98)",         "#22c55e"),
    ]
    for x, (title, value, subtitle, color) in zip(starts, cards):
        add_card(fig, x, top, card_w, card_h, title, value, subtitle, color)

    # ── Key findings text ────────────────────────────────────────────────
    ax_text = fig.add_axes([0.04, 0.08, 0.56, 0.48], facecolor="#1e40af")
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    ax_text.text(0.5, 0.93, "Key Findings", ha="center", fontsize=13,
                 fontweight="bold", color="white")
    findings = [
        "1.  AUC = 0.568 → model barely above random (0.5)",
        "2.  Top predictor: Glucose (LBXGLU) by SHAP",
        "3.  Most promising subgroup: Age ≤31 (+5.6% response)",
        "4.  No subgroup passed Bonferroni correction",
        "5.  Exploratory only — prospective validation needed",
    ]
    for i, line in enumerate(findings):
        ax_text.text(0.06, 0.75 - i * 0.16, line, va="center", fontsize=10.5, color="white")

    # ── Recommendation box ───────────────────────────────────────────────
    ax_rec = fig.add_axes([0.63, 0.08, 0.34, 0.48], facecolor="#7c3aed")
    ax_rec.set_xlim(0, 1); ax_rec.set_ylim(0, 1)
    ax_rec.axis("off")
    ax_rec.text(0.5, 0.93, "Recommendation", ha="center", fontsize=13,
                fontweight="bold", color="white")
    rec_lines = [
        "Signal is weak (AUC < 0.6).",
        "",
        "No compelling evidence of",
        "a rescuable subgroup.",
        "",
        "Next steps if pursuing:",
        "• Validate in independent cohort",
        "• Prospective enriched trial",
        "• Biological mechanism check",
    ]
    for i, line in enumerate(rec_lines):
        ax_rec.text(0.06, 0.78 - i * 0.085, line, va="center", fontsize=10, color="white")

    path = PLOTS_DIR / "05_summary_slide.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor=PALETTE["primary"])
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Top Features — Lollipop Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_feature_lollipop():
    df = pd.read_csv(PLOTS_DIR / "top_features.csv").head(10)
    df["label"] = df["feature"].map(lambda x: FEATURE_NAMES.get(x, x))
    df = df.sort_values("abs_mean")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hlines(df["label"], 0, df["abs_mean"] * 1000,
              color=PALETTE["gray"], linewidth=1.5, linestyle="--")
    ax.scatter(df["abs_mean"] * 1000, df["label"],
               color=PALETTE["accent"], s=120, zorder=5, edgecolors="white", linewidths=0.8)

    for _, row in df.iterrows():
        ax.text(row["abs_mean"] * 1000 + 0.1, row["label"],
                f'{row["abs_mean"]*1000:.2f}', va="center", fontsize=9, color="#333")

    ax.set_xlabel("Mean |SHAP| value (×10⁻³)", fontsize=11)
    ax.set_title("Top 10 Predictive Biomarkers\n(SHAP Mean Absolute Impact)", fontsize=14,
                 fontweight="bold", color=PALETTE["primary"])
    ax.set_xlim(0, df["abs_mean"].max() * 1000 * 1.25)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    path = PLOTS_DIR / "06_feature_lollipop.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. AUC Interpretation Gauge
# ═══════════════════════════════════════════════════════════════════════════
def plot_auc_gauge():
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"aspect": "equal"})

    auc = 0.568
    zones = [
        (0.50, 0.60, "#ef4444", "Fail\n(0.5–0.6)"),
        (0.60, 0.70, "#f97316", "Weak\n(0.6–0.7)"),
        (0.70, 0.80, "#eab308", "Fair\n(0.7–0.8)"),
        (0.80, 0.90, "#22c55e", "Good\n(0.8–0.9)"),
        (0.90, 1.00, "#3b82f6", "Excellent\n(0.9–1.0)"),
    ]

    # Draw half-donut arcs
    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc, Wedge

    for lo, hi, color, label in zones:
        theta1 = 180 - (lo - 0.5) * 360
        theta2 = 180 - (hi - 0.5) * 360
        wedge = Wedge((0, 0), 1.0, theta2, theta1, width=0.3, facecolor=color, alpha=0.85)
        ax.add_patch(wedge)
        mid_theta = np.radians((theta1 + theta2) / 2)
        lx = 0.83 * np.cos(mid_theta)
        ly = 0.83 * np.sin(mid_theta)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    # Needle
    needle_angle = np.radians(180 - (auc - 0.5) * 360)
    ax.annotate("", xy=(0.65 * np.cos(needle_angle), 0.65 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["primary"], lw=2.5))
    ax.add_patch(plt.Circle((0, 0), 0.08, color=PALETTE["primary"], zorder=5))

    ax.text(0, -0.18, f"AUC = {auc}", ha="center", fontsize=16,
            fontweight="bold", color=PALETTE["primary"])
    ax.text(0, -0.35, "Weak signal — barely above random", ha="center", fontsize=11,
            color=PALETTE["danger"], style="italic")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.55, 1.15)
    ax.axis("off")
    ax.set_title("Model Discrimination (AUC-ROC)", fontsize=14,
                 fontweight="bold", color=PALETTE["primary"], pad=10)

    plt.tight_layout()
    path = PLOTS_DIR / "07_auc_gauge.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating presentation images...")
    print()

    plot_model_performance()
    plot_feature_importance()
    plot_subgroup_forest()
    plot_subgroup_bars()
    plot_summary_slide()
    plot_feature_lollipop()
    plot_auc_gauge()

    print()
    print("All done! Existing images in outputs/plots/:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {p.name}")
