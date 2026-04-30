"""
Generate a data snapshot figure for the Data slide.
Top: styled table of representative patients (one per subgroup tier)
Bottom: distribution plots for LDL, ALT, Glucose with threshold lines
Saves to outputs/plots/data_snapshot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from pathlib import Path

DATA_PATH = "data/synthetic_trial_with_subgroups.csv"
OUT_DIR   = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy":     "#1A2E4A",
    "blue":     "#1E88E5",
    "green":    "#3BB273",
    "red":      "#E84855",
    "gold":     "#F4A261",
    "purple":   "#7B2D8B",
    "teal":     "#2E86AB",
    "light_bg": "#F8F9FA",
    "header_bg":"#1A2E4A",
    "row_alt":  "#EEF2F7",
    "grid":     "#D0D7E3",
    "muted":    "#90A4AE",
}

# Subgroup tier definitions (for labeling representative patients)
def assign_tier(row):
    ldl = row["LBDLDL"]; alt = row["LBXALT"]
    glu = row["LBXGLU"]; gen = row["RIAGENDR"]; age = row["RIDAGEYR"]
    if ldl >= 145 and alt < 30 and glu < 100:
        return ("Triple Positive",       PALETTE["green"],  "88%")
    if ldl >= 145 and alt < 30:
        return ("LDL + ALT",             PALETTE["teal"],   "78%")
    if ldl >= 145 and glu < 100:
        return ("LDL + Glucose",         PALETTE["blue"],   "73%")
    if ldl >= 145 and gen == 2 and 50 <= age <= 75:
        return ("LDL + Post-menopausal", PALETTE["purple"], "68%")
    if ldl >= 145:
        return ("LDL ≥ 145 only",        PALETTE["gold"],   "55%")
    return ("LDL < 145 (no indication)", PALETTE["red"],    "5%")


def pick_representatives(df):
    """Pick one clean treated-arm patient per subgroup tier."""
    drug = df[df["treatment_arm"] == 1].copy()
    tiers_wanted = [
        lambda r: r["LBDLDL"] >= 145 and r["LBXALT"] < 30 and r["LBXGLU"] < 100,
        lambda r: r["LBDLDL"] >= 145 and r["LBXALT"] < 30 and r["LBXGLU"] >= 100,
        lambda r: r["LBDLDL"] >= 145 and r["LBXGLU"] < 100 and r["LBXALT"] >= 30,
        lambda r: r["LBDLDL"] >= 145 and r["RIAGENDR"] == 2 and 50 <= r["RIDAGEYR"] <= 75
                  and r["LBXALT"] >= 30 and r["LBXGLU"] >= 100,
        lambda r: r["LBDLDL"] >= 145 and r["LBXALT"] >= 30 and r["LBXGLU"] >= 100
                  and not (r["RIAGENDR"] == 2 and 50 <= r["RIDAGEYR"] <= 75),
        lambda r: r["LBDLDL"] < 145,
    ]
    rows = []
    for fn in tiers_wanted:
        sub = drug[drug.apply(fn, axis=1)]
        if len(sub) > 0:
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


# ── Panel A: Patient snapshot table ──────────────────────────────────────────

def draw_patient_table(ax, reps):
    ax.axis("off")

    display_cols = ["Age", "Sex", "LDL\n(mg/dL)", "ALT\n(U/L)", "Glucose\n(mg/dL)",
                    "HDL\n(mg/dL)", "BMI", "Subgroup Tier", "Expected\nResponse"]
    col_widths = [0.8, 0.6, 1.0, 0.9, 1.1, 1.0, 0.7, 2.2, 1.1]
    xs = np.cumsum([0] + col_widths)
    total_w = xs[-1]
    row_h = 0.38
    header_h = 0.48
    n = len(reps)

    # Header
    for i, (lbl, w) in enumerate(zip(display_cols, col_widths)):
        rect = plt.Rectangle((xs[i], n * row_h), w, header_h,
                              facecolor=PALETTE["header_bg"], edgecolor="white",
                              linewidth=0.6, clip_on=False, transform=ax.transData)
        ax.add_patch(rect)
        ax.text(xs[i] + w / 2, n * row_h + header_h / 2, lbl,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", clip_on=False)

    # Data rows
    for r, (_, row) in enumerate(reps.iterrows()):
        tier_info = assign_tier(row)
        tier_label, tier_color, exp_rr = tier_info
        y = (n - 1 - r) * row_h
        bg = PALETTE["row_alt"] if r % 2 == 0 else "white"

        rect = plt.Rectangle((0, y), total_w, row_h,
                              facecolor=bg, edgecolor=PALETTE["grid"],
                              linewidth=0.4, clip_on=False)
        ax.add_patch(rect)

        sex_str = "F" if row["RIAGENDR"] == 2 else "M"
        values = [
            f"{row['RIDAGEYR']:.0f}",
            sex_str,
            f"{row['LBDLDL']:.0f}",
            f"{row['LBXALT']:.0f}",
            f"{row['LBXGLU']:.0f}",
            f"{row['LBDHDD']:.0f}",
            f"{row['BMXBMI']:.1f}",
            tier_label,
            exp_rr,
        ]
        for i, (val, w) in enumerate(zip(values, col_widths)):
            is_tier = (i == 7)
            is_rr   = (i == 8)
            color = tier_color if (is_tier or is_rr) else PALETTE["navy"]
            fw    = "bold" if (is_tier or is_rr) else "normal"
            align = "left" if is_tier else "center"
            xpos  = xs[i] + (0.07 if align == "left" else w / 2)
            ax.text(xpos, y + row_h / 2, val,
                    ha=align, va="center", fontsize=8,
                    color=color, fontweight=fw, clip_on=False)

    # Border
    border = plt.Rectangle((0, 0), total_w, n * row_h + header_h,
                            fill=False, edgecolor=PALETTE["navy"],
                            linewidth=1.2, clip_on=False)
    ax.add_patch(border)

    ax.set_xlim(-0.1, total_w + 0.1)
    ax.set_ylim(-0.1, n * row_h + header_h + 0.1)
    ax.set_title("Representative Patient Profiles — One Per Subgroup Tier  (treated arm)",
                 fontsize=10, fontweight="bold", color=PALETTE["navy"], pad=8)


# ── Panel B: Biomarker distributions with threshold lines ────────────────────

def draw_distributions(axes, df):
    configs = [
        ("LBDLDL", "LDL Cholesterol (mg/dL)", 145, "≥145\nthreshold",
         PALETTE["green"],  (60, 210)),
        ("LBXALT", "ALT — Liver Enzyme (U/L)", 30,  "<30\nthreshold",
         PALETTE["teal"],   (5,  100)),
        ("LBXGLU", "Glucose (mg/dL)",          100, "<100\nthreshold",
         PALETTE["blue"],   (65, 175)),
    ]

    drug    = df[df["treatment_arm"] == 1]
    placebo = df[df["treatment_arm"] == 0]

    for ax, (col, xlabel, thresh, thresh_label, color, xlim) in zip(axes, configs):
        ax.set_facecolor(PALETTE["light_bg"])

        for arm, label, alpha, lw in [(placebo, "Placebo", 0.25, 1.4),
                                       (drug,    "Drug arm", 0.40, 2.0)]:
            vals = arm[col].dropna()
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
            kde  = gaussian_kde(vals, bw_method=0.18)
            xs   = np.linspace(xlim[0], xlim[1], 300)
            ys   = kde(xs)
            ax.fill_between(xs, ys, alpha=alpha, color=color)
            ax.plot(xs, ys, color=color, lw=lw, label=label)

        # Threshold line
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.05
        ax.axvline(thresh, color=PALETTE["red"], lw=1.8, linestyle="--", zorder=5)
        ax.text(thresh + (xlim[1] - xlim[0]) * 0.02, ymax * 0.88,
                thresh_label, fontsize=8, color=PALETTE["red"],
                va="top", linespacing=1.3)

        ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["navy"])
        ax.set_xlim(*xlim)
        ax.set_ylabel("Density", fontsize=8.5, color=PALETTE["navy"])
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        ax.tick_params(colors=PALETTE["navy"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["grid"])

    axes[0].legend(fontsize=8, framealpha=0.85, edgecolor=PALETTE["grid"])
    axes[1].set_title("Key Biomarker Distributions — Drug vs. Placebo Arm  (subgroup thresholds marked)",
                      fontsize=10, fontweight="bold", color=PALETTE["navy"], pad=8,
                      x=1.6)   # center across 3 axes via offset


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df   = pd.read_csv(DATA_PATH)
    reps = pick_representatives(df)

    fig  = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(PALETTE["light_bg"])

    gs   = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1.15, 1],
                             hspace=0.52, wspace=0.38)

    ax_table = fig.add_subplot(gs[0, :])   # full-width top
    ax_ldl   = fig.add_subplot(gs[1, 0])
    ax_alt   = fig.add_subplot(gs[1, 1])
    ax_glu   = fig.add_subplot(gs[1, 2])

    draw_patient_table(ax_table, reps)
    draw_distributions([ax_ldl, ax_alt, ax_glu], df)

    # Summary stats strip
    n_total   = len(df)
    n_treated = (df["treatment_arm"] == 1).sum()
    rr_drug   = df[df["treatment_arm"] == 1]["Responded"].mean()
    rr_pbo    = df[df["treatment_arm"] == 0]["Responded"].mean()
    n_feats   = df.shape[1] - 2   # exclude treatment + outcome

    fig.text(0.5, 0.98,
             f"N = {n_total:,} patients   |   "
             f"{n_feats} clinical & demographic features   |   "
             f"1:1 randomization   |   "
             f"Drug arm RR = {rr_drug:.1%}   |   "
             f"Placebo arm RR = {rr_pbo:.1%}   |   "
             f"ITT appears as failed trial",
             ha="center", va="top", fontsize=9.5,
             color="white",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor=PALETTE["header_bg"],
                       edgecolor="none"),
             transform=fig.transFigure)

    out = OUT_DIR / "data_snapshot.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["light_bg"])
    plt.close()
    print(f"Saved: {out}")
