"""
Generate a presentation-quality results table figure.
Saves to outputs/plots/step3_results_table.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_CSV = "outputs/reports/experiment_results.csv"
OUT_DIR     = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy":      "#1A2E4A",
    "blue":      "#1E88E5",
    "green":     "#3BB273",
    "red":       "#E84855",
    "gold":      "#F4A261",
    "purple":    "#7B2D8B",
    "light_bg":  "#F8F9FA",
    "header_bg": "#1A2E4A",
    "row_alt":   "#EEF2F7",
    "grid":      "#D0D7E3",
    "muted":     "#B0BEC5",
    "best_bg":   "#E8F5E9",
}

MODEL_LABELS   = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
LEARNER_LABELS = {"s": "S-Learner", "t": "T-Learner", "x": "X-Learner", "dr": "DR-Learner"}

# Column definitions: (csv_key, display_header, format_fn, higher_is_better)
COLUMNS = [
    ("model",             "Model",          lambda v: MODEL_LABELS.get(str(v), v),   None),
    ("learner",           "Meta-Learner",   lambda v: LEARNER_LABELS.get(str(v), v), None),
    ("auc",               "AUC",            lambda v: f"{float(v):.3f}",             True),
    ("f1",                "F1",             lambda v: f"{float(v):.3f}",             True),
    ("mean_cate_treated", "Mean CATE",      lambda v: f"{float(v):.3f}",             True),
    ("pehe",              "PEHE",           lambda v: f"{float(v):.3f}",             False),
    ("best_subgroup_rr",  "Best RR",        lambda v: f"{float(v):.1%}",             True),
    ("n_significant_subgroups", "Sig. Subgroups", lambda v: str(int(v)),             True),
]


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values("pehe").reset_index(drop=True)
    return df


def best_in_col(df, key, higher_is_better):
    if higher_is_better is None:
        return None
    vals = pd.to_numeric(df[key], errors="coerce")
    return vals.max() if higher_is_better else vals.min()


if __name__ == "__main__":
    df = load_data()
    n_rows = len(df)
    n_cols = len(COLUMNS)

    col_widths = [2.2, 1.8, 0.9, 0.9, 1.1, 0.9, 1.0, 1.3]
    fig_w = sum(col_widths) + 0.4
    row_h = 0.52
    header_h = 0.65
    fig_h = header_h + n_rows * row_h + 1.1   # extra for title + footer

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(PALETTE["light_bg"])
    ax.set_facecolor(PALETTE["light_bg"])
    ax.axis("off")

    # cumulative x positions
    xs = np.cumsum([0] + col_widths)
    total_w = xs[-1]
    y_header = n_rows * row_h
    y_top    = y_header + header_h

    # ── Header row ────────────────────────────────────────────────────────────
    header_labels = [c[1] for c in COLUMNS]
    for i, (label, w) in enumerate(zip(header_labels, col_widths)):
        rect = plt.Rectangle((xs[i], y_header), w, header_h,
                              facecolor=PALETTE["header_bg"], edgecolor="white",
                              linewidth=0.8, clip_on=False)
        ax.add_patch(rect)
        ax.text(xs[i] + w / 2, y_header + header_h / 2, label,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", clip_on=False)

    # ── Data rows ─────────────────────────────────────────────────────────────
    best_vals = {c[0]: best_in_col(df, c[0], c[3]) for c in COLUMNS}

    # Best row = row 0 (sorted by PEHE ascending → best PEHE first)
    best_row_idx = 0

    for r, (_, row) in enumerate(df.iterrows()):
        y = (n_rows - 1 - r) * row_h
        is_best = (r == best_row_idx)
        bg = PALETTE["best_bg"] if is_best else (PALETTE["row_alt"] if r % 2 == 0 else "white")

        # Row background
        rect = plt.Rectangle((0, y), total_w, row_h,
                              facecolor=bg, edgecolor=PALETTE["grid"],
                              linewidth=0.5, clip_on=False)
        ax.add_patch(rect)

        # Best-row star marker in left margin
        if is_best:
            ax.text(-0.18, y + row_h / 2, "★",
                    ha="center", va="center", fontsize=11,
                    color=PALETTE["gold"], clip_on=False)

        for i, (key, _, fmt, hib) in enumerate(COLUMNS):
            val = row.get(key, "")
            text = fmt(val) if val != "" else "—"

            # Determine if this cell holds the column best
            is_col_best = False
            if hib is not None and best_vals[key] is not None:
                try:
                    fv = float(pd.to_numeric(val, errors="coerce"))
                    bv = float(best_vals[key])
                    is_col_best = abs(fv - bv) < 1e-5
                except Exception:
                    pass

            # Text style
            fw    = "bold" if is_col_best else "normal"
            color = PALETTE["blue"] if is_col_best else PALETTE["navy"]

            # Special color coding
            if key == "pehe":
                try:
                    fv = float(val)
                    color = PALETTE["green"]  if fv <= 0.09  else \
                            PALETTE["gold"]   if fv <= 0.15  else \
                            PALETTE["red"]
                    if is_col_best:
                        fw = "bold"
                except Exception:
                    pass

            if key == "auc":
                try:
                    fv = float(val)
                    color = PALETTE["green"] if fv >= 0.87 else PALETTE["navy"]
                    if is_col_best: fw = "bold"
                except Exception:
                    pass

            align = "left" if i < 2 else "center"
            xpos  = xs[i] + (0.10 if align == "left" else col_widths[i] / 2)

            ax.text(xpos, y + row_h / 2, text,
                    ha=align, va="center", fontsize=9,
                    color=color, fontweight=fw, clip_on=False)

    # ── Outer border ──────────────────────────────────────────────────────────
    border = plt.Rectangle((0, 0), total_w, y_top,
                            fill=False, edgecolor=PALETTE["navy"],
                            linewidth=1.4, clip_on=False)
    ax.add_patch(border)

    # Vertical column dividers
    for x in xs[1:-1]:
        ax.plot([x, x], [0, y_top], color=PALETTE["grid"],
                linewidth=0.6, clip_on=False)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(total_w / 2, y_top + 0.38,
            "Model Comparison — Subgroup Discovery Pipeline",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
            color=PALETTE["navy"], clip_on=False)
    ax.text(total_w / 2, y_top + 0.12,
            "Sorted by PEHE (ascending) · Best PEHE row highlighted in green · ★ = best overall",
            ha="center", va="bottom", fontsize=8.5,
            color=PALETTE["muted"], clip_on=False)

    # ── Footer legend ─────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=PALETTE["green"],  label="PEHE ≤ 0.09  (high causal accuracy)"),
        mpatches.Patch(color=PALETTE["gold"],   label="PEHE 0.09–0.15  (moderate)"),
        mpatches.Patch(color=PALETTE["red"],    label="PEHE > 0.15  (low causal accuracy)"),
        mpatches.Patch(color=PALETTE["blue"],   label="Column best value"),
        mpatches.Patch(color=PALETTE["best_bg"],label="Best overall row (lowest PEHE)", ec=PALETTE["grid"]),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(total_w / 2, -0.68), ncol=3,
              fontsize=8, framealpha=0.9, edgecolor=PALETTE["grid"],
              bbox_transform=ax.transData)

    ax.set_xlim(-0.3, total_w + 0.1)
    ax.set_ylim(-0.72, y_top + 0.55)

    plt.tight_layout(pad=0.3)
    out = OUT_DIR / "step3_results_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=PALETTE["light_bg"])
    plt.close()
    print(f"Saved: {out}")
