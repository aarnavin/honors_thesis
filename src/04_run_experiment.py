#!/usr/bin/env python3
"""
Modular experiment runner for subgroup discovery comparison.

Usage examples:
  python src/04_run_experiment.py --model lr  --learner s   --data data/synthetic_trial_with_subgroups.csv
  python src/04_run_experiment.py --model rf  --learner t   --data data/synthetic_trial_with_subgroups.csv
  python src/04_run_experiment.py --model xgb --learner x   --data data/synthetic_trial_with_subgroups.csv
  python src/04_run_experiment.py --model rf  --learner dr  --data data/synthetic_trial_with_subgroups.csv
  python src/04_run_experiment.py --model xgb --learner dr  --data data/synthetic_trial_with_subgroups.csv --note "tuned XGB"

Each run appends one row to:   outputs/reports/experiment_results.csv
After every run it regenerates: outputs/reports/results_table.tex
Drop the table into the thesis with: \\input{outputs/reports/results_table.tex}

Models   : lr (Logistic Regression)  |  rf (Random Forest)  |  xgb (XGBoost)
Learners : s (S-Learner)  |  t (T-Learner)  |  x (X-Learner)  |  dr (DR-Learner)

Meta-learner implementations are intentionally from scratch (scikit-learn only),
using the chosen base model.  This avoids the econml dependency while remaining
fully compatible with the formulae cited in Künzel et al. (2019) PNAS.
"""

import argparse
import csv
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_CSV = Path("outputs/reports/experiment_results.csv")
RESULTS_TEX = Path("outputs/reports/results_table.tex")

# Column order in the results CSV
RESULT_COLS = [
    "run_id", "timestamp", "model", "learner", "n_train", "n_test",
    "auc", "f1", "accuracy", "precision", "recall",
    "mean_cate_treated", "pehe",
    "best_subgroup_rr", "best_subgroup_n", "best_subgroup_label",
    "n_significant_subgroups",
    "note",
]

# ──────────────────────────────────────────────────────────────────────────────
# Base model factories
# ──────────────────────────────────────────────────────────────────────────────

def make_classifier(model_type: str, seed: int = 42):
    """Return a scikit-learn-compatible classifier."""
    if model_type == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed,
                                  class_weight="balanced", C=1.0)
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=20,
                                      min_samples_leaf=4, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=6,
                                 learning_rate=0.05, subsample=0.8,
                                 scale_pos_weight=3,
                                 eval_metric="logloss",
                                 random_state=seed, verbosity=0)
        except ImportError:
            raise SystemExit("xgboost not installed. Run: pip install xgboost")
    raise ValueError(f"Unknown model '{model_type}'. Choose: lr | rf | xgb")


def make_regressor(model_type: str, seed: int = 42):
    """Return a regressor for continuous pseudo-outcome regression in meta-learners."""
    if model_type == "lr":
        return Ridge(alpha=1.0)
    if model_type == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=20,
                                     min_samples_leaf=4,
                                     random_state=seed, n_jobs=-1)
    if model_type == "xgb":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300, max_depth=6,
                                learning_rate=0.05, subsample=0.8,
                                random_state=seed, verbosity=0)
        except ImportError:
            raise SystemExit("xgboost not installed.")
    raise ValueError(f"Unknown model '{model_type}'.")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str, seed: int = 42):
    """
    Load trial CSV, encode categoricals, return train/test splits.
    Keeps 'treatment_arm' separate; drops SEQN.
    """
    df = pd.read_csv(data_path)

    if "SEQN" in df.columns:
        df = df.drop(columns=["SEQN"])

    # Resolve outcome column name
    if "Responded" in df.columns:
        target = "Responded"
    elif "responded" in df.columns:
        target = "responded"
    else:
        raise ValueError("No 'Responded' column found.")

    # Resolve treatment column
    if "treatment_arm" in df.columns:
        treat_raw = df["treatment_arm"]
        # Normalise to 0/1
        if treat_raw.dtype == object:
            treatment = (treat_raw.str.lower().str.strip() == "drug").astype(int)
        else:
            treatment = treat_raw.astype(int)
        df = df.drop(columns=["treatment_arm"])
    else:
        treatment = None

    y = df[target]
    X = df.drop(columns=[target])

    # Encode categoricals
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

    # Scale for LR
    X_arr = X.values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    T = treatment.values if treatment is not None else np.zeros(len(y), dtype=int)

    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X_scaled, y, T, test_size=0.20, random_state=seed, stratify=y
    )

    # Also keep unscaled version for ground-truth CATE lookup (original feature values)
    _, X_raw_test = train_test_split(
        X, test_size=0.20, random_state=seed, stratify=y
    )

    return (X_train, X_test, y_train, y_test,
            T_train, T_test, X_raw_test)


# ──────────────────────────────────────────────────────────────────────────────
# Predictive model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_predictive(model_type: str, X_train, y_train, X_test, y_test,
                     T_train=None, T_test=None, seed=42):
    """
    Train classifier and return metrics.
    When T_train/T_test are provided, treatment_arm is appended as a feature —
    this lets the model learn the interaction between biomarkers and treatment
    assignment, which is necessary for meaningful AUC in a mixed trial population.
    """
    if T_train is not None:
        X_tr = X_train.copy()
        X_te = X_test.copy()
        X_tr["treatment_arm"] = T_train.values if hasattr(T_train, "values") else T_train
        X_te["treatment_arm"] = T_test.values if hasattr(T_test, "values") else T_test
    else:
        X_tr, X_te = X_train, X_test

    clf = make_classifier(model_type, seed)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    metrics = {
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth CATE (known because data is synthetic)
# ──────────────────────────────────────────────────────────────────────────────

def compute_true_cate(X_raw: pd.DataFrame) -> np.ndarray:
    """
    Assign ground-truth CATE values based on atorvastatin subgroup hierarchy.
    Only possible because this is synthetic data with known structure.
    Placebo baseline = 0.04; treatment baseline (non-responder) = 0.05.
    CATE = treatment response prob - placebo response prob (0.04).
    """
    true_cate = np.full(len(X_raw), 0.05 - 0.04)  # default: 0.01

    cols = X_raw.columns.tolist()
    has_ldl  = "LBDLDL"  in cols
    has_alt  = "LBXALT"  in cols
    has_glu  = "LBXGLU"  in cols
    has_gen  = "RIAGENDR" in cols
    has_age  = "RIDAGEYR" in cols

    if not has_ldl:
        return true_cate  # Can't assign subgroups without LDL

    ldl = X_raw["LBDLDL"].values
    alt = X_raw["LBXALT"].values  if has_alt else np.full(len(X_raw), 999)
    glu = X_raw["LBXGLU"].values  if has_glu else np.full(len(X_raw), 999)
    gen = X_raw["RIAGENDR"].values if has_gen else np.zeros(len(X_raw))
    age = X_raw["RIDAGEYR"].values if has_age else np.zeros(len(X_raw))

    # Assign from lowest to highest priority so the best subgroup wins
    m_ldl        = ldl >= 145
    m_triple     = m_ldl & (alt < 30) & (glu < 100)
    m_ldl_alt    = m_ldl & (alt < 30) & (glu >= 100)
    m_ldl_glu    = m_ldl & (glu < 100) & (alt >= 30)
    m_postmeno   = m_ldl & (gen == 2) & (age >= 50) & (age <= 75) & (alt >= 30) & (glu >= 100)
    m_ldl_only   = m_ldl & (alt >= 30) & (glu >= 100) & ((gen != 2) | (age < 50) | (age > 75))

    true_cate[m_ldl_only]  = 0.55 - 0.04   # 0.51
    true_cate[m_postmeno]  = 0.68 - 0.04   # 0.64
    true_cate[m_ldl_glu]   = 0.73 - 0.04   # 0.69
    true_cate[m_ldl_alt]   = 0.78 - 0.04   # 0.74
    true_cate[m_triple]    = 0.88 - 0.04   # 0.84

    return true_cate


# ──────────────────────────────────────────────────────────────────────────────
# Meta-learner CATE estimators
# (Implemented from scratch per Künzel et al. 2019 PNAS)
# ──────────────────────────────────────────────────────────────────────────────

def cate_s_learner(model_type, X_train, y_train, T_train, X_test, seed=42):
    """
    S-Learner: train one model on (X, T).
    CATE(x) = f(x, 1) – f(x, 0)
    """
    # Append treatment indicator as a feature
    X_train_s = np.column_stack([X_train.values, T_train])
    X_test_1  = np.column_stack([X_test.values,  np.ones(len(X_test))])
    X_test_0  = np.column_stack([X_test.values,  np.zeros(len(X_test))])

    clf = make_classifier(model_type, seed)
    clf.fit(X_train_s, y_train)

    mu1 = clf.predict_proba(X_test_1)[:, 1]
    mu0 = clf.predict_proba(X_test_0)[:, 1]
    return mu1 - mu0


def cate_t_learner(model_type, X_train, y_train, T_train, X_test, seed=42):
    """
    T-Learner: two separate outcome models.
    CATE(x) = mu_1(x) – mu_0(x)
    """
    mask_t = T_train == 1
    mask_c = T_train == 0

    clf1 = make_classifier(model_type, seed)
    clf0 = make_classifier(model_type, seed)

    clf1.fit(X_train[mask_t], y_train[mask_t])
    clf0.fit(X_train[mask_c], y_train[mask_c])

    mu1 = clf1.predict_proba(X_test.values)[:, 1]
    mu0 = clf0.predict_proba(X_test.values)[:, 1]
    return mu1 - mu0


def cate_x_learner(model_type, X_train, y_train, T_train, X_test, seed=42):
    """
    X-Learner (Künzel et al. 2019):
    Stage 1: fit mu_0, mu_1 (T-Learner outcome models)
    Stage 2: impute D_0 = mu_1(X_c) – Y_c; D_1 = Y_t – mu_0(X_t)
    Stage 3: fit tau_0 on (X_c, D_0); fit tau_1 on (X_t, D_1)
    Final: CATE(x) = e(x)*tau_0(x) + (1–e(x))*tau_1(x)
      where e(x) = 0.5 in an RCT (equal randomisation)
    """
    mask_t = T_train == 1
    mask_c = T_train == 0

    X_t = X_train[mask_t].values
    X_c = X_train[mask_c].values
    y_t = y_train[mask_t].values
    y_c = y_train[mask_c].values

    # Stage 1
    clf1 = make_classifier(model_type, seed)
    clf0 = make_classifier(model_type, seed)
    clf1.fit(X_t, y_t)
    clf0.fit(X_c, y_c)

    # Stage 2 pseudo-outcomes
    D1 = y_t - clf0.predict_proba(X_t)[:, 1]   # treated imputed effect
    D0 = clf1.predict_proba(X_c)[:, 1] - y_c    # control imputed effect

    # Stage 3 CATE models
    tau1 = make_regressor(model_type, seed)
    tau0 = make_regressor(model_type, seed)
    tau1.fit(X_t, D1)
    tau0.fit(X_c, D0)

    # Combine (e=0.5 for balanced RCT)
    cate = 0.5 * tau0.predict(X_test.values) + 0.5 * tau1.predict(X_test.values)
    return cate


def cate_dr_learner(model_type, X_train, y_train, T_train, X_test, seed=42):
    """
    Doubly Robust (DR) Learner:
    1. Estimate outcome models mu_0, mu_1 and propensity e(x)
    2. Compute pseudo-outcome:
         Y_DR = mu_1(X) – mu_0(X)
                + T/e(X) * (Y – mu_1(X))
                – (1–T)/(1–e(X)) * (Y – mu_0(X))
    3. Regress Y_DR on X to get CATE
    """
    mask_t = T_train == 1
    mask_c = T_train == 0

    X_tr = X_train.values
    y_tr = y_train.values
    T_tr = T_train

    # Outcome models (fitted on respective arms)
    clf1 = make_classifier(model_type, seed)
    clf0 = make_classifier(model_type, seed)
    clf1.fit(X_tr[mask_t], y_tr[mask_t])
    clf0.fit(X_tr[mask_c], y_tr[mask_c])

    mu1_tr = clf1.predict_proba(X_tr)[:, 1]
    mu0_tr = clf0.predict_proba(X_tr)[:, 1]

    # Propensity model (e = 0.5 in balanced RCT; estimate for robustness)
    prop_clf = LogisticRegression(max_iter=500, random_state=seed)
    prop_clf.fit(X_tr, T_tr)
    e_tr = prop_clf.predict_proba(X_tr)[:, 1].clip(0.05, 0.95)

    # Pseudo-outcomes
    pseudo = ((mu1_tr - mu0_tr)
              + (T_tr / e_tr) * (y_tr - mu1_tr)
              - ((1 - T_tr) / (1 - e_tr)) * (y_tr - mu0_tr))

    # Stage 2: regress pseudo on X
    reg = make_regressor(model_type, seed)
    reg.fit(X_tr, pseudo)

    return reg.predict(X_test.values)


# ──────────────────────────────────────────────────────────────────────────────
# Subgroup discovery from CATE estimates
# ──────────────────────────────────────────────────────────────────────────────

def discover_subgroups_from_cate(cate: np.ndarray, y_test, T_test,
                                  min_n: int = 30):
    """
    Identify patients with estimated CATE > threshold as a candidate subgroup.
    Report observed response rate in that subgroup (treatment arm only).
    Also applies Wilson CI and chi-square test.
    """
    results = []
    overall_rr = y_test[T_test == 1].mean() if (T_test == 1).sum() > 0 else y_test.mean()
    overall_n  = int((T_test == 1).sum())

    # Varying CATE thresholds
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25]:
        mask_high_cate = (cate >= thresh) & (T_test == 1)
        sub_n = mask_high_cate.sum()
        if sub_n < min_n:
            continue

        sub_y = y_test[mask_high_cate]
        rr    = sub_y.mean()
        n_resp = int(sub_y.sum())

        # Wilson CI
        p = rr
        z = 1.96
        denom = 1 + z**2 / sub_n
        centre = (p + z**2 / (2*sub_n)) / denom
        margin = z * np.sqrt(p*(1-p)/sub_n + z**2/(4*sub_n**2)) / denom
        ci_lo  = max(0, centre - margin)
        ci_hi  = min(1, centre + margin)

        # Chi-square vs overall
        cont  = [[n_resp, sub_n - n_resp],
                 [int(overall_rr * overall_n),
                  overall_n - int(overall_rr * overall_n)]]
        try:
            _, pval, _, _ = stats.chi2_contingency(cont)
        except Exception:
            pval = 1.0

        results.append({
            "label":   f"CATE>={thresh:.2f}",
            "n":       int(sub_n),
            "rr":      round(rr, 4),
            "ci":      f"[{ci_lo:.2f}, {ci_hi:.2f}]",
            "delta_rr":round(rr - overall_rr, 4),
            "pval":    round(pval, 4),
        })

    # Sort by response rate
    results.sort(key=lambda r: r["rr"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Results logging
# ──────────────────────────────────────────────────────────────────────────────

def _next_run_id() -> int:
    if not RESULTS_CSV.exists():
        return 1
    df = pd.read_csv(RESULTS_CSV)
    if df.empty or "run_id" not in df.columns:
        return 1
    return int(df["run_id"].max()) + 1


def log_result(row: dict):
    """Append a result row to the CSV, creating file and header if needed."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLS})
    print(f"\n  Logged -> {RESULTS_CSV}")


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_latex_table():
    """
    Read experiment_results.csv and write a publication-quality LaTeX table.
    Designed to be dropped directly into the thesis with \\input{}.
    """
    if not RESULTS_CSV.exists():
        print("  No results CSV found - skipping LaTeX generation.")
        return

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        return

    model_labels  = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
    learner_labels = {"s": "S-Learner", "t": "T-Learner", "x": "X-Learner", "dr": "DR-Learner"}

    lines = []
    lines.append(r"% Auto-generated by src/04_run_experiment.py - do not edit manually")
    lines.append(r"% Include in thesis with: \input{outputs/reports/results_table.tex}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of model configurations for subgroup discovery.")
    lines.append(r"  AUC and F1 measure predictive performance on test set.")
    lines.append(r"  PEHE (Precision in Estimation of Heterogeneous Effects) measures")
    lines.append(r"  causal accuracy against known ground-truth CATEs.")
    lines.append(r"  Best Subgroup RR is the observed response rate in the highest-CATE")
    lines.append(r"  subgroup (treatment arm only).}")
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Meta-Learner & AUC & F1 & PEHE & "
                 r"Best RR & N (best) & Sig.\ Subgroups \\")
    lines.append(r"\midrule")

    best_auc  = df["auc"].max()
    best_f1   = df["f1"].max()
    best_pehe = df["pehe"].min() if df["pehe"].notna().any() else None

    for _, row in df.iterrows():
        mdl_lbl = model_labels.get(str(row["model"]).lower(), str(row["model"]))
        lrn_lbl = learner_labels.get(str(row["learner"]).lower(), str(row["learner"]))

        def fmt(val, best_val=None, higher_is_better=True, precision=3):
            """Format a value; bold if it matches the best."""
            if pd.isna(val) or val == "":
                return "--"
            fval = f"{float(val):.{precision}f}"
            if best_val is not None:
                if higher_is_better and float(val) >= float(best_val) - 1e-6:
                    return r"\textbf{" + fval + r"}"
                if not higher_is_better and float(val) <= float(best_val) + 1e-6:
                    return r"\textbf{" + fval + r"}"
            return fval

        auc_s   = fmt(row.get("auc"),  best_auc,  True)
        f1_s    = fmt(row.get("f1"),   best_f1,   True)
        pehe_s  = fmt(row.get("pehe"), best_pehe, False)
        rr_s    = fmt(row.get("best_subgroup_rr"), None, True)
        n_s     = str(int(row["best_subgroup_n"])) if pd.notna(row.get("best_subgroup_n")) else "--"
        sig_s   = str(int(row["n_significant_subgroups"])) if pd.notna(row.get("n_significant_subgroups")) else "--"

        lines.append(f"{mdl_lbl} & {lrn_lbl} & {auc_s} & {f1_s} & "
                     f"{pehe_s} & {rr_s} & {n_s} & {sig_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    RESULTS_TEX.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX table -> {RESULTS_TEX}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 62)
    print(f"  Experiment: model={args.model.upper()}  learner={args.learner.upper()}")
    print(f"  Data:  {args.data}")
    print("=" * 62)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    (X_train, X_test, y_train, y_test,
     T_train, T_test, X_raw_test) = load_data(args.data, args.seed)

    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Treatment arm (test): {T_test.sum()} treated / {(T_test==0).sum()} control")

    # ── 2. Predictive model ───────────────────────────────────────
    print(f"\n[2/5] Training predictive model ({args.model.upper()})...")
    pred_metrics = train_predictive(
        args.model, X_train, y_train, X_test, y_test,
        T_train=T_train, T_test=T_test, seed=args.seed
    )
    print(f"  AUC={pred_metrics['auc']:.3f}  "
          f"F1={pred_metrics['f1']:.3f}  "
          f"Acc={pred_metrics['accuracy']:.3f}")

    # ── 3. CATE estimation ────────────────────────────────────────
    print(f"\n[3/5] Estimating CATE ({args.learner.upper()}-Learner)...")
    learner = args.learner.lower()

    if learner == "s":
        cate_est = cate_s_learner(args.model, X_train, y_train, T_train,
                                   X_test, args.seed)
    elif learner == "t":
        cate_est = cate_t_learner(args.model, X_train, y_train, T_train,
                                   X_test, args.seed)
    elif learner == "x":
        cate_est = cate_x_learner(args.model, X_train, y_train, T_train,
                                   X_test, args.seed)
    elif learner == "dr":
        cate_est = cate_dr_learner(args.model, X_train, y_train, T_train,
                                    X_test, args.seed)
    else:
        raise ValueError(f"Unknown learner '{args.learner}'. Choose: s | t | x | dr")

    mean_cate_treated = float(np.mean(cate_est[T_test == 1]))
    print(f"  Mean estimated CATE (treated): {mean_cate_treated:.4f}")

    # ── 4. PEHE against ground truth ─────────────────────────────
    print(f"\n[4/5] Computing PEHE against known ground truth...")
    true_cate = compute_true_cate(X_raw_test)
    pehe = float(np.sqrt(np.mean((cate_est - true_cate) ** 2)))
    print(f"  PEHE = {pehe:.4f}  (lower is better)")
    print(f"  True CATE range: [{true_cate.min():.3f}, {true_cate.max():.3f}]")
    print(f"  Est. CATE range: [{cate_est.min():.3f}, {cate_est.max():.3f}]")

    # ── 5. Subgroup discovery ─────────────────────────────────────
    print(f"\n[5/5] Discovering subgroups from CATE estimates...")
    subgroups = discover_subgroups_from_cate(
        cate_est, y_test.values, T_test, min_n=30
    )

    best_rr = best_n = best_label = ""
    n_sig = 0
    bonferroni_thresh = 0.05 / max(len(subgroups), 1)

    for sg in subgroups:
        print(f"  {sg['label']:18s}  N={sg['n']:4d}  "
              f"RR={sg['rr']:.3f}  dRR={sg['delta_rr']:+.3f}  "
              f"p={sg['pval']:.3f}")
        if sg["pval"] < bonferroni_thresh:
            n_sig += 1

    if subgroups:
        best = subgroups[0]
        best_rr    = best["rr"]
        best_n     = best["n"]
        best_label = best["label"]

    # ── Log result ────────────────────────────────────────────────
    run_id = _next_run_id()
    row = {
        "run_id":                  run_id,
        "timestamp":               datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model":                   args.model.lower(),
        "learner":                 args.learner.lower(),
        "n_train":                 len(X_train),
        "n_test":                  len(X_test),
        **pred_metrics,
        "mean_cate_treated":       round(mean_cate_treated, 4),
        "pehe":                    round(pehe, 4),
        "best_subgroup_rr":        round(float(best_rr), 4) if best_rr != "" else "",
        "best_subgroup_n":         int(best_n) if best_n != "" else "",
        "best_subgroup_label":     best_label,
        "n_significant_subgroups": n_sig,
        "note":                    args.note,
    }
    log_result(row)
    generate_latex_table()

    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  AUC   = {pred_metrics['auc']:.3f}")
    print(f"  F1    = {pred_metrics['f1']:.3f}")
    print(f"  PEHE  = {pehe:.4f}")
    if best_rr:
        print(f"  Best subgroup RR = {float(best_rr):.1%}  (N={best_n})")
    print(f"  Bonferroni-significant subgroups: {n_sig}")
    print(f"\n  Run #{run_id} saved -> {RESULTS_CSV}")
    print(f"  LaTeX  table  ready -> {RESULTS_TEX}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one model/learner experiment and log results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/04_run_experiment.py --model lr  --learner s  --data data/synthetic_trial_with_subgroups.csv
  python src/04_run_experiment.py --model rf  --learner t  --data data/synthetic_trial_with_subgroups.csv
  python src/04_run_experiment.py --model xgb --learner dr --data data/synthetic_trial_with_subgroups.csv --note "high LDL subset"
        """
    )
    parser.add_argument("--model",  required=True, choices=["lr", "rf", "xgb"],
                        help="Base model: lr | rf | xgb")
    parser.add_argument("--learner", required=True, choices=["s", "t", "x", "dr"],
                        help="Meta-learner: s | t | x | dr")
    parser.add_argument("--data",   required=True,
                        help="Path to trial CSV (e.g. data/synthetic_trial_with_subgroups.csv)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--note",   default="",
                        help="Optional note for the run log (e.g. 'tuned XGB')")

    args = parser.parse_args()
    main(args)
