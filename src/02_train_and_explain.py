#!/usr/bin/env python3
"""
Train ML model and generate SHAP explanations for clinical trial data.
Adapted from model.py - Enhanced with subgroup discovery
Person B - ML Engineer
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def wilson_ci(successes, n, confidence=0.95):
    """Wilson score confidence interval for proportions."""
    if n == 0:
        return 0, 0
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def main(args):
    print("Loading Loading data...")
    data = pd.read_csv(args.data)
    
    print(f"   Columns: {data.columns.tolist()}")
    
    # DATA OVERVIEW
    print(f"\n✓ Data Overview")
    print(f"   Total samples: {len(data)}")
    
    # Drop ID column if present
    if 'SEQN' in data.columns:
        data = data.drop(columns=['SEQN'])
    
    # Handle the target variable (check both naming conventions)
    target_col = None
    if 'Responded' in data.columns:
        target_col = 'Responded'
    elif 'responded' in data.columns:
        target_col = 'responded'
    else:
        raise ValueError("No 'Responded' or 'responded' column found in dataset")
    
    print(f"   Outcome distribution: {data[target_col].value_counts().to_dict()}")
    print(f"   Response rate: {data[target_col].mean():.1%}")
    
    # Prepare features
    cols_to_drop = [target_col]
    if 'treatment_arm' in data.columns:
        cols_to_drop.append('treatment_arm')
    
    feature_cols = [col for col in data.columns if col not in cols_to_drop]
    
    X = data[feature_cols].copy()
    y = data[target_col]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"   Encoding categorical: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Keep only numeric
    X = X.select_dtypes(include=[np.number])
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"   Features: {len(X.columns)}")
    
    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Data Split")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Feature selection - keep top features
    if X_train.shape[1] > 15:
        print(f"   Selecting top features...")
        selector = SelectKBest(f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        selected_features = X_train.columns[selector.get_support()].tolist()
        X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        print(f"   Selected {len(selected_features)} features")
    
    # TRAIN MODEL WITH HYPERPARAMETER TUNING
    print(f"\nTraining Model with optimized hyperparameters...")
    
    param_distributions = {
        'n_estimators': [300, 500],
        'max_depth': [25, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, bootstrap=True),
        param_distributions,
        n_iter=8,  # Fast: only 8 combinations
        cv=3,  # Reduced from 5 folds
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    print(f"   Best CV AUC: {random_search.best_score_:.3f}")
    print(f"   Best params: {random_search.best_params_}")
    
    # EVALUATE MODEL
    print(f"\n✓ Model Performance")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Test AUC: {auc:.3f}")
    print(f"   F1: {f1:.3f}")
    print(f"   Accuracy: {acc:.3f}")
    
    # Save model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': model,
        'features': X.columns.tolist(),
        'metrics': {'auc': auc, 'f1': f1, 'accuracy': acc}
    }, args.model_out)
    print(f"\nSaved Model saved to {args.model_out}")
    
    # Create output directories
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    print(f"\nGenerating Generating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.plots_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP ANALYSIS
    print(f"\nComputing Computing SHAP values...")
    
    # Sample background for efficiency
    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]  # Take positive class
    
    print(f"   SHAP values computed: {shap_values.shape}")
    
    # SHAP Summary Plot (Beeswarm)
    print(f"   Generating SHAP visualizations...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)
    plt.title("SHAP Feature Impact Distribution", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{args.plots_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP Feature Importance (Bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
    plt.title("SHAP Feature Importance", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{args.plots_dir}/shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_names = X_test.columns.tolist()
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'abs_mean': feature_importance
    }).sort_values('abs_mean', ascending=False)
    
    # Save top features
    feature_importance_df.to_csv(f"{args.plots_dir}/top_features.csv", index=False)
    print(f"   ✓ Top features saved")
    
    # DISCOVER SUBGROUPS (DEMOGRAPHIC-FOCUSED)
    print(f"\nDiscovering Discovering Responder Subgroups...")
    
    # CRITICAL: Filter to treatment arm only if available
    # (Subgroups should respond to DRUG, not just random variation)
    test_full = X_test.copy()
    test_full['outcome'] = y_test.values
    
    # Check if we have treatment arm info in original data
    if 'treatment_arm' in data.columns:
        # Get treatment arm for test indices
        test_treatment = data.loc[X_test.index, 'treatment_arm']
        test_full['treatment_arm'] = test_treatment.values
        
        # Filter to treatment arm only (value = 1 for numeric, 'Drug' for string)
        if test_treatment.dtype == 'object':
            mask_treatment = test_full['treatment_arm'] == 'Drug'
        else:
            mask_treatment = test_full['treatment_arm'] == 1
        
        test_full = test_full[mask_treatment].copy()
        print(f"   🎯 Analyzing TREATMENT ARM ONLY (N={len(test_full)})")
        print(f"   (Subgroups should respond to drug, not placebo)")
    
    # Prioritize demographic features
    demographic_features = [f for f in feature_names if any(x in f for x in ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'age', 'gender', 'sex', 'race', 'ethnicity'])]
    
    # Get top biomarkers too
    top_biomarker_idx = np.argsort(feature_importance)[-5:][::-1]
    top_biomarkers = [feature_names[i] for i in top_biomarker_idx if feature_names[i] not in demographic_features][:3]
    
    # Combine: demographics first, then top biomarkers
    analysis_features = demographic_features + top_biomarkers
    
    print(f"   Analyzing features: {analysis_features}")
    
    overall_rr = test_full['outcome'].mean()
    overall_n = len(test_full)
    
    print(f"   Overall response rate: {overall_rr:.1%} (N={overall_n})")
    
    # Find subgroups
    subgroups = []
    min_size = 30
    
    for feature in analysis_features:
        n_unique = test_full[feature].nunique()
        
        if n_unique <= 10:  # Categorical
            for val in test_full[feature].unique():
                mask = test_full[feature] == val
                sub = test_full[mask]
                
                if len(sub) >= min_size:
                    n = len(sub)
                    resp = sub['outcome'].sum()
                    rr = resp / n
                    
                    # Chi-square test
                    cont = [[resp, n - resp], 
                           [int(overall_rr * overall_n), overall_n - int(overall_rr * overall_n)]]
                    _, pval, _, _ = stats.chi2_contingency(cont)
                    
                    risk_inc = rr - overall_rr
                    rel_risk = rr / overall_rr if overall_rr > 0 else np.inf
                    ci_l, ci_h = wilson_ci(int(resp), n)
                    
                    subgroups.append({
                        'Subgroup': f"{feature} = {val}",
                        'N': n,
                        'Responders': int(resp),
                        'Response_Rate': f"{rr:.1%}",
                        'CI_95': f"[{ci_l:.1%}, {ci_h:.1%}]",
                        'Risk_Increase': f"{risk_inc:+.1%}",
                        'Relative_Risk': f"{rel_risk:.2f}",
                        'P_Value': f"{pval:.4f}",
                        'risk_inc_num': risk_inc,
                        'pval_num': pval
                    })
        
        else:  # Numeric - quartile analysis
            q75 = test_full[feature].quantile(0.75)
            q25 = test_full[feature].quantile(0.25)
            
            for label, mask in [(f"≥Q3 ({q75:.2f})", test_full[feature] >= q75),
                               (f"≤Q1 ({q25:.2f})", test_full[feature] <= q25)]:
                sub = test_full[mask]
                
                if len(sub) >= min_size:
                    n = len(sub)
                    resp = sub['outcome'].sum()
                    rr = resp / n
                    
                    cont = [[resp, n - resp],
                           [int(overall_rr * overall_n), overall_n - int(overall_rr * overall_n)]]
                    _, pval, _, _ = stats.chi2_contingency(cont)
                    
                    risk_inc = rr - overall_rr
                    rel_risk = rr / overall_rr if overall_rr > 0 else np.inf
                    ci_l, ci_h = wilson_ci(int(resp), n)
                    
                    subgroups.append({
                        'Subgroup': f"{feature} {label}",
                        'N': n,
                        'Responders': int(resp),
                        'Response_Rate': f"{rr:.1%}",
                        'CI_95': f"[{ci_l:.1%}, {ci_h:.1%}]",
                        'Risk_Increase': f"{risk_inc:+.1%}",
                        'Relative_Risk': f"{rel_risk:.2f}",
                        'P_Value': f"{pval:.4f}",
                        'risk_inc_num': risk_inc,
                        'pval_num': pval
                    })
    
    # MULTI-FEATURE COMBINATIONS - Find 2-way and 3-way interactions
    print(f"\n   Searching for multi-feature combinations...")
    
    # Get demographic + top biomarkers for combinations
    gender_feat = next((f for f in demographic_features if 'RIAGENDR' in f or 'gender' in f.lower()), None)
    age_feat = next((f for f in demographic_features if 'RIDAGEYR' in f or 'age' in f.lower()), None)
    
    # Define age bins
    age_bins = []
    if age_feat and age_feat in test_full.columns:
        age_bins = [
            ('Young (<40)', test_full[age_feat] < 40),
            ('Middle-aged (40-60)', (test_full[age_feat] >= 40) & (test_full[age_feat] <= 60)),
            ('Older (60-80)', (test_full[age_feat] >= 60) & (test_full[age_feat] <= 80)),
        ]
    
    # Gender values
    gender_vals = []
    if gender_feat and gender_feat in test_full.columns:
        gender_vals = [(f'Gender={int(v)}', test_full[gender_feat] == v) 
                       for v in test_full[gender_feat].unique()]
    
    # Top biomarkers - discretize into high/low
    biomarker_conditions = []
    for bio_feat in top_biomarkers[:3]:  # Top 3 biomarkers
        if bio_feat in test_full.columns:
            q75 = test_full[bio_feat].quantile(0.75)
            q25 = test_full[bio_feat].quantile(0.25)
            median = test_full[bio_feat].median()
            biomarker_conditions.append((f'{bio_feat}>Q3({q75:.1f})', test_full[bio_feat] >= q75))
            biomarker_conditions.append((f'{bio_feat}<Q1({q25:.1f})', test_full[bio_feat] <= q25))
            biomarker_conditions.append((f'{bio_feat}>median({median:.1f})', test_full[bio_feat] > median))
    
    # 2-way combinations: Gender + Age
    if gender_vals and age_bins:
        for gender_label, gender_mask in gender_vals:
            for age_label, age_mask in age_bins:
                mask = gender_mask & age_mask
                sub = test_full[mask]
                
                if len(sub) >= min_size:
                    n = len(sub)
                    resp = sub['outcome'].sum()
                    rr = resp / n
                    
                    cont = [[resp, n - resp],
                           [int(overall_rr * overall_n), overall_n - int(overall_rr * overall_n)]]
                    _, pval, _, _ = stats.chi2_contingency(cont)
                    
                    risk_inc = rr - overall_rr
                    rel_risk = rr / overall_rr if overall_rr > 0 else np.inf
                    ci_l, ci_h = wilson_ci(int(resp), n)
                    
                    subgroups.append({
                        'Subgroup': f"{gender_label} + {age_label}",
                        'N': n,
                        'Responders': int(resp),
                        'Response_Rate': f"{rr:.1%}",
                        'CI_95': f"[{ci_l:.1%}, {ci_h:.1%}]",
                        'Risk_Increase': f"{risk_inc:+.1%}",
                        'Relative_Risk': f"{rel_risk:.2f}",
                        'P_Value': f"{pval:.4f}",
                        'risk_inc_num': risk_inc,
                        'pval_num': pval
                    })
    
    # 3-way combinations: Gender + Age + Biomarker
    if gender_vals and age_bins and biomarker_conditions:
        for gender_label, gender_mask in gender_vals:
            for age_label, age_mask in age_bins:
                for bio_label, bio_mask in biomarker_conditions[:6]:  # Limit to prevent explosion
                    mask = gender_mask & age_mask & bio_mask
                    sub = test_full[mask]
                    
                    if len(sub) >= min_size:
                        n = len(sub)
                        resp = sub['outcome'].sum()
                        rr = resp / n
                        
                        cont = [[resp, n - resp],
                               [int(overall_rr * overall_n), overall_n - int(overall_rr * overall_n)]]
                        _, pval, _, _ = stats.chi2_contingency(cont)
                        
                        risk_inc = rr - overall_rr
                        rel_risk = rr / overall_rr if overall_rr > 0 else np.inf
                        ci_l, ci_h = wilson_ci(int(resp), n)
                        
                        subgroups.append({
                            'Subgroup': f"{gender_label} + {age_label} + {bio_label}",
                            'N': n,
                            'Responders': int(resp),
                            'Response_Rate': f"{rr:.1%}",
                            'CI_95': f"[{ci_l:.1%}, {ci_h:.1%}]",
                            'Risk_Increase': f"{risk_inc:+.1%}",
                            'Relative_Risk': f"{rel_risk:.2f}",
                            'P_Value': f"{pval:.4f}",
                            'risk_inc_num': risk_inc,
                            'pval_num': pval
                        })
    
    print(f"   ✓ Found {len(subgroups)} total subgroups (single + combinations)")
    
    # Save subgroups
    if subgroups:
        df_sub = pd.DataFrame(subgroups)
        
        # Bonferroni correction
        n_tests = len(df_sub)
        df_sub['P_Bonferroni'] = (df_sub['pval_num'] * n_tests).clip(upper=1.0)
        df_sub['Significant'] = df_sub['P_Bonferroni'] < 0.05
        
        # Sort by effect size
        df_sub = df_sub.sort_values('risk_inc_num', ascending=False)
        
        # Save
        cols = ['Subgroup', 'N', 'Responders', 'Response_Rate', 'CI_95', 
                'Risk_Increase', 'Relative_Risk', 'P_Value', 'Significant']
        df_sub[cols].to_csv(f'{args.plots_dir}/subgroups.csv', index=False)
        print(f"   ✓ Subgroups saved ({len(df_sub)} discovered)")
    
    # CREATE DEMOGRAPHIC BREAKDOWN
    print(f"\nAnalyzing Demographics...")
    demographic_breakdown = []
    
    for demo_feat in demographic_features:
        if demo_feat in test_full.columns:
            for val in test_full[demo_feat].unique():
                mask = test_full[demo_feat] == val
                n = mask.sum()
                if n >= 20:
                    rr = test_full.loc[mask, 'outcome'].mean()
                    demographic_breakdown.append({
                        'Demographic': demo_feat,
                        'Value': val,
                        'N': n,
                        'Response_Rate': f"{rr:.1%}",
                        'Risk_vs_Overall': f"{(rr - overall_rr):.1%}"
                    })
    
    demographic_df = pd.DataFrame(demographic_breakdown)
    if len(demographic_df) > 0:
        demographic_df.to_csv(f"{args.plots_dir}/demographic_breakdown.csv", index=False)
        print(f"   ✓ Demographic breakdown saved")
    
    # GENERATE REPORT
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w') as f:
        f.write("# Sublytics Analysis Report\n\n")
        f.write(f"## What This Model Predicts\n\n")
        f.write(f"**Target**: Treatment Response (1 = Responded, 0 = Did Not Respond)\n\n")
        f.write(f"**Question**: Which patient characteristics predict positive response to treatment?\n\n")
        f.write(f"## Model Performance\n\n")
        f.write(f"- **AUC**: {auc:.3f} (ability to distinguish responders from non-responders)\n")
        f.write(f"- **F1 Score**: {f1:.3f}\n")
        f.write(f"- **Accuracy**: {acc:.3f}\n")
        f.write(f"- **Overall Response Rate**: {overall_rr:.1%}\n\n")
        f.write(f"## Top 10 Predictive Features\n\n")
        for idx, row in feature_importance_df.head(10).iterrows():
            f.write(f"- **{row['feature']}**: {row['abs_mean']:.4f}\n")
        
        if len(demographic_df) > 0:
            f.write(f"\n## Demographic Response Patterns\n\n")
            for idx, row in demographic_df.head(15).iterrows():
                f.write(f"- **{row['Demographic']} = {row['Value']}**: {row['Response_Rate']} (N={row['N']}, {row['Risk_vs_Overall']} vs overall)\n")
        
        if subgroups:
            f.write(f"\n## Discovered Subgroups\n\n")
            
            # Show subgroups with large effect sizes (even if not Bonferroni-significant)
            promising = df_sub[df_sub['risk_inc_num'] > 0.05]  # >5% increase
            significant = df_sub[df_sub['Significant']]
            
            if not promising.empty:
                f.write("**🎯 Most Promising Subgroups** (>5% response rate increase):\n\n")
                for idx, row in promising.head(15).iterrows():
                    sig_marker = "✓ SIGNIFICANT" if row['Significant'] else f"p={row['P_Value']}"
                    f.write(f"- **{row['Subgroup']}**: {row['Response_Rate']} ")
                    f.write(f"(N={row['N']}, {row['Risk_Increase']}, RR={row['Relative_Risk']}) [{sig_marker}]\n")
                
                if significant.empty:
                    f.write(f"\n⚠️ **Note**: None passed strict Bonferroni correction (p < {0.05/len(df_sub):.4f}). ")
                    f.write(f"However, effect sizes are substantial and warrant further investigation.\n")
            else:
                f.write("No subgroups with >5% risk increase found.\n")
        
        f.write(f"\n## ⚠️ CRITICAL LIMITATIONS & FALSE POSITIVE RISKS\n\n")
        f.write(f"**This is exploratory, post-hoc analysis. Key risks:**\n\n")
        f.write(f"1. **Multiple Testing**: Testing {len(df_sub) if subgroups else 0} subgroups inflates false positive rate\n")
        f.write(f"2. **Overfitting**: Model may find patterns in noise, not true signal\n")
        f.write(f"3. **Small Sample Bias**: Subgroups with N<50 are especially unreliable\n")
        f.write(f"4. **No Causal Inference**: Associations ≠ causation\n\n")
        f.write(f"**AUC Interpretation:**\n")
        if auc < 0.6:
            f.write(f"- AUC {auc:.3f} is WEAK - barely better than random (0.5)\n")
            f.write(f"- **Recommendation**: No compelling subgroup signal. Consider abandoning rescue attempt.\n")
        elif auc < 0.7:
            f.write(f"- AUC {auc:.3f} is MODEST - some predictive value but high uncertainty\n")
            f.write(f"- **Recommendation**: Proceed with extreme caution. Require independent validation.\n")
        elif auc < 0.8:
            f.write(f"- AUC {auc:.3f} is GOOD - meaningful predictive signal\n")
            f.write(f"- **Recommendation**: Potential subgroup worth investigating in prospective trial.\n")
        else:
            f.write(f"- AUC {auc:.3f} is STRONG - high predictive accuracy\n")
            f.write(f"- **Caution**: Unusually high AUC may indicate overfitting. Validate externally.\n")
        f.write(f"\n**Next Steps:**\n")
        f.write(f"- Independent validation cohort REQUIRED\n")
        f.write(f"- Prospective randomized trial in identified subgroup\n")
        f.write(f"- Biological mechanism investigation\n")
        f.write(f"- Stakeholder impact assessment (patients, regulators, payers)\n\n")
        f.write(f"## Generated Files\n\n")
        f.write(f"- ROC curve: `{args.plots_dir}/roc_curve.png`\n")
        f.write(f"- SHAP summary: `{args.plots_dir}/shap_summary.png`\n")
        f.write(f"- SHAP importance: `{args.plots_dir}/shap_importance.png`\n")
        f.write(f"- Subgroups: `{args.plots_dir}/subgroups.csv`\n")
        if len(demographic_df) > 0:
            f.write(f"- Demographics: `{args.plots_dir}/demographic_breakdown.csv`\n")
        f.write(f"\n---\n*These findings are exploratory and hypothesis-generating only. Do NOT use for clinical decisions without prospective validation.*\n")
    
    print(f"\nDone Report saved to {args.report}")
    print(f"\nComplete Analysis complete!")
    print(f"\nLoading Generated files:")
    print(f"   • {args.plots_dir}/roc_curve.png")
    print(f"   • {args.plots_dir}/shap_summary.png")
    print(f"   • {args.plots_dir}/shap_importance.png")
    print(f"   • {args.plots_dir}/top_features.csv")
    if subgroups:
        print(f"   • {args.plots_dir}/subgroups.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model, explain with SHAP, discover subgroups")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--model-out", default="models/model.pkl", help="Path to save model")
    parser.add_argument("--plots-dir", default="outputs/plots", help="Directory for plots")
    parser.add_argument("--report", default="outputs/reports/summary.md", help="Path to report")
    
    args = parser.parse_args()
    main(args)
