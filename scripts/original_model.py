import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve)
import shap
import matplotlib.pyplot as plt
from scipy import stats


DATA_FILE = 'sample_data_with_outcome.csv'  # Change to your data file path
OUTCOME_COL = 'Responded'  # ← Change to your outcome column name
EXCLUDE_COLS = []  # Empty list = use ALL columns as features (except outcome)

data = pd.read_csv(DATA_FILE)

print("CLINICAL TRIAL SUBGROUP DISCOVERY")

# PREPARE FEATURES AND OUTCOME
print("\n Data Overview")
print(f"Total samples: {len(data)}")
print(f"\nOutcome distribution:")
print(data[OUTCOME_COL].value_counts())
print(f"\nResponse rate: {data[OUTCOME_COL].mean():.1%}")

# Separate features and outcome
all_exclude = EXCLUDE_COLS + [OUTCOME_COL]
feature_cols = [col for col in data.columns if col not in all_exclude]

X = data[feature_cols]
y = data[OUTCOME_COL]

print(f"\nFeatures: {len(feature_cols)}")

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Data Split")
print(f"Training: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# TRAIN MODEL WITH HYPERPARAMETER TUNING
print(f"\n Training Model...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

print(f"Best CV AUC: {grid_search.best_score_:.3f}")
print(f"Best params: {grid_search.best_params_}")

# EVALUATE MODEL
print(f"\n Model Performance")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Test AUC-ROC: {auc_score:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {auc_score:.3f}')
plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# SHAP ANALYSIS
print(f"\n🔍 Computing SHAP Values...")

# Sample background for efficiency
background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(model, background)
shap_values = explainer.shap_values(X_test)

# For binary classification, use positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("SHAP values computed")

# SHAP Summary Plots
print("\n Generating SHAP Visualizations...")

# 1. Feature Importance (Bar)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
plt.title("SHAP Feature Importance", fontweight='bold')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Feature Impact Distribution (Beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.title("SHAP Feature Impact Distribution", fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Dependence Plots for Top 3 Features
feature_importance = np.abs(shap_values).mean(axis=0)
top_3_idx = np.argsort(feature_importance)[-3:][::-1]
feature_names = X_test.columns.tolist()

for idx in top_3_idx:
    feat = feature_names[idx]
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Dependence: {feat}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{feat}.png', dpi=300, bbox_inches='tight')
    plt.show()

print("✓ SHAP plots saved")


# DISCOVER SUBGROUPS

print(f"\n Discovering Responder Subgroups...")

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

# Get top 5 features
top_5_idx = np.argsort(feature_importance)[-5:][::-1]
top_5_features = [feature_names[i] for i in top_5_idx]

print(f"\nTop 5 features:")
for i, feat in enumerate(top_5_features, 1):
    print(f"  {i}. {feat}")

# Prepare test data with outcomes
test_full = X_test.copy()
test_full['outcome'] = y_test.values

overall_rr = y_test.mean()
overall_n = len(y_test)

print(f"\nOverall response rate: {overall_rr:.1%} (N={overall_n})")

# Find subgroups
subgroups = []
min_size = 30

for feature in top_5_features:
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

# Results
if subgroups:
    df_sub = pd.DataFrame(subgroups)
    
    # Bonferroni correction
    n_tests = len(df_sub)
    df_sub['P_Bonferroni'] = (df_sub['pval_num'] * n_tests).clip(upper=1.0)
    df_sub['Significant'] = df_sub['P_Bonferroni'] < 0.05
    
    # Sort by effect size
    df_sub = df_sub.sort_values('risk_inc_num', ascending=False)
    
    # Display
    cols = ['Subgroup', 'N', 'Responders', 'Response_Rate', 'CI_95', 
            'Risk_Increase', 'Relative_Risk', 'P_Value', 'Significant']
    
    print("\n" + "="*70)
    print("DISCOVERED SUBGROUPS (Ranked by Effect Size)")
    print("="*70)
    print(f"Tests performed: {n_tests} | Bonferroni correction applied\n")
    print(df_sub[cols].head(15).to_string(index=False))
    
    # Promising subgroups
    promising = df_sub[(df_sub['Significant']) & (df_sub['risk_inc_num'] > 0.10)]
    
    if not promising.empty:
        print("\n" + "="*70)
        print("⭐ MOST PROMISING SUBGROUPS")
        print("="*70)
        print("(Significant after Bonferroni + Risk increase > 10%)\n")
        print(promising[cols].to_string(index=False))
    
    # Save
    df_sub[cols].to_csv('subgroups.csv', index=False)
    print("\n✓ Saved to 'subgroups.csv'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  • roc_curve.png")
print("  • shap_importance.png")
print("  • shap_summary.png")
print("  • shap_dependence_*.png")
print("  • subgroups.csv")
print("\n These findings are exploratory - require prospective validation!")
print("="*70)