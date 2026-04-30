# Sublytics Analysis Report

## What This Model Predicts

**Target**: Treatment Response (1 = Responded, 0 = Did Not Respond)

**Question**: Which patient characteristics predict positive response to treatment?

## Model Performance

- **AUC**: 0.568 (ability to distinguish responders from non-responders)
- **F1 Score**: 0.173
- **Accuracy**: 0.689
- **Overall Response Rate**: 41.3%

## Top 10 Predictive Features

- **LBXGLU**: 0.0251
- **LBXSATSI**: 0.0153
- **INDFMPIR**: 0.0134
- **RIDAGEYR**: 0.0117
- **LBXSAL**: 0.0104
- **LBDLDL**: 0.0097
- **LBXTST**: 0.0089
- **LBXSAPSI**: 0.0086
- **LBXTLG**: 0.0084
- **LBXSASSI**: 0.0082

## Demographic Response Patterns

- **RIDAGEYR = 80.0**: 26.1% (N=23, -15.2% vs overall)

## Discovered Subgroups

**Most Promising Subgroups** (>5% response rate increase):

- **RIDAGEYR <=Q1 (31.00)**: 46.9% (N=98, +5.6%, RR=1.14) [p=0.3737]

⚠️ **Note**: None passed strict Bonferroni correction (p < 0.0063). However, effect sizes are substantial and warrant further investigation.

## CRITICAL LIMITATIONS & FALSE POSITIVE RISKS

**This is exploratory, post-hoc analysis. Key risks:**

1. **Multiple Testing**: Testing 8 subgroups inflates false positive rate
2. **Overfitting**: Model may find patterns in noise, not true signal
3. **Small Sample Bias**: Subgroups with N<50 are especially unreliable
4. **No Causal Inference**: Associations != causation

**AUC Interpretation:**
- AUC 0.568 is WEAK - barely better than random (0.5)
- **Recommendation**: No compelling subgroup signal. Consider abandoning rescue attempt.

**Next Steps:**
- Independent validation cohort REQUIRED
- Prospective randomized trial in identified subgroup
- Biological mechanism investigation
- Stakeholder impact assessment (patients, regulators, payers)

## Generated Files

- ROC curve: `outputs/plots/roc_curve.png`
- SHAP summary: `outputs/plots/shap_summary.png`
- SHAP importance: `outputs/plots/shap_importance.png`
- Subgroups: `outputs/plots/subgroups.csv`
- Demographics: `outputs/plots/demographic_breakdown.csv`

---
*These findings are exploratory and hypothesis-generating only. Do NOT use for clinical decisions without prospective validation.*
