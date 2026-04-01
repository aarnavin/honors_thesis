# Sublytics

Clinical Trial Subgroup Analytics Platform

## Problem

A **$2.5 billion** Phase 3 trial fails. The drug shows no significant benefit over placebo for the entire population. But what if it worked extremely well for a small, undiscovered subgroup (e.g., "females over 65 with high HDL")?

**This tool finds that subgroup.**

## How It Works

1. **Upload** trial data (NHANES clinical data)
2. **Train** ML model with hyperparameter tuning to predict responders
3. **Explain** using SHAP (SHapley Additive exPlanations) to identify which patient features matter most
4. **Discover** hidden subgroups with statistical significance testing (demographics-first)
5. **Get AI insights** using Claude with:
   - Honest assessment (will say "no signal" if evidence is weak)
   - False positive risk warnings (multiple testing, overfitting)
   - Stakeholder impact analysis (patients, pharma, regulators, payers, clinicians)

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Get Anthropic API Key (for AI explanations)

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up / log in
3. Create an API key
4. Copy it

### 3. Run the Web App

```bash
streamlit run src/03_app.py
```

Then:
- **Tab 1 (Data Upload)**: Upload your CSV **OR** use the sample NHANES data (4,000 patients)
- **Tab 2 (Model & Analysis)**: Click "Run Analysis" to train the model
- **Tab 3 (Results & Insights)**: View SHAP results, demographic patterns, and discovered subgroups
  - Enter your Anthropic API key in the sidebar
  - Click "Generate Explanation" to get brutally honest AI insights
- **Tab 4 (Adaptive Design Advisor)**: Paste your trial protocol to assess if adaptive design is more appropriate (preventative approach)

### 4. Or Run Analysis Directly (No UI)

```bash
python src/02_train_and_explain.py \
  --data data/processed/nhanes_trial_data.csv \
  --model-out models/model.pkl \
  --plots-dir outputs/plots \
  --report outputs/reports/summary.md
```

##  Data Format

Your CSV should have these columns:

### Required Columns:
- **Outcome**: `Responded` or `responded` (0/1 binary)
- **Features**: Any clinical/demographic variables

### Sample NHANES Columns:
- **Biomarkers**: `LBXGLU` (glucose), `LBDHDD` (HDL), `LBXTC` (cholesterol), `LBXTLG` (triglycerides), `LBDLDL` (LDL)
- **Demographics**: `RIAGENDR` (Male/Female), `RIDAGEYR` (age), `RIDRETH3` (race/ethnicity)
- **Optional**: `treatment_arm` (Drug/Placebo) - will be excluded from features
- **ID**: `SEQN` (patient ID) - will be excluded from analysis

### 🧪 Test with Synthetic Dataset (Strong Subgroup Signals)

Want to see the tool find REAL subgroups? We've included a synthetic dataset with ground truth responder subgroups:

```bash
python scripts/generate_synthetic_data.py
```

This creates `data/synthetic_trial_with_subgroups.csv` with:
- **Overall response rate**: ~23% (appears as "failed trial")
- **Hidden subgroup 1**: Females aged 60-80 with HDL > 60 → **62% response rate**
- **Hidden subgroup 2**: Males aged 40-60 with glucose < 100 → **58% response rate**
- **Hidden subgroup 3**: Young patients (<40) with high HDL + low triglycerides → **50% response rate**

Upload this CSV in the Streamlit app to see the model successfully discover these subgroups!

##  Features

### Core Analysis
- **Automated data preprocessing** (one-hot encoding, missing value handling)
- **Hyperparameter tuning** with GridSearchCV (optimizes AUC)
- **Multiple evaluation metrics** (AUC, F1, Accuracy, ROC curve)
- **SHAP explanations** (beeswarm plot + feature importance)
- **Subgroup discovery** with statistical testing (Chi-square, Bonferroni correction)
- **Quartile analysis** for continuous variables
- **Wilson confidence intervals** for proportions

### Web Interface
- **4 tabs for complete workflow**:
  1. **Data Upload**: Upload CSV or use sample NHANES data
  2. **Model & Analysis**: Train ML model with one click
  3. **Results & Insights**: SHAP plots, demographic breakdown, subgroups, AI analysis
  4. **Adaptive Design Advisor**: Preventative approach
- **Interactive visualizations** (ROC curve, SHAP plots, demographic breakdown)
- **Discovered subgroups table** (statistically significant groups)
- **Claude AI integration** with scientific rigor prompts
  - Explicit false positive warnings
  - "No signal" recommendations when appropriate
  - Comprehensive stakeholder impact analysis
- **Adaptive Design Assessment**: Paste trial protocol, get AI recommendation on whether to use adaptive design instead of post-hoc rescue
- **Export reports** (CSV, PNG, Markdown)

##  Tech Stack

- **Data**: Pandas, NumPy, SciPy
- **ML**: Scikit-learn, XGBoost, GridSearchCV
- **Explainability**: SHAP
- **Statistics**: SciPy stats (chi-square, Wilson CI)
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **AI**: Anthropic Claude Sonnet 4

##  What Makes This Special

### 1. **Subgroup Discovery with Statistical Rigor**
- Not just "show me correlations" → discovers actionable patient subgroups
- Bonferroni correction for multiple testing
- Wilson confidence intervals (better than simple proportions)
- Quartile analysis for continuous biomarkers
- **Demographics-first analysis**

### 2. **Explainable AI (SHAP)**
- Shows *why* the model predicts response
- Feature interactions (e.g., "older females with high HDL")
- Visual beeswarm plots for clinical understanding

### 3. **Brutally Honest AI Analysis**
- **Will explicitly state if there's NO rescuable subgroup** (not just optimistic findings)
- Comprehensive false positive risk assessment (multiple testing, overfitting, p-hacking)
- AUC-based signal strength interpretation (weak/modest/good/strong)
- Clear "abandon vs. rescue" recommendations based on evidence quality

### 4. **Stakeholder Impact Analysis**
Claude evaluates how findings affect:
- **Patients**: False hope vs. access; health equity implications
- **Pharma**: $100M+ pivot-or-abandon decisions; reputational risk
- **Regulators**: Safety vs. innovation trade-offs; validation requirements
- **Payers**: Coverage decisions; cost-effectiveness in narrow populations
- **Clinicians**: Biomarker testing infrastructure; liability considerations

### 5. **Production-Ready Pipeline**
- Works with real NHANES data (4,000 patients)
- Provided preprocessing notebook (handles categorical, missing data)
- Fast training (~30-60 seconds with optimized hyperparameter search)
- Scalable to multiple trials

##  Sample Output

After running the tool, you'll get:

### 1. **SHAP Summary Plot**
Visual breakdown of feature importance with directionality

### 2. **Top Features Table**
Ranked list of predictive variables with SHAP values

### 3. **Discovered Subgroups**
```
Subgroup                        N    Response_Rate  Risk_Increase  P_Value
LBXGLU ≥Q3 (156.00)            200       42.5%        +13.3%       0.0012
RIAGENDR = Female              420       35.7%         +6.5%       0.0234
RIDAGEYR ≥Q3 (65.00)           200       38.0%         +8.8%       0.0089
```

### 4. **Claude AI Report**
*"Analysis reveals females over 65 with HDL > 50 mg/dL show 3× higher response rates. This subgroup represents 15% of participants but accounts for 45% of responders. **Recommendation**: Phase 2b trial targeting this demographic. **Market**: ~2.1M US patients, $380M projected revenue."*

##  Workflow Example

```bash
# 1. Train model and discover subgroups
python src/02_train_and_explain.py \
  --data data/processed/nhanes_trial_data.csv

# Outputs:
# ✓ models/model.pkl (trained RF classifier)
# ✓ outputs/plots/roc_curve.png
# ✓ outputs/plots/shap_summary.png
# ✓ outputs/plots/shap_importance.png
# ✓ outputs/plots/subgroups.csv (discovered groups)
# ✓ outputs/reports/summary.md

# 2. View in web interface
streamlit run src/03_app.py
# → Upload data → Run Analysis → View results → Generate AI explanation
```
##  Model Performance (Sample Dataset)

- **Training**: 3,196 patients (80%)
- **Test**: 800 patients (20%)
- **Best CV AUC**: 0.521 (from GridSearch)
- **Test AUC**: 0.585
- **Discovered subgroups**: 10 (statistical testing applied)

##  Use Cases

1. **Pharma R&D**: Rescue failed Phase 3 trials by identifying responder subgroups
2. **Regulatory**: Generate FDA submissions with subgroup analyses
3. **Clinical Practice**: Personalized medicine - match patients to treatments
4. **Research**: Exploratory analysis of clinical datasets

##  Important Note

**These findings are exploratory and require prospective validation!**

This tool is designed for:
- Hypothesis generation
- Exploratory subgroup discovery
- Identifying promising patient profiles for follow-up studies

NOT for:
- Final clinical decisions without validation
- Replacing rigorous clinical trials
- Direct patient care without physician oversight

##  Resources

- [NHANES Data](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [FDA Guidance on Subgroup Analysis](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/enrichment-strategies-clinical-trials-support-determination-effectiveness-human-drugs-and-biological)
