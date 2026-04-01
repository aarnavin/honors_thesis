import streamlit as st
import pandas as pd
import joblib
import subprocess
import os
from pathlib import Path
import anthropic
import json

# Page config
st.set_page_config(
    page_title="Sublytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner, modern UI
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* AI Box */
    .ai-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .ai-content {
        background-color: white;
        color: #1e293b;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Reduce padding in streamlit elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Sublytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clinical Trial Subgroup Analytics Platform</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Claude API key input
    anthropic_api_key = st.text_input(
        "🔑 Claude API Key",
        type="password",
        help="Get your key at console.anthropic.com"
    )
    
    with st.expander("Advanced Options", expanded=False):
        model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost"], index=0)
        n_estimators = st.slider("Number of Trees", 50, 500, 300, 50)
        test_size = st.slider("Test Split Percentage", 10, 40, 20, 5)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Upload", "🤖 Model & Analysis", "📈 Results & Insights", "🔄 Adaptive Design Advisor"])

# Tab 1: Data Upload
with tab1:
    st.subheader("📁 Upload Trial Data")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
    use_sample = st.checkbox("Use sample NHANES dataset (3,996 patients)", value=True)
    
    # Use the main dataset
    data_path = "data/processed/nhanes_trial_data.csv"
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.to_csv(data_path, index=False)
        st.session_state['data_loaded'] = True
        st.session_state['data_path'] = data_path
    elif use_sample and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.session_state['data_loaded'] = True
        st.session_state['data_path'] = data_path
    else:
        st.session_state['data_loaded'] = False
    
    if st.session_state.get('data_loaded', False):
        df = pd.read_csv(st.session_state['data_path'])
        
        # Stats in nice cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total Patients", len(df))
        with col2:
            # Handle both naming conventions
            resp_col = 'Responded' if 'Responded' in df.columns else 'responded'
            if resp_col in df.columns:
                responders = df[resp_col].sum()
                st.metric("Responders", f"{responders} ({(responders/len(df)*100):.0f}%)")
        with col3:
            if 'treatment_arm' in df.columns:
                # Handle both string ('Drug') and numeric (1) formats
                if df['treatment_arm'].dtype == 'object':
                    treated = (df['treatment_arm'] == 'Drug').sum()
                else:
                    treated = (df['treatment_arm'] == 1).sum()
                st.metric("💊 Treatment Group", f"{treated} ({(treated/len(df)*100):.0f}%)")
            elif 'treatment' in df.columns:
                treated = df['treatment'].sum()
                st.metric("💊 Treatment Group", f"{treated} ({(treated/len(df)*100):.0f}%)")
        
        # Collapsible preview
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Please select sample data or upload a CSV file to begin.")

# Tab 2: Model & Analysis
with tab2:
    st.subheader("Train Model & Analyze")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please upload data first.")
    else:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Training model and computing SHAP values..."):
                try:
                    result = subprocess.run(
                        [
                            "python", 
                            "src/02_train_and_explain.py",
                            "--data", st.session_state['data_path'],
                            "--model-out", "models/model.pkl",
                            "--plots-dir", "outputs/plots",
                            "--report", "outputs/reports/summary.md"
                        ],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    st.session_state['model_trained'] = True
                    st.success("Analysis complete. View results in the Results & Insights tab.")
                    st.balloons()
                    
                except subprocess.CalledProcessError as e:
                    st.error(f"Error: {e.stderr}")
                    st.session_state['model_trained'] = False
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state['model_trained'] = False
        
        if st.session_state.get('model_trained', False):
            st.success("Model trained successfully. Proceed to Results & Insights tab.")

# Tab 3: Results & Insights
with tab3:
    if not st.session_state.get('model_trained', False):
        st.warning("Please train the model first.")
    else:
        # What the model predicts
        st.info("**What this model predicts:** Treatment Response (1 = Responded, 0 = Did Not Respond). The AUC measures how well the model can distinguish responders from non-responders (0.5 = random guessing, 1.0 = perfect separation).")
        
        # Model performance
        st.subheader("Model Performance")
        if os.path.exists("outputs/reports/summary.md"):
            with open("outputs/reports/summary.md") as f:
                report_content = f.read()
                lines = report_content.split('\n')
                
                col1, col2, col3 = st.columns(3)
                for line in lines:
                    if "AUC:" in line or "**AUC**" in line:
                        auc_val = line.split(":")[-1].strip().split()[0]
                        col1.metric("AUC-ROC", auc_val, help="Area Under ROC Curve")
                    if "F1:" in line or "**F1 Score**" in line:
                        f1_val = line.split(":")[-1].strip().split()[0]
                        col2.metric("F1 Score", f1_val)
                    if "Accuracy:" in line or "**Accuracy**" in line:
                        acc_val = line.split(":")[-1].strip().split()[0]
                        col3.metric("Accuracy", acc_val)
        
        st.divider()
        
        # ROC Curve
        st.subheader("Model Discrimination")
        roc_plot_path = "outputs/plots/roc_curve.png"
        if os.path.exists(roc_plot_path):
            st.image(roc_plot_path, use_container_width=True)
        
        st.divider()
        
        # Demographic Breakdown
        demographic_path = "outputs/plots/demographic_breakdown.csv"
        if os.path.exists(demographic_path):
            st.subheader("Demographic Response Patterns")
            demo_df = pd.read_csv(demographic_path)
            if len(demo_df) > 0:
                st.dataframe(
                    demo_df.head(15),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No demographic patterns available.")
        
        st.divider()
        
        # SHAP plots
        st.subheader("Feature Importance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**SHAP Impact Distribution**")
            shap_plot_path = "outputs/plots/shap_summary.png"
            if os.path.exists(shap_plot_path):
                st.image(shap_plot_path, use_container_width=True)
        
        with col2:
            st.markdown("**SHAP Feature Importance**")
            shap_importance_path = "outputs/plots/shap_importance.png"
            if os.path.exists(shap_importance_path):
                st.image(shap_importance_path, use_container_width=True)
        
        st.divider()
        
        # Claude AI
        st.subheader("AI-Generated Clinical Insights")
        
        st.warning("⚠️ **Scientific Rigor Notice**: This analysis prioritizes honesty over optimism. The AI will explicitly state if there's no rescuable subgroup, highlight false pattern risks, and discuss stakeholder impacts including potential harms.")
        
        with st.expander("📋 Why Stakeholder Analysis Matters"):
            st.markdown("""
            **Subgroup analysis affects multiple parties with competing interests:**
            
            - **Patients**: False hope vs. access to potentially life-saving treatment
            - **Pharma**: $100M+ decision to pursue or abandon program  
            - **Regulators**: Patient safety vs. innovation access
            - **Payers**: Coverage decisions affect millions
            - **Clinicians**: Liability for treatment selection
            
            **Key Ethical Concerns:**
            - Health equity if subgroups are demographic (age, race, gender)
            - Off-label use if approved for narrow indication
            - Biomarker testing infrastructure and cost
            - Risk of pharmaceutical companies "gaming" post-hoc analyses
            
            This is why the AI analysis will be brutally honest about signal quality.
            """)
        
        top_features_path = "outputs/plots/top_features.csv"
        
        if anthropic_api_key:
            if st.button("Generate Explanation", type="primary", use_container_width=True):
                with st.spinner("Generating AI analysis..."):
                    try:
                        df = pd.read_csv(st.session_state['data_path'])
                        top_features_df = pd.read_csv(top_features_path)
                        
                        with open("outputs/reports/summary.md") as f:
                            report = f.read()
                        
                        # Get response count
                        resp_col = 'Responded' if 'Responded' in df.columns else 'responded'
                        resp_count = df[resp_col].sum()
                        
                        prompt = f"""You are a clinical biostatistician and trial expert. Analyze this failed Phase 3 trial with EXTREME SCIENTIFIC RIGOR.

**Trial Summary:**
- Total patients: {len(df)}
- Responders: {resp_count} ({(resp_count/len(df)*100):.1f}%)

**Top Predictive Features (SHAP Analysis):**
{top_features_df.head(8).to_string(index=False)}

**Full Report:**
{report[:1000]}

**Biomarker Key:**
- LBXGLU = Glucose, LBDHDD = HDL cholesterol, LBXTC = Total cholesterol
- LBXTLG = Triglycerides, LBDLDL = LDL cholesterol
- RIDAGEYR = Age, RIAGENDR = Gender, RIDRETH3 = Race/Ethnicity

**CRITICAL - AVOIDING FALSE PATTERNS:**
This is exploratory, post-hoc subgroup analysis with HIGH risk of:
1. **Multiple testing**: Testing many subgroups inflates false positive rate
2. **Overfitting**: Finding patterns in noise, not signal
3. **P-hacking**: Cherry-picking significant results
4. **Regression to the mean**: Extreme values in small subgroups

**Your analysis MUST:**
- Be skeptical by default. If the signal looks weak or inconsistent, SAY SO.
- If there's no rescuable subgroup, explicitly state: "This trial shows no compelling evidence of a responder subgroup."
- Only identify subgroups with STRONG statistical support (Bonferroni-corrected p<0.05, large effect sizes, biological plausibility)
- Acknowledge uncertainty and call out red flags (small N, weak p-values, implausible biology)

**STAKEHOLDER IMPACT ANALYSIS:**
Discuss how your findings affect:

1. **Patients**: 
   - Those who might benefit if subgroup is real
   - Those who might be harmed by false hope or unnecessary treatment
   - Health equity implications if subgroup is demographic

2. **Pharmaceutical Sponsor**:
   - Financial decision (pivot vs. abandon program)
   - Regulatory strategy for potential follow-on trial
   - Reputational risk of failed rescue attempt

3. **Regulators (FDA/EMA)**:
   - Burden of proof for subgroup claims
   - Need for prospective validation
   - Patient safety vs. access trade-offs

4. **Payers/Insurers**:
   - Coverage decisions if drug approved for subgroup
   - Cost-effectiveness in narrow population
   - Potential for off-label use

5. **Clinicians**:
   - Clinical decision-making complexity
   - Need for biomarker testing infrastructure
   - Liability if subgroup misidentified

**Your Deliverable:**
1. **Signal Strength Assessment** (weak/moderate/strong) - be honest
2. **Responder Profile** (if any exists) - demographics, biomarkers, thresholds
3. **Red Flags & Limitations** - what could be wrong with this analysis
4. **Recommendation** - rescue attempt OR abandon (with rationale)
5. **Stakeholder Considerations** - key impacts and ethical concerns

Use clear sections with bullet points. Prioritize scientific integrity over optimism."""

                        client = anthropic.Anthropic(api_key=anthropic_api_key)
                        message = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=3000,  # Increased for comprehensive stakeholder analysis
                            messages=[{"role": "user", "content": prompt}]
                        )
                        
                        explanation = message.content[0].text
                        
                        # Save it
                        Path("outputs/reports").mkdir(parents=True, exist_ok=True)
                        with open("outputs/reports/claude_explanation.md", "w") as f:
                            f.write(explanation)
                        
                        # Display in pretty box
                        st.markdown(f'<div class="ai-box"><h3>Analysis Results</h3><div class="ai-content">{explanation}</div></div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Enter your Claude API key in the sidebar to enable AI-generated insights.")
        
        # Show previous explanation if exists
        claude_path = "outputs/reports/claude_explanation.md"
        if os.path.exists(claude_path):
            with st.expander("View Previous Analysis"):
                with open(claude_path) as f:
                    st.markdown(f.read())
        
        st.divider()
        
        # Discovered Subgroups
        st.subheader("Discovered Responder Subgroups")
        
        subgroups_path = "outputs/plots/subgroups.csv"
        if os.path.exists(subgroups_path):
            subgroups_df = pd.read_csv(subgroups_path)
            
            # Check if we have the columns we need
            if 'Significant' in subgroups_df.columns:
                # Show only significant ones
                promising = subgroups_df[subgroups_df['Significant'] == True]
            else:
                promising = subgroups_df
            
            if len(promising) > 0:
                st.success(f"Found {len(promising)} statistically significant subgroups.")
                
                # Display top 5 with available columns
                display_cols = ['Subgroup', 'N', 'Response_Rate', 'Risk_Increase', 'P_Value']
                available_cols = [col for col in display_cols if col in promising.columns]
                
                st.dataframe(
                    promising.head(5)[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                with st.expander("View All Subgroups"):
                    st.dataframe(subgroups_df, use_container_width=True)
            else:
                st.info("No statistically significant subgroups found.")
        
        st.divider()
        
        # Export buttons
        st.subheader("Download Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if os.path.exists(top_features_path):
            with col1:
                with open(top_features_path, 'rb') as f:
                    st.download_button("Features CSV", f, "features.csv", use_container_width=True)
        
        if os.path.exists(subgroups_path):
            with col2:
                with open(subgroups_path, 'rb') as f:
                    st.download_button("Subgroups CSV", f, "subgroups.csv", use_container_width=True)
        
        if os.path.exists(demographic_path):
            with col3:
                with open(demographic_path, 'rb') as f:
                    st.download_button("Demographics CSV", f, "demographics.csv", use_container_width=True)
        
        if os.path.exists(claude_path):
            with col4:
                with open(claude_path, 'rb') as f:
                    st.download_button("AI Report", f, "report.md", use_container_width=True)
        
        st.markdown("##### Visualizations")
        col4, col5, col6 = st.columns(3)
        
        if os.path.exists(roc_plot_path):
            with col4:
                with open(roc_plot_path, 'rb') as f:
                    st.download_button("ROC Curve", f, "roc_curve.png", use_container_width=True)
        
        if os.path.exists(shap_plot_path):
            with col5:
                with open(shap_plot_path, 'rb') as f:
                    st.download_button("SHAP Impact", f, "shap_impact.png", use_container_width=True)
        
        shap_importance_path = "outputs/plots/shap_importance.png"
        if os.path.exists(shap_importance_path):
            with col6:
                with open(shap_importance_path, 'rb') as f:
                    st.download_button("SHAP Importance", f, "shap_importance.png", use_container_width=True)

# Tab 4: Adaptive Design Advisor
with tab4:
    st.subheader("🔄 Adaptive Design Advisor")
    
    st.info("**Preventative Approach**: Instead of rescuing failed trials, assess if adaptive design would be more appropriate BEFORE the trial starts.")
    
    with st.expander("📘 What is Adaptive Design?"):
        st.markdown("""
        **Adaptive designs** allow pre-planned modifications to ongoing trials based on interim results:
        
        **Types of Adaptive Designs:**
        - **Group Sequential**: Early stopping for efficacy or futility
        - **Sample Size Re-estimation**: Adjust N based on interim effect sizes
        - **Adaptive Enrichment**: Focus enrollment on responder subgroups
        - **Adaptive Randomization**: Allocate more patients to effective arms
        - **Seamless Phase 2/3**: Combine dose-finding and confirmatory phases
        - **Biomarker-Adaptive**: Enrich for biomarker-positive patients
        
        **Benefits:**
        - Faster, more efficient trials
        - Reduced exposure to ineffective treatments
        - Higher probability of success
        - Better patient outcomes
        
        **Trade-offs:**
        - More complex design and analysis
        - Requires pre-specified adaptation rules
        - Higher upfront planning costs
        - Regulatory scrutiny
        """)
    
    st.divider()
    
    # Input: Trial Protocol
    st.markdown("### 📋 Enter Your Trial Protocol")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        protocol_text = st.text_area(
            "Trial Description",
            placeholder="""Example:
Phase 3 trial for Novel Drug X in advanced lung cancer patients.
- Primary endpoint: Progression-free survival (PFS)
- Secondary: Overall survival, response rate
- Sample size: 500 patients (2:1 randomization)
- Duration: 24 months enrollment + 12 months follow-up
- Population: Adults with stage IV NSCLC, prior chemotherapy
- Hypothesis: 30% improvement in median PFS (from 5 to 6.5 months)
- Cost: $50M total
- Known challenges: Heterogeneous patient population, some may have EGFR mutations""",
            height=300,
            help="Describe your trial design, endpoints, population, timeline, budget, and any anticipated challenges"
        )
    
    with col2:
        st.markdown("**Include:**")
        st.markdown("- Disease/indication")
        st.markdown("- Target population")
        st.markdown("- Primary endpoint")
        st.markdown("- Sample size & duration")
        st.markdown("- Anticipated challenges")
        st.markdown("- Budget constraints")
        st.markdown("- Known biomarkers")
        st.markdown("- Subgroup hypotheses")
    
    st.divider()
    
    # Assessment
    if anthropic_api_key:
        if st.button("🔍 Assess Adaptive Design Suitability", type="primary", use_container_width=True):
            if not protocol_text.strip():
                st.error("Please enter your trial protocol above.")
            else:
                with st.spinner("Analyzing trial design with AI..."):
                    try:
                        # Build comprehensive prompt
                        prompt = f"""You are a clinical trial design expert and biostatistician. Assess whether ADAPTIVE DESIGN would be beneficial for the following trial protocol.

**TRIAL PROTOCOL:**
{protocol_text}

**YOUR ANALYSIS MUST INCLUDE:**

## 1. Adaptive Design Suitability Assessment
- **Overall Score**: LOW / MODERATE / HIGH / VERY HIGH
- **Primary Rationale**: Why is adaptive design suitable (or not)?

## 2. Recommended Adaptive Design Type(s)
For each recommended type, explain:
- **Which type**: Group sequential, enrichment, sample size re-estimation, etc.
- **When to adapt**: Timing of interim analyses
- **Adaptation rules**: What triggers the adaptation
- **Expected benefits**: Time savings, cost savings, increased success probability

## 3. Specific Implementation Plan
- **Interim analysis schedule**: When to look at data (e.g., after 50%, 75% enrollment)
- **Decision criteria**: Statistical thresholds for stopping/adapting
- **Biomarker strategy**: If applicable, which biomarkers to use for enrichment
- **Sample size considerations**: Min/max sample size range
- **Regulatory considerations**: FDA/EMA requirements for adaptive trials

## 4. Risk Assessment
**Benefits:**
- Quantify expected improvements (time, cost, success rate)

**Risks:**
- Operational complexity
- Statistical penalties (multiple testing)
- Regulatory acceptance challenges
- Need for real-time data monitoring

**Mitigation strategies**

## 5. Comparison: Adaptive vs. Fixed Design
Create a table comparing:
- Expected duration
- Expected cost
- Probability of success
- Risk of false positives
- Regulatory burden

## 6. Final Recommendation
**PROCEED WITH ADAPTIVE DESIGN** or **STICK WITH FIXED DESIGN**
- Clear rationale
- Key success factors
- Next steps

**IMPORTANT:**
- Be specific with numbers (sample sizes, timelines, costs)
- Reference real-world examples if applicable
- Consider both statistical and operational feasibility
- Address regulatory acceptance explicitly
- If the trial is NOT suitable for adaptive design, clearly state why and recommend alternatives"""

                        client = anthropic.Anthropic(api_key=anthropic_api_key)
                        message = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=4000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        
                        explanation = message.content[0].text
                        
                        # Save report
                        Path("outputs/reports").mkdir(parents=True, exist_ok=True)
                        with open("outputs/reports/adaptive_design_assessment.md", "w") as f:
                            f.write("# Adaptive Design Assessment Report\n\n")
                            f.write(f"## Input Protocol\n\n{protocol_text}\n\n")
                            f.write("---\n\n")
                            f.write(explanation)
                        
                        # Display
                        st.success("✓ Assessment Complete!")
                        st.markdown(explanation)
                        
                        # Download button
                        st.divider()
                        with open("outputs/reports/adaptive_design_assessment.md", 'rb') as f:
                            st.download_button(
                                "📥 Download Full Assessment",
                                f,
                                "adaptive_design_assessment.md",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter your Claude API key in the sidebar to use the Adaptive Design Advisor.")
    
    st.divider()
    
    # Educational resources
    with st.expander("📚 Learn More About Adaptive Designs"):
        st.markdown("""
        ### Key Resources:
        
        **FDA Guidance:**
        - [Adaptive Designs for Clinical Trials of Drugs and Biologics (2019)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry)
        
        **When Adaptive Designs Work Best:**
        1. **Uncertain effect size**: Wide range of plausible treatment effects
        2. **Known biomarkers**: Can enrich for responders during trial
        3. **Multiple doses**: Need to select optimal dose
        4. **Rare diseases**: Can't afford large fixed-sample trials
        5. **Heterogeneous populations**: Subgroups likely to respond differently
        6. **High costs**: Need to minimize sample size
        
        **When to Avoid Adaptive Designs:**
        1. **Well-understood disease/drug**: Effect size well-characterized from Phase 2
        2. **Simple hypothesis**: Single dose, homogeneous population
        3. **Short recruitment period**: No time for interim looks
        4. **Limited infrastructure**: Can't support real-time data monitoring
        5. **Regulatory concerns**: Novel indication where regulators prefer traditional design
        
        ### Real-World Examples:
        
        **Success Story**: I-SPY 2 Trial (Breast Cancer)
        - Adaptive platform trial testing multiple drugs
        - Biomarker-guided enrichment
        - Graduated drugs 2-3x faster than traditional trials
        - Cost savings: ~$100M per drug
        
        **Lesson Learned**: COVID-19 Vaccine Trials
        - Adaptive designs allowed rapid dose selection
        - Seamless Phase 2/3 reduced timeline by months
        - Real-time data monitoring enabled faster decisions
        """)

