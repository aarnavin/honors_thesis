"""
Generate synthetic clinical trial data with REAL responder subgroups.

This creates a dataset where:
- Overall response rate is low (~25% - "failed trial")
- BUT specific subgroups have high response rates (50-70%)
- Subgroups are based on demographics + biomarkers
"""

import pandas as pd
import numpy as np

np.random.seed(42)

def generate_synthetic_trial_data(n_patients=5000):
    """
    Generate synthetic trial with hidden responder subgroups.
    
    Ground truth subgroups:
    1. Females aged 60-80 with HDL > 60: 65% response rate
    2. Males aged 40-60 with low glucose (<100): 55% response rate
    3. All others: 15% response rate (trial appears to fail)
    """
    
    data = []
    
    for i in range(n_patients):
        # Demographics
        age = np.random.normal(55, 15)
        age = np.clip(age, 20, 85)
        gender = np.random.choice([1, 2])  # 1=Male, 2=Female
        race = np.random.choice([1, 2, 3, 4, 5])  # NHANES race codes
        
        # Biomarkers (correlated with age/gender)
        if gender == 2:  # Female - higher HDL
            hdl = np.random.normal(55, 12)
        else:
            hdl = np.random.normal(45, 10)
        
        glucose = np.random.normal(100, 20) + (age - 55) * 0.3  # Increases with age
        total_chol = np.random.normal(200, 35)
        ldl = total_chol - hdl - np.random.normal(30, 10)
        triglycerides = np.random.normal(150, 50)
        
        # Clip to realistic ranges
        hdl = np.clip(hdl, 20, 100)
        glucose = np.clip(glucose, 70, 200)
        total_chol = np.clip(total_chol, 120, 300)
        ldl = np.clip(ldl, 50, 200)
        triglycerides = np.clip(triglycerides, 50, 400)
        
        # Treatment arm (randomized)
        treatment = np.random.choice([0, 1])  # 0=Placebo, 1=Drug
        
        # GROUND TRUTH: Response based on subgroup membership
        base_response_prob = 0.15  # Overall low response
        
        # Only drug arm can have enhanced response
        if treatment == 1:
            # Subgroup 1: Older females with high HDL
            if gender == 2 and 60 <= age <= 80 and hdl > 60:
                response_prob = 0.65  # STRONG responders
            # Subgroup 2: Middle-aged males with low glucose
            elif gender == 1 and 40 <= age <= 60 and glucose < 100:
                response_prob = 0.55  # GOOD responders
            # Subgroup 3: Young patients (any gender) with high HDL + low triglycerides
            elif age < 40 and hdl > 55 and triglycerides < 120:
                response_prob = 0.50
            else:
                response_prob = base_response_prob
        else:
            # Placebo arm - low response across all subgroups
            response_prob = base_response_prob * 0.8  # Slightly lower than drug
        
        # Generate response
        responded = np.random.binomial(1, response_prob)
        
        # Additional noise features (irrelevant)
        bmi = np.random.normal(28, 5)
        systolic_bp = np.random.normal(130, 15)
        diastolic_bp = np.random.normal(80, 10)
        
        data.append({
            'SEQN': i + 1000,
            'RIDAGEYR': age,
            'RIAGENDR': gender,
            'RIDRETH3': race,
            'LBDHDD': hdl,
            'LBXGLU': glucose,
            'LBXTC': total_chol,
            'LBDLDL': ldl,
            'LBXTLG': triglycerides,
            'BMXBMI': bmi,
            'BPXSY1': systolic_bp,
            'BPXDI1': diastolic_bp,
            'treatment_arm': treatment,
            'Responded': responded
        })
    
    df = pd.DataFrame(data)
    
    # Print ground truth statistics
    print("=" * 60)
    print("SYNTHETIC DATASET GROUND TRUTH")
    print("=" * 60)
    
    drug_df = df[df['treatment_arm'] == 1]
    overall_rr = drug_df['Responded'].mean()
    print(f"\nOverall Response Rate (Drug): {overall_rr:.1%}")
    print("(Would appear as 'FAILED TRIAL')")
    
    print("\n--- HIDDEN RESPONDER SUBGROUPS ---")
    
    # Subgroup 1
    mask1 = (drug_df['RIAGENDR'] == 2) & (drug_df['RIDAGEYR'] >= 60) & (drug_df['RIDAGEYR'] <= 80) & (drug_df['LBDHDD'] > 60)
    sg1 = drug_df[mask1]
    print(f"\n1. Females aged 60-80 with HDL > 60:")
    print(f"   N = {len(sg1)}")
    print(f"   Response Rate = {sg1['Responded'].mean():.1%}")
    
    # Subgroup 2
    mask2 = (drug_df['RIAGENDR'] == 1) & (drug_df['RIDAGEYR'] >= 40) & (drug_df['RIDAGEYR'] <= 60) & (drug_df['LBXGLU'] < 100)
    sg2 = drug_df[mask2]
    print(f"\n2. Males aged 40-60 with glucose < 100:")
    print(f"   N = {len(sg2)}")
    print(f"   Response Rate = {sg2['Responded'].mean():.1%}")
    
    # Subgroup 3
    mask3 = (drug_df['RIDAGEYR'] < 40) & (drug_df['LBDHDD'] > 55) & (drug_df['LBXTLG'] < 120)
    sg3 = drug_df[mask3]
    print(f"\n3. Young (<40) with HDL > 55 and triglycerides < 120:")
    print(f"   N = {len(sg3)}")
    print(f"   Response Rate = {sg3['Responded'].mean():.1%}")
    
    # Non-responders
    mask_none = ~(mask1 | mask2 | mask3)
    sg_none = drug_df[mask_none]
    print(f"\n4. All other patients:")
    print(f"   N = {len(sg_none)}")
    print(f"   Response Rate = {sg_none['Responded'].mean():.1%}")
    
    print("\n" + "=" * 60)
    print("✓ Dataset should show STRONG subgroup signals!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_trial_data(n_patients=5000)
    
    output_path = "data/synthetic_trial_with_subgroups.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    print(f"✓ Upload this file in the Streamlit app to see real subgroup discovery!")

