"""
Generate synthetic clinical trial data simulating an atorvastatin Phase III trial.

This creates a dataset where:
- Overall response rate is low (~25% on intent-to-treat - "failed trial")
- BUT specific subgroups have high response rates (42-72%)
- Subgroup thresholds are grounded in atorvastatin pharmacology literature:

  LDL >= 145 mg/dL  -- ACC/AHA 'borderline high' threshold; patients in this range
                        show the greatest absolute cardiovascular risk reduction per
                        unit LDL lowering (CTT Collaboration, Lancet 2010)

  ALT < 30 U/L      -- Normal hepatic function; proxy for efficient CYP3A4-mediated
                        atorvastatin metabolism. Elevated ALT indicates hepatocellular
                        stress reducing CYP3A4 expression (Lennernas, Clin Pharmacokinet 2003)

  Glucose < 100     -- ADA normoglycemia threshold; non-diabetic patients receive
  mg/dL               full cardiovascular benefit without the statin-diabetes risk
                        offset (~10% T2DM increase; FDA label warning 2012; Sattar et al.
                        Lancet 2010; Ridker et al. Lancet 2012)

  Female, Age 50-75 -- Post-menopausal primary prevention window. Pre-menopausal
                        estrogen suppresses LDL, reducing marginal statin benefit.
                        After menopause, LDL rises and CVD risk increases, improving
                        the risk-benefit ratio (Mora et al. Circulation 2010;
                        ACC/AHA 2018 primary prevention guidelines)
"""

import pandas as pd
import numpy as np

np.random.seed(42)


def generate_synthetic_trial_data(n_patients=5000):
    """
    Generate synthetic atorvastatin trial with hidden responder subgroups.

    Ground truth subgroup hierarchy (biologically grounded):
      1. LDL >= 145 + ALT < 30 + Glucose < 100  →  72% response (triple positive)
      2. LDL >= 145 + ALT < 30                   →  63% response
      3. LDL >= 145 + Glucose < 100              →  58% response
      4. LDL >= 145 + Female + Age 50-75         →  54% response (post-menopausal)
      5. LDL >= 145 (alone)                      →  42% response
      6. All others (LDL < 145)                  →  15% response  ← trial "fails"
      Placebo arm (all profiles)                 →  12% background

    The intent-to-treat analysis yields ~25% overall response, appearing as a
    failed trial. Hidden subgroups contain 54-72% responders.
    """

    data = []

    for i in range(n_patients):
        # ------------------------------------------------------------------ #
        # Demographics
        # ------------------------------------------------------------------ #
        age = np.random.normal(55, 15)
        age = np.clip(age, 20, 85)
        gender = np.random.choice([1, 2])       # 1 = Male, 2 = Female
        race = np.random.choice([1, 2, 3, 4, 5])  # NHANES race/ethnicity codes

        # ------------------------------------------------------------------ #
        # Biomarkers (realistic distributions, some correlated with age/gender)
        # ------------------------------------------------------------------ #

        # HDL: females average ~10 mg/dL higher than males (NHANES reference)
        if gender == 2:
            hdl = np.random.normal(55, 12)
        else:
            hdl = np.random.normal(45, 10)

        # Glucose increases ~0.3 mg/dL per year of age above 55
        glucose = np.random.normal(100, 20) + (age - 55) * 0.3

        # Total cholesterol and derived LDL
        total_chol = np.random.normal(200, 35)
        ldl = total_chol - hdl - np.random.normal(30, 10)   # Friedewald equation

        # Triglycerides
        triglycerides = np.random.normal(150, 50)

        # ALT (alanine aminotransferase) -- liver enzyme, proxy for CYP3A4 function
        # Log-normal reflects the known right-skewed population distribution
        # Males average ~28 U/L; females average ~20 U/L (NHANES reference ranges)
        if gender == 1:
            alt = np.random.lognormal(np.log(28), 0.50)
        else:
            alt = np.random.lognormal(np.log(20), 0.45)

        # Clip all biomarkers to physiologically realistic ranges
        hdl          = np.clip(hdl,          20,  100)
        glucose      = np.clip(glucose,      70,  200)
        total_chol   = np.clip(total_chol,  120,  300)
        ldl          = np.clip(ldl,          50,  200)
        triglycerides= np.clip(triglycerides, 50, 400)
        alt          = np.clip(alt,            7,  120)

        # ------------------------------------------------------------------ #
        # Treatment assignment (1:1 randomization)
        # ------------------------------------------------------------------ #
        treatment = np.random.choice([0, 1])    # 0 = Placebo, 1 = Drug

        # ------------------------------------------------------------------ #
        # GROUND TRUTH: Atorvastatin responder subgroup hierarchy
        # Each threshold is justified by published pharmacology (see module docstring)
        # ------------------------------------------------------------------ #
        base_response_prob = 0.15

        if treatment == 1:
            if ldl >= 145 and alt < 30 and glucose < 100:
                # Triple positive: elevated LDL need + efficient hepatic metabolism
                # + no glycemic offset → strongest benefit
                response_prob = 0.72
            elif ldl >= 145 and alt < 30:
                # High LDL + efficient CYP3A4 metabolism; some glycemic risk present
                response_prob = 0.63
            elif ldl >= 145 and glucose < 100:
                # High LDL + no glycemic offset; less-optimal hepatic metabolism
                response_prob = 0.58
            elif ldl >= 145 and gender == 2 and 50 <= age <= 75:
                # High LDL + post-menopausal primary prevention window
                response_prob = 0.54
            elif ldl >= 145:
                # High LDL alone; benefit attenuated by other factors
                response_prob = 0.42
            else:
                # LDL < 145: no compelling pharmacological indication;
                # minimal marginal benefit (dilutes overall trial signal)
                response_prob = base_response_prob
        else:
            # Placebo arm: uniformly low background response
            response_prob = base_response_prob * 0.80

        # Binary outcome via Bernoulli draw
        responded = np.random.binomial(1, response_prob)

        # ------------------------------------------------------------------ #
        # Additional noise features (not in ground-truth subgroup logic)
        # ------------------------------------------------------------------ #
        bmi          = np.random.normal(28, 5)
        systolic_bp  = np.random.normal(130, 15)
        diastolic_bp = np.random.normal(80, 10)

        data.append({
            'SEQN':         i + 1000,
            'RIDAGEYR':     age,
            'RIAGENDR':     gender,
            'RIDRETH3':     race,
            'LBDHDD':       hdl,
            'LBXGLU':       glucose,
            'LBXTC':        total_chol,
            'LBDLDL':       ldl,
            'LBXTLG':       triglycerides,
            'LBXALT':       alt,        # NEW: ALT liver enzyme (CYP3A4 proxy)
            'BMXBMI':       bmi,
            'BPXSY1':       systolic_bp,
            'BPXDI1':       diastolic_bp,
            'treatment_arm': treatment,
            'Responded':    responded,
        })

    df = pd.DataFrame(data)

    # ------------------------------------------------------------------ #
    # Print ground-truth statistics for verification
    # ------------------------------------------------------------------ #
    print("=" * 65)
    print("SYNTHETIC ATORVASTATIN TRIAL DATASET — GROUND TRUTH SUMMARY")
    print("=" * 65)

    drug_df    = df[df['treatment_arm'] == 1]
    placebo_df = df[df['treatment_arm'] == 0]

    overall_rr         = drug_df['Responded'].mean()
    placebo_overall_rr = placebo_df['Responded'].mean()

    print(f"\nOverall Response Rate — Drug arm:    {overall_rr:.1%}")
    print(f"Overall Response Rate — Placebo arm: {placebo_overall_rr:.1%}")
    print("(Intent-to-treat analysis would appear as a FAILED TRIAL)")

    print("\n--- HIDDEN RESPONDER SUBGROUPS (DRUG ARM) ---")

    # Masks use hierarchical ELIF logic to avoid double-counting
    m_triple = (
        (drug_df['LBDLDL'] >= 145) &
        (drug_df['LBXALT'] < 30) &
        (drug_df['LBXGLU'] < 100)
    )
    m_ldl_alt = (
        (drug_df['LBDLDL'] >= 145) &
        (drug_df['LBXALT'] < 30) &
        ~(drug_df['LBXGLU'] < 100)
    )
    m_ldl_glu = (
        (drug_df['LBDLDL'] >= 145) &
        (drug_df['LBXGLU'] < 100) &
        ~(drug_df['LBXALT'] < 30)
    )
    m_postmeno = (
        (drug_df['LBDLDL'] >= 145) &
        (drug_df['RIAGENDR'] == 2) &
        (drug_df['RIDAGEYR'] >= 50) &
        (drug_df['RIDAGEYR'] <= 75) &
        ~(drug_df['LBXALT'] < 30) &
        ~(drug_df['LBXGLU'] < 100)
    )
    m_ldl_only = (
        (drug_df['LBDLDL'] >= 145) &
        ~(drug_df['LBXALT'] < 30) &
        ~(drug_df['LBXGLU'] < 100) &
        ~((drug_df['RIAGENDR'] == 2) &
          (drug_df['RIDAGEYR'] >= 50) &
          (drug_df['RIDAGEYR'] <= 75))
    )
    m_low_ldl = drug_df['LBDLDL'] < 145

    for label, mask, expected_rr in [
        ("1. LDL>=145 + ALT<30 + Glucose<100  (triple positive)", m_triple,   "~72%"),
        ("2. LDL>=145 + ALT<30                (liver efficient)", m_ldl_alt,  "~63%"),
        ("3. LDL>=145 + Glucose<100           (non-diabetic)",    m_ldl_glu,  "~58%"),
        ("4. LDL>=145 + Female age 50-75      (post-menopausal)", m_postmeno, "~54%"),
        ("5. LDL>=145 only                    (moderate benefit)",m_ldl_only, "~42%"),
        ("6. LDL<145                          (no indication)",   m_low_ldl,  "~15%"),
    ]:
        sg = drug_df[mask]
        observed = f"{sg['Responded'].mean():.1%}" if len(sg) > 0 else "N/A"
        print(f"\n  {label}")
        print(f"     N = {len(sg):4d}  |  Observed RR = {observed}  |  Expected ≈ {expected_rr}")

    print("\n" + "=" * 65)
    print("Threshold justifications:")
    print("  LDL >= 145 mg/dL : ACC/AHA 'borderline high→high' transition;")
    print("                     greatest absolute CVD risk reduction per CTT 2010")
    print("  ALT < 30 U/L     : Normal hepatic function; efficient CYP3A4")
    print("                     atorvastatin metabolism (Lennernas 2003)")
    print("  Glucose < 100    : ADA normoglycemia; no statin-diabetes offset")
    print("                     (Sattar 2010; Ridker 2012 JUPITER sub-analysis)")
    print("  Female 50-75     : Post-menopausal primary prevention window")
    print("                     (Mora 2010; ACC/AHA 2018 guidelines)")
    print("=" * 65)

    return df


if __name__ == "__main__":
    df = generate_synthetic_trial_data(n_patients=5000)

    output_path = "data/synthetic_trial_with_subgroups.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Upload this file in the Streamlit app to see real subgroup discovery!")
