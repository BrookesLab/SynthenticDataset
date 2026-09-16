import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# =====================================================================
# 📊 EVALUATION CONFIGURATION & THRESHOLDS
# =====================================================================
REAL_DIR = "hma/Preprocessed_Data"
SYNTHETIC_DIR = "hma_prod/Synthetic_Output"
MODEL_DIR = "hma_prod/Trained_Models"
DELIMITER = ","
READ_ENCODING = "latin-1"

# Target Acceptance Guardrail Thresholds
THRESHOLDS = {
    "min_ks_score": 0.70,         # Minimum acceptable Kolmogorov-Smirnov fidelity score
    "max_mean_age_diff": 4.0,     # Maximum allowed absolute year deviation for mean age
    "max_profile_overlap": 0,     # Strict zero tolerance for exact demographic record clones
    "mia_ratio_min": 0.85,        # Lower bound for train/hold-out MIA distance ratio
    "mia_ratio_max": 1.15         # Upper bound for train/hold-out MIA distance ratio
}

files = {
    "patients": ("SRPatient.csv", "synthetic_SRPatient.csv"),
    "codes": ("SRCode.csv", "synthetic_SRCode.csv"),
    "medications": ("SRPrimaryCareMedication.csv", "synthetic_SRPrimaryCareMedication.csv"),
    "immunisations": ("SRImmunisation.csv", "synthetic_SRImmunisation.csv"),
}

def load_data():
    print("📂 Loading real and synthetic datasets for evaluation...")
    real_data, synthetic_data = {}, {}
    
    for table_key, (real_file, synth_file) in files.items():
        r_path = os.path.join(REAL_DIR, real_file)
        s_path = os.path.join(SYNTHETIC_DIR, synth_file)
        
        if os.path.exists(r_path) and os.path.exists(s_path):
            real_data[table_key] = pd.read_csv(r_path, sep=DELIMITER, encoding=READ_ENCODING, low_memory=False, dtype=str)
            synthetic_data[table_key] = pd.read_csv(s_path, sep=DELIMITER, encoding=READ_ENCODING, low_memory=False, dtype=str)
            
            # Safely typecast numerical columns
            for df in [real_data[table_key], synthetic_data[table_key]]:
                for col in ["AgeIn2026", "AgeAtDeath", "AgeAtEvent", "AgeAtMedicationStart", "AgeAtMedicationEnd", "DiseaseMedRatio"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        
    return real_data, synthetic_data


# =====================================================================
# PILLAR 1: STATISTICAL FIDELITY METRICS
# =====================================================================
def evaluate_fidelity(real_data, synthetic_data):
    print("\n" + "="*60)
    print("🔍 PILLAR 1: STATISTICAL FIDELITY METRICS")
    print("="*60)
    
    fidelity_status = True
    for table_key, r_df in real_data.items():
        if table_key not in synthetic_data:
            continue
        s_df = synthetic_data[table_key]
        num_cols = r_df.select_dtypes(include=[np.number]).columns
        
        print(f"\n--- Table: {table_key} ---")
        for col in num_cols:
            if col in s_df.columns:
                r_clean = r_df[col].dropna()
                s_clean = s_df[col].dropna()
                if len(r_clean) > 0 and len(s_clean) > 0:
                    ks_res = ks_2samp(r_clean, s_clean)
                    ks_score = 1.0 - ks_res.statistic
                    
                    mean_diff = abs(r_clean.mean() - s_clean.mean()) / (r_clean.mean() + 1e-6)
                    std_diff = abs(r_clean.std() - s_clean.std()) / (r_clean.std() + 1e-6)
                    
                    status = "✅ PASS" if ks_score >= THRESHOLDS["min_ks_score"] else "⚠️ REVIEW"
                    print(f"  [{col}]")
                    print(f"    • KS Complement Score : {ks_score:.4f} {status}")
                    print(f"    • Mean Relative Error : {mean_diff:.4f}")
                    print(f"    • Std Dev Error       : {std_diff:.4f}")
                    
                    if ks_score < THRESHOLDS["min_ks_score"]:
                        fidelity_status = False
                        
    return fidelity_status


# =====================================================================
# PILLAR 2: CLINICAL UTILITY METRICS
# =====================================================================
def evaluate_utility(real_data, synthetic_data):
    print("\n" + "="*60)
    print("⚙️ PILLAR 2: CLINICAL UTILITY METRICS")
    print("="*60)
    
    if "patients" not in real_data or "patients" not in synthetic_data:
        print("Skipping utility: Patient table missing.")
        return False
        
    r_pat = real_data["patients"]
    s_pat = synthetic_data["patients"]
    utility_status = True
    
    print("\n--- Metric: Categorical Marginal Proportion (Gender) ---")
    r_g = r_pat["Gender"].value_counts(normalize=True).to_dict()
    s_g = s_pat["Gender"].value_counts(normalize=True).to_dict()
    print(f"  Real Distribution      : {r_g}")
    print(f"  Synthetic Distribution : {s_g}")
    
    print("\n--- Metric: Central Tendency Parity (AgeIn2026) ---")
    if "AgeIn2026" in r_pat.columns and "AgeIn2026" in s_pat.columns:
        r_mean = r_pat["AgeIn2026"].mean()
        s_mean = s_pat["AgeIn2026"].mean()
        age_delta = abs(r_mean - s_mean)
        
        print(f"  Real Mean Age          : {r_mean:.2f}")
        print(f"  Synthetic Mean Age     : {s_mean:.2f}")
        print(f"  Absolute Delta         : {age_delta:.2f} years")
        
        if age_delta <= THRESHOLDS["max_mean_age_diff"]:
            print("  ✅ Age Parity Check    : PASS")
        else:
            print("  ⚠️ Age Parity Check    : WARNING (Drift detected)")
            utility_status = False

    print("\n--- Metric: Structural Volume Multipliers ---")
    num_real_p = len(r_pat)
    num_synth_p = len(s_pat)
    print(f"  Parent Cohort Ratio (Synth/Real) : {num_synth_p / num_real_p:.2f}x")
    
    for t_key in ["codes", "medications", "immunisations"]:
        if t_key in real_data and t_key in synthetic_data:
            r_ratio = len(real_data[t_key]) / num_real_p
            s_ratio = len(synthetic_data[t_key]) / num_synth_p
            print(f"  {t_key} Density per Patient -> Real: {r_ratio:.2f} | Synth: {s_ratio:.2f}")

    return utility_status


# =====================================================================
# PILLAR 3: PRIVACY, DISCLOSURE RISK & PIPELINE-ALIGNED MIA
# =====================================================================
def evaluate_privacy_and_mia(real_data, synthetic_data):
    print("\n" + "="*60)
    print("🛡️ PILLAR 3: PRIVACY, DISCLOSURE RISK & MIA METRICS")
    print("="*60)
    
    if "patients" not in real_data or "patients" not in synthetic_data:
        print("Skipping privacy/MIA: Patient table missing.")
        return False

    r_pat = real_data["patients"].dropna(subset=["AgeIn2026", "AgeAtDeath", "DiseaseMedRatio"]).copy()
    s_pat = synthetic_data["patients"].dropna(subset=["AgeIn2026", "AgeAtDeath"]).copy()
    
    privacy_passed = True
    
    print("\n--- Metric: Distance-to-Closest-Record (DCR) ---")
    if len(r_pat) > 0 and len(s_pat) > 0:
        features = ["AgeIn2026", "AgeAtDeath"]
        r_mat = r_pat[features].head(2000).to_numpy()
        s_mat = s_pat[features].head(2000).to_numpy()
        
        dcr_list = [np.min(np.linalg.norm(r_mat - row, axis=1)) for row in s_mat]
        print(f"  • Mean DCR Distance    : {np.mean(dcr_list):.4f}")
        print(f"  • Median DCR Distance  : {np.median(dcr_list):.4f}")
        print(f"  • Minimum DCR Distance : {np.min(dcr_list):.4f} (Non-zero confirms absence of clone records)")

    print("\n--- Metric: Exact Attribute Combination Overlap ---")
    r_set = set(r_pat.apply(lambda r: f"{r.get('Gender')}_{r.get('AgeIn2026')}", axis=1))
    s_set = set(s_pat.apply(lambda r: f"{r.get('Gender')}_{r.get('AgeIn2026')}", axis=1))
    overlap_count = len(r_set.intersection(s_set))
    
    print(f"  • Exact Matches Found  : {overlap_count:,} unique demographic profiles")
    if overlap_count <= THRESHOLDS["max_profile_overlap"]:
        print("  ✅ Profile Overlap     : PASS (Zero record re-identification overlap)")
    else:
        print("  ❌ Profile Overlap     : FAIL (Exact attribute matches found!)")
        privacy_passed = False

    print("\n--- Metric: Empirical Membership Inference Attack (MIA) [Pipeline-Aligned] ---")
    if len(r_pat) > 100 and len(s_pat) > 100:
        # Replicate exact stratified logic used during training split
        r_pat["AgeGroup"] = pd.qcut(r_pat["AgeIn2026"].fillna(50), q=5, labels=False, duplicates="drop")
        r_pat["UtilGroup"] = pd.qcut(r_pat["DiseaseMedRatio"].fillna(0), q=4, labels=False, duplicates="drop")
        r_pat["StratifyKey"] = r_pat["Gender"].astype(str) + "_" + r_pat["AgeGroup"].astype(str) + "_" + r_pat["UtilGroup"].astype(str)

        def stratified_split_sample(g):
            n_safe = max(1, int(len(g) * 0.03))
            return g.sample(n=n_safe, random_state=42)

        r_train_df = r_pat.groupby("StratifyKey", group_keys=False).apply(stratified_split_sample)
        r_holdout_df = r_pat.drop(r_train_df.index)

        eval_features = ["AgeIn2026", "AgeAtDeath"]
        r_train = r_train_df[eval_features].dropna().to_numpy()
        r_holdout = r_holdout_df[eval_features].dropna().to_numpy()
        s_sample_mat = s_pat[eval_features].dropna().head(1000).to_numpy()
        
        if len(r_train) > 0 and len(r_holdout) > 0 and len(s_sample_mat) > 0:
            train_dists = [np.min(np.linalg.norm(r_train - row, axis=1)) for row in s_sample_mat]
            holdout_dists = [np.min(np.linalg.norm(r_holdout - row, axis=1)) for row in s_sample_mat]
            
            mean_train_dist = np.mean(train_dists)
            mean_holdout_dist = np.mean(holdout_dists)
            dist_ratio = mean_train_dist / (mean_holdout_dist + 1e-6)
            
            print(f"  • Mean Distance to Train Subsample    : {mean_train_dist:.4f}")
            print(f"  • Mean Distance to Hold-out Subsample : {mean_holdout_dist:.4f}")
            print(f"  • Train/Holdout Distance Ratio        : {dist_ratio:.4f} (Ideal $\\approx$ 1.0)")
            
            mia_passed = THRESHOLDS["mia_ratio_min"] <= dist_ratio <= THRESHOLDS["mia_ratio_max"]
            if mia_passed:
                print("  ✅ MIA Vulnerability Check            : PASS (No preferential training subset memorization)")
            else:
                print("  ⚠️ MIA Vulnerability Check            : WARNING (Overfitting/memorization tendency detected)")
                privacy_passed = False

    return privacy_passed


# =====================================================================
# PILLAR 4: ACTIVE ISSUE & CLINICAL RULE VIOLATION AUDIT
# =====================================================================
def audit_synthetic_issues(synthetic_data):
    print("\n" + "="*60)
    print("🚨 PILLAR 4: ACTIVE SYNTHETIC ISSUE & CONSTRAINT AUDIT")
    print("="*60)
    
    issues_found = 0
    
    if "patients" not in synthetic_data or "codes" not in synthetic_data:
        print("Skipping issue audit: Missing patient or code tables.")
        return 0

    s_pat = synthetic_data["patients"]
    s_codes = synthetic_data["codes"]
    
    # Create patient lookup map for gender and death ages
    pat_map = s_pat.set_index("IDPatient")[["Gender", "AgeAtDeath"]].to_dict(orient="index")
    
    # 1. Check Sex-Exclusive Code Leaks
    sex_rules_path = os.path.join(MODEL_DIR, "sex_exclusive_codes.json")
    if os.path.exists(sex_rules_path) and not s_codes.empty:
        with open(sex_rules_path, "r") as f:
            sex_rules = json.load(f)
            fem_exc = set(sex_rules.get("female_exclusive", []))
            male_exc = set(sex_rules.get("male_exclusive", []))
            
        print("\n--- Audit: Checking Sex-Exclusive Code Violations ---")
        violation_count = 0
        for _, row in s_codes.iterrows():
            pid = row.get("IDPatient")
            code = str(row.get("SNOMEDCode")).strip()
            p_info = pat_map.get(pid, {})
            gender = p_info.get("Gender", "U")
            
            if gender == "M" and code in fem_exc:
                violation_count += 1
            elif gender == "F" and code in male_exc:
                violation_count += 1
                
        if violation_count == 0:
            print("  ✅ Sex-Exclusive Rules Check : PASS (0 code leakage violations found)")
        else:
            print(f"  ❌ Sex-Exclusive Rules Check : FAIL ({violation_count:,} violations found!)")
            issues_found += violation_count

    # 2. Check Post-Mortem Event Violations
    if not s_codes.empty and "AgeAtEvent" in s_codes.columns:
        print("\n--- Audit: Checking Post-Mortem Event Violations ---")
        pm_violations = 0
        for _, row in s_codes.iterrows():
            pid = row.get("IDPatient")
            event_age = pd.to_numeric(row.get("AgeAtEvent"), errors="coerce")
            p_info = pat_map.get(pid, {})
            death_age = p_info.get("AgeAtDeath", np.nan)
            
            if not np.isnan(death_age) and not np.isnan(event_age) and event_age > death_age:
                pm_violations += 1
                
        if pm_violations == 0:
            print("  ✅ Post-Mortem Events Check  : PASS (0 events recorded after death)")
        else:
            print(f"  ❌ Post-Mortem Events Check  : FAIL ({pm_violations:,} post-mortem events found!)")
            issues_found += pm_violations

    return issues_found


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    real, synth = load_data()
    if real and synth:
        f_pass = evaluate_fidelity(real, synth)
        u_pass = evaluate_utility(real, synth)
        p_pass = evaluate_privacy_and_mia(real, synth)
        issues_count = audit_synthetic_issues(synth)
        
        print("\n" + "="*60)
        print("🏁 FINAL THREE-PILLAR & COMPLIANCE AUDIT SUMMARY")
        print("="*60)
        print(f"  • Pillar 1 (Fidelity)         : {'PASSED' if f_pass else 'REVIEW REQUIRED'}")
        print(f"  • Pillar 2 (Utility)          : {'PASSED' if u_pass else 'REVIEW REQUIRED'}")
        print(f"  • Pillar 3 (Privacy & MIA)    : {'PASSED' if p_pass else 'FAILED/WARNING'}")
        print(f"  • Pillar 4 (Active Issues)    : {issues_count} constraint violations detected")
        print("============================================================")
        print("✅ COMPLETE EVALUATION SUITE EXECUTED SUCCESSFULLY.")
    else:
        print("❌ Error: Missing preprocessed or synthetic output files for evaluation.")