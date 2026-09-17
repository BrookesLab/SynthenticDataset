import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

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
    "min_ks_score": 0.70,         # Minimum acceptable Kolmogorov-Smirnov complement score (1 - statistic)
    "max_mean_age_diff": 4.0,     # Maximum allowed absolute year deviation for mean age
    "min_spearman_rho": 0.80,     # Minimum acceptable Spearman rank correlation for vocabulary
    "max_profile_overlap": 0,     # Strict zero tolerance for exact high-dimensional feature record clones
    "mia_ratio_min": 0.85,        # Lower bound for balanced train/hold-out MIA distance ratio
    "mia_ratio_max": 1.15         # Upper bound for balanced train/hold-out MIA distance ratio
}

files = {
    "patients": ("SRPatient.csv", "synthetic_SRPatient.csv"),
    "codes": ("SRCode.csv", "synthetic_SRCode.csv"),
    "medications": ("SRPrimaryCareMedication.csv", "synthetic_SRPrimaryCareMedication.csv"),
    "immunisations": ("SRImmunisation.csv", "synthetic_SRImmunisation.csv"),
}

# Adaptive vocabulary candidate column names across MurMur / LLR-DfR schemas
VOCAB_CANDIDATES = {
    "codes": ["SNOMEDCode", "Code", "ConceptID", "SRCode"],
    "medications": ["MultilexProductID", "MultilexDrugID", "ProductID", "DMDCode", "MedicationCode", "DrugName"],
    "immunisations": ["VaccineCode", "ImmunisationCode", "Code", "VaccinationName"]
}

def resolve_vocab_column(df, candidates):
    """Finds the first existing candidate column in a dataframe."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

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
            num_cols = ["AgeIn2026", "AgeAtDeath", "AgeAtEvent", "AgeAtMedicationStart", "AgeAtMedicationEnd", "DiseaseMedRatio"]
            for df in [real_data[table_key], synthetic_data[table_key]]:
                for col in num_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        
    return real_data, synthetic_data


# =====================================================================
# PILLAR 1: STATISTICAL FIDELITY & DISTRIBUTION PARITY
# =====================================================================
def evaluate_fidelity(real_data, synthetic_data):
    print("\n" + "="*70)
    print("🔍 PILLAR 1: STATISTICAL FIDELITY & DISTRIBUTION PARITY")
    print("="*70)
    
    fidelity_status = True
    
    # 1. Continuous Distribution Matching (KS & Wasserstein)
    for table_key, r_df in real_data.items():
        if table_key not in synthetic_data:
            continue
        s_df = synthetic_data[table_key]
        num_cols = r_df.select_dtypes(include=[np.number]).columns
        
        print(f"\n--- Table: {table_key} (Continuous Features) ---")
        for col in num_cols:
            if col in s_df.columns:
                r_clean = r_df[col].dropna()
                s_clean = s_df[col].dropna()
                if len(r_clean) > 0 and len(s_clean) > 0:
                    ks_res = ks_2samp(r_clean, s_clean)
                    ks_score = 1.0 - ks_res.statistic
                    w_dist = wasserstein_distance(r_clean, s_clean)
                    
                    mean_diff = abs(r_clean.mean() - s_clean.mean()) / (abs(r_clean.mean()) + 1e-6)
                    std_diff = abs(r_clean.std() - s_clean.std()) / (abs(r_clean.std()) + 1e-6)
                    
                    status = "✅ PASS" if ks_score >= THRESHOLDS["min_ks_score"] else "⚠️ REVIEW"
                    print(f"  [{col}]")
                    print(f"    • KS Complement Score    : {ks_score:.4f} {status}")
                    print(f"    • Wasserstein Distance   : {w_dist:.4f}")
                    print(f"    • Mean Relative Error    : {mean_diff:.4f}")
                    print(f"    • Std Dev Relative Error : {std_diff:.4f}")
                    
                    if ks_score < THRESHOLDS["min_ks_score"]:
                        fidelity_status = False

    # 2. Vocabulary Spearman Rank Correlation & Prevalence Parity
    print("\n--- Vocabulary Rank Correlation & Code Prevalence ---")
    for t_key, candidates in VOCAB_CANDIDATES.items():
        if t_key in real_data and t_key in synthetic_data:
            r_df = real_data[t_key]
            s_df = synthetic_data[t_key]
            
            r_col = resolve_vocab_column(r_df, candidates)
            s_col = resolve_vocab_column(s_df, candidates)
            
            if not r_col or not s_col:
                print(f"  [{t_key.upper()}] ⚠️ Skipped: Column not found matching {candidates}")
                print(f"    • Available columns in real table: {list(r_df.columns)}")
                continue
                
            r_counts = r_df[r_col].dropna().astype(str).str.strip().value_counts()
            s_counts = s_df[s_col].dropna().astype(str).str.strip().value_counts()
            
            vocab_df = pd.DataFrame({"real": r_counts, "synth": s_counts}).fillna(0)
            
            print(f"  [{t_key.upper()} - {r_col}]")
            print(f"    • Total Unique Real Vocab         : {len(r_counts):,}")
            print(f"    • Total Unique Synthetic Vocab    : {len(s_counts):,}")
            
            if len(vocab_df) >= 10:
                rho, p_val = spearmanr(vocab_df["real"], vocab_df["synth"])
                status = "✅ PASS" if rho >= THRESHOLDS["min_spearman_rho"] else "⚠️ REVIEW"
                print(f"    • Spearman Rank Correlation (rho) : {rho:.4f} {status} (p = {p_val:.2e})")
                if rho < THRESHOLDS["min_spearman_rho"]:
                    fidelity_status = False
            else:
                print(f"    • Spearman Rank Correlation (rho) : ℹ️ Skipped (Insufficient vocabulary size: {len(vocab_df)} codes < 10)")
                
            # Top-15 Prevalence Check
            top15_real = set(r_counts.head(15).index)
            top15_synth = set(s_counts.head(15).index)
            top_overlap = len(top15_real.intersection(top15_synth))
            print(f"    • Top-15 Code Set Intersection    : {top_overlap}/15")
                    
    return fidelity_status


# =====================================================================
# PILLAR 2: CLINICAL & RELATIONAL UTILITY
# =====================================================================
def evaluate_utility(real_data, synthetic_data):
    print("\n" + "="*70)
    print("⚙️ PILLAR 2: CLINICAL & RELATIONAL UTILITY")
    print("="*70)
    
    if "patients" not in real_data or "patients" not in synthetic_data:
        print("Skipping utility: Patient table missing.")
        return False
        
    r_pat = real_data["patients"]
    s_pat = synthetic_data["patients"]
    utility_status = True
    
    print("\n--- Categorical Marginal Proportion (Gender) ---")
    r_g = r_pat["Gender"].value_counts(normalize=True).to_dict()
    s_g = s_pat["Gender"].value_counts(normalize=True).to_dict()
    print(f"  • Real Proportions      : {r_g}")
    print(f"  • Synthetic Proportions : {s_g}")
    
    # Total Variation Distance (TVD) for Gender
    all_genders = set(r_g.keys()).union(set(s_g.keys()))
    tvd_gender = 0.5 * sum(abs(r_g.get(g, 0.0) - s_g.get(g, 0.0)) for g in all_genders)
    print(f"  • Gender Total Variation Distance (TVD): {tvd_gender:.4f}")
    
    print("\n--- Central Tendency Parity (AgeIn2026) ---")
    if "AgeIn2026" in r_pat.columns and "AgeIn2026" in s_pat.columns:
        r_mean = r_pat["AgeIn2026"].mean()
        s_mean = s_pat["AgeIn2026"].mean()
        age_delta = abs(r_mean - s_mean)
        
        print(f"  • Real Mean Age         : {r_mean:.2f} years")
        print(f"  • Synthetic Mean Age    : {s_mean:.2f} years")
        print(f"  • Absolute Delta        : {age_delta:.2f} years")
        
        if age_delta <= THRESHOLDS["max_mean_age_diff"]:
            print("  ✅ Mean Age Parity Check: PASS")
        else:
            print("  ⚠️ Mean Age Parity Check: WARNING (Drift detected)")
            utility_status = False

    print("\n--- Relational Child-to-Parent Event Densities ---")
    num_real_p = len(r_pat)
    num_synth_p = len(s_pat)
    print(f"  • Synthetic/Real Patient Scale Factor: {num_synth_p / num_real_p:.2f}x")
    
    for t_key in ["codes", "medications", "immunisations"]:
        if t_key in real_data and t_key in synthetic_data:
            r_density = len(real_data[t_key]) / num_real_p
            s_density = len(synthetic_data[t_key]) / num_synth_p
            print(f"  • {t_key.capitalize()} Density per Patient -> Real: {r_density:.2f} | Synth: {s_density:.2f}")

    return utility_status


# =====================================================================
# PILLAR 3: PRIVACY, DISCLOSURE RISK & BALANCED MIA
# =====================================================================
def evaluate_privacy_and_mia(real_data, synthetic_data):
    print("\n" + "="*70)
    print("🛡️ PILLAR 3: PRIVACY, DISCLOSURE RISK & BALANCED MIA METRICS")
    print("="*70)
    
    if "patients" not in real_data or "patients" not in synthetic_data:
        print("Skipping privacy/MIA: Patient table missing.")
        return False

    r_pat = real_data["patients"].copy()
    s_pat = synthetic_data["patients"].copy()
    privacy_passed = True

    # 1. Feature Engineering: Attach Event Volume Profiles
    def get_event_counts(data_dict):
        pat_df = data_dict["patients"][["IDPatient", "Gender", "AgeIn2026", "AgeAtDeath"]].copy()
        for t_key in ["codes", "medications", "immunisations"]:
            if t_key in data_dict:
                counts = data_dict[t_key]["IDPatient"].value_counts().rename(f"Count_{t_key}")
                pat_df = pat_df.merge(counts, on="IDPatient", how="left")
            else:
                pat_df[f"Count_{t_key}"] = 0
        return pat_df.fillna(0)

    r_profiles = get_event_counts(real_data)
    s_profiles = get_event_counts(synthetic_data)

    # 2. Exact High-Dimensional Record Clone Rate
    print("\n--- Exact High-Dimensional Profile Clone Audit ---")
    feature_keys = ["Gender", "AgeIn2026", "AgeAtDeath", "Count_codes", "Count_medications", "Count_immunisations"]
    
    r_signatures = set(r_profiles[feature_keys].astype(str).agg("_".join, axis=1))
    s_signatures = s_profiles[feature_keys].astype(str).agg("_".join, axis=1)
    
    exact_clones = s_signatures.isin(r_signatures).sum()
    clone_rate = (exact_clones / len(s_profiles)) * 100.0
    
    print(f"  • Exact Real Clones Identified : {exact_clones:,}")
    print(f"  • Exact Clone Rate             : {clone_rate:.4f}%")
    if exact_clones <= THRESHOLDS["max_profile_overlap"]:
        print("  ✅ Zero-Clone Check            : PASS")
    else:
        print("  ❌ Zero-Clone Check            : FAIL (Identical feature vectors detected; post-purge required)")
        privacy_passed = False

    # 3. Distance to Closest Record (DCR) in Normalized Feature Space
    print("\n--- Normalized Distance to Closest Record (DCR) ---")
    dcr_cols = ["AgeIn2026", "AgeAtDeath", "Count_codes", "Count_medications", "Count_immunisations"]
    
    sample_size = min(5000, len(r_profiles), len(s_profiles))
    r_sub = r_profiles[dcr_cols].head(sample_size).to_numpy(dtype=float)
    s_sub = s_profiles[dcr_cols].head(sample_size).to_numpy(dtype=float)
    
    mins = np.nanmin(np.vstack([r_sub, s_sub]), axis=0)
    maxs = np.nanmax(np.vstack([r_sub, s_sub]), axis=0)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    
    r_norm = np.nan_to_num((r_sub - mins) / ranges)
    s_norm = np.nan_to_num((s_sub - mins) / ranges)
    
    dcr_values = [np.min(np.linalg.norm(r_norm - row, axis=1)) for row in s_norm]
    dcr_5th = np.percentile(dcr_values, 5)
    
    print(f"  • Mean DCR Distance            : {np.mean(dcr_values):.4f}")
    print(f"  • Median DCR Distance          : {np.median(dcr_values):.4f}")
    print(f"  • 5th Percentile DCR (DCR_5%)   : {dcr_5th:.4f} (Safety buffer against memorization)")

    # 4. Pipeline-Aligned Balanced Membership Inference Attack (MIA)
    print("\n--- Balanced Empirical Membership Inference Attack (MIA) ---")
    if "DiseaseMedRatio" in real_data["patients"].columns:
        r_mia = real_data["patients"].dropna(subset=["AgeIn2026", "DiseaseMedRatio"]).copy()
    else:
        r_mia = real_data["patients"].dropna(subset=["AgeIn2026"]).copy()
        r_mia["DiseaseMedRatio"] = 0.0

    if len(r_mia) >= 200:
        r_mia["AgeGroup"] = pd.qcut(r_mia["AgeIn2026"].fillna(50), q=5, labels=False, duplicates="drop")
        r_mia["UtilGroup"] = pd.qcut(r_mia["DiseaseMedRatio"].fillna(0), q=4, labels=False, duplicates="drop")
        r_mia["StratifyKey"] = r_mia["Gender"].astype(str) + "_" + r_mia["AgeGroup"].astype(str) + "_" + r_mia["UtilGroup"].astype(str)

        def stratified_split(g):
            n_sample = max(1, int(len(g) * 0.03))
            return g.sample(n=n_sample, random_state=42)

        train_split = r_mia.groupby("StratifyKey", group_keys=False).apply(stratified_split)
        unseen_pool = r_mia.drop(train_split.index)

        # Draw an equal-sized holdout sample (1:1 ratio) to eliminate point-density bias
        n_eval = min(len(train_split), len(unseen_pool))
        train_sample = train_split.sample(n=n_eval, random_state=42)
        holdout_sample = unseen_pool.sample(n=n_eval, random_state=42)

        eval_vars = ["AgeIn2026", "AgeAtDeath"]
        t_mat = train_sample[eval_vars].fillna(0).to_numpy()
        h_mat = holdout_sample[eval_vars].fillna(0).to_numpy()
        synth_eval_mat = s_pat[eval_vars].fillna(0).head(1000).to_numpy()

        train_min_dists = [np.min(np.linalg.norm(t_mat - row, axis=1)) for row in synth_eval_mat]
        holdout_min_dists = [np.min(np.linalg.norm(h_mat - row, axis=1)) for row in synth_eval_mat]

        mean_t_dist = np.mean(train_min_dists)
        mean_h_dist = np.mean(holdout_min_dists)
        mia_dist_ratio = mean_t_dist / (mean_h_dist + 1e-6)
        mia_prob = np.mean([1 if td < hd else 0 for td, hd in zip(train_min_dists, holdout_min_dists)]) * 100.0

        print(f"  • Mean Distance to Train Sample  : {mean_t_dist:.4f}")
        print(f"  • Mean Distance to Holdout Sample: {mean_h_dist:.4f}")
        print(f"  • Distance Ratio (Train/Holdout) : {mia_dist_ratio:.4f} (Ideal ≈ 1.0)")
        print(f"  • Empirical MIA Advantage Score  : {mia_prob:.2f}% (Ideal baseline ≈ 50.0%)")

        if THRESHOLDS["mia_ratio_min"] <= mia_dist_ratio <= THRESHOLDS["mia_ratio_max"]:
            print("  ✅ Balanced MIA Check            : PASS (No membership leakage detected)")
        else:
            print("  ⚠️ Balanced MIA Check            : WARNING (Overfitting/memorization drift detected)")
            privacy_passed = False

    return privacy_passed


# =====================================================================
# PILLAR 4: COMPREHENSIVE CLINICAL RULE & INTEGRITY AUDIT
# =====================================================================
def audit_clinical_rules(synthetic_data):
    print("\n" + "="*70)
    print("🚨 PILLAR 4: COMPREHENSIVE CLINICAL RULE & INTEGRITY AUDIT")
    print("="*70)
    
    total_violations = 0
    if "patients" not in synthetic_data:
        print("Skipping integrity audit: Patient table missing.")
        return 0

    s_pat = synthetic_data["patients"]
    pat_meta = s_pat.set_index("IDPatient")[["Gender", "AgeAtDeath"]].to_dict(orient="index")

    # 1. Post-Mortem Event Violations Across All Child Tables
    print("\n--- Post-Mortem Event Violations (AgeAtEvent > AgeAtDeath) ---")
    event_age_cols = {
        "codes": "AgeAtEvent",
        "medications": "AgeAtMedicationStart",
        "immunisations": "AgeAtEvent"
    }
    
    pm_violations_found = 0
    for t_key, col in event_age_cols.items():
        if t_key in synthetic_data and col in synthetic_data[t_key].columns:
            df = synthetic_data[t_key]
            v_count = 0
            for _, row in df.iterrows():
                pid = row.get("IDPatient")
                e_age = row.get(col)
                p_death = pat_meta.get(pid, {}).get("AgeAtDeath", np.nan)
                if pd.notna(p_death) and pd.notna(e_age) and float(e_age) > float(p_death):
                    v_count += 1
            print(f"  • Post-Mortem Violations in {t_key.capitalize():<14} : {v_count:,}")
            pm_violations_found += v_count

    if pm_violations_found == 0:
        print("  ✅ Global Post-Mortem Check : PASS (100% compliant)")
    else:
        print(f"  ❌ Global Post-Mortem Check : FAIL ({pm_violations_found:,} total events occurred post-mortem)")
    total_violations += pm_violations_found

    # 2. Biological Sex-Exclusive Clinical Code Violations
    print("\n--- Biological Sex-Exclusive Code Integrity ---")
    sex_rules_path = os.path.join(MODEL_DIR, "sex_exclusive_codes.json")
    sex_violations_found = 0
    
    if os.path.exists(sex_rules_path) and "codes" in synthetic_data:
        with open(sex_rules_path, "r") as f:
            sex_rules = json.load(f)
            fem_exc = set(str(c).strip() for c in sex_rules.get("female_exclusive", []))
            male_exc = set(str(c).strip() for c in sex_rules.get("male_exclusive", []))

        s_codes = synthetic_data["codes"]
        code_col = resolve_vocab_column(s_codes, VOCAB_CANDIDATES["codes"])
        
        if code_col:
            for _, row in s_codes.iterrows():
                pid = row.get("IDPatient")
                code = str(row.get(code_col)).strip()
                gender = pat_meta.get(pid, {}).get("Gender", "U")

                if gender == "M" and code in fem_exc:
                    sex_violations_found += 1
                elif gender == "F" and code in male_exc:
                    sex_violations_found += 1

            if sex_violations_found == 0:
                print("  ✅ Sex-Exclusive Rules Check: PASS (0 biological code leaks detected)")
            else:
                print(f"  ❌ Sex-Exclusive Rules Check: FAIL ({sex_violations_found:,} code leaks detected)")
    else:
        print("  ℹ️ Skipped: sex_exclusive_codes.json not found in model directory.")
    total_violations += sex_violations_found

    # 3. Medication Chronology Integrity (AgeAtStart <= AgeAtEnd)
    print("\n--- Medication Chronology Validation (Start <= End) ---")
    if "medications" in synthetic_data:
        s_meds = synthetic_data["medications"]
        if "AgeAtMedicationStart" in s_meds.columns and "AgeAtMedicationEnd" in s_meds.columns:
            valid_dates = s_meds.dropna(subset=["AgeAtMedicationStart", "AgeAtMedicationEnd"])
            chrono_violations = (valid_dates["AgeAtMedicationStart"] > valid_dates["AgeAtMedicationEnd"]).sum()
            
            if chrono_violations == 0:
                print("  ✅ Chronology Logic Check   : PASS (0 inverted prescription intervals)")
            else:
                print(f"  ❌ Chronology Logic Check   : FAIL ({chrono_violations:,} intervals start after ending)")
                total_violations += chrono_violations

    return total_violations


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    real, synth = load_data()
    if real and synth:
        f_pass = evaluate_fidelity(real, synth)
        u_pass = evaluate_utility(real, synth)
        p_pass = evaluate_privacy_and_mia(real, synth)
        issues_count = audit_clinical_rules(synth)
        
        print("\n" + "="*70)
        print("🏁 COMPLETE THREE-PILLAR & COMPLIANCE AUDIT REPORT")
        print("="*70)
        print(f"  • Pillar 1 (Statistical Fidelity) : {'PASSED' if f_pass else 'REVIEW REQUIRED'}")
        print(f"  • Pillar 2 (Clinical Utility)     : {'PASSED' if u_pass else 'REVIEW REQUIRED'}")
        print(f"  • Pillar 3 (Privacy & MIA)        : {'PASSED' if p_pass else 'FAILED/WARNING'}")
        print(f"  • Pillar 4 (Rule Compliance)      : {'PASSED (0 violations)' if issues_count == 0 else f'FAILED ({issues_count:,} violations)'}")
        print("="*70)
        print("✅ VALIDATION SUITE EXECUTION COMPLETED.")
    else:
        print("❌ Error: Missing preprocessed or synthetic output files for evaluation.")