import gc
import json
import os
import re
import shutil
import sys
import numpy as np
import pandas as pd
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer

# Opt-in to modern pandas downcasting behavior
pd.set_option("future.no_silent_downcasting", True)

# =====================================================================
# 🎛️ PIPELINE CONFIGURATION & CONTROL SWITCHES
# =====================================================================
TEST_MODE = False  # Set to False for the full 500k production run

BASE_PATH = "Subsamples"
PREPROCESSED_DIR = "hma/Preprocessed_Data"
CURRENT_YEAR = 2026
DELIMITER = ","  # 👈 CSV Comma Separator
READ_ENCODING = "latin-1"
CHUNKSIZE = 1_000_000

if TEST_MODE:
    print("⚠️ [EXECUTION MODE: TEST DRY RUN - 10 PATIENTS ONLY]")
    OUTPUT_DIR = "hma_test/Synthetic_Output"
    MODEL_DIR = "hma_test/Trained_Models"
    TEMP_CHUNK_DIR = "hma_test/Temp_Chunks"
    TARGET_PATIENTS = 10
    NUM_GEN_CHUNKS = 1
    SAMPLE_SCALE_PER_CHUNK = 1.0
else:
    print("🚀 [EXECUTION MODE: PRODUCTION RUN - FULL LLR-DfR COHORT]")
    OUTPUT_DIR = "hma_prod/Synthetic_Output"
    MODEL_DIR = "hma_prod/Trained_Models"
    TEMP_CHUNK_DIR = "hma_prod/Temp_Chunks"
    TRAIN_SAMPLE_FRACTION = 0.03  # 3% stratified sample (~15k patients)
    NUM_GEN_CHUNKS = 8
    SAMPLE_SCALE_PER_CHUNK = 4.1667  # 👈 Scale factor to hit 500k total

os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)

# LLR-DfR CSV File Mappings
files = {
    "patients": "SRPatient.csv",
    "codes": "SRCode.csv",
    "medications": "SRPrimaryCareMedication.csv",
    "immunisations": "SRImmunisation.csv",
}

COL_TYPES = {
    "patients": {
        "IDPatient": "str",
        "Gender": "str",
        "AgeIn2026": "float",
        "AgeAtDeath": "float",
        "DiseaseMedRatio": "float",
    },
    "codes": {
        "CTV3Code": "str",
        "SNOMEDCode": "str",
        "EpisodeType": "str",
        "IDEvent": "str",
        "IDPatient": "str",
        "AgeAtEvent": "float",
    },
    "immunisations": {
        "Dose": "str",
        "Location": "str",
        "ImmsReadCode": "str",
        "ImmsSNOMEDCode": "str",
        "IDPatient": "str",
        "AgeAtEvent": "float",
    },
    "medications": {
        "IDMultiLexProduct": "str",
        "NameOfMedication": "str",
        "MedicationDosage": "str",
        "IDPatient": "str",
        "AgeAtMedicationStart": "float",
        "AgeAtMedicationEnd": "float",
    },
}

# Clinical Biology Regex Keywords (Used ONLY for Medications text descriptions)
FEMALE_KEYWORDS = [
    r"\bpregnancy\b", r"\bpregnant\b", r"\bcervical\b", r"\bovarian\b",
    r"\bantenatal\b", r"\bpostnatal\b", r"\bmenopause\b", r"\bcontraceptive\b",
    r"\bhrt\b", r"\bintrauterine\b", r"\bvaginal\b", r"\bmammogram\b"
]
MALE_KEYWORDS = [
    r"\bprostate\b", r"\bprostatic\b", r"\btesticular\b", r"\berectile\b",
    r"\bvasectomy\b", r"\bsemen\b"
]

FEMALE_REGEX = re.compile("|".join(FEMALE_KEYWORDS), re.IGNORECASE)
MALE_REGEX = re.compile("|".join(MALE_KEYWORDS), re.IGNORECASE)

np.random.seed(42)


def parse_year(series: pd.Series) -> pd.Series:
    """Extracts 4-digit years (1900-2099) from YYYY, YYYYMM, or YYYYMMDD formats."""
    s = series.astype(str).str.strip()
    year_str = s.str.extract(r"(19\d{2}|20\d{2})")[0]
    return pd.to_numeric(year_str, errors="coerce").astype("Int64")


# =====================================================================
# STEP 1: PREPROCESSING RAW DATA & ANONYMIZATION
# =====================================================================
def run_step1_preprocessing():
    pat_in = os.path.join(BASE_PATH, files["patients"])
    pat_out = os.path.join(PREPROCESSED_DIR, files["patients"])
    print(f"\n--- STEP 1: Processing Patients from {pat_in} ---")

    df = pd.read_csv(
        pat_in,
        sep=DELIMITER,
        encoding=READ_ENCODING,
        low_memory=False,
        on_bad_lines="skip",
        dtype=str,
    )
    df.columns = df.columns.str.strip()

    df["IDPatient"] = df["IDPatient"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df = df[~df["IDPatient"].isin(["", "nan", "None", "-1", "<NA>"])].drop_duplicates(subset=["IDPatient"])

    df["Gender"] = df.get("Gender", "U").astype(str).str.strip().str.upper()
    df["Gender"] = df["Gender"].replace({"FEMALE": "F", "MALE": "M"})
    df["Gender"] = df["Gender"].apply(lambda x: x if x in ["F", "M"] else "U")

    b_col = "DateBirth" if "DateBirth" in df.columns else "YearOfBirth"
    d_col = "DateDeath" if "DateDeath" in df.columns else "YearOfDeath"

    b_year = parse_year(df[b_col]) if b_col in df.columns else pd.Series(np.nan, index=df.index, dtype="Int64")
    d_year = parse_year(df[d_col]) if d_col in df.columns else pd.Series(np.nan, index=df.index, dtype="Int64")

    df["AgeIn2026"] = np.where(b_year.notna() & d_year.isna(), CURRENT_YEAR - b_year, np.nan)
    df["AgeAtDeath"] = np.where(b_year.notna() & d_year.notna() & (d_year >= b_year), d_year - b_year, np.nan)

    df["AgeIn2026"] = pd.to_numeric(df["AgeIn2026"], errors="coerce")
    df["AgeAtDeath"] = pd.to_numeric(df["AgeAtDeath"], errors="coerce")

    # Memory-safe chunked accumulation for counts to build DiseaseMedRatio
    code_path = os.path.join(BASE_PATH, files["codes"])
    med_path = os.path.join(BASE_PATH, files["medications"])
    
    code_counts, med_counts = {}, {}

    if os.path.exists(code_path):
        for chunk in pd.read_csv(code_path, sep=DELIMITER, usecols=["IDPatient"], chunksize=CHUNKSIZE, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip", dtype=str):
            chunk["IDPatient"] = chunk["IDPatient"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            for pid, cnt in chunk["IDPatient"].value_counts().items():
                code_counts[pid] = code_counts.get(pid, 0) + cnt

    if os.path.exists(med_path):
        for chunk in pd.read_csv(med_path, sep=DELIMITER, usecols=["IDPatient"], chunksize=CHUNKSIZE, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip", dtype=str):
            chunk["IDPatient"] = chunk["IDPatient"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            for pid, cnt in chunk["IDPatient"].value_counts().items():
                med_counts[pid] = med_counts.get(pid, 0) + cnt

    s_codes = pd.Series(code_counts, dtype=float)
    s_meds = pd.Series(med_counts, dtype=float)

    log_codes = np.log1p(df["IDPatient"].map(s_codes).fillna(0))
    log_meds = np.log1p(df["IDPatient"].map(s_meds).fillna(0))
    df["DiseaseMedRatio"] = log_codes / (log_meds + 1e-5)

    dob_map = dict(zip(df["IDPatient"], b_year))

    df = df[["IDPatient", "Gender", "AgeIn2026", "AgeAtDeath", "DiseaseMedRatio"]]
    df.to_csv(pat_out, sep=DELIMITER, index=False, na_rep="")
    print(f" -> Saved {len(df):,} primary patient records with DiseaseMedRatio feature.")

    del df, s_codes, s_meds
    gc.collect()

    # Process Child Tables
    child_date_map = {
        "codes": (files["codes"], ["DateEvent"]),
        "medications": (files["medications"], ["DateEvent", "DateMedicationStart", "DateMedicationEnd"]),
        "immunisations": (files["immunisations"], ["DateEvent"]),
    }

    for key, (filename, date_cols) in child_date_map.items():
        file_in = os.path.join(BASE_PATH, filename)
        file_out = os.path.join(PREPROCESSED_DIR, filename)
        if not os.path.exists(file_in):
            continue
            
        print(f" -> Processing child table: {filename}...")
        chunk_iter = pd.read_csv(
            file_in, 
            sep=DELIMITER, 
            chunksize=CHUNKSIZE, 
            encoding=READ_ENCODING, 
            low_memory=False, 
            on_bad_lines="skip"
        )
        for i, chunk in enumerate(chunk_iter):
            chunk.columns = chunk.columns.str.strip()
            chunk["IDPatient"] = chunk["IDPatient"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            chunk = chunk[chunk["IDPatient"].isin(dob_map)]
            
            p_births = chunk["IDPatient"].map(dob_map)
            
            for d_col in date_cols:
                if d_col in chunk.columns:
                    e_years = parse_year(chunk[d_col])
                    age_col = d_col.replace("Date", "AgeAt")
                    age_series = e_years - p_births
                    valid_mask = (age_series >= 0) & age_series.notna()
                    chunk[age_col] = np.where(valid_mask, age_series, np.nan)
                    chunk[age_col] = pd.to_numeric(chunk[age_col], errors="coerce")
            
            chunk.drop(columns=date_cols, inplace=True, errors="ignore")
            
            # Structural sanity cleans & exact LLR-DfR schema mappings
            if key == "codes" and "SNOMEDCode" in chunk.columns:
                chunk["SNOMEDCode"] = chunk["SNOMEDCode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                chunk = chunk[~chunk["SNOMEDCode"].isin(["", "nan", "None", "-1", "<NA>"])]
            elif key == "medications" and "IDMultiLexProduct" in chunk.columns:
                chunk["IDMultiLexProduct"] = chunk["IDMultiLexProduct"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                chunk = chunk[~chunk["IDMultiLexProduct"].isin(["", "nan", "None", "-1", "<NA>"])]
            elif key == "immunisations":
                if "ImmsSNOMEDCode" in chunk.columns:
                    chunk["ImmsSNOMEDCode"] = chunk["ImmsSNOMEDCode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                if "ImmsReadCode" in chunk.columns:
                    chunk["ImmsReadCode"] = chunk["ImmsReadCode"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

            chunk.to_csv(file_out, sep=DELIMITER, mode="w" if i == 0 else "a", header=(i == 0), index=False, na_rep="")
    print("✅ PREPROCESSING COMPLETE.")


# =====================================================================
# STEP 2 & 3: FEATURE FLATTENING, MODEL TRAINING & FITTING
# =====================================================================
def run_step2_and_3_training():
    print("\n--- STEP 2 & 3: Feature Vector Flattening & HMA Model Fitting ---")
    
    patients_path = os.path.join(PREPROCESSED_DIR, files["patients"])
    df_patients_all = pd.read_csv(patients_path, sep=DELIMITER, encoding=READ_ENCODING, dtype=COL_TYPES["patients"])
    df_patients_all.columns = df_patients_all.columns.str.strip()
    df_patients_all = df_patients_all.dropna(subset=["IDPatient"]).drop_duplicates(subset=["IDPatient"])
    
    real_code_vocab_freq = {}
    code_counts, med_counts, imm_counts = {}, {}, {}
    
    # Track frequencies and compute summary counts
    for table_key, storage_dict in [("codes", code_counts), ("medications", med_counts), ("immunisations", imm_counts)]:
        f_path = os.path.join(PREPROCESSED_DIR, files[table_key])
        if os.path.exists(f_path):
            vocab_counter = {}
            chunk_iter = pd.read_csv(f_path, sep=DELIMITER, chunksize=1000000, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip")
            code_col = "SNOMEDCode" if table_key == "codes" else ("IDMultiLexProduct" if table_key == "medications" else "ImmsSNOMEDCode")
            
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.str.strip()
                if "IDPatient" in chunk.columns:
                    counts = chunk["IDPatient"].value_counts()
                    for pid, cnt in counts.items():
                        pid_str = str(pid)
                        storage_dict[pid_str] = storage_dict.get(pid_str, 0) + cnt
                if code_col in chunk.columns:
                    vc = chunk[code_col].dropna().value_counts()
                    for item_code, cnt in vc.items():
                        vocab_counter[str(item_code)] = vocab_counter.get(str(item_code), 0) + cnt
            real_code_vocab_freq[table_key] = vocab_counter

    with open(os.path.join(MODEL_DIR, "real_vocab_frequencies.json"), "w") as f:
        json.dump(real_code_vocab_freq, f)

    # Learn data-driven sex-exclusive SNOMED codes for the codes table
    snomed_gender_tracker = {}
    code_path_prep = os.path.join(PREPROCESSED_DIR, files["codes"])
    if os.path.exists(code_path_prep):
        pat_gender_dict = dict(zip(df_patients_all["IDPatient"].str.strip(), df_patients_all["Gender"].str.strip()))
        for chunk in pd.read_csv(code_path_prep, sep=DELIMITER, chunksize=1000000, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip", dtype=str):
            chunk.columns = chunk.columns.str.strip()
            if "IDPatient" in chunk.columns and "SNOMEDCode" in chunk.columns:
                chunk["IDPatient"] = chunk["IDPatient"].astype(str).str.strip()
                chunk["Gender"] = chunk["IDPatient"].map(pat_gender_dict)
                for _, row in chunk.dropna(subset=["SNOMEDCode", "Gender"]).iterrows():
                    code = str(row["SNOMEDCode"]).strip()
                    gender = str(row["Gender"]).strip()
                    if code not in snomed_gender_tracker:
                        snomed_gender_tracker[code] = set()
                    snomed_gender_tracker[code].add(gender)
                    
        exclusive_female_codes = {code for code, genders in snomed_gender_tracker.items() if genders == {"F"}}
        exclusive_male_codes = {code for code, genders in snomed_gender_tracker.items() if genders == {"M"}}
        
        with open(os.path.join(MODEL_DIR, "sex_exclusive_codes.json"), "w") as f:
            json.dump({
                "female_exclusive": list(exclusive_female_codes),
                "male_exclusive": list(exclusive_male_codes)
            }, f)

    # Inject Log-Count Features onto Parent Table
    df_patients_all["LogNumCodes"] = np.log1p(df_patients_all["IDPatient"].map(code_counts).fillna(0))
    df_patients_all["LogNumMeds"] = np.log1p(df_patients_all["IDPatient"].map(med_counts).fillna(0))
    df_patients_all["LogNumImms"] = np.log1p(df_patients_all["IDPatient"].map(imm_counts).fillna(0))

    # Safe Stratified Sampling Setup with Fallback for Tied/Small Cohorts
    try:
        df_patients_all["UtilGroup"] = pd.qcut(
            df_patients_all["LogNumCodes"] + df_patients_all["LogNumMeds"], 
            q=4, 
            labels=["L", "M", "H", "VH"], 
            duplicates="drop"
        )
    except ValueError:
        df_patients_all["UtilGroup"] = pd.cut(
            df_patients_all["LogNumCodes"] + df_patients_all["LogNumMeds"], 
            bins=min(4, max(1, df_patients_all["LogNumCodes"].nunique())), 
            labels=False, 
            include_lowest=True
        ).astype(str)

    try:
        df_patients_all["AgeGroup"] = pd.qcut(
            df_patients_all["AgeIn2026"].fillna(50), 
            q=5, 
            labels=False, 
            duplicates="drop"
        )
    except ValueError:
        df_patients_all["AgeGroup"] = pd.cut(
            df_patients_all["AgeIn2026"].fillna(50), 
            bins=min(5, max(1, df_patients_all["AgeIn2026"].nunique())), 
            labels=False, 
            include_lowest=True
        )

    df_patients_all["StratifyKey"] = df_patients_all["Gender"].astype(str) + "_" + df_patients_all["AgeGroup"].astype(str) + "_" + df_patients_all["UtilGroup"].astype(str)

    # SAFE SAMPLING HELPER
    def safe_sample(g, target_fraction):
        n_requested = max(1, int(len(g) * target_fraction))
        n_safe = min(len(g), n_requested)
        return g.sample(n=n_safe, random_state=42)

    if TEST_MODE:
        target_frac = TARGET_PATIENTS / len(df_patients_all)
        sampled_df = df_patients_all.groupby("StratifyKey", group_keys=False).apply(
            lambda g: safe_sample(g, target_frac)
        )
        if len(sampled_df) > TARGET_PATIENTS:
            sampled_df = sampled_df.sample(n=TARGET_PATIENTS, random_state=42)
    else:
        sampled_df = df_patients_all.groupby("StratifyKey", group_keys=False).apply(
            lambda g: safe_sample(g, TRAIN_SAMPLE_FRACTION)
        )

    sampled_patient_ids = set(sampled_df["IDPatient"])
    df_patients_train = df_patients_all[df_patients_all["IDPatient"].isin(sampled_patient_ids)].drop(columns=["UtilGroup", "AgeGroup", "StratifyKey"]).copy()

    train_data = {"patients": df_patients_train.replace(r"^\s*$", np.nan, regex=True)}

    for table_key in ["codes", "medications", "immunisations"]:
        f_path = os.path.join(PREPROCESSED_DIR, files[table_key])
        if not os.path.exists(f_path):
            train_data[table_key] = pd.DataFrame(columns=list(COL_TYPES[table_key].keys()))
            continue

        chunks = []
        chunk_iter = pd.read_csv(f_path, sep=DELIMITER, chunksize=1000000, encoding=READ_ENCODING, dtype=str, on_bad_lines="skip")
        for chunk in chunk_iter:
            chunk.columns = chunk.columns.str.strip()
            filtered_chunk = chunk[chunk["IDPatient"].isin(sampled_patient_ids)]
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)

        if chunks:
            df_child_train = pd.concat(chunks, ignore_index=True)
            if table_key == "medications" and "AgeAtMedicationEnd" in df_child_train.columns and "AgeAtMedicationStart" in df_child_train.columns:
                df_child_train["AgeAtMedicationEnd"] = df_child_train["AgeAtMedicationEnd"].fillna(df_child_train["AgeAtMedicationStart"])
            train_data[table_key] = df_child_train.replace(r"^\s*$", np.nan, regex=True)
        else:
            train_data[table_key] = pd.DataFrame(columns=list(COL_TYPES[table_key].keys()))

    # Build Metadata & Train Model
    global_metadata = MultiTableMetadata()
    global_metadata.detect_from_dataframes(data=train_data)

    if "patients" in global_metadata.tables:
        global_metadata.update_column(table_name="patients", column_name="IDPatient", sdtype="id")
        global_metadata.set_primary_key(table_name="patients", column_name="IDPatient")

    for child_table in ["codes", "medications", "immunisations"]:
        if child_table in global_metadata.tables and "IDPatient" in global_metadata.tables[child_table].columns:
            global_metadata.update_column(table_name=child_table, column_name="IDPatient", sdtype="id")
            try:
                global_metadata.add_relationship(parent_table_name="patients", child_table_name=child_table, parent_primary_key="IDPatient", child_foreign_key="IDPatient")
            except Exception:
                pass

    global_metadata.validate()
    metadata_json_path = os.path.join(MODEL_DIR, "global_metadata.json")
    if os.path.exists(metadata_json_path):
        os.remove(metadata_json_path)

    global_metadata.save_to_json(filepath=metadata_json_path)

    model_path = os.path.join(MODEL_DIR, "global_murmur_synthesizer.pkl")
    if os.path.exists(model_path):
        print(f" -> Loading pre-trained HMASynthesizer from {model_path}...")
        synthesizer = HMASynthesizer.load(filepath=model_path)
    else:
        print(" -> Fitting Unified HMASynthesizer model...")
        synthesizer = HMASynthesizer(global_metadata)
        
        if "medications" in train_data and len(train_data["medications"]) > 0:
            try:
                med_constraint = {
                    "constraint_class": "Inequality",
                    "table_name": "medications",
                    "constraint_parameters": {
                        "low_column_name": "AgeAtMedicationStart",
                        "high_column_name": "AgeAtMedicationEnd",
                    },
                }
                synthesizer.add_constraints(constraints=[med_constraint])
            except Exception as e:
                print(f" -> Constraint note: {e}")
                
        synthesizer.fit(train_data)
        synthesizer.save(filepath=model_path)

    return synthesizer


# =====================================================================
# STEP 4 & 5: CHUNKED GENERATION & CLINICAL POST-PROCESSING
# =====================================================================
def run_step4_and_5_generation(synthesizer):
    print("\n--- STEP 4 & 5: Chunked Generation & Clinical Rule Filtering ---")
    
    # 4. Generate Chunks
    for chunk_idx in range(NUM_GEN_CHUNKS):
        chunk_check = os.path.join(TEMP_CHUNK_DIR, f"patients_chunk_{chunk_idx}.csv")
        if os.path.exists(chunk_check):
            continue

        sampled_data = synthesizer.sample(scale=SAMPLE_SCALE_PER_CHUNK)
        prefix = f"C{chunk_idx}_"

        for table_name, df in sampled_data.items():
            if len(df) > 0:
                if "IDPatient" in df.columns:
                    df["IDPatient"] = prefix + df["IDPatient"].astype(str)
                chunk_file = os.path.join(TEMP_CHUNK_DIR, f"{table_name}_chunk_{chunk_idx}.csv")
                df.to_csv(chunk_file, sep=DELIMITER, index=False, encoding="utf-8")

        del sampled_data
        gc.collect()

    # 5. Clinical Post-Processing & Remapping
    with open(os.path.join(MODEL_DIR, "real_vocab_frequencies.json"), "r") as f:
        real_code_vocab_freq = json.load(f)

    sex_rules_path = os.path.join(MODEL_DIR, "sex_exclusive_codes.json")
    fem_exc, male_exc = set(), set()
    if os.path.exists(sex_rules_path):
        with open(sex_rules_path, "r") as f:
            sex_rules = json.load(f)
            fem_exc = set(sex_rules.get("female_exclusive", []))
            male_exc = set(sex_rules.get("male_exclusive", []))

    patient_context = {}
    unique_patient_ids = set()

    for chunk_idx in range(NUM_GEN_CHUNKS):
        pat_file = os.path.join(TEMP_CHUNK_DIR, f"patients_chunk_{chunk_idx}.csv")
        if os.path.exists(pat_file):
            df_p = pd.read_csv(pat_file, sep=DELIMITER, dtype=str)
            for _, row in df_p.iterrows():
                pid = row["IDPatient"]
                unique_patient_ids.add(pid)
                patient_context[pid] = {
                    "Gender": row.get("Gender", "U"),
                    "AgeAtDeath": pd.to_numeric(row.get("AgeAtDeath"), errors="coerce"),
                    "AgeIn2026": pd.to_numeric(row.get("AgeIn2026"), errors="coerce"),
                }

    patient_map = {old_id: f"SYN_PAT_{i+1:08d}" for i, old_id in enumerate(sorted(unique_patient_ids))}
    global_med_event_counter = 1

    for table_name, orig_filename in files.items():
        final_output_path = os.path.join(OUTPUT_DIR, f"synthetic_{orig_filename}")
        real_counts = real_code_vocab_freq.get(table_name, {})
        code_col = "SNOMEDCode" if table_name == "codes" else ("IDMultiLexProduct" if table_name == "medications" else "ImmsSNOMEDCode")

        first_chunk = True
        for chunk_idx in range(NUM_GEN_CHUNKS):
            chunk_file = os.path.join(TEMP_CHUNK_DIR, f"{table_name}_chunk_{chunk_idx}.csv")
            if not os.path.exists(chunk_file):
                continue

            chunk_df = pd.read_csv(chunk_file, sep=DELIMITER, dtype=str)

            if table_name == "patients":
                chunk_df = chunk_df.drop(columns=["LogNumCodes", "LogNumMeds", "LogNumImms", "DiseaseMedRatio"], errors="ignore")

            if "IDPatient" in chunk_df.columns:
                chunk_df["_Gender"] = chunk_df["IDPatient"].map(lambda x: patient_context.get(x, {}).get("Gender", "U"))
                chunk_df["_AgeAtDeath"] = chunk_df["IDPatient"].map(lambda x: patient_context.get(x, {}).get("AgeAtDeath", np.nan))
                chunk_df["IDPatient"] = chunk_df["IDPatient"].map(patient_map)
                chunk_df = chunk_df.dropna(subset=["IDPatient"])

            # Clinical Filtering Logic
            if table_name in ["codes", "medications", "immunisations"]:
                
                # Enforce data-driven sex-exclusive SNOMED code filters for the codes table
                if table_name == "codes" and "SNOMEDCode" in chunk_df.columns:
                    male_violation = (chunk_df["_Gender"] == "M") & (chunk_df["SNOMEDCode"].isin(fem_exc))
                    fem_violation = (chunk_df["_Gender"] == "F") & (chunk_df["SNOMEDCode"].isin(male_exc))
                    chunk_df = chunk_df[~(male_violation | fem_violation)]

                # Apply text-based biological regex ONLY to medications text descriptions
                if table_name == "medications" and "NameOfMedication" in chunk_df.columns:
                    male_mask = chunk_df["_Gender"] == "M"
                    fem_mask = chunk_df["_Gender"] == "F"
                    female_code_match = chunk_df["NameOfMedication"].str.contains(FEMALE_REGEX, na=False)
                    male_code_match = chunk_df["NameOfMedication"].str.contains(MALE_REGEX, na=False)

                    chunk_df = chunk_df[~(male_mask & female_code_match)]
                    chunk_df = chunk_df[~(fem_mask & male_code_match)]

                # Post-Mortem Event Filter
                age_col = "AgeAtEvent" if "AgeAtEvent" in chunk_df.columns else "AgeAtMedicationStart"
                if age_col in chunk_df.columns:
                    event_age = pd.to_numeric(chunk_df[age_col], errors="coerce")
                    post_mortem_mask = chunk_df["_AgeAtDeath"].notna() & event_age.notna() & (event_age > chunk_df["_AgeAtDeath"])
                    chunk_df = chunk_df[~post_mortem_mask]

                # Copula Noise Filter (< 5 occurrences in real baseline)
                if code_col in chunk_df.columns:
                    rare_real_codes = {k for k, v in real_counts.items() if v < 5}
                    chunk_df = chunk_df[~chunk_df[code_col].isin(rare_real_codes)]

                chunk_df = chunk_df.drop(columns=["_Gender", "_AgeAtDeath"], errors="ignore")

            if table_name == "medications":
                num_rows = len(chunk_df)
                global_med_event_counter += num_rows

            # Enforce strict Privacy-Safe final output columns matching LLR-DfR schemas
            if table_name == "patients":
                chunk_df = chunk_df[["IDPatient", "Gender", "AgeIn2026", "AgeAtDeath"]]
            elif table_name == "codes":
                chunk_df = chunk_df[["CTV3Code", "SNOMEDCode", "EpisodeType", "IDEvent", "IDPatient", "AgeAtEvent"]]
            elif table_name == "medications":
                chunk_df = chunk_df[["IDMultiLexProduct", "NameOfMedication", "MedicationDosage", "IDPatient", "AgeAtMedicationStart", "AgeAtMedicationEnd"]]
            elif table_name == "immunisations":
                chunk_df = chunk_df[["Dose", "Location", "ImmsReadCode", "ImmsSNOMEDCode", "IDPatient", "AgeAtEvent"]]

            chunk_df.to_csv(final_output_path, sep=DELIMITER, mode="w" if first_chunk else "a", header=first_chunk, index=False, encoding="utf-8")
            first_chunk = False

    shutil.rmtree(TEMP_CHUNK_DIR, ignore_errors=True)
    print(f"\nSUCCESS: Pipeline execution complete. Output saved to: {OUTPUT_DIR}")


# =====================================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_step1_preprocessing()
    model = run_step2_and_3_training()
    run_step4_and_5_generation(model)