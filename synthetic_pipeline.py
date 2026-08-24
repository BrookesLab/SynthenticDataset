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
TEST_MODE = False  # Set to True for a dry-run on 10 patients

BASE_PATH = "Subsamples"
PREPROCESSED_DIR = "Preprocessed_Data"
CURRENT_YEAR = 2026
DELIMITER = ","
READ_ENCODING = "latin-1"

if TEST_MODE:
    print("⚠️ [EXECUTION MODE: TEST DRY RUN - 10 PATIENTS ONLY]")
    OUTPUT_DIR = "Synthetic_Output_test"
    MODEL_DIR = "Trained_Models_test"
    TEMP_CHUNK_DIR = "Temp_Chunks_test"
    TARGET_PATIENTS = 10
    NUM_GEN_CHUNKS = 1
    SAMPLE_SCALE_PER_CHUNK = 1.0
else:
    print("🚀 [EXECUTION MODE: PRODUCTION RUN - FULL COHORT]")
    OUTPUT_DIR = "Synthetic_Output"
    MODEL_DIR = "Trained_Models"
    TEMP_CHUNK_DIR = "Temp_Chunks"
    TRAIN_SAMPLE_FRACTION = 0.10  # 10% stratified sample (~50k patients, ~16M child rows)
    NUM_GEN_CHUNKS = 10           # 10 chunks x 1.0 scale to reconstitute 100% volume
    SAMPLE_SCALE_PER_CHUNK = 1.0

os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)

# File names matching your exact input data
files = {
    "patients": "SRPatient.csv",
    "codes": "SRCode.csv",
    "medications": "SRPrimaryCareMedication.csv",
    "immunisations": "SRImmunisation.csv",
}

# Explicit dtypes matching your exact tables
COL_TYPES = {
    "patients": {
        "IDPatient": "str",
        "Gender": "str",
        "AgeIn2026": "float",
        "AgeAtDeath": "float",
    },
    "codes": {
        "IDEvent": "str",
        "IDPatient": "str",
        "CTV3Code": "str",
        "SNOMEDCode": "str",
        "EpisodeType": "str",
        "AgeAtEvent": "float",
    },
    "medications": {
        "IDEvent": "str",
        "IDPatient": "str",
        "IDMultiLexProduct": "str",
        "IDMultiLexDMD": "str",
        "NameOfMedication": "str",
        "AgeAtMedicationStart": "float",
        "AgeAtMedicationEnd": "float",
    },
    "immunisations": {
        "IDPatient": "str",
        "IDImmunisationContent": "str",
        "Dose": "str",
        "Location": "str",
        "Method": "str",
        "ImmsReadCode": "str",
        "ImmsSNOMEDCode": "str",
        "VaccPart": "str",
        "AgeAtEvent": "float",
    },
}

# Clinical Biology Regex Keywords
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

# =====================================================================
# UTILITIES
# =====================================================================
def parse_date_to_year(series: pd.Series) -> pd.Series:
    """Parses standard ISO dates (YYYY-MM-DD) or extract 4-digit years."""
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    years = parsed.dt.year
    unparsed = years.isna()
    if unparsed.any():
        extracted = series[unparsed].astype(str).str.extract(r"(\b19\d{2}\b|\b20\d{2}\b)")[0]
        years.loc[unparsed] = pd.to_numeric(extracted, errors="coerce")
    return years.astype("Int64")


# =====================================================================
# STEP 1: PREPROCESSING & RELATIVE AGE CALCULATION
# =====================================================================
def run_step1_preprocessing():
    print("\n--- STEP 1: Preprocessing & Anonymizing Dates to Ages ---")
    
    # 1A. Process SRPatient.csv
    pat_in = os.path.join(BASE_PATH, files["patients"])
    pat_out = os.path.join(PREPROCESSED_DIR, files["patients"])
    
    df_pat = pd.read_csv(pat_in, sep=DELIMITER, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip")
    df_pat.columns = df_pat.columns.str.strip()
    df_pat["IDPatient"] = df_pat["IDPatient"].astype(str).str.strip()
    df_pat = df_pat.dropna(subset=["IDPatient"]).drop_duplicates(subset=["IDPatient"])

    b_years = parse_date_to_year(df_pat["DateBirth"])
    d_years = parse_date_to_year(df_pat["DateDeath"]) if "DateDeath" in df_pat.columns else pd.Series(pd.NA, index=df_pat.index, dtype="Int64")

    df_pat["AgeIn2026"] = pd.Series(dtype="Int64", index=df_pat.index)
    df_pat["AgeAtDeath"] = pd.Series(dtype="Int64", index=df_pat.index)

    death_mask = b_years.notna() & d_years.notna() & (d_years >= b_years)
    living_mask = b_years.notna() & d_years.isna()

    df_pat.loc[death_mask, "AgeAtDeath"] = d_years[death_mask] - b_years[death_mask]
    df_pat.loc[living_mask, "AgeIn2026"] = CURRENT_YEAR - b_years[living_mask]

    df_pat.drop(columns=["DateBirth", "DateDeath"], inplace=True, errors="ignore")
    df_pat.to_csv(pat_out, sep=DELIMITER, index=False, encoding="utf-8")

    dob_map = dict(zip(df_pat["IDPatient"], b_years))
    print(f" -> Processed {len(df_pat):,} patients. Birth lookup loaded.")

    # 1B. Chunk-Process Child Tables
    child_specs = {
        "codes": (files["codes"], ["DateEvent"]),
        "medications": (files["medications"], ["DateMedicationStart", "DateMedicationEnd"]),
        "immunisations": (files["immunisations"], ["DateEvent"]),
    }

    for key, (filename, date_cols) in child_specs.items():
        file_in = os.path.join(BASE_PATH, filename)
        file_out = os.path.join(PREPROCESSED_DIR, filename)
        if not os.path.exists(file_in):
            continue

        print(f" -> Preprocessing child table: {filename}...")
        chunk_iter = pd.read_csv(file_in, sep=DELIMITER, chunksize=1500000, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip")
        
        for i, chunk in enumerate(chunk_iter):
            chunk.columns = chunk.columns.str.strip()
            chunk["IDPatient"] = chunk["IDPatient"].astype(str).str.strip()
            chunk = chunk.dropna(subset=["IDPatient"])

            p_births = chunk["IDPatient"].map(dob_map)

            for d_col in date_cols:
                if d_col in chunk.columns:
                    e_years = parse_date_to_year(chunk[d_col])
                    age_col = d_col.replace("Date", "AgeAt")
                    age_series = e_years - p_births
                    valid_mask = (age_series >= 0) & age_series.notna()
                    
                    # Fixed: using pandas .where() to properly support nullable "Int64"
                    chunk[age_col] = age_series.where(valid_mask, np.nan).astype("Int64")

            chunk.drop(columns=[c for c in date_cols if c in chunk.columns] + ["DateEvent"], inplace=True, errors="ignore")
            chunk.to_csv(file_out, sep=DELIMITER, mode="w" if i == 0 else "a", header=(i == 0), index=False, encoding="utf-8")


# =====================================================================
# STEP 2 & 3: STRATIFIED SAMPLING & HMA FITTING (UPDATED)
# =====================================================================
def run_step2_and_3_training():
    print("\n--- STEP 2 & 3: Stratified Sampling & HMASynthesizer Fitting ---")
    
    pat_path = os.path.join(PREPROCESSED_DIR, files["patients"])
    df_patients_all = pd.read_csv(pat_path, sep=DELIMITER, encoding=READ_ENCODING, dtype=COL_TYPES["patients"])
    df_patients_all.columns = df_patients_all.columns.str.strip()

    real_code_vocab_freq = {}
    code_counts, med_counts, imm_counts = {}, {}, {}

    for table_key, storage_dict in [("codes", code_counts), ("medications", med_counts), ("immunisations", imm_counts)]:
        f_path = os.path.join(PREPROCESSED_DIR, files[table_key])
        if os.path.exists(f_path):
            vocab_counter = {}
            code_col = "SNOMEDCode" if table_key == "codes" else ("IDMultiLexProduct" if table_key == "medications" else "ImmsSNOMEDCode")
            chunk_iter = pd.read_csv(f_path, sep=DELIMITER, chunksize=1500000, encoding=READ_ENCODING, low_memory=False, on_bad_lines="skip")
            
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.str.strip()
                if "IDPatient" in chunk.columns:
                    vc_pat = chunk["IDPatient"].value_counts()
                    for pid, cnt in vc_pat.items():
                        storage_dict[str(pid)] = storage_dict.get(str(pid), 0) + cnt
                if code_col in chunk.columns:
                    vc_code = chunk[code_col].dropna().value_counts()
                    for c_val, cnt in vc_code.items():
                        vocab_counter[str(c_val)] = vocab_counter.get(str(c_val), 0) + cnt
            real_code_vocab_freq[table_key] = vocab_counter

    with open(os.path.join(MODEL_DIR, "real_vocab_frequencies.json"), "w") as f:
        json.dump(real_code_vocab_freq, f)

    # Feature engineering for stratified sampling
    df_patients_all["LogNumCodes"] = np.log1p(df_patients_all["IDPatient"].map(code_counts).fillna(0))
    df_patients_all["LogNumMeds"] = np.log1p(df_patients_all["IDPatient"].map(med_counts).fillna(0))
    df_patients_all["LogNumImms"] = np.log1p(df_patients_all["IDPatient"].map(imm_counts).fillna(0))

    df_patients_all["UtilGroup"] = pd.qcut(df_patients_all["LogNumCodes"] + df_patients_all["LogNumMeds"], q=4, labels=["L", "M", "H", "VH"], duplicates="drop")
    df_patients_all["AgeGroup"] = pd.qcut(df_patients_all["AgeIn2026"].fillna(50), q=5, labels=False, duplicates="drop")
    df_patients_all["StratifyKey"] = df_patients_all["Gender"].astype(str) + "_" + df_patients_all["AgeGroup"].astype(str) + "_" + df_patients_all["UtilGroup"].astype(str)

    def safe_sample(g, frac):
        n = max(1, min(len(g), int(len(g) * frac)))
        return g.sample(n=n, random_state=42)

    # Fixed pandas FutureWarning by passing include_groups=False
    if TEST_MODE:
        target_frac = TARGET_PATIENTS / len(df_patients_all)
        sampled_df = df_patients_all.groupby("StratifyKey", group_keys=False).apply(
            lambda g: safe_sample(g, target_frac), 
            include_groups=False
        )
        if len(sampled_df) > TARGET_PATIENTS:
            sampled_df = sampled_df.sample(n=TARGET_PATIENTS, random_state=42)
    else:
        sampled_df = df_patients_all.groupby("StratifyKey", group_keys=False).apply(
            lambda g: safe_sample(g, TRAIN_SAMPLE_FRACTION), 
            include_groups=False
        )

    sampled_patient_ids = set(sampled_df["IDPatient"].astype(str))
    df_patients_train = df_patients_all[df_patients_all["IDPatient"].astype(str).isin(sampled_patient_ids)].drop(
        columns=["UtilGroup", "AgeGroup", "StratifyKey"], errors="ignore"
    ).copy()

    # Ensure String IDs across all tables to prevent type mismatch during validation
    df_patients_train["IDPatient"] = df_patients_train["IDPatient"].astype(str)
    train_data = {"patients": df_patients_train.replace(r"^\s*$", np.nan, regex=True)}

    # Collect and filter child tables
    for table_key in ["codes", "medications", "immunisations"]:
        f_path = os.path.join(PREPROCESSED_DIR, files[table_key])
        if not os.path.exists(f_path):
            train_data[table_key] = pd.DataFrame(columns=list(COL_TYPES[table_key].keys()))
            continue

        chunks = []
        chunk_iter = pd.read_csv(f_path, sep=DELIMITER, chunksize=1500000, encoding=READ_ENCODING, dtype=COL_TYPES[table_key], on_bad_lines="skip")
        for chunk in chunk_iter:
            chunk.columns = chunk.columns.str.strip()
            chunk["IDPatient"] = chunk["IDPatient"].astype(str)
            filtered = chunk[chunk["IDPatient"].isin(sampled_patient_ids)]
            if not filtered.empty:
                chunks.append(filtered)

        if chunks:
            df_child = pd.concat(chunks, ignore_index=True)
            if table_key == "medications" and "AgeAtMedicationEnd" in df_child.columns and "AgeAtMedicationStart" in df_child.columns:
                df_child["AgeAtMedicationEnd"] = df_child["AgeAtMedicationEnd"].fillna(df_child["AgeAtMedicationStart"])
            df_child["IDPatient"] = df_child["IDPatient"].astype(str)
            train_data[table_key] = df_child.replace(r"^\s*$", np.nan, regex=True)
        else:
            train_data[table_key] = pd.DataFrame(columns=list(COL_TYPES[table_key].keys()))

    # Build Metadata explicitly to prevent auto-detecting invalid foreign keys
    global_metadata = MultiTableMetadata()
    
    # Detect table columns without auto-inferring cross-table relationships
    for table_name, df_table in train_data.items():
        if len(df_table) > 0:
            global_metadata.detect_table_from_dataframe(table_name=table_name, data=df_table)
        else:
            global_metadata.add_table(table_name=table_name)

    # Configure ID and Primary Key for parent
    if "patients" in global_metadata.tables:
        global_metadata.update_column(table_name="patients", column_name="IDPatient", sdtype="id")
        global_metadata.set_primary_key(table_name="patients", column_name="IDPatient")

    # Set explicit 1-to-N relationships solely on IDPatient
    for child_table in ["codes", "medications", "immunisations"]:
        if child_table in global_metadata.tables and "IDPatient" in global_metadata.tables[child_table].columns:
            global_metadata.update_column(table_name=child_table, column_name="IDPatient", sdtype="id")
            
            # Explicitly mark IDEvent as an ID if present
            if "IDEvent" in global_metadata.tables[child_table].columns:
                global_metadata.update_column(table_name=child_table, column_name="IDEvent", sdtype="id")
                
            global_metadata.add_relationship(
                parent_table_name="patients",
                child_table_name=child_table,
                parent_primary_key="IDPatient",
                child_foreign_key="IDPatient"
            )

    # Ensure no lingering foreign key mappings on numerical columns
    global_metadata.validate()
    global_metadata.save_to_json(filepath=os.path.join(MODEL_DIR, "global_metadata.json"))

    model_path = os.path.join(MODEL_DIR, "global_murmur_synthesizer.pkl")
    if os.path.exists(model_path):
        print(f" -> Loading pre-trained HMASynthesizer from {model_path}...")
        synthesizer = HMASynthesizer.load(filepath=model_path)
    else:
        print(" -> Fitting Unified HMASynthesizer model...")
        synthesizer = HMASynthesizer(global_metadata)
        
        # Add Inequality constraint if using newer SDV constraints module or skip if using basic HMA
        if "medications" in train_data and len(train_data["medications"]) > 0:
            try:
                from sdv.constraints import Inequality
                med_constraint = Inequality(
                    table_name="medications",
                    low_column_name="AgeAtMedicationStart",
                    high_column_name="AgeAtMedicationEnd"
                )
                synthesizer.add_constraints(constraints=[med_constraint])
            except Exception as e:
                # If constraints module is structured differently in your SDV version, proceed with standard fit
                pass

        synthesizer.fit(train_data)
        synthesizer.save(filepath=model_path)

    return synthesizer


# =====================================================================
# STEP 4 & 5: CHUNKED GENERATION & CLINICAL POST-FILTERING
# =====================================================================
def run_step4_and_5_generation(synthesizer):
    print("\n--- STEP 4 & 5: Scaled Chunk Generation & Clinical Rule Enforcement ---")
    
    # Generate temporary chunks
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
                df.to_csv(os.path.join(TEMP_CHUNK_DIR, f"{table_name}_chunk_{chunk_idx}.csv"), sep=DELIMITER, index=False, encoding="utf-8")

        del sampled_data
        gc.collect()

    with open(os.path.join(MODEL_DIR, "real_vocab_frequencies.json"), "r") as f:
        real_code_vocab_freq = json.load(f)

    # Build cross-table context dictionary
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
    
    global_counters = {
        "codes": 1,
        "medications": 1,
    }

    for table_name, orig_filename in files.items():
        final_output_path = os.path.join(OUTPUT_DIR, f"synthetic_{orig_filename}")
        real_counts = real_code_vocab_freq.get(table_name, {})
        code_col = "SNOMEDCode" if table_name == "codes" else ("IDMultiLexProduct" if table_name == "medications" else "ImmsSNOMEDCode")
        desc_col = "NameOfMedication" if table_name == "medications" else ("SNOMEDCode" if table_name == "codes" else "ImmsReadCode")

        first_chunk = True
        for chunk_idx in range(NUM_GEN_CHUNKS):
            chunk_file = os.path.join(TEMP_CHUNK_DIR, f"{table_name}_chunk_{chunk_idx}.csv")
            if not os.path.exists(chunk_file):
                continue

            chunk_df = pd.read_csv(chunk_file, sep=DELIMITER, dtype=str)

            if table_name == "patients":
                chunk_df = chunk_df.drop(columns=["LogNumCodes", "LogNumMeds", "LogNumImms"], errors="ignore")

            if "IDPatient" in chunk_df.columns:
                chunk_df["_Gender"] = chunk_df["IDPatient"].map(lambda x: patient_context.get(x, {}).get("Gender", "U"))
                chunk_df["_AgeAtDeath"] = chunk_df["IDPatient"].map(lambda x: patient_context.get(x, {}).get("AgeAtDeath", np.nan))
                chunk_df["IDPatient"] = chunk_df["IDPatient"].map(patient_map)
                chunk_df = chunk_df.dropna(subset=["IDPatient"])

            if table_name in ["codes", "medications", "immunisations"]:
                # Sex-specific keyword filtering
                if desc_col in chunk_df.columns:
                    male_mask = chunk_df["_Gender"] == "M"
                    fem_mask = chunk_df["_Gender"] == "F"
                    female_match = chunk_df[desc_col].str.contains(FEMALE_REGEX, na=False)
                    male_match = chunk_df[desc_col].str.contains(MALE_REGEX, na=False)

                    chunk_df = chunk_df[~(male_mask & female_match)]
                    chunk_df = chunk_df[~(fem_mask & male_match)]

                # Post-mortem event removal
                age_col = "AgeAtEvent" if "AgeAtEvent" in chunk_df.columns else "AgeAtMedicationStart"
                if age_col in chunk_df.columns:
                    event_age = pd.to_numeric(chunk_df[age_col], errors="coerce")
                    post_mortem = chunk_df["_AgeAtDeath"].notna() & event_age.notna() & (event_age > chunk_df["_AgeAtDeath"])
                    chunk_df = chunk_df[~post_mortem]

                # Noise threshold filter
                if code_col in chunk_df.columns:
                    rare_codes = {k for k, v in real_counts.items() if v < 20}
                    chunk_df = chunk_df[~chunk_df[code_col].isin(rare_codes)]

                chunk_df = chunk_df.drop(columns=["_Gender", "_AgeAtDeath"], errors="ignore")

            # Sequential Event ID generation for event tables
            if table_name in ["codes", "medications"]:
                num_rows = len(chunk_df)
                start_c = global_counters[table_name]
                prefix_code = "EVT_C" if table_name == "codes" else "EVT_M"
                chunk_df["IDEvent"] = [f"SYN_{prefix_code}_{i:09d}" for i in range(start_c, start_c + num_rows)]
                global_counters[table_name] += num_rows

            chunk_df.to_csv(final_output_path, sep=DELIMITER, mode="w" if first_chunk else "a", header=first_chunk, index=False, encoding="utf-8")
            first_chunk = False

    shutil.rmtree(TEMP_CHUNK_DIR, ignore_errors=True)
    print(f"\nSUCCESS: Pipeline complete. Generated outputs written to: {OUTPUT_DIR}")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_step1_preprocessing()
    model = run_step2_and_3_training()
    run_step4_and_5_generation(model)