import os
import pandas as pd

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_DIR = "hma/Preprocessed_Data"
READ_ENCODING = "latin-1"
DELIMITER = "|"  # Will auto-detect or default based on file inspection
CHUNKSIZE = 1_000_000

EXPECTED_SCHEMAS = {
    "SRPatient.csv": ["IDPatient", "Gender", "AgeIn2026", "AgeAtDeath", "DiseaseMedRatio"],
    "SRCode.csv": ["CTV3Code", "SNOMEDCode", "EpisodeType", "IDEvent", "IDPatient", "AgeAtEvent"],
    "SRPrimaryCareMedication.csv": ["IDMultiLexProduct", "NameOfMedication", "MedicationDosage", "IDPatient", "AgeAtEvent", "AgeAtMedicationStart", "AgeAtMedicationEnd"],
    "SRImmunisation.csv": ["Dose", "Location", "ImmsReadCode", "ImmsSNOMEDCode", "IDPatient", "AgeAtEvent"]
}
def detect_delimiter(file_path):
    with open(file_path, "r", encoding=READ_ENCODING, errors="ignore") as f:
        first_line = f.readline()
        return "," if "," in first_line else "|"

def validate_preprocessing():
    print("=====================================================================")
    print(" 🔍 STARTING PREPROCESSING VALIDATION SUITE")
    print(f" Target Directory: {DATA_DIR}")
    print("=====================================================================\n")

    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Directory '{DATA_DIR}' does not exist. Run preprocessing first.")
        return

    # =====================================================================
    # 1. VALIDATE SRPATIENT (Parent Table)
    # =====================================================================
    pat_path = os.path.join(DATA_DIR, "SRPatient.csv")
    if not os.path.exists(pat_path):
        print("❌ CRITICAL ERROR: 'SRPatient.csv' is missing.")
        return

    delimiter = detect_delimiter(pat_path)
    print(f"--- Validating Parent Table: SRPatient.csv (Delimiter: '{delimiter}') ---")
    
    pat_df = pd.read_csv(pat_path, sep=delimiter, encoding=READ_ENCODING, low_memory=False, dtype=str)
    
    # Check Schema
    assert list(pat_df.columns) == EXPECTED_SCHEMAS["SRPatient.csv"], f"Schema mismatch in SRPatient. Found: {list(pat_df.columns)}"
    print("  ✅ Schema matches expected format.")

    # Check Uniqueness of IDPatient
    dup_count = pat_df["IDPatient"].duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate patient IDs in SRPatient!"
    print("  ✅ All patient IDs are unique.")

    # Check Gender values
    valid_genders = {"F", "M", "U"}
    invalid_genders = set(pat_df["Gender"].unique()) - valid_genders
    assert not invalid_genders, f"Found invalid gender values: {invalid_genders}"
    print("  ✅ Gender values are strictly valid (F, M, U).")

    valid_patient_ids = set(pat_df["IDPatient"])
    print(f"  📊 Total Unique Patients: {len(valid_patient_ids):,}\n")

    # =====================================================================
    # 2. VALIDATE CHILD TABLES (SRCode, SRPrimaryCareMedication, SRImmunisation)
    # =====================================================================
    child_files = ["SRCode.csv", "SRPrimaryCareMedication.csv", "SRImmunisation.csv"]

    for filename in child_files:
        file_path = os.path.join(DATA_DIR, filename)
        print(f"--- Validating Child Table: {filename} ---")

        if not os.path.exists(file_path):
            print(f"  ⚠️ Warning: {filename} not found. Skipping.")
            continue

        total_rows = 0
        orphan_rows = 0
        patients_with_events = set()

        chunk_iter = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=READ_ENCODING,
            chunksize=CHUNKSIZE,
            low_memory=False,
            dtype=str
        )

        for chunk in chunk_iter:
            # Verify schema on first chunk
            if total_rows == 0:
                assert list(chunk.columns) == EXPECTED_SCHEMAS[filename], f"Schema mismatch in {filename}. Found: {list(chunk.columns)}"
                print(f"  ✅ Schema for {filename} matches expected format.")

            total_rows += len(chunk)
            
            # Track unique patients appearing in this child table
            patients_with_events.update(chunk["IDPatient"].dropna().unique())

            # Check for orphan records (foreign key integrity)
            orphans = ~chunk["IDPatient"].isin(valid_patient_ids)
            orphan_rows += orphans.sum()

        assert orphan_rows == 0, f"❌ INTEGRITY FAILURE: Found {orphan_rows:,} orphan records in {filename} with no matching IDPatient in SRPatient!"
        print(f"  ✅ Foreign Key Integrity Verified: 0 orphan records out of {total_rows:,} rows.")
        print(f"  📊 Total Valid Rows: {total_rows:,}")
        print(f"  📊 Unique Patients with Records: {len(patients_with_events):,}\n")

    print("=====================================================================")
    print(" 🎉 ALL VALIDATION CHECKS PASSED SUCCESSFULLY! Ready for SDV HMA.")
    print("=====================================================================")

if __name__ == "__main__":
    validate_preprocessing()