import pandas as pd
import numpy as np
import os

INPUT_DIR = "."  # Direct directory containing your files
CHUNK_SIZE = 500000

files = {
    'patients': 'SRPatient.csv',
    'codes': 'SRCode.csv',
    'medications': 'SRPrimaryCareMedication.csv',
    'immunisations': 'SRImmunisation.csv'
}

def preprocess_child_tables():
    patient_path = os.path.join(INPUT_DIR, files['patients'])
    print("Loading patient registry baseline index...")
    srpatient_df = pd.read_csv(patient_path, usecols=["IDPatient", "DateBirth"])
    srpatient_df["DateBirth"] = pd.to_datetime(srpatient_df["DateBirth"], errors='coerce')
    srpatient_df["IDPatient"] = pd.to_numeric(srpatient_df["IDPatient"], errors='coerce').astype("Int64")

    # Define child table configurations matching your actual headers
    child_configs = {
        'codes': (files['codes'], ["DateEvent"]),
        'medications': (files['medications'], ["DateEvent", "DateMedicationStart", "DateMedicationEnd"]),
        'immunisations': (files['immunisations'], ["DateEvent"])
    }

    for key, (filename, date_cols) in child_configs.items():
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(file_path):
            print(f"File {filename} not found, skipping...")
            continue
            
        print(f"\nProcessing child table: {filename}...")
        output_file = file_path + ".tmp"
        chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
        
        for i, chunk in enumerate(chunk_iter):
            print(f" -> Processing chunk {i + 1}")
            chunk["IDPatient"] = pd.to_numeric(chunk["IDPatient"], errors='coerce').astype("Int64")
            
            merged = chunk.merge(srpatient_df, on="IDPatient", how="left")
            
            for col in date_cols:
                if col in merged.columns:
                    merged[col] = pd.to_datetime(merged[col], errors='coerce')
                    
                    valid_mask = (
                        merged[col].notna() &
                        merged["DateBirth"].notna() &
                        (merged[col] >= merged["DateBirth"])
                    )
                    
                    age_col_name = col.replace("Date", "AgeAt")
                    merged[age_col_name] = pd.Series(dtype="Int64")
                    
                    if valid_mask.any():
                        ages_days = (merged.loc[valid_mask, col] - merged.loc[valid_mask, "DateBirth"]).dt.days
                        merged.loc[valid_mask, age_col_name] = np.ceil(ages_days / 365.25).astype("Int64")

            # Drop baseline DOB and raw datetime columns
            cols_to_drop = date_cols + ["DateBirth"]
            merged.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            write_header = (i == 0)
            merged.to_csv(output_file, index=False, mode='w' if write_header else 'a', header=write_header)
            
        os.replace(output_file, file_path)
        print(f" -> Success! Updated {filename} with age-relative timeline transformations.")

if __name__ == "__main__":
    preprocess_child_tables()