import os
import pandas as pd
import numpy as np

INPUT_DIR = "."
PATIENT_FILE = "SRPatient.csv"
CURRENT_YEAR = 2026

def process_parent_patients():
    path = os.path.join(INPUT_DIR, PATIENT_FILE)
    print(f"Reading patient registry: {path}...")
    
    df = pd.read_csv(path, sep=',', low_memory=False)
    df.columns = df.columns.str.strip()
    
    # Standardize Patient ID to nullable integer
    df['IDPatient'] = pd.to_numeric(df['IDPatient'], errors='coerce').astype("Int64")
    df['DateBirth'] = pd.to_datetime(df['DateBirth'], errors='coerce')
    df['DateDeath'] = pd.to_datetime(df['DateDeath'], errors='coerce')
    
    # Initialize the target columns as Int64
    df['AgeIn2026'] = pd.Series(dtype="Int64")
    df['AgeAtDeath'] = pd.Series(dtype="Int64")
    
    # Calculate AgeIn2026
    birth_year_mask = df['DateBirth'].notna()
    if birth_year_mask.any():
        df.loc[birth_year_mask, 'AgeIn2026'] = (CURRENT_YEAR - df.loc[birth_year_mask, 'DateBirth'].dt.year).astype("Int64")
    
    # Calculate AgeAtDeath
    death_mask = df['DateBirth'].notna() & df['DateDeath'].notna() & (df['DateDeath'] >= df['DateBirth'])
    if death_mask.any():
        death_days = (df.loc[death_mask, 'DateDeath'] - df.loc[death_mask, 'DateBirth']).dt.days
        df.loc[death_mask, 'AgeAtDeath'] = np.ceil(death_days / 365.25).astype("Int64")
        
    # Drop direct identifying date columns (Patient ID is already not in this table)
    df = df.drop(columns=['DateBirth', 'DateDeath'], errors='ignore')
    
    # Overwrite registry with safe schema
    df.to_csv(path, index=False, sep=',')
    print("SUCCESS: Parent patient dataset is fully anonymized!")

if __name__ == "__main__":
    process_parent_patients()