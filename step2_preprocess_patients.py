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
    
    # Initialize target columns as nullable Int64
    df['AgeIn2026'] = pd.Series(dtype="Int64")
    df['AgeAtDeath'] = pd.Series(dtype="Int64")
    
    # 1. Calculate AgeAtDeath for DECEASED patients
    death_mask = df['DateBirth'].notna() & df['DateDeath'].notna() & (df['DateDeath'] >= df['DateBirth'])
    if death_mask.any():
        death_days = (df.loc[death_mask, 'DateDeath'] - df.loc[death_mask, 'DateBirth']).dt.days
        df.loc[death_mask, 'AgeAtDeath'] = np.ceil(death_days / 365.25).astype("Int64")
        
    # 2. Calculate AgeIn2026 ONLY for LIVING patients (DateDeath is null/blank)
    living_mask = df['DateBirth'].notna() & df['DateDeath'].isna()
    if living_mask.any():
        df.loc[living_mask, 'AgeIn2026'] = (CURRENT_YEAR - df.loc[living_mask, 'DateBirth'].dt.year).astype("Int64")
        
    # Drop raw direct date columns to ensure complete anonymization
    df = df.drop(columns=['DateBirth', 'DateDeath'], errors='ignore')
    
    # Overwrite registry with safe schema
    df.to_csv(path, index=False, sep=',')
    print("SUCCESS: Parent patient dataset is fully anonymized and logically separated!")

if __name__ == "__main__":
    process_parent_patients()