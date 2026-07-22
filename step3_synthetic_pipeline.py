import os
import shutil
import sys
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer

# =====================================================================
# CONFIGURATION
# =====================================================================
INPUT_DIR = "./processed"
OUTPUT_DIR = "./Synthetic_Output"
MODEL_DIR = "./Trained_Models"

# Scale batches up so each model processes ~1.2M rows instead of ~14M rows
NUM_BATCHES = 50  

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

files = {
    'patients': 'SRPatient.csv',
    'codes': 'SRCode.csv',
    'medications': 'SRPrimaryCareMedication.csv',
    'immunisations': 'SRImmunisation.csv'
}

# Step 1: Assign Patients to Batches
print("Step 1: Reading patients registry and assigning batch IDs...")
patients_path = os.path.join(INPUT_DIR, files['patients'])
df_patients = pd.read_csv(patients_path, low_memory=False)
df_patients['IDPatient'] = pd.to_numeric(df_patients['IDPatient'], errors='coerce').astype("Int64")
df_patients = df_patients.dropna(subset=['IDPatient'])

np.random.seed(42)
patient_ids = df_patients['IDPatient'].unique()
patient_to_batch = {pid: np.random.randint(0, NUM_BATCHES) for pid in patient_ids}
df_patients['BatchID'] = df_patients['IDPatient'].map(patient_to_batch)

# Step 2: Training & Sampling
for batch_idx in range(NUM_BATCHES):
    print(f"\n--- PROCESSING BATCH {batch_idx + 1}/{NUM_BATCHES} ---")
    
    batch_patients = df_patients[df_patients['BatchID'] == batch_idx].drop(columns=['BatchID'])
    batch_patient_ids = set(batch_patients['IDPatient'])
    
    batch_data = {'patients': batch_patients}
    
    for table_key in ['codes', 'medications', 'immunisations']:
        file_path = os.path.join(INPUT_DIR, files[table_key])
        chunks = []
        
        # Ingestion with fast row filtering
        for chunk in pd.read_csv(file_path, chunksize=500000, low_memory=False):
            chunk['IDPatient'] = pd.to_numeric(chunk['IDPatient'], errors='coerce').astype("Int64")
            
            # Sub-slice using standard set membership
            filtered_chunk = chunk[chunk['IDPatient'].isin(batch_patient_ids)]
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
            
        df_child = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=['IDPatient'])
        df_child = df_child.replace(r'^\s*$', np.nan, regex=True)
        
        # Merge constraint lookup attributes
        if 'AgeAtDeath' in batch_patients.columns:
            df_child = df_child.merge(batch_patients[['IDPatient', 'AgeAtDeath']], on='IDPatient', how='left')
            
        batch_data[table_key] = df_child

    # Detect metadata from batch data schema
    metadata = Metadata.detect_from_dataframes(data=batch_data, infer_keys='primary_only')
    metadata.update_column(table_name='patients', column_name='IDPatient', sdtype='id')
    metadata.set_primary_key(table_name='patients', column_name='IDPatient')
    
    for table_name in ['codes', 'medications', 'immunisations']:
        metadata.update_column(table_name=table_name, column_name='IDPatient', sdtype='id')
        metadata.add_relationship(
            parent_table_name='patients', child_table_name=table_name, 
            parent_primary_key='IDPatient', child_foreign_key='IDPatient'
        )

    # Instantiate synthesizer per batch
    synthesizer = HMASynthesizer(metadata)
    
    # Configure timeline boundaries
    constraints_list = []
    for t_name, age_col in [('codes', 'AgeAtEvent'), ('medications', 'AgeAtMedicationStart'), ('immunisations', 'AgeAtEvent')]:
        if age_col in batch_data[t_name].columns and 'AgeAtDeath' in batch_data[t_name].columns:
            constraints_list.append({
                'constraint_class': 'Inequality',
                'table_name': t_name,
                'constraint_parameters': {'low_column_name': age_col, 'high_column_name': 'AgeAtDeath'}
            })

    if 'AgeAtMedicationStart' in batch_data['medications'].columns and 'AgeAtMedicationEnd' in batch_data['medications'].columns:
        constraints_list.append({
            'constraint_class': 'Inequality',
            'table_name': 'medications',
            'constraint_parameters': {'low_column_name': 'AgeAtMedicationStart', 'high_column_name': 'AgeAtMedicationEnd'}
        })

    if constraints_list:
        synthesizer.add_constraints(constraints=constraints_list)

    print(" -> Fitting HMA Model...")
    synthesizer.fit(batch_data)
    
    print(" -> Generating synthetic tables...")
    batch_synthetic = synthesizer.sample(scale=1.0)
    
    # Clean up output temporary columns
    for table_name, df in batch_synthetic.items():
        if table_name != 'patients' and 'AgeAtDeath' in df.columns:
            df = df.drop(columns=['AgeAtDeath'])
            
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{table_name}")
        os.makedirs(temp_dir, exist_ok=True)
        df.to_csv(os.path.join(temp_dir, f"batch_{batch_idx}.csv"), index=False)

# Step 3: Reassemble
print("\nMerging completed batches...")
for table_name, orig_filename in files.items():
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{table_name}")
    final_output_path = os.path.join(OUTPUT_DIR, f"synthetic_{orig_filename}")
    
    batch_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not batch_files:
        continue
        
    shutil.copyfile(batch_files[0], final_output_path)
    with open(final_output_path, 'a') as outfile:
        for batch_file in batch_files[1:]:
            with open(batch_file, 'r') as infile:
                next(infile)  # Skip CSV header
                shutil.copyfileobj(infile, outfile)
                
    shutil.rmtree(temp_dir)

print("\nSUCCESS: Multi-batch synthetic data output pipeline completed!")