import os
import sys
import shutil
import pandas as pd
import numpy as np
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer

# =====================================================================
# CONFIGURATION
# =====================================================================
INPUT_DIR = "."
OUTPUT_DIR = "./Synthetic_Output"
MODEL_DIR = "./Trained_Models"
NUM_BATCHES = 5  

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

files = {
    'patients': 'SRPatient.csv',
    'codes': 'SRCode.csv',
    'medications': 'SRPrimaryCareMedication.csv',
    'immunisations': 'SRImmunisation.csv'
}

# =====================================================================
# 1. BATCH PARTITIONING
# =====================================================================
print("Step 1: Partitioning patients into relational batches...")
patients_path = os.path.join(INPUT_DIR, files['patients'])
if not os.path.exists(patients_path):
    print(f"CRITICAL ERROR: Parent file not found at {patients_path}")
    sys.exit(1)

df_patients = pd.read_csv(patients_path, sep=',', low_memory=False)
df_patients.columns = df_patients.columns.str.strip()
df_patients['IDPatient'] = pd.to_numeric(df_patients['IDPatient'], errors='coerce').astype("Int64")
df_patients = df_patients.dropna(subset=['IDPatient'])

total_original_patients = len(df_patients)
print(f" -> Found {total_original_patients:,} total real patient records.")

# Assign Batch IDs
np.random.seed(42)
patient_ids = df_patients['IDPatient'].unique()
patient_to_batch = {pid: np.random.randint(0, NUM_BATCHES) for pid in patient_ids}
df_patients['BatchID'] = df_patients['IDPatient'].map(patient_to_batch)

# =====================================================================
# 2. TRAINING & GENERATION LOOP
# =====================================================================
print(f"\nStep 2: Training HMA model on {NUM_BATCHES} batches...")

for batch_idx in range(NUM_BATCHES):
    print(f"\n--- PROCESSING BATCH {batch_idx + 1}/{NUM_BATCHES} ---")
    
    # Isolate patients assigned to this batch
    batch_patients = df_patients[df_patients['BatchID'] == batch_idx].drop(columns=['BatchID'])
    batch_patient_ids = set(batch_patients['IDPatient'])
    
    batch_data = {'patients': batch_patients}
    
    for table_key in ['codes', 'medications', 'immunisations']:
        file_path = os.path.join(INPUT_DIR, files[table_key])
        chunks = []
        
        for chunk in pd.read_csv(file_path, sep=',', chunksize=250000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunk['IDPatient'] = pd.to_numeric(chunk['IDPatient'], errors='coerce').astype("Int64")
            
            filtered_chunk = chunk[chunk['IDPatient'].isin(batch_patient_ids)]
            chunks.append(filtered_chunk)
            
        df_child = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=['IDPatient'])
        df_child = df_child.replace(r'^\s*$', np.nan, regex=True)
        
        # Bring AgeAtDeath down to the child table for intra-table constraint handling
        if 'AgeAtDeath' in batch_patients.columns:
            df_child = df_child.merge(batch_patients[['IDPatient', 'AgeAtDeath']], on='IDPatient', how='left')
            
        batch_data[table_key] = df_child
        
    print(f" -> Batch contains {len(batch_patients):,} patients.")
    batch_data['patients'] = batch_data['patients'].replace(r'^\s*$', np.nan, regex=True)

    # Metadata Schema Mapping
    metadata = Metadata.detect_from_dataframes(data=batch_data, infer_keys='primary_only')
    metadata.update_column(table_name='patients', column_name='IDPatient', sdtype='id')
    metadata.set_primary_key(table_name='patients', column_name='IDPatient')
    
    for table_name in ['codes', 'medications', 'immunisations']:
        metadata.update_column(table_name=table_name, column_name='IDPatient', sdtype='id')
        metadata.add_relationship(
            parent_table_name='patients', child_table_name=table_name, 
            parent_primary_key='IDPatient', child_foreign_key='IDPatient'
        )

    # Initialize Synthesizer
    synthesizer = HMASynthesizer(metadata)
    constraints_list = []
    
    # 1. Timeline Consistency Constraints via clean Dict format (Event age <= AgeAtDeath)
    for t_name, age_col in [('codes', 'AgeAtEvent'), ('medications', 'AgeAtMedicationStart'), 
                            ('immunisations', 'AgeAtEvent')]:
        if age_col in batch_data[t_name].columns and 'AgeAtDeath' in batch_data[t_name].columns:
            constraints_list.append({
                'constraint_class': 'Inequality',
                'table_name': t_name,
                'constraint_parameters': {
                    'low_column_name': age_col,
                    'high_column_name': 'AgeAtDeath'
                }
            })

    # 2. Medication start age <= end age consistency boundary
    if 'AgeAtMedicationStart' in batch_data['medications'].columns and 'AgeAtMedicationEnd' in batch_data['medications'].columns:
        constraints_list.append({
            'constraint_class': 'Inequality',
            'table_name': 'medications',
            'constraint_parameters': {
                'low_column_name': 'AgeAtMedicationStart',
                'high_column_name': 'AgeAtMedicationEnd'
            }
        })

    # Supply the compiled constraints list to the multi-table synthesizer
    if constraints_list:
        synthesizer.add_constraints(constraints=constraints_list)

    print(" -> Training Relational Synthesizer...")
    synthesizer.fit(batch_data)
    
    # Save trained model block to disk
    model_path = os.path.join(MODEL_DIR, f"safe_synthesizer_batch_{batch_idx}.pkl")
    synthesizer.save(filepath=model_path)
    
    # Sample synthetic generation
    print(" -> Generating synthetic tables...")
    batch_synthetic = synthesizer.sample(scale=1.0)
    
    # Drop temporary AgeAtDeath processing columns from final child outputs
    for table_name, df in batch_synthetic.items():
        if table_name != 'patients' and 'AgeAtDeath' in df.columns:
            df = df.drop(columns=['AgeAtDeath'])
            
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{table_name}")
        os.makedirs(temp_dir, exist_ok=True)
        df.to_csv(os.path.join(temp_dir, f"batch_{batch_idx}.csv"), sep=',', index=False)

# =====================================================================
# 3. REASSEMBLE FILES
# =====================================================================
print("\nStep 3: Merging synthetic batches into consolidated CSV outputs...")

for table_name, orig_filename in files.items():
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_{table_name}")
    final_output_path = os.path.join(OUTPUT_DIR, f"synthetic_{orig_filename}")
    
    print(f" -> Merging {table_name} tables into {final_output_path}...")
    
    batch_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not batch_files:
        continue
        
    first_file = batch_files[0]
    shutil.copyfile(first_file, final_output_path)
    
    with open(final_output_path, 'a') as outfile:
        for batch_file in batch_files[1:]:
            with open(batch_file, 'r') as infile:
                next(infile)  # Skip CSV header row
                shutil.copyfileobj(infile, outfile)
                
    shutil.rmtree(temp_dir)

print("\nSUCCESS: Dataset successfully preprocessed and synthesized using integer limits!")