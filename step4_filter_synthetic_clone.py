import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# =====================================================================
# CONFIGURATION
# =====================================================================
REAL_PATIENT_PATH = "SRPatient.csv"
SYNTHETIC_PATIENT_PATH = "./Synthetic_Output/synthetic_SRPatient.csv"
FILTERED_OUTPUT_PATH = "./Synthetic_Output/synthetic_SRPatient_secure.csv"

# CLONE THRESHOLD:
# 0.0 means the synthetic row must be an absolute exact 100% clone to be dropped.
# 0.05 is a safe threshold (drops anything that is 95%+ identical to a real person).
CLONE_THRESHOLD = 0.02 

def secure_synthetic_patients():
    if not os.path.exists(REAL_PATIENT_PATH) or not os.path.exists(SYNTHETIC_PATIENT_PATH):
        print("Error: Ensure both real and synthetic patient CSV files exist.")
        return

    print("Step 1: Loading real and synthetic patient files...")
    df_real = pd.read_csv(REAL_PATIENT_PATH, sep=',')
    df_synth = pd.read_csv(SYNTHETIC_PATIENT_PATH, sep=',')
    
    # Keep track of original synthetic IDs so we can drop them later
    original_synth_ids = df_synth['IDPatient'].copy()
    
    # Define features to calculate distance on (excluding unique relational IDs)
    distance_features = ['Gender', 'AgeIn2026', 'AgeAtDeath']
    
    # Isolate calculation columns
    real_feats = df_real[distance_features].copy()
    synth_feats = df_synth[distance_features].copy()

    print("Step 2: Preprocessing and normalizing features...")
    # Fill missing age values with a sentinel value (e.g., -1) so the distance engine can compute them
    real_feats = real_feats.fillna(-1)
    synth_feats = synth_feats.fillna(-1)
    
    # Convert categorical 'Gender' to numeric (One-Hot Encoding)
    combined = pd.concat([real_feats, synth_feats], keys=['real', 'synth'])
    combined_encoded = pd.get_dummies(combined, columns=['Gender'], drop_first=True)
    
    real_encoded = combined_encoded.xs('real')
    synth_encoded = combined_encoded.xs('synth')

    # Normalize age variables to scale of [0, 1]
    scaler = MinMaxScaler()
    numerical_cols = ['AgeIn2026', 'AgeAtDeath']
    
    real_encoded[numerical_cols] = scaler.fit_transform(real_encoded[numerical_cols])
    synth_encoded[numerical_cols] = scaler.transform(synth_encoded[numerical_cols])

    print("Step 3: Finding the nearest real-world neighbor for each synthetic patient...")
    # Fit the nearest neighbors engine on the REAL patient registry
    # Using L1 distance (Manhattan distance) works exceptionally well for mixed data
    nn = NearestNeighbors(n_neighbors=1, metric='manhattan', n_jobs=-1)
    nn.fit(real_encoded)

    # For every synthetic record, find the distance to its closest real neighbor
    distances, indices = nn.kneighbors(synth_encoded)
    
    # Flatten the distance array
    distances = distances.flatten()

    # Identify clone candidates
    clone_mask = distances <= CLONE_THRESHOLD
    num_clones = np.sum(clone_mask)
    total_synth = len(df_synth)
    
    print(f"\nAnalysis Completed:")
    print(f" -> Total Synthetic Patients Checked: {total_synth:,}")
    print(f" -> Potential Clones Identified: {num_clones:,} ({ (num_clones / total_synth) * 100:.2f}%)")

    # Step 4: Filtering out the clones
    if num_clones > 0:
        print(f" -> Removing clone records with distance <= {CLONE_THRESHOLD}...")
        secure_df_synth = df_synth[~clone_mask].copy()
        
        # Save the secure patient dataset
        secure_df_synth.to_csv(FILTERED_OUTPUT_PATH, index=False, sep=',')
        print(f" -> Safe dataset written to: {FILTERED_OUTPUT_PATH}")
        
        # (Optional) Clean up child tables to match
        print("\nStep 5: Cascading deletion to child tables...")
        allowed_patient_ids = set(secure_df_synth['IDPatient'].astype(str))
        
        child_files = ['synthetic_SRCode.csv', 'synthetic_SRPrimaryCareMedication.csv', 'synthetic_SRImmunisation.csv']
        for child_file in child_files:
            child_path = os.path.join("./Synthetic_Output", child_file)
            if os.path.exists(child_path):
                print(f" -> Filtering {child_file}...")
                df_child = pd.read_csv(child_path, sep=',')
                df_child['IDPatient'] = df_child['IDPatient'].astype(str)
                
                # Only keep events for patients who survived the privacy filter
                filtered_child = df_child[df_child['IDPatient'].isin(allowed_patient_ids)]
                filtered_child.to_csv(child_path, index=False, sep=',')
                
    else:
        print(" -> Perfect! No synthetic clones detected. Your data is secure.")
        df_synth.to_csv(FILTERED_OUTPUT_PATH, index=False, sep=',')

if __name__ == "__main__":
    secure_synthetic_patients()