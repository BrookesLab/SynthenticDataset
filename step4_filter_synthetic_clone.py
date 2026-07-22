import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# =====================================================================
# CONFIGURATION
# =====================================================================
REAL_PATIENT_PATH = "./processed/SRPatient.csv"
SYNTHETIC_PATIENT_PATH = "./Synthetic_Output/synthetic_SRPatient.csv"
FILTERED_OUTPUT_PATH = "./Synthetic_Output/synthetic_SRPatient_secure.csv"
SYNTHETIC_DIR = "./Synthetic_Output"

# Threshold for distance (DCR)
CLONE_THRESHOLD = 0.001  # Set strictly to drop near-identical matches
CHUNK_SIZE = 500_000


def secure_synthetic_patients():
    if not os.path.exists(REAL_PATIENT_PATH) or not os.path.exists(
        SYNTHETIC_PATIENT_PATH
    ):
        print("Error: Ensure both real and synthetic patient CSV files exist.")
        return

    print("Step 1: Loading patient registries...")
    df_real = pd.read_csv(REAL_PATIENT_PATH, low_memory=False)
    df_synth = pd.read_csv(SYNTHETIC_PATIENT_PATH, low_memory=False)

    distance_features = ["Gender", "AgeIn2026", "AgeAtDeath"]
    available_cols = [
        c for c in distance_features if c in df_real.columns and c in df_synth.columns
    ]

    print(f" -> Calculating distance across features: {available_cols}")

    # Prepare real features
    real_df = df_real[available_cols].copy()
    synth_df = df_synth[available_cols].copy()

    # Fill NaNs safely
    real_df = real_df.fillna(-1)
    synth_df = synth_df.fillna(-1)

    # Encode Gender efficiently
    if "Gender" in available_cols:
        gender_map = {
            val: idx for idx, val in enumerate(real_df["Gender"].unique())
        }
        real_df["Gender"] = real_df["Gender"].map(gender_map).fillna(-1)
        synth_df["Gender"] = synth_df["Gender"].map(gender_map).fillna(-1)

    # Scale numeric columns
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_df.astype(np.float32))
    synth_scaled = scaler.transform(synth_df.astype(np.float32))

    # Free memory
    del df_real, real_df, synth_df
    gc.collect()

    print("Step 2: Fitting Nearest Neighbors model (float32 optimized)...")
    nn = NearestNeighbors(
        n_neighbors=1, metric="manhattan", algorithm="kd_tree", n_jobs=-1
    )
    nn.fit(real_scaled)

    print("Step 3: Calculating distance to closest real neighbor...")
    distances, _ = nn.kneighbors(synth_scaled)
    distances = distances.flatten()

    # Identify true clone matches
    clone_mask = distances <= CLONE_THRESHOLD
    num_clones = int(np.sum(clone_mask))
    total_synth = len(df_synth)

    print("\nAnalysis Completed:")
    print(f" -> Total Synthetic Patients Checked: {total_synth:,}")
    print(
        f" -> Potential Clones Flagged: {num_clones:,} ({(num_clones / total_synth) * 100:.2f}%)"
    )

    # Step 4: Write filtered parent dataset
    if num_clones > 0:
        print(f" -> Dropping {num_clones:,} clone records...")
        secure_df_synth = df_synth[~clone_mask].copy()
    else:
        print(" -> No clones identified under threshold.")
        secure_df_synth = df_synth.copy()

    secure_df_synth.to_csv(FILTERED_OUTPUT_PATH, index=False)
    print(f" -> Secure parent registry saved to: {FILTERED_OUTPUT_PATH}")

    # Step 5: Streaming Cascade Removal for Child Tables
    print("\nStep 5: Streaming cascaded filtering on child tables...")
    allowed_ids = set(
        pd.to_numeric(secure_df_synth["IDPatient"], errors="coerce")
        .dropna()
        .astype("Int64")
    )

    child_files = [
        "synthetic_SRCode.csv",
        "synthetic_SRPrimaryCareMedication.csv",
        "synthetic_SRImmunisation.csv",
    ]

    for child_file in child_files:
        child_path = os.path.join(SYNTHETIC_DIR, child_file)
        if not os.path.exists(child_path):
            continue

        print(f" -> Filtering {child_file} in chunks...")
        temp_child_path = child_path + ".tmp"

        chunk_iter = pd.read_csv(child_path, chunksize=CHUNK_SIZE, low_memory=False)

        for i, chunk in enumerate(chunk_iter):
            chunk["IDPatient"] = pd.to_numeric(
                chunk["IDPatient"], errors="coerce"
            ).astype("Int64")
            filtered_chunk = chunk[chunk["IDPatient"].isin(allowed_ids)]

            write_header = i == 0
            filtered_chunk.to_csv(
                temp_child_path,
                index=False,
                mode="w" if write_header else "a",
                header=write_header,
            )

        os.replace(temp_child_path, child_path)
        print(f" -> Successfully updated {child_file}")


if __name__ == "__main__":
    secure_synthetic_patients()