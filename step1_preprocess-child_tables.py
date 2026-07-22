import os
import numpy as np
import pandas as pd

INPUT_DIR = "."  # Source directory containing raw CSV files
OUTPUT_DIR = "./processed"  # Target directory for transformed CSV files
CHUNK_SIZE = 1_000_000  # High vectorized throughput batch size

files = {
    "patients": "SRPatient.csv",
    "codes": "SRCode.csv",
    "medications": "SRPrimaryCareMedication.csv",
    "immunisations": "SRImmunisation.csv",
}


def preprocess_child_tables():
    # Ensure the destination output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patient_path = os.path.join(INPUT_DIR, files["patients"])
    print("Loading patient registry baseline into fast lookup map...")

    # Load only necessary columns and set index for O(1) map lookups
    srpatient_df = pd.read_csv(
        patient_path,
        usecols=["IDPatient", "DateBirth"],
        dtype={"IDPatient": "Int64"},
    )
    srpatient_df["DateBirth"] = pd.to_datetime(
        srpatient_df["DateBirth"], errors="coerce"
    )

    # Convert to a fast series map indexed by IDPatient
    dob_lookup = srpatient_df.set_index("IDPatient")["DateBirth"]
    del srpatient_df  # Free memory immediately

    # Child table configs
    child_configs = {
        "codes": (files["codes"], ["DateEvent"]),
        "medications": (
            files["medications"],
            ["DateEvent", "DateMedicationStart", "DateMedicationEnd"],
        ),
        "immunisations": (files["immunisations"], ["DateEvent"]),
    }

    for key, (filename, date_cols) in child_configs.items():
        file_path = os.path.join(INPUT_DIR, filename)
        output_file = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(file_path):
            print(f"File {filename} not found in {INPUT_DIR}, skipping...")
            continue

        print(f"\nProcessing child table: {filename} -> {output_file}...")

        chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)

        for i, chunk in enumerate(chunk_iter):
            print(
                f" -> Processing chunk {i + 1} (Rows {i * CHUNK_SIZE:,} - {(i + 1) * CHUNK_SIZE:,})..."
            )

            # Cast IDPatient efficiently
            chunk["IDPatient"] = pd.to_numeric(
                chunk["IDPatient"], errors="coerce"
            ).astype("Int64")

            # O(1) Map lookup instead of heavy DataFrame merge
            dob_series = chunk["IDPatient"].map(dob_lookup)

            for col in date_cols:
                if col in chunk.columns:
                    event_date = pd.to_datetime(chunk[col], errors="coerce")

                    # Calculate age in days directly with vectorized operations
                    valid_mask = (
                        event_date.notna()
                        & dob_series.notna()
                        & (event_date >= dob_series)
                    )

                    age_col_name = col.replace("Date", "AgeAt")

                    # Compute age in years using NumPy vectorization
                    ages = np.full(len(chunk), np.nan)
                    if valid_mask.any():
                        days_diff = (
                            event_date[valid_mask] - dob_series[valid_mask]
                        ).dt.days
                        ages[valid_mask] = np.ceil(days_diff / 365.25)

                    chunk[age_col_name] = pd.Series(
                        ages, index=chunk.index
                    ).astype("Int64")

            # Drop original datetime columns to conserve disk write IO
            chunk.drop(columns=date_cols, inplace=True, errors="ignore")

            write_header = i == 0
            chunk.to_csv(
                output_file,
                index=False,
                mode="w" if write_header else "a",
                header=write_header,
            )

        print(f" -> Success! Output saved to: {output_file}")


if __name__ == "__main__":
    preprocess_child_tables()