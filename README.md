# Synthetic Dataset Generation Pipeline

A production-grade, memory-safe pipeline for generating **privacy-preserving, relationally consistent synthetic clinical datasets** from **SystmOne** primary care records.

The pipeline is designed to process very large healthcare datasets efficiently while preserving longitudinal relationships between tables. It removes direct patient identifiers, converts absolute timestamps into relative age-based features, and applies post-generation privacy auditing to reduce the risk of synthetic data memorization.

---

# Project Structure

```
.
├── SRPatient.csv                    # Raw patient demographics
├── SRCode.csv                       # Clinical event records
├── SRPrimaryCareMedication.csv      # Medication history
├── SRImmunisation.csv               # Immunisation records
├── step1_preprocess_child_tables.py # Convert dates to relative age metrics
├── step2_preprocess_patients.py     # Anonymise patient table
├── step3_synthetic_pipeline.py      # Train HMASynthesizer and generate synthetic data
├── step4_filter_synthetic_clones.py # Remove potential memorised synthetic records
├── requirements.txt                 # Python dependencies
├── processed/                       # Intermediate anonymised datasets
├── Trained_Models/                  # Serialized synthesizer models
└── Synthetic_Output/                # Final privacy-filtered synthetic datasets
```

---

# Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

# Execution Pipeline

Run the pipeline sequentially:

```bash
python step1_preprocess_child_tables.py
python step2_preprocess_patients.py
python step3_synthetic_pipeline.py
python step4_filter_synthetic_clones.py
```

---

# Pipeline Architecture

## Step 1 – Preprocess Child Tables

Processes large clinical event tables in **1,000,000-row chunks** to minimise memory consumption.

### Operations

* Reads raw clinical datasets incrementally
* Maps patient date of birth onto event records
* Computes:

  * `AgeAtEvent`
  * `AgeAtMedicationStart`
  * `AgeAtMedicationEnd`
* Removes all absolute event timestamps
* Writes anonymised outputs to `processed/`

---

## Step 2 – Preprocess Patient Table

Anonymises the patient registry while preserving clinically meaningful age information.

### Operations

* Converts patient identifiers to nullable 64-bit integers
* Computes:

  * `AgeAtDeath` (deceased patients)
  * `AgeIn2026` (living patients)
* Removes:

  * `DateBirth`
  * `DateDeath`
* Saves the processed registry to:

```
processed/SRPatient.csv
```

---

## Step 3 – Synthetic Data Generation

Generates relationally consistent synthetic datasets using the **SDV HMASynthesizer**.

### Memory Optimisation

To support datasets containing millions of records, patients are partitioned into **50 independent batches**, allowing model training within typical workstation memory limits.

### Model Constraints

The synthesizer enforces logical temporal consistency, including:

* `AgeAtEvent ≤ AgeAtDeath`
* `AgeAtMedicationStart ≤ AgeAtMedicationEnd`

### Outputs

* Trained models stored in:

```
Trained_Models/
```

* Synthetic datasets stored in:

```
Synthetic_Output/
```

---

## Step 4 – Privacy Audit and Clone Removal

Performs a final privacy validation step to reduce the likelihood of releasing memorised patient records.

### Process

* Compares synthetic patients with the original registry using **Manhattan Distance**
* Removes records below the configured clone threshold:

```
CLONE_THRESHOLD = 0.001
```

* Cascades deleted patient IDs through all related child tables
* Produces the final privacy-filtered synthetic datasets

Example output:

```
Synthetic_Output/synthetic_SRPatient_secure.csv
```

---

# Privacy and Security Features

| Feature                | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| Absolute anonymisation | All birth dates, death dates and event timestamps are removed after preprocessing. |
| Relational consistency | Parent-child relationships and longitudinal event structure are preserved.         |
| Temporal privacy       | Absolute dates are replaced with relative age-based features.                      |
| Memory-safe processing | Streaming and chunked processing enables handling of very large datasets.          |
| Anti-memorisation      | Near-identical synthetic records are detected and removed before release.          |

---

# Processing Workflow

```
Raw SystmOne Tables
        │
        ▼
Step 1
Child Table Preprocessing
        │
        ▼
Step 2
Patient Table Anonymisation
        │
        ▼
Step 3
HMASynthesizer Training
        │
        ▼
Synthetic Dataset Generation
        │
        ▼
Step 4
Clone Detection & Privacy Audit
        │
        ▼
Final Secure Synthetic Dataset
```

---

# Output

The completed pipeline produces a fully synthetic, relationally consistent clinical dataset that:

* preserves statistical characteristics of the source data
* maintains relationships between patients and clinical events
* removes direct identifiers and absolute temporal information
* supports scalable processing of very large datasets
* includes post-generation privacy auditing to reduce memorisation risk

---

**Status:** ✅ Production-ready pipeline for scalable synthetic clinical data generation with integrated privacy preservation and relational integrity.
