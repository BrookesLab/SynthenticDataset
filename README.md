# Synthetic Dataset Generation Pipeline

A multi-table synthetic data generation pipeline built on SDV’s Hierarchical Modeling Algorithm (HMA). The pipeline ingests raw primary care electronic health records, anonymizes temporal attributes into relative age offsets, fits a relational generative model using stratified sampling, and generates realistic synthetic cohorts subject to strict clinical rule validation.


# Project Structure

```
.
├── SRPatient.csv                    # Raw patient demographics
├── SRCode.csv                       # Clinical event records
├── SRPrimaryCareMedication.csv      # Medication history
├── SRImmunisation.csv               # Immunisation records
├── synthetic_pipeline.py
```

---


# Execution Pipeline

Run the pipeline sequentially:

```bash
python synthetic_pipeline.py

```

# Architecture & Data Schema

The pipeline models a relational 1-to-N hierarchy where SRPatient serves as the root parent entity across three related child event tables:

                  ┌──────────────────────┐
                  │    SRPatient.csv     │
                  │ (Parent Primary Key) │
                  └──────────┬───────────┘
                             │ (IDPatient)
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  SRCode.csv  │     │SRPrimaryCare │     │SRImmunisation│
│(Clinical Obs)│     │Medication.csv│     │    .csv      │
└──────────────┘     └──────────────┘     └──────────────┘