# Synthetic Dataset Generation Pipeline

This repository contains a production-grade, memory-safe pipeline for generating privacy-preserving, relationally consistent synthetic clinical datasets from SystmOne primary care records. 

The pipeline handles massive health datasets by processing records in structured chunks, sanitizes direct patient identifiers, converts absolute timestamps into robust relative age metrics (`AgeAtEvent`), and protects against AI model memorization attacks.

---

## 📂 Project Structure

Your working directory should contain your clean source files and the pipeline scripts structured as follows:

```text
├── SRPatient.csv                  # Raw patient demographics baseline file
├── SRCode.csv                     # Raw clinical codes event entries
├── SRPrimaryCareMedication.csv    # Raw standard prescriptions history
├── SRImmunisation.csv             # Raw vaccination logs
├── step1_preprocess_child_tables.py
├── step2_preprocess_patients.py
├── step3_synthetic_pipeline.py
├── step4_filter_synthetic_clones.py
└── Synthetic_Output/              # Directory automatically created for final exports