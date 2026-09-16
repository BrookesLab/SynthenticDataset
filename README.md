# Synthetic Dataset Generation Pipeline

A production-grade, multi-table synthetic data generation and validation pipeline built on SDV’s Hierarchical Modeling Algorithm (HMA). The pipeline ingests raw primary care electronic health records, anonymizes temporal attributes into relative age offsets, fits a relational generative model using stratified sampling, and generates privacy-compliant, clinically validated synthetic cohorts.

# Project Structure

```
.
├── SRPatient.csv                    # Raw patient demographics
├── SRCode.csv                       # Clinical event records
├── SRPrimaryCareMedication.csv      # Medication history
├── SRImmunisation.csv               # Immunisation records
├── synthetic_pipeline.py            # Main generation & training script
├── evaluate_pipeline.py             # Three-pillar evaluation & compliance suite
├── validate_preprocessing.py        # Schema & structural data validation suite
└── requirements.txt                 # Python dependencies
```

---


# Execution Pipeline

Run the pipeline sequentially:

```bash
python synthetic_pipeline.py

```
# Pipline Executio Workflow

        Step 1: Out-of-core temporal parsing and age conversion (AgeIn2026, AgeAtEvent, etc.).

        Step 2: Log-count utilization engineering (DiseaseMedRatio) and multi-attribute stratified sampling (3% fraction).

        Step 3: Relational metadata configuration, temporal inequality constraint setup, and HMASynthesizer training & serialization.

        Step 4 & 5: Chunked synthetic generation, data-driven sex-exclusive code filtering, post-mortem event removal, tail frequency suppression, and ID remapping.

# Evaluation Suite

Once the pipeline completes execution, run the evaluation framework to audit statistical fidelity, clinical utility, disclosure risks, and Membership Inference Attack (MIA) resistance.

        Pillar 1 (Statistical Fidelity): Validates distribution overlap using Kolmogorov-Smirnov (KS) complement scores and relative errors.

        Pillar 2 (Clinical Utility): Checks categorical marginal proportions (Gender balance) and central tendency age parity (Delta years).

        Pillar 3 (Privacy & MIA): Evaluates Distance-to-Closest-Record (DCR) metrics, exact demographic profile overlap counts (0 tolerance), and pipeline-aligned train vs. holdout MIA distance ratios.
        
        Pillar 4 (Constraint Audit): Scans final outputs for any residual sex-exclusive code leakage or post-mortem record violations