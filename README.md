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
# Pipline Executio Workflow

        Step 1: Out-of-Core Temporal PreprocessingComputes an in-memory patient birth lookup map (DateBirth).Streams child event files in 1.5M-row chunks to prevent out-of-memory (OOM) errors.Transforms explicit dates (YYYY-MM-DD) into integer ages: $\text{AgeAtEvent} = \text{Year(Event)} - \text{Year(Birth)}$.Strips original raw timestamp strings to eliminate exact temporal re-identification risks.

        Step 2: Utilization Feature Engineering & StratificationAggregates event counts per patient and calculates log-transformed features: $\log(1 + \text{count})$.Creates composite stratification bins: $\text{Gender} \times \text{Age Group} \times \text{Healthcare Utilization Group}$.Extracts a representative 10–15% stratified cohort (~50k patients) and isolates their corresponding records across all child tables.

        Step 3: Relational Modeling & SDV FittingConfigures MultiTableMetadata with per-table schema detection, eliminating spurious cross-table foreign key inferences.Defines temporal inequality constraints: $\text{AgeAtMedicationStart} \le \text{AgeAtMedicationEnd}$.Fits and serializes the HMASynthesizer copula model to disk.

        Step 4 & 5: Scaled Generation & Clinical Post-ProcessingSamples synthetic cohorts in modular chunks to maintain low memory overhead.Sex-specific sanity checks: Regex validation removes biologically inconsistent diagnoses and prescriptions (e.g., pregnancy records assigned to male profiles, prostate screenings assigned to female profiles).Post-mortem filtering: Drops event records occurring after a patient's generated AgeAtDeath.Copula tail filtering: Suppresses ultra-rare synthetic codes (< 20 occurrences in baseline).Generates continuous, collision-free synthetic identifiers (SYN_PAT_xxxxxxxx, SYN_EVT_C_xxxxxxxxx, SYN_EVT_M_xxxxxxxxx).