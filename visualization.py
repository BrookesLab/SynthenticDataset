import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# =====================================================================
# 📊 CONFIGURATION & PATHS (Fixed Real Directory Name)
# =====================================================================
REAL_DIR = "hma/Preprocessed_Data"
SYNTHETIC_DIR = "hma_prod/Synthetic_Output"
PLOT_DIR = "Evaluation_Plots"
os.makedirs(PLOT_DIR, exist_ok=True)

DELIMITER = ","
READ_ENCODING = "latin-1"

print("📂 Loading datasets for visual evaluation...")
real_path = os.path.join(REAL_DIR, "SRPatient.csv")
synth_path = os.path.join(SYNTHETIC_DIR, "synthetic_SRPatient.csv")

if not os.path.exists(real_path):
    raise FileNotFoundError(f"❌ Could not find real data at: {real_path}. Check your directory path!")
if not os.path.exists(synth_path):
    raise FileNotFoundError(f"❌ Could not find synthetic data at: {synth_path}. Run your generation pipeline first!")

real_patients = pd.read_csv(real_path, sep=DELIMITER, encoding=READ_ENCODING, low_memory=False)
synth_patients = pd.read_csv(synth_path, sep=DELIMITER, encoding=READ_ENCODING, low_memory=False)

# Clean numeric columns safely
for df in [real_patients, synth_patients]:
    if "AgeIn2026" in df.columns:
        df["AgeIn2026"] = pd.to_numeric(df["AgeIn2026"], errors="coerce")
    if "AgeAtDeath" in df.columns:
        df["AgeAtDeath"] = pd.to_numeric(df["AgeAtDeath"], errors="coerce")

# =====================================================================
# PILLAR 1: FIDELITY VISUALIZATION (Distribution Overlap)
# =====================================================================
print("\n🎨 Generating Pillar 1: Fidelity Plots (Age Distributions)...")
if "AgeIn2026" in real_patients.columns and "AgeIn2026" in synth_patients.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=real_patients, x="AgeIn2026", label="Real Data", 
        color="royalblue", fill=True, alpha=0.4, common_norm=False
    )
    sns.kdeplot(
        data=synth_patients, x="AgeIn2026", label="Synthetic Data", 
        color="darkorange", fill=True, alpha=0.4, common_norm=False
    )
    plt.title("Pillar 1: Statistical Fidelity — Patient Age Distribution (2026)", fontsize=14, weight="bold")
    plt.xlabel("Age (Years)")
    plt.ylabel("Density")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(PLOT_DIR, "pillar1_fidelity_age_distribution.png"), dpi=300)
    plt.close()
    print("   -> Saved pillar1_fidelity_age_distribution.png")


# =====================================================================
# PILLAR 2: UTILITY VISUALIZATION (Clinical Marginals / Gender Proportions)
# =====================================================================
print("🎨 Generating Pillar 2: Utility Plots (Gender Proportions)...")
if "Gender" in real_patients.columns and "Gender" in synth_patients.columns:
    real_counts = real_patients["Gender"].value_counts(normalize=True).reset_index()
    real_counts.columns = ["Gender", "Proportion"]
    real_counts["Dataset"] = "Real"

    synth_counts = synth_patients["Gender"].value_counts(normalize=True).reset_index()
    synth_counts.columns = ["Gender", "Proportion"]
    synth_counts["Dataset"] = "Synthetic"

    combined_gender = pd.concat([real_counts, synth_counts], ignore_index=True)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=combined_gender, x="Gender", y="Proportion", hue="Dataset", palette=["royalblue", "darkorange"])
    plt.title("Pillar 2: Clinical Utility — Gender Category Proportions", fontsize=14, weight="bold")
    plt.xlabel("Gender Category")
    plt.ylabel("Proportion")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(PLOT_DIR, "pillar2_utility_gender_proportions.png"), dpi=300)
    plt.close()
    print("   -> Saved pillar2_utility_gender_proportions.png")


# =====================================================================
# PILLAR 3: PRIVACY VISUALIZATION (Distance-to-Closest-Record / DCR)
# =====================================================================
print("🎨 Generating Pillar 3: Privacy Plots (DCR Distribution)...")
if "AgeIn2026" in real_patients.columns:
    # Use AgeIn2026 and fill NaNs for AgeAtDeath so living patients aren't discarded
    r_sub = real_patients[["AgeIn2026", "AgeAtDeath"]].copy().fillna(0)
    s_sub = synth_patients[["AgeIn2026", "AgeAtDeath"]].copy().fillna(0)

    r_mat = r_sub.head(2000).to_numpy()
    s_mat = s_sub.head(2000).to_numpy()

    if len(r_mat) > 0 and len(s_mat) > 0:
        dcr_values = []
        for s_row in s_mat:
            distances = np.linalg.norm(r_mat - s_row, axis=1)
            dcr_values.append(np.min(distances))

        plt.figure(figsize=(10, 6))
        sns.histplot(dcr_values, kde=True, color="forestgreen", bins=30)
        plt.axvline(x=0, color="red", linestyle="--", label="Exact Match Threshold (0)")
        plt.title("Pillar 3: Privacy Risk — Distance-to-Closest-Record (DCR)", fontsize=14, weight="bold")
        plt.xlabel("Minimum Euclidean Distance to Real Record")
        plt.ylabel("Synthetic Patient Count")
        plt.legend(frameon=True)
        plt.savefig(os.path.join(PLOT_DIR, "pillar3_privacy_dcr_distribution.png"), dpi=300)
        plt.close()
        print("   -> Saved pillar3_privacy_dcr_distribution.png")

print(f"\n✅ All visualization plots successfully generated and saved to: {os.path.abspath(PLOT_DIR)}/")