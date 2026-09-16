#!/usr/bin/env python3
"""
Standalone utility script to calculate SAMPLE_SCALE_PER_CHUNK 
for SDV HMASynthesizer on the LLR-DfR dataset.
"""

def main():
    # ==========================================
    # LLR-DfR Cohort Configuration Parameters
    # ==========================================
    TOTAL_TARGET_PATIENTS = 500_000
    TRAIN_SAMPLE_FRACTION = 0.03  # 3% training sample
    NUM_GEN_CHUNKS = 8 #16 or 20 for 64 GB RAM, 4 for 256GB+

    # ==========================================
    # Calculation Logic
    # ==========================================
    train_patients = TOTAL_TARGET_PATIENTS * TRAIN_SAMPLE_FRACTION
    total_scale = TOTAL_TARGET_PATIENTS / train_patients
    scale_per_chunk = total_scale / NUM_GEN_CHUNKS

    # ==========================================
    # Output Results
    # ==========================================
    print("=" * 60)
    print(" 📊 LLR-DfR SYNTHESIS SCALING CALCULATOR")
    print("=" * 60)
    print(f" Total Target Cohort Size : {TOTAL_TARGET_PATIENTS:,} patients")
    print(f" Training Sample Fraction : {TRAIN_SAMPLE_FRACTION * 100:.1f}% (~{train_patients:,.0f} patients)")
    print(f" Generation Chunks        : {NUM_GEN_CHUNKS}")
    print(f" Total Scale Multiplier   : {total_scale:.4f}x")
    print("-" * 60)
    print(f" 👉 SAMPLE_SCALE_PER_CHUNK: {scale_per_chunk:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()