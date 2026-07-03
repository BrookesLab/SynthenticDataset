# SynthenticDataset

Synthetic dataset generation utilities for a primary care dataset with linked
patient, code event, medication, and immunisation records.

## Generate synthetic primary care files

Use the script below to generate synthetic files with the same table structure
as:

- `SRPatient.txt`
- `SRCode.txt`
- `SRPrimaryCareMedication.txt`
- `SRImmunisation.txt`

The generator uses randomized synthetic values (seeded for reproducibility) and
does not copy any source records, which helps preserve privacy while mimicking
dataset patterns such as event density per patient, linked IDs, date windows,
and realistic sparse/long-tail distributions.

```bash
python /home/runner/work/SynthenticDataset/SynthenticDataset/generate_synthetic_primary_care.py \
  --output-dir /home/runner/work/SynthenticDataset/SynthenticDataset/output \
  --patients 10000 \
  --avg-events-per-patient 30 \
  --medication-ratio 0.70 \
  --immunisation-ratio 0.20 \
  --seed 42
```

### Notes

- `SRPatient.txt` and `SRCode.txt` are written as comma-separated files.
- `SRPrimaryCareMedication.txt` and `SRImmunisation.txt` are written as
  pipe-separated files (`|`) to match their source format.
- Use a larger `--patients` value to scale row counts upward.
