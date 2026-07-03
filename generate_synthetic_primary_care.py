#!/usr/bin/env python3
"""Generate a synthetic primary care dataset without using source patient data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from pathlib import Path


CTV3_CODES = ["Xa1A1", "Xa2B2", "Xa3C3", "Xa4D4", "Xa5E5", "Xa6F6"]
SNOMED_CODES = ["44054006", "233604007", "27113001", "195967001", "386661006", "49727002"]
EPISODE_TYPES = ["Consultation", "FollowUp", "Review", "Acute", "Chronic"]
IMM_CONTENT = ["FLU", "MMR", "COVID19", "TDAP", "HEPB", "HPV"]
IMM_LOCATIONS = ["GP", "Community", "Hospital", "Pharmacy", "School"]
IMM_METHODS = ["IM", "SC", "Oral", "Nasal"]
IMM_READ_CODES = ["65E..", "65F..", "65G..", "65H..", "65I.."]
IMM_SNOMED_CODES = ["86198006", "33879002", "1119349007", "871751000000105", "6142004"]
VACC_PART = ["Whole", "PartA", "PartB"]
MEDICATIONS = [
    ("100100", "dmd100100", "Metformin 500mg tablets"),
    ("100200", "dmd100200", "Atorvastatin 20mg tablets"),
    ("100300", "dmd100300", "Lisinopril 10mg tablets"),
    ("100400", "dmd100400", "Omeprazole 20mg capsules"),
    ("100500", "dmd100500", "Amlodipine 5mg tablets"),
    ("100600", "dmd100600", "Salbutamol inhaler"),
]
MIN_DEATH_AGE_YEARS = 18
MAX_DEATH_LOOKBACK_YEARS = 25


def random_date(start: dt.date, end: dt.date) -> dt.date:
    delta = (end - start).days
    return start + dt.timedelta(days=random.randint(0, delta))


def generate_patients(count: int) -> list[dict]:
    today = dt.date.today()
    patients = []
    for patient_id in range(1, count + 1):
        age = min(max(int(random.gauss(45, 20)), 0), 100)
        dob = today - dt.timedelta(days=age * 365 + random.randint(0, 364))
        has_dod = random.random() < 0.15 and age > 55
        if has_dod:
            dod = random_date(
                max(
                    dob + dt.timedelta(days=MIN_DEATH_AGE_YEARS * 365),
                    today - dt.timedelta(days=MAX_DEATH_LOOKBACK_YEARS * 365),
                ),
                today,
            )
            dod_value = dod.isoformat()
        else:
            dod_value = ""
        patients.append(
            {
                "IDPatient": patient_id,
                "DateBirth": dob.isoformat(),
                "DateDeath": dod_value,
                "Gender": random.choices(["F", "M", "U"], weights=[0.495, 0.495, 0.01], k=1)[0],
            }
        )
    return patients


def events_for_patient(avg_events: int) -> int:
    base = max(1, int(random.expovariate(1 / max(1, avg_events))))
    return min(base, avg_events * 8)


def generate_events(patients: list[dict], avg_events: int) -> list[dict]:
    start_date = dt.date(2000, 1, 1)
    today = dt.date.today()
    events = []
    event_id = 1
    for patient in patients:
        patient_id = patient["IDPatient"]
        dob = dt.date.fromisoformat(patient["DateBirth"])
        latest = dt.date.fromisoformat(patient["DateDeath"]) if patient["DateDeath"] else today
        earliest_event = max(start_date, dob + dt.timedelta(days=365))
        if earliest_event > latest:
            earliest_event = latest
        for _ in range(events_for_patient(avg_events)):
            event_date = random_date(earliest_event, latest)
            events.append(
                {
                    "DateEvent": event_date.isoformat(),
                    "CTV3Code": random.choice(CTV3_CODES),
                    "SNOMEDCode": random.choice(SNOMED_CODES),
                    "EpisodeType": random.choice(EPISODE_TYPES),
                    "IDEvent": event_id,
                    "IDPatient": patient_id,
                }
            )
            event_id += 1
    return events


def generate_medications(events: list[dict], medication_ratio: float) -> list[dict]:
    medications = []
    for event in events:
        if random.random() > medication_ratio:
            continue
        med = random.choice(MEDICATIONS)
        start = dt.date.fromisoformat(event["DateEvent"])
        duration = random.randint(7, 365)
        end = start + dt.timedelta(days=duration) if random.random() < 0.85 else None
        medications.append(
            {
                "IDMultiLexProduct": med[0],
                "IDMultiLexDMD": med[1],
                "NameOfMedication": med[2],
                "DateMedicationStart": start.isoformat(),
                "DateMedicationEnd": end.isoformat() if end else "",
                "IDEvent": event["IDEvent"],
                "IDPatient": event["IDPatient"],
            }
        )
    return medications


def generate_immunisations(events: list[dict], immunisation_ratio: float) -> list[dict]:
    immunisations = []
    for event in events:
        if random.random() > immunisation_ratio:
            continue
        immunisations.append(
            {
                "IDPatient": event["IDPatient"],
                "DateEvent": event["DateEvent"],
                "IDImmunisationContent": random.choice(IMM_CONTENT),
                "Dose": random.randint(1, 4),
                "Location": random.choice(IMM_LOCATIONS),
                "Method": random.choice(IMM_METHODS),
                "ImmsReadCode": random.choice(IMM_READ_CODES),
                "ImmsSNOMEDCode": random.choice(IMM_SNOMED_CODES),
                "VaccPart": random.choice(VACC_PART),
            }
        )
    return immunisations


def write_file(path: Path, rows: list[dict], delimiter: str) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where synthetic files are written")
    parser.add_argument("--patients", type=int, default=10000, help="Number of synthetic patients")
    parser.add_argument("--avg-events-per-patient", type=int, default=30, help="Average SRCode events per patient")
    parser.add_argument(
        "--medication-ratio",
        type=float,
        default=0.70,
        help="Probability in [0, 1] that an event generates a medication record",
    )
    parser.add_argument(
        "--immunisation-ratio",
        type=float,
        default=0.70,
        help="Probability in [0, 1] that an event generates an immunisation record",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    args = parser.parse_args()

    if args.patients < 1:
        raise ValueError("--patients must be > 0")
    if args.avg_events_per_patient < 1:
        raise ValueError("--avg-events-per-patient must be > 0")
    if not (0.0 <= args.medication_ratio <= 1.0):
        raise ValueError("--medication-ratio must be in [0, 1]")
    if not (0.0 <= args.immunisation_ratio <= 1.0):
        raise ValueError("--immunisation-ratio must be in [0, 1]")

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patients = generate_patients(args.patients)
    events = generate_events(patients, args.avg_events_per_patient)
    medications = generate_medications(events, args.medication_ratio)
    immunisations = generate_immunisations(events, args.immunisation_ratio)

    write_file(args.output_dir / "SRPatient.txt", patients, delimiter=",")
    write_file(args.output_dir / "SRCode.txt", events, delimiter=",")
    write_file(args.output_dir / "SRPrimaryCareMedication.txt", medications, delimiter="|")
    write_file(args.output_dir / "SRImmunisation.txt", immunisations, delimiter="|")

    print(f"Generated {len(patients):,} patients")
    print(f"Generated {len(events):,} code events")
    print(f"Generated {len(medications):,} medications")
    print(f"Generated {len(immunisations):,} immunisations")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
