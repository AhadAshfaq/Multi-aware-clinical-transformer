"""
Construct the patient-excluded General Cancer (GC) cohort for pretraining.

This script creates the external GC cohort used for self-supervised
pretraining in the workflow. It identifies MIMIC-IV admissions with
neoplasm diagnoses, then excludes every cancer admission belonging to patients
represented in the downstream Aplasia (AP) or Neutropenic Fever (NF) cohorts.

The script writes two restricted-data outputs:
1. A GC cohort file containing admission identifiers, derived age, gender, and
   a compatibility label field.
2. A temporal laboratory-event file restricted to the supplied top-100 clinical
   variables and the first 14 days (336 hours) after hospital admission.

The resulting outputs are derived from restricted MIMIC-IV data and therefore
cannot be committed to a public repository.

Expected input layout under --data-root:
    mimic-iv/
        admissions.csv.gz
        patients.csv.gz
        diagnoses_icd.csv.gz
        labevents.csv.gz
    mimic_chemo/
        cohorts/
            mimic_cohort_aplasia_45_days.csv
            mimic_cohort_NF_30_days.csv
        top_features/
            mimic_top100_features.pkl

Outputs are written under:
    mimic_chemo/
        cohorts/mimic_cohort_gc.csv
        temporal_features/mimic_cohort_gc_admissions_labs_14_days_to_ts.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm


TOP_FEATURE_COUNT = 100
OBSERVATION_HOURS = 336
LABEVENTS_CHUNK_SIZE = 1_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for restricted-data input locations."""
    parser = argparse.ArgumentParser(
        description="Construct a patient-excluded GC cohort from MIMIC-IV."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing authorized MIMIC-IV files and the "
            "derived AP/NF cohort-extraction inputs."
        ),
    )
    return parser.parse_args()


def load_downstream_subjects(
    chemo_dir: Path,
    admissions_path: Path,
) -> tuple[set[int], set[int], pd.DataFrame]:
    """
    Identify AP/NF downstream admissions and their associated MIMIC-IV patients.

    Returns
    -------
    downstream_hadms:
        Union of AP and NF admission identifiers.
    downstream_subjects:
        Patient identifiers associated with those downstream admissions.
    admissions:
        Admissions lookup table with subject_id, hadm_id, and admittime.
    """
    ap_path = chemo_dir / "cohorts" / "mimic_cohort_aplasia_45_days.csv"
    nf_path = chemo_dir / "cohorts" / "mimic_cohort_NF_30_days.csv"

    ap_cohort = pd.read_csv(ap_path, usecols=["hadm_id"])
    nf_cohort = pd.read_csv(nf_path, usecols=["hadm_id"])

    downstream_hadms = set(ap_cohort["hadm_id"]).union(nf_cohort["hadm_id"])

    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "admittime"],
    )

    downstream_subjects = set(
        admissions.loc[
            admissions["hadm_id"].isin(downstream_hadms),
            "subject_id",
        ]
    )

    if not downstream_subjects:
        raise RuntimeError(
            "No downstream AP/NF subjects were identified. Check cohort files "
            "and admission identifier formats."
        )

    return downstream_hadms, downstream_subjects, admissions


def identify_cancer_admissions(diagnoses_path: Path) -> set[int]:
    """
    Identify candidate cancer admissions using workflow-defined diagnosis criteria.

    ICD-10 diagnoses beginning with 'C' and ICD-9 diagnoses with the first
    three digits in the inclusive range 140--239 are retained.
    """
    diagnoses = pd.read_csv(
        diagnoses_path,
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": "string"},
    ).dropna(subset=["hadm_id", "icd_code"])

    icd10_cancer = (
        (diagnoses["icd_version"] == 10)
        & diagnoses["icd_code"].str.startswith("C", na=False)
    )

    icd9_prefix = pd.to_numeric(
        diagnoses["icd_code"].str.slice(0, 3),
        errors="coerce",
    )
    icd9_cancer = (
        (diagnoses["icd_version"] == 9)
        & icd9_prefix.between(140, 239, inclusive="both")
    )

    return set(diagnoses.loc[icd10_cancer | icd9_cancer, "hadm_id"].unique())


def validate_exclusion(
    gc_hadms: set[int],
    downstream_hadms: set[int],
    downstream_subjects: set[int],
    admissions: pd.DataFrame,
) -> None:
    """Validate that the retained GC cohort has no AP/NF admission or patient overlap."""
    gc_subjects = set(
        admissions.loc[
            admissions["hadm_id"].isin(gc_hadms),
            "subject_id",
        ]
    )

    admission_overlap = gc_hadms.intersection(downstream_hadms)
    subject_overlap = gc_subjects.intersection(downstream_subjects)

    if admission_overlap:
        raise AssertionError(
            f"Admission-level overlap detected: {len(admission_overlap)} admissions."
        )

    if subject_overlap:
        raise AssertionError(
            f"Patient-level overlap detected: {len(subject_overlap)} patients."
        )


def build_gc_cohort(
    gc_hadms: set[int],
    admissions: pd.DataFrame,
    patients_path: Path,
) -> pd.DataFrame:
    """
    Build the GC demographic cohort table used by downstream preprocessing.

    The label column is retained only for compatibility with the existing
    preprocessing interface. GC labels are not used in self-supervised
    pretraining.
    """
    patients = pd.read_csv(
        patients_path,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )

    cohort = admissions.loc[
        admissions["hadm_id"].isin(gc_hadms)
    ].merge(
        patients,
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )

    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    if cohort["admittime"].isna().any():
        missing_count = int(cohort["admittime"].isna().sum())
        raise ValueError(
            f"{missing_count} GC admissions have invalid admission timestamps."
        )

    cohort["age"] = (
        cohort["anchor_age"]
        + (cohort["admittime"].dt.year - cohort["anchor_year"])
    )
    cohort["label"] = 0

    cohort = cohort[["hadm_id", "admittime", "age", "gender", "label"]].copy()
    if not cohort["hadm_id"].is_unique:
        raise AssertionError("GC cohort contains duplicate admission identifiers.")

    return cohort


def extract_gc_laboratory_events(
    labevents_path: Path,
    gc_hadms: set[int],
    top_features: set[int],
    admission_times: dict[int, pd.Timestamp],
) -> pd.DataFrame:
    """
    Extract top-100 laboratory events from GC admissions within 14 days (336 hours) after hospital admission.
    """
    records: list[pd.DataFrame] = []

    reader = pd.read_csv(
        labevents_path,
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=LABEVENTS_CHUNK_SIZE,
    )

    for chunk in tqdm(reader, desc="Extracting GC laboratory events"):
        chunk = chunk.loc[
            chunk["hadm_id"].isin(gc_hadms)
            & chunk["itemid"].isin(top_features)
        ].dropna(subset=["valuenum"])

        if chunk.empty:
            continue

        chunk["admittime"] = chunk["hadm_id"].map(admission_times)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["admittime", "charttime"])

        chunk["hour"] = (
            chunk["charttime"] - chunk["admittime"]
        ).dt.total_seconds() / 3600.0

        chunk = chunk.loc[
            chunk["hour"].between(0, OBSERVATION_HOURS, inclusive="both")
        ].rename(columns={"valuenum": "value"})

        if not chunk.empty:
            records.append(chunk[["hadm_id", "itemid", "hour", "value"]])

    if not records:
        raise RuntimeError(
            "No GC laboratory events were extracted. Check MIMIC-IV paths, "
            "top-100 feature IDs, cancer-admission selection, and timestamps."
        )

    events = pd.concat(records, ignore_index=True)

    if not events["hadm_id"].isin(gc_hadms).all():
        raise AssertionError("Extracted events include admissions outside the GC cohort.")

    return events


def main() -> None:
    """Construct and save the patient-excluded GC cohort and temporal event file."""
    args = parse_args()

    raw_dir = args.data_root
    mimic_dir = raw_dir / "mimic-iv"
    chemo_dir = raw_dir / "mimic_chemo"

    admissions_path = mimic_dir / "admissions.csv.gz"
    diagnoses_path = mimic_dir / "diagnoses_icd.csv.gz"
    patients_path = mimic_dir / "patients.csv.gz"
    labevents_path = mimic_dir / "labevents.csv.gz"
    top_features_path = chemo_dir / "top_features" / "mimic_top100_features.pkl"

    cohort_output_path = chemo_dir / "cohorts" / "mimic_cohort_gc.csv"
    events_output_path = (
        chemo_dir
        / "temporal_features"
        / "mimic_cohort_gc_admissions_labs_14_days_to_ts.csv"
    )

    required_paths = [
        admissions_path,
        diagnoses_path,
        patients_path,
        labevents_path,
        top_features_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required input files:\n{formatted}")

    print("Starting GC cohort construction.")

    downstream_hadms, downstream_subjects, admissions = load_downstream_subjects(
        chemo_dir=chemo_dir,
        admissions_path=admissions_path,
    )

    print(
        f"Identified {len(downstream_hadms):,} combined AP/NF admissions from "
        f"{len(downstream_subjects):,} downstream patients."
    )

    candidate_cancer_hadms = identify_cancer_admissions(diagnoses_path)
    gc_hadms = set(
        admissions.loc[
            admissions["hadm_id"].isin(candidate_cancer_hadms)
            & ~admissions["subject_id"].isin(downstream_subjects),
            "hadm_id",
        ]
    )

    print(f"Identified {len(candidate_cancer_hadms):,} candidate cancer admissions.")
    print(
        f"Retained {len(gc_hadms):,} GC admissions after patient-level exclusion."
    )

    validate_exclusion(
        gc_hadms=gc_hadms,
        downstream_hadms=downstream_hadms,
        downstream_subjects=downstream_subjects,
        admissions=admissions,
    )
    print("Validated zero downstream admission and patient overlap.")

    gc_cohort = build_gc_cohort(
        gc_hadms=gc_hadms,
        admissions=admissions,
        patients_path=patients_path,
    )

    cohort_output_path.parent.mkdir(parents=True, exist_ok=True)
    gc_cohort[["hadm_id", "age", "gender", "label"]].to_csv(
        cohort_output_path,
        index=False,
    )
    print(f"Saved GC cohort: {cohort_output_path}")

    with top_features_path.open("rb") as handle:
        top_features = set(pickle.load(handle))

    if len(top_features) != TOP_FEATURE_COUNT:
        print(
            f"Warning: expected {TOP_FEATURE_COUNT} top features, "
            f"found {len(top_features)}."
        )

    admission_times = gc_cohort.set_index("hadm_id")["admittime"].to_dict()
    gc_events = extract_gc_laboratory_events(
        labevents_path=labevents_path,
        gc_hadms=gc_hadms,
        top_features=top_features,
        admission_times=admission_times,
    )

    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    gc_events.to_csv(events_output_path, index=False)
    print(f"Saved GC temporal laboratory events: {events_output_path}")

    print("GC cohort construction completed successfully.")


if __name__ == "__main__":
    main()