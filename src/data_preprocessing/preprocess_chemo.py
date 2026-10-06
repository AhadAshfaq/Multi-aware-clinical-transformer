"""
Prepare GC, AP, and NF cohorts for the workflow.

This script converts restricted MIMIC-IV-derived cohort files into a common
event-table and fold-package format for the EMIT-based pipeline.

Cohort roles
------------
GC:
    External GC cohort used only for self-supervised pretraining.
    A deterministic 80/20 train/validation split is created. No test split is
    created because GC outcome labels are not used during pretraining.

AP and NF:
    Downstream AP and NF cohorts used for fold-specific supervised fine-tuning 
    and held-out test evaluation. The script preserves the five predefined 
    train/validation/test splits supplied with the cohort extraction protocol.
    
Processing steps
----------------
1. Map raw MIMIC-IV admission identifiers to compact, cohort-specific `ts_ind`
   values suitable for dense tensor construction.
2. Retain the supplied top-100 clinical variables and events recorded from
   hour 0 through hour 336 after admission.
3. Append Age and Gender as static hour-zero event records.
4. Save a preprocessed GC package and five fold-specific AP/NF packages.

Observation-window selection, chronological ordering, tail truncation, and
tensor construction occur later in the pretraining and fine-tuning pipelines.

All inputs and outputs are derived from restricted MIMIC-IV data. They must not
be committed to a public repository.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OBSERVATION_HOURS = 336
GC_TRAIN_FRACTION = 0.80
DEFAULT_SEED = 2021
VALID_COHORTS = ("AP", "NF", "GC")


def parse_args() -> argparse.Namespace:
    """Parse data locations, cohort selection, and GC split seed."""
    parser = argparse.ArgumentParser(
        description="Preprocess GC, AP, and NF cohorts for EMIT-based experiments."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Restricted data root containing the `mimic_chemo` directory with "
            "cohort, temporal-feature, top-feature, and fold inputs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Restricted directory where preprocessed cohort packages are saved.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=VALID_COHORTS,
        default=list(VALID_COHORTS),
        help="Cohorts to preprocess. Default: AP NF GC.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for the GC train/validation split. Default: {DEFAULT_SEED}.",
    )
    return parser.parse_args()


def cohort_paths(chemo_dir: Path, cohort_name: str) -> tuple[Path, Path, Path | None]:
    """
    Return input cohort, temporal-event, and fold-directory paths.

    Parameters
    ----------
    chemo_dir:
        Restricted directory containing MIMIC-IV-derived oncology inputs.
    cohort_name:
        One of AP, NF, or GC.
    """
    cohorts_dir = chemo_dir / "cohorts"
    temporal_dir = chemo_dir / "temporal_features"
    folds_dir = chemo_dir / "folds"

    if cohort_name == "AP":
        return (
            cohorts_dir / "mimic_cohort_aplasia_45_days.csv",
            temporal_dir / "mimic_cohort_aplasia_45_days_admissions_labs_14_days_to_ts.csv",
            folds_dir / "AP",
        )

    if cohort_name == "NF":
        return (
            cohorts_dir / "mimic_cohort_NF_30_days.csv",
            temporal_dir / "mimic_cohort_NF_30_days_admissions_labs_14_days_to_ts.csv",
            folds_dir / "NF",
        )

    if cohort_name == "GC":
        return (
            cohorts_dir / "mimic_cohort_gc.csv",
            temporal_dir / "mimic_cohort_gc_admissions_labs_14_days_to_ts.csv",
            None,
        )

    raise ValueError(f"Unsupported cohort: {cohort_name}")


def validate_paths(paths: Iterable[Path]) -> None:
    """Raise FileNotFoundError listing every missing required input path."""
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required input files:\n{formatted}")


def map_admissions_to_indices(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Map raw admission identifiers to contiguous zero-based cohort indices.

    Dense NumPy tensors require compact row indices. Raw MIMIC-IV admission IDs
    are therefore mapped to `ts_ind` values from 0 to N-1.
    """
    if cohort["hadm_id"].isna().any():
        raise ValueError("Cohort contains missing admission identifiers.")

    unique_hadms = sorted(cohort["hadm_id"].unique())
    hadm_to_ts_ind = {hadm_id: index for index, hadm_id in enumerate(unique_hadms)}

    cohort = cohort.copy()
    cohort["ts_ind"] = cohort["hadm_id"].map(hadm_to_ts_ind)

    if cohort["ts_ind"].isna().any():
        raise AssertionError("Failed to map one or more cohort admissions to ts_ind.")

    return cohort, hadm_to_ts_ind


def load_and_prepare_events(
    temporal_path: Path,
    top_features: set[int],
    hadm_to_ts_ind: dict[int, int],
) -> pd.DataFrame:
    """
    Load top-100 temporal events within the full 0--336 hour (14 days) observation period.

    Window selection for the full 14-day or last-7-day experiments occurs later
    in the corresponding pretraining or fine-tuning data formatter.
    """
    temporal_data = pd.read_csv(temporal_path)

    required_columns = {"hadm_id", "itemid", "hour", "value"}
    missing_columns = required_columns - set(temporal_data.columns)
    if missing_columns:
        raise ValueError(
            f"Temporal file is missing required columns: {sorted(missing_columns)}"
        )

    temporal_data = temporal_data.loc[
        temporal_data["itemid"].isin(top_features)
        & temporal_data["hadm_id"].isin(hadm_to_ts_ind)
        & temporal_data["hour"].between(0, OBSERVATION_HOURS, inclusive="both")
    ].copy()

    temporal_data["ts_ind"] = temporal_data["hadm_id"].map(hadm_to_ts_ind)
    if temporal_data["ts_ind"].isna().any():
        raise AssertionError("Temporal events contain unmapped admission identifiers.")

    temporal_data = temporal_data.rename(columns={"itemid": "variable"})
    events = temporal_data[["ts_ind", "variable", "value", "hour"]].dropna(
        subset=["value"]
    )

    if events.empty:
        raise RuntimeError("No temporal events remain after filtering.")

    return events


def append_static_demographics(events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Append Age and Gender as static hour-zero event records.

    These static records are retained in preprocessed cohort packages. The GC
    pretraining formatter includes them in its 102-variable vocabulary, whereas
    the downstream AP/NF fine-tuning formatter removes them before tensor
    construction.
    """
    required_columns = {"ts_ind", "age", "gender"}
    missing_columns = required_columns - set(cohort.columns)
    if missing_columns:
        raise ValueError(
            f"Cohort file is missing demographic columns: {sorted(missing_columns)}"
        )

    age_events = cohort[["ts_ind", "age"]].rename(columns={"age": "value"}).copy()
    age_events["variable"] = "Age"
    age_events["hour"] = 0.0

    gender_events = cohort[["ts_ind", "gender"]].copy()
    gender_events["value"] = gender_events["gender"].map({"M": 0.0, "F": 1.0})
    gender_events["variable"] = "Gender"
    gender_events["hour"] = 0.0
    gender_events = gender_events.drop(columns=["gender"])

    all_events = pd.concat(
        [events, age_events, gender_events],
        ignore_index=True,
    ).dropna(subset=["value"])

    return all_events[["ts_ind", "variable", "value", "hour"]]


def build_outcome_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Build the outcome table expected by the downstream fine-tuning interface.

    AP/NF labels are mapped to `in_hospital_mortality` only for compatibility
    with the inherited EMIT fine-tuning interface. They represent cohort-specific
    45-day AP or 30-day NF outcomes, not in-hospital mortality. GC labels are
    placeholders and are not used in self-supervised pretraining.
    """
    required_columns = {"ts_ind", "label"}
    missing_columns = required_columns - set(cohort.columns)
    if missing_columns:
        raise ValueError(
            f"Cohort file is missing outcome columns: {sorted(missing_columns)}"
        )

    return cohort[["ts_ind", "label"]].rename(
        columns={"label": "in_hospital_mortality"}
    )


def map_fold_admissions(
    raw_hadms: np.ndarray,
    hadm_to_ts_ind: dict[int, int],
    fold_path: Path,
    split_name: str,
) -> np.ndarray:
    """Map a raw fold admission-ID array to contiguous `ts_ind` values."""
    hadm_ids = raw_hadms[:, 1]
    missing_hadms = set(hadm_ids) - set(hadm_to_ts_ind)

    if missing_hadms:
        raise ValueError(
            f"{fold_path} {split_name} split contains {len(missing_hadms)} "
            "admissions absent from the cohort file."
        )

    return np.asarray([hadm_to_ts_ind[hadm_id] for hadm_id in hadm_ids], dtype=int)


def save_package(
    output_path: Path,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Save one preprocessing package in the format consumed by training scripts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as handle:
        pickle.dump(
            [events, outcomes, train_indices, valid_indices, test_indices],
            handle,
        )


def create_gc_split(cohort: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the deterministic 80/20 GC pretraining train/validation split."""
    indices = cohort["ts_ind"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(indices)

    split_index = int(len(shuffled_indices) * GC_TRAIN_FRACTION)
    train_indices = shuffled_indices[:split_index]
    valid_indices = shuffled_indices[split_index:]
    test_indices = np.asarray([], dtype=int)

    return train_indices, valid_indices, test_indices


def process_gc(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> None:
    """Save the single GC pretraining package with its deterministic 80/20 split."""
    train_indices, valid_indices, test_indices = create_gc_split(cohort, seed)

    output_path = output_dir / "chemo_GC_fold_0.pkl"
    save_package(
        output_path,
        events,
        outcomes,
        train_indices,
        valid_indices,
        test_indices,
    )
    print(f"Saved GC pretraining package: {output_path}")


def process_downstream_folds(
    cohort_name: str,
    folds_dir: Path,
    hadm_to_ts_ind: dict[int, int],
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Map and save the five predefined AP or NF train/validation/test folds."""
    for fold_index in range(5):
        fold_path = folds_dir / f"fold_{fold_index}.pkl"
        validate_paths([fold_path])

        with fold_path.open("rb") as handle:
            train_raw, valid_raw, test_raw = pickle.load(handle)

        train_indices = map_fold_admissions(
            train_raw, hadm_to_ts_ind, fold_path, "train"
        )
        valid_indices = map_fold_admissions(
            valid_raw, hadm_to_ts_ind, fold_path, "validation"
        )
        test_indices = map_fold_admissions(
            test_raw, hadm_to_ts_ind, fold_path, "test"
        )

        output_path = output_dir / f"chemo_{cohort_name}_fold_{fold_index}.pkl"
        save_package(
            output_path,
            events,
            outcomes,
            train_indices,
            valid_indices,
            test_indices,
        )
        print(f"Saved {cohort_name} fold {fold_index}: {output_path}")


def process_cohort(
    cohort_name: str,
    chemo_dir: Path,
    output_dir: Path,
    top_features: set[int],
    seed: int,
) -> None:
    """Prepare one cohort for GC pretraining or AP/NF downstream fine-tuning."""
    cohort_path, temporal_path, folds_dir = cohort_paths(chemo_dir, cohort_name)
    validate_paths([cohort_path, temporal_path])

    print(f"Processing {cohort_name} cohort.")

    cohort = pd.read_csv(cohort_path)
    cohort, hadm_to_ts_ind = map_admissions_to_indices(cohort)

    events = load_and_prepare_events(
        temporal_path=temporal_path,
        top_features=top_features,
        hadm_to_ts_ind=hadm_to_ts_ind,
    )
    events = append_static_demographics(events, cohort)
    outcomes = build_outcome_table(cohort)

    if cohort_name == "GC":
        process_gc(
            cohort=cohort,
            events=events,
            outcomes=outcomes,
            output_dir=output_dir,
            seed=seed,
        )
        return

    if folds_dir is None:
        raise ValueError(f"Missing fold directory for downstream cohort {cohort_name}.")

    process_downstream_folds(
        cohort_name=cohort_name,
        folds_dir=folds_dir,
        hadm_to_ts_ind=hadm_to_ts_ind,
        events=events,
        outcomes=outcomes,
        output_dir=output_dir,
    )


def main() -> None:
    """Preprocess selected GC/AP/NF cohorts into restricted training packages."""
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    chemo_dir = data_root / "mimic_chemo"
    output_dir = args.output_dir.expanduser().resolve()
    top_features_path = chemo_dir / "top_features" / "mimic_top100_features.pkl"

    validate_paths([top_features_path])

    with top_features_path.open("rb") as handle:
        top_features = set(pickle.load(handle))

    for cohort_name in args.cohorts:
        process_cohort(
            cohort_name=cohort_name,
            chemo_dir=chemo_dir,
            output_dir=output_dir,
            top_features=top_features,
            seed=args.seed,
        )

    print("Cohort preprocessing completed successfully.")


if __name__ == "__main__":
    main()