"""
Create GC pretraining tensors and event-position masks for the workflow.

This script converts the patient-excluded General Cancer (GC) cohort package
created by `preprocess_chemo.py` into dense arrays used for self-supervised
pretraining of the EMIT-based model.

The workflow uses GC only for self-supervised pretraining:
    GC pretraining tensors and masks -> GC checkpoint -> AP/NF fine-tuning.

The script performs the following steps:
1. Loads the preprocessed GC package and its deterministic train/validation split.
2. Applies the configured full-14-day or last-7-day observation window.
3. Builds a shared 102-variable vocabulary consisting of the supplied top-100
   clinical variables plus Age and Gender.
4. Supports Hybrid reconstruction--forecasting and Pure MAE pretraining.
5. Orders events chronologically, retains the most recent `max_len` events per
   admission, and creates padded time, value, variable-ID, and clinical-weight matrices.
6. Generates event-position masks using one configured strategy:
   `rate_of_change`, `random`, `clinical`, `frequency`, or `temporal`.
7. Saves train/validation tensors and train/validation masks.

All inputs and outputs are derived from the restricted MIMIC-IV data and cannot be committed to a public repository.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


OBSERVATION_HOURS = 336
FORECAST_START_HOUR = 312
TOP_FEATURE_COUNT = 100
PADDING_VARIABLE_ID = 0
FALLBACK_LOWER_BOUND = -9999.0
FALLBACK_UPPER_BOUND = 9999.0
CLINICAL_OUT_OF_RANGE_WEIGHT = 2.0


def parse_args() -> argparse.Namespace:
    """Parse paths for restricted cohort data, configuration, and outputs."""
    parser = argparse.ArgumentParser(
        description="Create GC pretraining tensors and event masks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the GC pretraining YAML configuration file.",
    )
    parser.add_argument(
        "--preprocessed-path",
        type=Path,
        required=True,
        help="Path to the restricted `chemo_GC_fold_0.pkl` package.",
    )
    parser.add_argument(
        "--top-features-path",
        type=Path,
        required=True,
        help="Path to the restricted top-100 feature pickle file.",
    )
    parser.add_argument(
        "--range-files",
        type=Path,
        nargs=2,
        metavar=("AP_RANGE_FILE", "NF_RANGE_FILE"),
        required=True,
        help=(
            "Restricted AP and NF reference-range files used to construct "
            "combined clinical bounds."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Restricted output directory for GC tensors and masks.",
    )
    return parser.parse_args()


def validate_paths(paths: list[Path]) -> None:
    """Raise an informative error if any required restricted input file is missing."""
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required input files:\n{formatted}")


def load_gc_package(
    package_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the GC preprocessed event table and train/validation/test index arrays.

    GC is used only for pretraining. The package must therefore contain an empty test-index array.
    """
    with package_path.open("rb") as handle:
        data, _, train_indices, valid_indices, test_indices = pickle.load(handle)

    train_indices = np.asarray(train_indices, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    if test_indices.size != 0:
        raise ValueError(
            "GC pretraining package must not contain a downstream test split."
        )

    return data.copy(), train_indices, valid_indices, test_indices


def apply_observation_window(data: pd.DataFrame, use_7_day_window: bool) -> pd.DataFrame:
    """Apply the configured 14-day or last-7-day pretraining observation window."""
    if use_7_day_window:
        filtered = data.loc[
            data["hour"].between(168, OBSERVATION_HOURS, inclusive="right")
        ].copy()
        print("Using last-7-day observation window: (168, 336] hours.")
    else:
        filtered = data.loc[
            data["hour"].between(0, OBSERVATION_HOURS, inclusive="both")
        ].copy()
        print("Using full 14-day observation window: [0, 336] hours.")

    if filtered.empty:
        raise RuntimeError("No GC events remain after applying the observation window.")

    return filtered


def build_vocabulary(top_features_path: Path) -> tuple[list[str], dict[str, int]]:
    """Build the fixed 102-variable GC pretraining vocabulary."""
    with top_features_path.open("rb") as handle:
        top_features = pickle.load(handle)

    vocabulary = sorted([str(item) for item in top_features] + ["Age", "Gender"])

    if len(vocabulary) != TOP_FEATURE_COUNT + 2:
        raise ValueError(
            f"Expected {TOP_FEATURE_COUNT + 2} vocabulary entries, "
            f"found {len(vocabulary)}."
        )

    variable_to_index = {
        variable: index + 1
        for index, variable in enumerate(vocabulary)
    }
    return vocabulary, variable_to_index


def load_clinical_bounds(range_paths: list[Path]) -> dict[str, tuple[float, float]]:
    """
    Construct conservative combined laboratory bounds from AP and NF range files.

    For each laboratory variable, the minimum available lower bound and maximum
    available upper bound across both files are retained.
    """
    range_frames = [pd.read_csv(path) for path in range_paths]
    ranges = pd.concat(range_frames, ignore_index=True)

    required_columns = {"itemid", "ref_range_lower", "ref_range_upper"}
    missing_columns = required_columns - set(ranges.columns)
    if missing_columns:
        raise ValueError(
            f"Range files are missing required columns: {sorted(missing_columns)}"
        )

    ranges = ranges.dropna(subset=["ref_range_lower", "ref_range_upper"])
    grouped = ranges.groupby("itemid", as_index=False).agg(
        lower_bound=("ref_range_lower", "min"),
        upper_bound=("ref_range_upper", "max"),
    )

    return {
        str(int(row.itemid)): (float(row.lower_bound), float(row.upper_bound))
        for row in grouped.itertuples()
    }


def attach_variable_indices(
    data: pd.DataFrame,
    variable_to_index: dict[str, int],
) -> pd.DataFrame:
    """Map event variable names to the fixed GC pretraining vocabulary."""
    indexed = data.copy()
    indexed["variable"] = indexed["variable"].astype(str)
    indexed["vind"] = indexed["variable"].map(variable_to_index)

    if indexed["vind"].isna().any():
        unknown_variables = sorted(indexed.loc[indexed["vind"].isna(), "variable"].unique())
        raise ValueError(
            "Found variables outside the fixed pretraining vocabulary: "
            f"{unknown_variables[:10]}"
        )

    indexed["vind"] = indexed["vind"].astype(int)
    return indexed


def create_forecasting_targets(
    data: pd.DataFrame,
    total_admissions: int,
    vocabulary_size: int,
    use_forecasting: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create EMIT-compatible forecasting targets and return encoder input events.

    In Hybrid mode, hours (312, 336) form the target interval and all events at
    or before hour 312 form encoder inputs. In Pure MAE mode, targets remain
    zero-filled and the complete configured observation window is retained.
    """
    targets = np.zeros((total_admissions, 2 * vocabulary_size), dtype=np.float32)

    if not use_forecasting:
        print("Pure MAE mode: forecasting targets are zero-filled.")
        return data, targets

    print("Hybrid mode: using (312, 336) hours as forecasting targets.")
    forecast_events = data.loc[data["hour"] > FORECAST_START_HOUR]
    encoder_events = data.loc[data["hour"] <= FORECAST_START_HOUR].copy()

    grouped = (
        forecast_events.groupby(["ts_ind", "vind"], as_index=False)["value"]
        .mean()
    )

    for row in grouped.itertuples():
        variable_position = row.vind - 1
        targets[row.ts_ind, variable_position] = row.value
        targets[row.ts_ind, variable_position + vocabulary_size] = 1.0

    return encoder_events, targets


def retain_recent_events(data: pd.DataFrame, max_len: int) -> pd.DataFrame:
    """ Order events chronologically and retain the most recent max_len per admission."""
    ordered = data[["ts_ind", "vind", "hour", "value"]].sort_values(
        by=["ts_ind", "hour", "vind"]
    )

    retained = ordered.groupby("ts_ind", group_keys=False).tail(max_len)
    retained = retained.sort_values(by=["ts_ind", "hour", "vind"]).reset_index(
        drop=True
    )

    retained["obs_ind"] = retained.groupby("ts_ind").cumcount()
    return retained


def build_dense_inputs(
    retained_events: pd.DataFrame,
    total_admissions: int,
    max_len: int,
    variable_to_index: dict[str, int],
    clinical_bounds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create padded time, value, variable-ID, and clinical-weight matrices."""
    bounds_by_index = {
        variable_to_index[variable]: clinical_bounds.get(
            variable,
            (FALLBACK_LOWER_BOUND, FALLBACK_UPPER_BOUND),
        )
        for variable in variable_to_index
    }

    times = np.zeros((total_admissions, max_len), dtype=np.float32)
    values = np.zeros((total_admissions, max_len), dtype=np.float32)
    variables = np.zeros((total_admissions, max_len), dtype=np.int32)
    clinical_weights = np.ones((total_admissions, max_len), dtype=np.float32)

    for row in retained_events.itertuples():
        times[row.ts_ind, row.obs_ind] = row.hour
        values[row.ts_ind, row.obs_ind] = row.value
        variables[row.ts_ind, row.obs_ind] = row.vind

        lower_bound, upper_bound = bounds_by_index[row.vind]
        if row.value < lower_bound or row.value > upper_bound:
            clinical_weights[row.ts_ind, row.obs_ind] = CLINICAL_OUT_OF_RANGE_WEIGHT

    return times, values, variables, clinical_weights


def generate_rate_of_change_masks(
    times: np.ndarray,
    values: np.ndarray,
    variables: np.ndarray,
    threshold: float,
    insignificant_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate RoC masks with rates assigned to the earlier event position.

    For each admission and variable, the rate calculated between consecutive
    measurements is assigned to the earlier event. Events above the configured
    threshold are masked with probability 1 - insignificant_probability; all
    remaining observed events are masked with insignificant_probability.
    """
    masks = np.zeros_like(values, dtype=bool)

    for admission_index in range(values.shape[0]):
        for variable_index in np.unique(variables[admission_index]):
            if variable_index == PADDING_VARIABLE_ID:
                continue

            positions = np.where(variables[admission_index] == variable_index)[0]
            rates = np.zeros(len(positions), dtype=np.float32)

            if len(positions) > 1:
                value_differences = np.diff(values[admission_index, positions])
                time_differences = np.diff(times[admission_index, positions])

                valid_deltas = time_differences != 0
                rates[:-1][valid_deltas] = (
                    value_differences[valid_deltas] / time_differences[valid_deltas]
                )

            significant = np.abs(rates) > threshold
            probabilities = np.where(
                significant,
                1.0 - insignificant_probability,
                insignificant_probability,
            )
            masks[admission_index, positions] = (
                rng.random(len(positions)) < probabilities
            )

    return masks


def generate_masks(
    strategy: str,
    times: np.ndarray,
    values: np.ndarray,
    variables: np.ndarray,
    clinical_weights: np.ndarray,
    config: OmegaConf,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate event-position masks for one configured adaptive strategy."""
    valid_events = variables > PADDING_VARIABLE_ID

    if strategy == "rate_of_change":
        return generate_rate_of_change_masks(
            times=times,
            values=values,
            variables=variables,
            threshold=float(config.get("mask_threshold", 0.001)),
            insignificant_probability=float(config.get("insignificant_prob", 0.40)),
            rng=rng,
        )

    random_values = rng.random(variables.shape)

    if strategy == "random":
        probability = float(config.get("random_mask_prob", 0.50))
        return valid_events & (random_values < probability)

    if strategy == "clinical":
        pathological_probability = float(config.get("pathological_mask_prob", 0.60))
        normal_probability = float(config.get("normal_mask_prob", 0.40))

        probabilities = np.where(
            clinical_weights == CLINICAL_OUT_OF_RANGE_WEIGHT,
            pathological_probability,
            normal_probability,
        )
        return valid_events & (random_values < probabilities)

    if strategy == "frequency":
        rare_probability = float(config.get("rare_mask_prob", 0.60))
        common_probability = float(config.get("common_mask_prob", 0.40))

        observed_variables = variables[valid_events]
        unique_variables, counts = np.unique(observed_variables, return_counts=True)

        count_map = np.zeros(int(variables.max()) + 1, dtype=np.float64)
        count_map[unique_variables] = counts

        min_count = counts.min()
        max_count = counts.max()
        count_range = max(max_count - min_count, 1)

        event_counts = count_map[variables]
        commonness = (event_counts - min_count) / count_range
        probabilities = rare_probability - commonness * (
            rare_probability - common_probability
        )

        return valid_events & (random_values < probabilities)

    if strategy == "temporal":
        recent_probability = float(config.get("recent_mask_prob", 0.60))
        old_probability = float(config.get("old_mask_prob", 0.40))

        positive_time_events = valid_events & (times > 0.0)
        safe_times = np.where(positive_time_events, times, np.nan)

        min_times = np.nanmin(safe_times, axis=1, keepdims=True)
        max_times = np.nanmax(safe_times, axis=1, keepdims=True)

        no_positive_time = np.isnan(min_times)
        min_times[no_positive_time] = 0.0
        max_times[no_positive_time] = 1.0

        time_ranges = np.maximum(max_times - min_times, 1e-5)
        relative_times = np.clip((times - min_times) / time_ranges, 0.0, 1.0)

        probabilities = old_probability + relative_times * (
            recent_probability - old_probability
        )
        return valid_events & (random_values < probabilities)

    valid_strategies = "rate_of_change, random, clinical, frequency, temporal"
    raise ValueError(
        f"Unknown masking strategy '{strategy}'. Choose one of: {valid_strategies}."
    )


def main() -> None:
    """Create and save GC pretraining tensors and event masks."""
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    preprocessed_path = args.preprocessed_path.expanduser().resolve()
    top_features_path = args.top_features_path.expanduser().resolve()
    range_paths = [path.expanduser().resolve() for path in args.range_files]
    output_dir = args.output_dir.expanduser().resolve()

    validate_paths([config_path, preprocessed_path, top_features_path, *range_paths])

    config = OmegaConf.load(config_path)
    rng = np.random.default_rng(int(config.get("seed", 2021)))

    data, train_indices, valid_indices, _ = load_gc_package(preprocessed_path)
    data = apply_observation_window(data, bool(config.USE_7_DAY_WINDOW))

    _, variable_to_index = build_vocabulary(top_features_path)
    data = attach_variable_indices(data, variable_to_index)

    total_admissions = int(data["ts_ind"].max()) + 1
    vocabulary_size = len(variable_to_index)

    encoder_events, forecasting_targets = create_forecasting_targets(
        data=data,
        total_admissions=total_admissions,
        vocabulary_size=vocabulary_size,
        use_forecasting=bool(config.USE_FORECASTING_ABLATION),
    )

    retained_events = retain_recent_events(
        data=encoder_events,
        max_len=int(config.max_len),
    )

    clinical_bounds = load_clinical_bounds(range_paths)
    times, values, variables, clinical_weights = build_dense_inputs(
        retained_events=retained_events,
        total_admissions=total_admissions,
        max_len=int(config.max_len),
        variable_to_index=variable_to_index,
        clinical_bounds=clinical_bounds,
    )

    all_inputs = np.stack([times, values, variables, clinical_weights], axis=0)
    masks = generate_masks(
        strategy=str(config.masking_strategy),
        times=times,
        values=values,
        variables=variables,
        clinical_weights=clinical_weights,
        config=config,
        rng=rng,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    tensors_path = output_dir / "pretraining_data_chemo_GC.npz"
    masks_path = output_dir / "chemo_GC_event_masks.npz"

    np.savez(
        tensors_path,
        fore_train_ip=all_inputs[:, train_indices],
        fore_train_op=forecasting_targets[train_indices],
        fore_valid_ip=all_inputs[:, valid_indices],
        fore_valid_op=forecasting_targets[valid_indices],
    )
    np.savez(
        masks_path,
        fore_train_masks=masks[train_indices],
        fore_val_masks=masks[valid_indices],
    )

    observed_event_count = int(np.sum(variables > PADDING_VARIABLE_ID))
    out_of_range_count = int(
        np.sum(
            (clinical_weights == CLINICAL_OUT_OF_RANGE_WEIGHT)
            & (variables > PADDING_VARIABLE_ID)
        )
    )
    clinical_percentage = (
        100.0 * out_of_range_count / observed_event_count
        if observed_event_count
        else 0.0
    )

    print(f"Saved GC pretraining tensors: {tensors_path}")
    print(f"Saved GC event masks: {masks_path}")
    print(f"GC admissions: {total_admissions:,}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Maximum sequence length: {config.max_len}")
    print(f"Masking strategy: {config.masking_strategy}")
    print(f"Out-of-range events: {clinical_percentage:.2f}%.")


if __name__ == "__main__":
    main()