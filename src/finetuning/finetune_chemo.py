"""
Downstream fine-tuning for AP and NF clinical event prediction.

This module loads a GC self-supervised pretraining checkpoint and fine-tunes
the EMIT-based oncology model for binary classification of:
- Chemotherapy-induced Aplasia (AP)
- Neutropenic Fever (NF)

Key thesis-relevant features
-----------------------------
- Loads GC-pretrained weights or trains from random initialization.
- Supports 7-day (168–336 h) and full 14-day (0–336 h) observation windows.
- Implements stratified batching with guaranteed minimum positive-class exposure
  per batch to address severe class imbalance.
- Tracks ROC-AUC and PR-AUC on the validation fold and selects the best
  checkpoint by the sum of these two metrics.
- Reports final performance on a held-out test set with ROC-AUC, PR-AUC,
  minimum of precision/recall, and maximum F1.

All tensors, labels, and results are derived from restricted MIMIC-IV data
and must not be committed to a public repository.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from src.model import build_strats

tqdm.pandas()


VALID_BATCHING_STRATEGIES = {"random_shuffle", "cycle_balanced"}


def set_global_determinism(seed: int) -> None:
    """
    Set Python, NumPy, and TensorFlow random seeds.

    Shell scripts should additionally set PYTHONHASHSEED,
    TF_DETERMINISTIC_OPS, and TF_CUDNN_DETERMINISTIC before TensorFlow starts.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


class BaseCycleIndex:
    """
    Iterate over dataset indices in fixed-size batches with optional shuffling.

    This class wraps around the index array and yields contiguous batches,
    reshuffling when the end of the array is reached.
    """

    def __init__(
        self,
        indices: list[int] | np.ndarray | int,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        if isinstance(indices, int):
            indices = np.arange(indices)

        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pointer = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def _next_indices(
        self,
        arr: np.ndarray,
        pointer: int,
        size: int,
    ) -> tuple[np.ndarray, bool, int]:
        end = pointer + size
        if end <= len(arr):
            out = arr[pointer:end]
            pointer = end % len(arr)
            end_reached = False
        else:
            out = np.concatenate((arr[pointer:], arr[: end % len(arr)]))
            pointer = end % len(arr)
            end_reached = True
        return out, end_reached, pointer

    def get_batch_ind(self) -> np.ndarray:
        batch, end_reached, self.pointer = self._next_indices(
            self.indices,
            self.pointer,
            self.batch_size,
        )
        if end_reached and self.shuffle:
            np.random.shuffle(self.indices)
        return batch


class CycleIndexBalanced(BaseCycleIndex):
    """
    Enforce a guaranteed minimum number of positive-class samples per batch.

    This stratified batching strategy is essential for highly imbalanced
    clinical event prediction tasks such as AP and NF.
    """

    def __init__(
        self,
        indices: list[int] | np.ndarray | int,
        y: list[int] | np.ndarray,
        batch_size: int,
        min_class1: int = 1,
        shuffle: bool = True,
    ) -> None:
        if isinstance(indices, int):
            indices = np.arange(indices)

        self.indices = np.asarray(indices)
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.min_class1 = min_class1
        self.shuffle = shuffle

        self.class1_inds = self.indices[self.y == 1]
        self.class0_inds = self.indices[self.y == 0]

        self.pointer0 = 0
        self.pointer1 = 0

        if self.shuffle:
            np.random.shuffle(self.class1_inds)
            np.random.shuffle(self.class0_inds)

    def get_batch_ind(self) -> np.ndarray:
        n1 = min(self.min_class1, len(self.class1_inds))
        class1_batch, end1_reached, self.pointer1 = self._next_indices(
            self.class1_inds,
            self.pointer1,
            n1,
        )

        n0 = self.batch_size - len(class1_batch)
        class0_batch, end0_reached, self.pointer0 = self._next_indices(
            self.class0_inds,
            self.pointer0,
            n0,
        )

        if self.shuffle and end1_reached:
            np.random.shuffle(self.class1_inds)
        if self.shuffle and end0_reached:
            np.random.shuffle(self.class0_inds)

        batch = np.concatenate((class1_batch, class0_batch))
        np.random.shuffle(batch)
        return batch


def load_and_generate_train_val_test_sets(
    data_path: Path,
    max_seq_length: int,
    use_7_day_window: bool,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load cross-validation folds and construct 3D input tensors.

    Parameters
    ----------
    data_path :
        Path to the pickled tuple containing raw event data, outcomes, and
        train/valid/test index arrays.
    max_seq_length :
        Maximum number of clinical events per admission. Events beyond this
        limit are truncated from the beginning of each sequence, preserving
        the most recent events.
    use_7_day_window :
        If True, restrict inputs to hours 168–336 (last 7 days).
        If False, use hours 0–336 (full 14-day window).

    Returns
    -------
    train_ip :
        List of three training input arrays: [times, values, variable IDs].
    valid_ip :
        Validation input arrays with the same structure.
    test_ip :
        Test input arrays with the same structure.
    train_op :
        Training binary labels (binary outcome labels from the fold package).
    valid_op :
        Validation binary labels.
    test_op :
        Test binary labels.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Fine-tuning data file not found: {data_path}")

    with data_path.open("rb") as handle:
        data, oc, train_ind, valid_ind, test_ind = pickle.load(handle)

    if use_7_day_window:
        data = data.loc[(data.hour > 168) & (data.hour <= 336)]
        print("[*] Applying 7-Day Window (168h - 336h)")
    else:
        data = data.loc[data.hour <= 336]
        print("[*] Applying Full 14-Day Window (0h - 336h)")

    all_indices = np.concatenate((train_ind, valid_ind, test_ind))
    data = data.loc[data.ts_ind.isin(all_indices)]
    oc = oc.loc[oc.ts_ind.isin(all_indices)]

    data.loc[(data.variable == "Age") & (data.value > 200), "value"] = 91.4

    y = (
        oc.sort_values(by="ts_ind")["in_hospital_mortality"]
        .to_numpy()
        .astype("float32")
    )
    N = int(data.ts_ind.max() + 1)

    static_varis = ["Age", "Gender"]
    static_data = data.loc[data.variable.isin(static_varis)]
    data = data.loc[~data.variable.isin(static_varis)]

    data = data.sort_values(by=["ts_ind", "hour", "variable"])
    data = data.groupby("ts_ind").tail(max_seq_length)

    N = int(data.ts_ind.max() + 1)
    varis = sorted(set(data.variable))
    V = len(varis)

    def inv_list(l: list, start: int = 0) -> dict:
        return {value: idx + start for idx, value in enumerate(l)}

    var_to_ind = inv_list(varis, start=1)
    data["vind"] = data.variable.map(var_to_ind)
    data = data[["ts_ind", "vind", "hour", "value"]].sort_values(
        by=["ts_ind", "vind", "hour"]
    )
    data = data.sort_values(by="ts_ind").reset_index(drop=True)
    data = data.reset_index().rename(columns={"index": "obs_ind"})

    min_obs = data.groupby("ts_ind")["obs_ind"].min().reset_index()
    min_obs = min_obs.rename(columns={"obs_ind": "first_obs_ind"})
    data = data.merge(min_obs, on="ts_ind")
    data["obs_ind"] = data["obs_ind"] - data["first_obs_ind"]

    times_inp = np.zeros((N, max_seq_length), dtype="float32")
    values_inp = np.zeros((N, max_seq_length), dtype="float32")
    varis_inp = np.zeros((N, max_seq_length), dtype="int32")

    for row in data.itertuples():
        ts_ind = int(row.ts_ind)
        l = int(row.obs_ind)
        times_inp[ts_ind, l] = float(row.hour)
        values_inp[ts_ind, l] = float(row.value)
        varis_inp[ts_ind, l] = int(row.vind)

    data.drop(columns=["obs_ind", "first_obs_ind"], inplace=True)

    train_ind = [i for i in train_ind if i < N]
    valid_ind = [i for i in valid_ind if i < N]
    test_ind = [i for i in test_ind if i < N]

    train_ip = [ip[train_ind] for ip in [times_inp, values_inp, varis_inp]]
    valid_ip = [ip[valid_ind] for ip in [times_inp, values_inp, varis_inp]]
    test_ip = [ip[test_ind] for ip in [times_inp, values_inp, varis_inp]]

    del times_inp, values_inp, varis_inp

    train_op = y[train_ind]
    valid_op = y[valid_ind]
    test_op = y[test_ind]
    del y

    return train_ip, valid_ip, test_ip, train_op, valid_op, test_op


def load_pretrained_base_model(
    weights_path: Path,
    max_seq_length: int,
    vocabulary_size: int,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
) -> tf.keras.Model:
    """
    Build the base EMIT architecture and load GC pretraining weights.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {weights_path}")

    model = build_strats(
        max_len=max_seq_length,
        V=vocabulary_size,
        d=embedding_dim,
        N=num_layers,
        he=num_heads,
        dropout=dropout,
    )
    model.load_weights(str(weights_path))
    return model


def build_random_base_model(
    max_seq_length: int,
    vocabulary_size: int,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
) -> tf.keras.Model:
    """
    Build the base EMIT architecture with random initialization.
    """
    return build_strats(
        max_len=max_seq_length,
        V=vocabulary_size,
        d=embedding_dim,
        N=num_layers,
        he=num_heads,
        dropout=dropout,
    )


def extend_model_with_binary_head(
    base_model: tf.keras.Model,
    max_seq_length: int,
) -> tf.keras.Model:
    """
    Attach a single-neuron sigmoid classification head for binary fine-tuning.

    The classifier is attached to the base model output, which is produced
    after global attention pooling and the existing forecast head.
    """
    input_layers = [
        Input(shape=(max_seq_length,)),
        Input(shape=(max_seq_length,)),
        Input(shape=(max_seq_length,)),
    ]

    base_output = base_model(input_layers)
    output_layer = Dense(1, activation="sigmoid")(base_output)

    return Model(inputs=input_layers, outputs=output_layer)


def mortality_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Unweighted binary cross-entropy for downstream fine-tuning.
    """
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute ROC-AUC, PR-AUC, minimum of precision/recall, and maximum F1.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    min_rp = float(np.minimum(precision, recall).max())
    roc_auc = float(roc_auc_score(y_true, y_pred))

    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator != 0),
    )
    max_f1 = float(np.max(f1_scores))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "min_rp": min_rp,
        "max_f1": max_f1,
    }


class CustomCallback(Callback):
    """
    Track validation ROC-AUC and PR-AUC after each epoch.

    The monitored metric is the sum of ROC-AUC and PR-AUC, as defined in the
    early-stopping configuration.
    """

    def __init__(
        self,
        validation_data: tuple[list[np.ndarray], np.ndarray],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.val_x, self.val_y = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}

        y_pred = self.model.predict(
            self.val_x,
            verbose=0,
            batch_size=self.batch_size,
        )
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        y_pred = np.asarray(y_pred).ravel()

        metrics = compute_clinical_metrics(self.val_y, y_pred)
        logs["custom_metric"] = metrics["roc_auc"] + metrics["pr_auc"]
        print(
            f"Epoch {epoch + 1} | "
            f"val ROC-AUC: {metrics['roc_auc']:.4f}, "
            f"val PR-AUC: {metrics['pr_auc']:.4f}"
        )


class RandomShuffleSequence(tf.keras.utils.Sequence):
    """
    Simple shuffled data generator for random-shuffle batching.
    """

    def __init__(
        self,
        input_data: list[np.ndarray],
        output_data: np.ndarray,
        batch_size: int,
    ) -> None:
        self.input_data = input_data
        self.output_data = output_data
        self.batch_size = batch_size
        self.indices = np.arange(len(output_data))
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size

    def __getitem__(
        self, batch_index: int
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        start = batch_index * self.batch_size
        end = start + self.batch_size
        batch_inds = self.indices[start:end]
        batch_x = [data[batch_inds] for data in self.input_data]
        batch_y = self.output_data[batch_inds]
        return tuple(batch_x), batch_y

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.indices)


def train_and_evaluate_model(
    model: tf.keras.Model,
    train_input: list[np.ndarray],
    train_output: np.ndarray,
    valid_input: list[np.ndarray],
    valid_output: np.ndarray,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    weight_decay: float,
    pos_ratio: float,
    batching_strategy: str,
) -> dict[str, list[float]]:
    """
    Fine-tune the classification head with clinical-metric monitoring.

    Parameters
    ----------
    model :
        Extended Keras model with a binary sigmoid head.
    train_input :
        List of three training input arrays.
    train_output :
        Training binary labels.
    valid_input :
        Validation input arrays.
    valid_output :
        Validation binary labels.
    batch_size :
        Mini-batch size.
    learning_rate :
        Base learning rate for the Adam optimizer.
    epochs :
        Maximum number of training epochs.
    patience :
        Number of epochs without improvement before early stopping.
    weight_decay :
        L2 regularization strength in the optimizer.
    pos_ratio :
        Target fraction of positive-class samples per balanced batch.
    batching_strategy :
        Either "random_shuffle" or "cycle_balanced".

    Returns
    -------
    history :
        Dictionary containing epoch-wise training and validation metrics.
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100_000,
        decay_rate=0.96,
        staircase=True,
    )

    model.compile(
        loss=mortality_loss,
        optimizer=Adam(learning_rate=lr_schedule, weight_decay=weight_decay),
    )

    es = EarlyStopping(
        monitor="custom_metric",
        patience=patience,
        mode="max",
        restore_best_weights=True,
    )
    cus = CustomCallback(
        validation_data=(valid_input, valid_output),
        batch_size=batch_size,
    )

    train_indices = np.arange(len(train_output))
    steps_per_epoch = max(1, len(train_indices) // batch_size)

    min_positives_per_batch = max(1, int(batch_size * pos_ratio))

    def balanced_generator():
        cycler = CycleIndexBalanced(
            indices=train_indices,
            y=train_output,
            batch_size=batch_size,
            min_class1=min_positives_per_batch,
            shuffle=True,
        )
        while True:
            batch_inds = cycler.get_batch_ind()
            batch_x = [data[batch_inds] for data in train_input]
            batch_y = train_output[batch_inds]
            yield tuple(batch_x), batch_y

    if batching_strategy == "random_shuffle":
        training_generator = RandomShuffleSequence(
            train_input,
            train_output,
            batch_size,
        )
        print("[*] Using random-shuffle batches with natural fold class distribution.")
        fit_kwargs = {
            "epochs": epochs,
            "verbose": 1,
            "callbacks": [cus, es],
        }
    elif batching_strategy == "cycle_balanced":
        training_generator = balanced_generator()
        print(
            f"[*] Using CycleIndexBalanced batches with "
            f"min_positives_per_batch={min_positives_per_batch}."
        )
        fit_kwargs = {
            "epochs": epochs,
            "verbose": 1,
            "callbacks": [cus, es],
            "steps_per_epoch": steps_per_epoch,
        }
    else:
        raise ValueError(
            "batching_strategy must be 'random_shuffle' or 'cycle_balanced'."
        )

    history = model.fit(training_generator, **fit_kwargs).history
    return history


def evaluate_on_test_set(
    model: tf.keras.Model,
    test_input: list[np.ndarray],
    test_output: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """
    Evaluate the final fine-tuned model on the held-out test set.
    """
    y_pred = model.predict(test_input, verbose=0, batch_size=batch_size)
    if isinstance(y_pred, list):
        y_pred = y_pred[0]
    y_pred = np.asarray(y_pred).ravel()

    return compute_clinical_metrics(test_output, y_pred)


@hydra.main(config_path="../../config", config_name=None, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run AP or NF downstream fine-tuning with cross-validation repeats.
    """
    if cfg.cohort_name not in {"AP", "NF"}:
        raise ValueError(
            "Fine-tuning supports AP and NF cohorts only. "
            f"Received cohort_name={cfg.cohort_name!r}."
        )

    if cfg.batching_strategy not in VALID_BATCHING_STRATEGIES:
        valid_strategies = ", ".join(sorted(VALID_BATCHING_STRATEGIES))
        raise ValueError(
            f"Unsupported batching_strategy '{cfg.batching_strategy}'. "
            f"Choose one of: {valid_strategies}."
        )

    seed = int(cfg.get("seed", 2021))
    set_global_determinism(seed)

    data_path = Path(hydra.utils.to_absolute_path(cfg.data_path))
    pt_model_weights_path = Path(
        hydra.utils.to_absolute_path(cfg.pt_model_weights_path)
    )
    results_path = Path(hydra.utils.to_absolute_path(cfg.results_path))
    results_path.parent.mkdir(parents=True, exist_ok=True)

    use_pretrained_weights = bool(cfg.get("use_pretrained_weights", True))
    lds = list(cfg.lds)
    repeats = int(cfg.repeats)
    max_seq_length = int(cfg.max_seq_length)
    vocabulary_size = int(cfg.V)
    embedding_dim = int(cfg.d)
    num_layers = int(cfg.N)
    num_heads = int(cfg.he)
    dropout = float(cfg.dropout)
    batch_size = int(cfg.batch_size)
    learning_rate = float(cfg.learning_rate)
    epochs = int(cfg.epochs)
    patience = int(cfg.patience)
    weight_decay = float(cfg.weight_decay)
    pos_ratio = float(cfg.get("pos_ratio", 0.5))
    batching_strategy = str(cfg.batching_strategy)
    use_7_day_window = bool(cfg.USE_7_DAY_WINDOW)

    print(f"Fine-tuning cohort: {cfg.cohort_name}")
    print(f"Observation window: {'7-day' if use_7_day_window else '14-day'}")
    print(f"Pretrained initialization: {use_pretrained_weights}")
    print(f"Batching strategy: {batching_strategy}")
    print(f"Pos ratio per balanced batch: {pos_ratio}")

    (
        train_ip,
        valid_ip,
        test_ip,
        train_op,
        valid_op,
        test_op,
    ) = load_and_generate_train_val_test_sets(
        data_path=data_path,
        max_seq_length=max_seq_length,
        use_7_day_window=use_7_day_window,
    )

    train_inds = np.arange(len(train_op))
    valid_inds = np.arange(len(valid_op))

    gen_res: dict[int, list[tuple[float, float]]] = {}

    for ld in lds:
        np.random.shuffle(train_inds)
        np.random.shuffle(valid_inds)

        train_subset_size = int(ld * len(train_inds) / 100)
        valid_subset_size = int(ld * len(valid_inds) / 100)

        train_starts = [
            int(i)
            for i in np.linspace(
                0,
                len(train_inds) - train_subset_size,
                repeats,
            )
        ]
        valid_starts = [
            int(i)
            for i in np.linspace(
                0,
                len(valid_inds) - valid_subset_size,
                repeats,
            )
        ]

        all_test_res: list[dict[str, float]] = []

        for i in range(repeats):
            print(f"Repeat {i}, ld {ld}")

            curr_train_ind = train_inds[
                train_starts[i] : train_starts[i] + train_subset_size
            ]
            curr_valid_ind = valid_inds[
                valid_starts[i] : valid_starts[i] + valid_subset_size
            ]

            curr_train_ip = [data[curr_train_ind] for data in train_ip]
            curr_train_op = train_op[curr_train_ind]
            curr_valid_ip = [data[curr_valid_ind] for data in valid_ip]
            curr_valid_op = valid_op[curr_valid_ind]

            if use_pretrained_weights:
                base_model = load_pretrained_base_model(
                    weights_path=pt_model_weights_path,
                    max_seq_length=max_seq_length,
                    vocabulary_size=vocabulary_size,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                print("[*] Initializing fine-tuning from GC pretrained weights.")
            else:
                base_model = build_random_base_model(
                    max_seq_length=max_seq_length,
                    vocabulary_size=vocabulary_size,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                print("[*] Initializing fine-tuning from random model weights.")

            extended_model = extend_model_with_binary_head(
                base_model,
                max_seq_length=max_seq_length,
            )

            _ = train_and_evaluate_model(
                model=extended_model,
                train_input=curr_train_ip,
                train_output=curr_train_op,
                valid_input=curr_valid_ip,
                valid_output=curr_valid_op,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                weight_decay=weight_decay,
                pos_ratio=pos_ratio,
                batching_strategy=batching_strategy,
            )

            test_metrics = evaluate_on_test_set(
                model=extended_model,
                test_input=test_ip,
                test_output=test_op,
                batch_size=batch_size,
            )
            all_test_res.append(test_metrics)

            K.clear_session()

        def aggregate_metric(
            metric_name: str,
        ) -> tuple[float, float]:
            values = [r[metric_name] for r in all_test_res]
            return float(np.mean(values)), float(np.std(values))

        gen_res[int(ld)] = [
            aggregate_metric("roc_auc"),
            aggregate_metric("pr_auc"),
            aggregate_metric("min_rp"),
            aggregate_metric("max_f1"),
        ]

    with results_path.open("wb") as handle:
        pickle.dump(gen_res, handle)
        print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()