"""Utilities for the wafer-map CNN experiments."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score
)
from skimage.transform import resize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Rescaling,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tqdm.auto import tqdm


def is_valid_label(label):
    # Handle nested lists or numpy arrays
    while isinstance(label, (np.ndarray, list)):
        if len(label) == 0:
            return False  # Discard empty arrays []
        label = label[0]
    
    # Convert to lowercase string to be safe
    label_str = str(label).strip().lower()
    
    # Return True only if it's explicitly training or test
    return label_str in ['training', 'test']


def set_global_determinism(seed: int = 42, clear_session: bool = False) -> None:
    """Reset Python, NumPy, and TensorFlow random state for a new experiment."""
    if clear_session:
        tf.keras.backend.clear_session()

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def clean_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested label values and standardize their text representation."""
    cleaned = df.copy()

    if "trianTestLabel" in cleaned.columns:
        cleaned = cleaned.rename(columns={"trianTestLabel": "trainTestLabel"})

    def extract_text(value: Any) -> str:
        while isinstance(value, (np.ndarray, list)):
            if len(value) == 0:
                return "unknown"
            value = value[0]
        return str(value).strip().lower()

    for column in ("trainTestLabel", "failureType"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(extract_text)

    return cleaned


def cap_indices_per_class(
    indices: np.ndarray,
    labels: np.ndarray,
    max_samples_per_class: int = 3000,
    seed: int = 42,
) -> np.ndarray:
    """Cap only the supplied index set while preserving every minority example."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []

    for class_value in np.unique(labels[indices]):
        class_indices = indices[labels[indices] == class_value]
        if len(class_indices) > max_samples_per_class:
            class_indices = rng.choice(
                class_indices,
                size=max_samples_per_class,
                replace=False,
            )
        selected.append(np.asarray(class_indices, dtype=np.int64))

    capped = np.concatenate(selected)
    rng.shuffle(capped)
    return capped


def resize_wafer_maps(
    df: pd.DataFrame,
    target_shape: tuple[int, int] = (56, 56),
    column: str = "waferMap",
) -> np.ndarray:
    """Resize discrete wafer maps with nearest-neighbor interpolation."""
    if column not in df.columns:
        raise KeyError(f"Missing required column: {column}")

    resized_maps = []
    print(f"Resizing {len(df):,} wafer maps to {target_shape}...")

    for image in tqdm(df[column].to_numpy(), total=len(df)):
        resized = resize(
            image,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        resized_maps.append(resized.astype(np.uint8))

    return np.expand_dims(np.asarray(resized_maps, dtype=np.uint8), axis=-1)


def _resolve_optimizer(optimizer: str | tf.keras.optimizers.Optimizer):
    if not isinstance(optimizer, str):
        return optimizer

    optimizer_name = optimizer.strip().lower()
    registry = {
        "adam": lambda: Adam(learning_rate=0.001),
        "nadam": lambda: Nadam(learning_rate=0.001),
        "rmsprop": lambda: RMSprop(learning_rate=0.001),
        "sgd_momentum": lambda: SGD(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=False,
        ),
        "sgd_nesterov": lambda: SGD(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True,
        ),
    }

    if optimizer_name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported optimizer '{optimizer}'. Available values: {available}"
        )

    return registry[optimizer_name]()


def build_optimized_model(
    use_he: bool = False,
    use_l2: bool = False,
    optimizer: str | tf.keras.optimizers.Optimizer = "adam",
    input_shape: tuple[int, int, int] = (56, 56, 1),
    num_classes: int = 9,
    l2_strength: float = 0.001,
) -> tf.keras.Model:
    """Build the CNN used throughout the controlled experiments."""
    initializer = "he_normal" if use_he else "glorot_uniform"
    regularizer = l2(l2_strength) if use_l2 else None

    model = Sequential(
        [
            Input(shape=input_shape, name="wafer_map"),
            Rescaling(scale=0.5, name="normalize_discrete_values"),
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="conv_layer_1",
            ),
            BatchNormalization(name="batch_norm_1"),
            MaxPooling2D(pool_size=(2, 2), name="max_pool_1"),
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="conv_layer_2",
            ),
            BatchNormalization(name="batch_norm_2"),
            MaxPooling2D(pool_size=(2, 2), name="max_pool_2"),
            Conv2D(
                128,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="conv_layer_3",
            ),
            BatchNormalization(name="batch_norm_3"),
            MaxPooling2D(pool_size=(2, 2), name="max_pool_3"),
            Flatten(name="flatten"),
            Dense(
                128,
                activation="relu",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="dense_features",
            ),
            Dropout(0.5, name="dropout"),
            Dense(num_classes, activation="softmax", name="class_probabilities"),
        ],
        name="Wafer_CNN",
    )

    model.compile(
        optimizer=_resolve_optimizer(optimizer),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks(
    checkpoint_path: str | Path,
    patience: int = 5,
    reduce_lr_patience: int = 3,
) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks that agree on the same validation objective."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def evaluate_classifier(
    model: tf.keras.Model,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: np.ndarray | list[str],
    batch_size: int = 256,
):
    """Return aggregate metrics, a class report, a confusion matrix, and predictions."""
    labels = np.asarray(labels, dtype=np.int64)
    class_names = np.asarray(class_names)
    class_ids = np.arange(len(class_names))

    probabilities = model.predict(features, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "Balanced Accuracy": balanced_accuracy_score(labels, predictions),
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
    }

    report = pd.DataFrame(
        classification_report(
            labels,
            predictions,
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    return metrics, report, matrix, predictions


def plot_training_history(history_obj, experiment_name: str = "") -> None:
    """Plot training and validation accuracy and loss."""
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history_obj.history["accuracy"], label="Training accuracy")
    axes[0].plot(history_obj.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title(f"Accuracy — {experiment_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history_obj.history["loss"], label="Training loss")
    axes[1].plot(history_obj.history["val_loss"], label="Validation loss")
    axes[1].set_title(f"Loss — {experiment_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: np.ndarray | list[str],
    title: str,
    normalize: bool = True,
) -> None:
    """Plot a raw or row-normalized confusion matrix."""
    matrix_to_plot = matrix.astype(np.float64)

    if normalize:
        row_totals = matrix_to_plot.sum(axis=1, keepdims=True)
        matrix_to_plot = np.divide(
            matrix_to_plot,
            row_totals,
            out=np.zeros_like(matrix_to_plot),
            where=row_totals != 0,
        )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix_to_plot,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()


def visualize_wafer_samples(
    df: pd.DataFrame,
    class_column: str = "failureType",
    image_column: str = "waferMap",
) -> None:
    """Show one original wafer map for each class."""
    classes = sorted(df[class_column].unique())
    columns = 3
    rows = math.ceil(len(classes) / columns)

    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.asarray(axes).reshape(-1)

    for axis, class_name in zip(axes, classes):
        sample = df.loc[df[class_column] == class_name, image_column].iloc[0]
        axis.imshow(sample, cmap="inferno")
        axis.set_title(class_name)
        axis.axis("off")

    for axis in axes[len(classes):]:
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def visualize_original_and_maps(
    model: tf.keras.Model,
    input_image: np.ndarray,
    label_index: int,
    class_names: np.ndarray | list[str],
) -> None:
    """Display an input wafer map and the feature maps of every convolutional layer."""
    convolutional_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    activation_model = tf.keras.Model(
        inputs=model.input,
        outputs=[layer.output for layer in convolutional_layers],
    )
    activations = activation_model.predict(input_image[np.newaxis, ...], verbose=0)

    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(np.squeeze(input_image), cmap="gray")
    plt.title(f"Original\nClass: {class_names[label_index]}")
    plt.axis("off")
    plt.show()

    for layer, activation in zip(convolutional_layers, activations):
        filter_count = activation.shape[-1]
        columns = 8
        rows = math.ceil(filter_count / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(columns * 1.3, rows * 1.3),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        figure.suptitle(f"{layer.name} — {filter_count} feature maps")

        for filter_index in range(filter_count):
            flat_axes[filter_index].imshow(
                activation[0, :, :, filter_index],
                cmap="viridis",
            )
            flat_axes[filter_index].axis("off")

        for axis in flat_axes[filter_count:]:
            axis.axis("off")

        plt.tight_layout()
        plt.show()


def add_bar_labels(axis, bars, suffix: str = "%") -> None:
    """Add compact labels above a collection of bars."""
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


class ValidationMacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        x_validation,
        y_validation,
        checkpoint_path,
        batch_size=256,
        patience=8,
        min_delta=0.001,
    ):
        super().__init__()

        self.x_validation = x_validation
        self.y_validation = np.asarray(y_validation)
        self.checkpoint_path = str(checkpoint_path)
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta

        self.best_macro_f1 = -np.inf
        self.best_epoch = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        probabilities = self.model.predict(
            self.x_validation,
            batch_size=self.batch_size,
            verbose=0,
        )
        predictions = np.argmax(probabilities, axis=1)

        macro_f1 = f1_score(
            self.y_validation,
            predictions,
            average="macro",
            zero_division=0,
        )
        balanced_accuracy = balanced_accuracy_score(
            self.y_validation,
            predictions,
        )

        logs["val_macro_f1"] = macro_f1
        logs["val_balanced_accuracy"] = balanced_accuracy

        print(
            f" — val_macro_f1: {macro_f1:.4f}"
            f" — val_balanced_accuracy: {balanced_accuracy:.4f}"
        )

        if macro_f1 > self.best_macro_f1 + self.min_delta:
            self.best_macro_f1 = macro_f1
            self.best_epoch = epoch + 1
            self.wait = 0
            self.model.save_weights(self.checkpoint_path)

            print(
                f"Saved new best Macro F1 checkpoint "
                f"at epoch {self.best_epoch}."
            )
        else:
            self.wait += 1

            if self.wait >= self.patience:
                print(
                    "Early stopping based on validation Macro F1. "
                    f"Best epoch: {self.best_epoch}."
                )
                self.model.stop_training = True


def make_macro_f1_callbacks(
    x_validation,
    y_validation,
    checkpoint_path,
    batch_size=256,
    patience=8,
    min_delta=0.001,
):
    return [
        ValidationMacroF1Checkpoint(
            x_validation=x_validation,
            y_validation=y_validation,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

def build_intermediate_model(
    use_he=False,
    use_l2=False,
    optimizer="adam",
    input_shape=(56, 56, 1),
    num_classes=9,
    l2_strength=0.001,
):
    """
    Build the intermediate-width wafer-map CNN.

    Architecture:
        Conv2D: 24 -> 48 -> 96 filters
        Dense: 96 units

    The model keeps the same preprocessing, pooling, normalization,
    regularization, dropout, and classifier design as the full CNN.
    """
    initializer = "he_normal" if use_he else "glorot_uniform"
    regularizer = (
        tf.keras.regularizers.l2(l2_strength)
        if use_l2
        else None
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=input_shape,
                name="wafer_map",
            ),
            tf.keras.layers.Rescaling(
                scale=0.5,
                name="normalize_discrete_values",
            ),
            tf.keras.layers.Conv2D(
                24,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="conv_layer_1",
            ),
            tf.keras.layers.BatchNormalization(
                name="batch_norm_1"
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="max_pool_1",
            ),
            tf.keras.layers.Conv2D(
                48,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="conv_layer_2",
            ),
            tf.keras.layers.BatchNormalization(
                name="batch_norm_2"
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="max_pool_2",
            ),
            tf.keras.layers.Conv2D(
                96,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="conv_layer_3",
            ),
            tf.keras.layers.BatchNormalization(
                name="batch_norm_3"
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="max_pool_3",
            ),
            tf.keras.layers.Flatten(
                name="flatten"
            ),
            tf.keras.layers.Dense(
                96,
                activation="relu",
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                name="dense_features",
            ),
            tf.keras.layers.Dropout(
                0.5,
                name="dropout",
            ),
            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                name="class_probabilities",
            ),
        ],
        name="Intermediate_Wafer_CNN",
    )

    model.compile(
        optimizer=_resolve_optimizer(optimizer),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model