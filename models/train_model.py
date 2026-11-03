"""
VGG16 Transfer Learning Training Pipeline for Fire & Smoke Detection
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Set environment variables for clean logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
from keras.applications.vgg16 import VGG16
from keras import layers, models, optimizers, callbacks
import numpy as np

# Add parent directory to sys.path for config import
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config


def build_vgg16_transfer_model(num_classes: int = 2) -> models.Model:
    """
    Constructs a CNN Transfer Learning Architecture using VGG16:
      - Frozen VGG16 Convolutional Base (Pretrained on ImageNet)
      - Global Average Pooling 2D
      - Dense(256, ReLU) + BatchNormalization + Dropout(0.4)
      - Softmax Dense Output Layer for Class Probabilities
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    base_model.trainable = False  # Freeze convolutional feature extraction layers

    inputs = layers.Input(shape=config.INPUT_SHAPE, name="input_cctv_frame")
    # Automatic VGG16 standard RGB->BGR & mean subtraction layer
    x = layers.Lambda(keras.applications.vgg16.preprocess_input, name="vgg16_preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_feature_layer")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.4, name="dropout_regularization")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VGG16_FireGuard_Transfer")
    
    loss_fn = "categorical_crossentropy" if num_classes > 1 else "binary_crossentropy"
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model


def train_on_dataset(
    dataset_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    output_model_path: Path = config.DEFAULT_MODEL_PATH
):
    """
    Loads dataset from directories, verifies classes, and trains VGG16.
    Expects dataset organized as:
      dataset_dir/
        ├── Fire/
        └── Normal/  (or Non-Fire, Smoke, etc.)
    """
    print("=" * 70)
    print("FIREGUARD AI - VGG16 TRAINING ENGINE")
    print("=" * 70)
    print(f"Checking dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"\n[ERROR] Dataset folder '{dataset_dir}' does not exist.")
        print("Please create the folder and copy your Kaggle FIRE Dataset into it:")
        print("  data/dataset/Fire/")
        print("  data/dataset/Normal/ (or Non-Fire / Smoke)")
        return False

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(subdirs) < 2:
        print(f"\n[ERROR] Found fewer than 2 class folders in '{dataset_dir}'.")
        print(f"Current folders: {[d.name for d in subdirs]}")
        print("Kaggle FIRE Dataset requires at least 2 categories (e.g., 'Fire' and 'Normal').")
        return False

    classes = sorted([d.name for d in subdirs])
    num_classes = len(classes)
    print(f"\nDiscovered {num_classes} classes: {classes}")

    # Build tf.data datasets
    try:
        train_ds = keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical"
        )
        val_ds = keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=config.IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="categorical"
        )
        # Cache and prefetch for high-speed training
        train_ds = train_ds.cache().prefetch(buffer_size=32)
        val_ds = val_ds.cache().prefetch(buffer_size=32)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset images: {e}")
        return False

    # Build model
    print(f"\nBuilding VGG16 architecture with {num_classes} output heads...")
    model = build_vgg16_transfer_model(num_classes=num_classes)
    model.summary()

    # Callbacks
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(output_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("\nStarting Transfer Learning Fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cb_early_stop, cb_checkpoint]
    )

    # Save classes map
    class_map = {str(i): cname for i, cname in enumerate(classes)}
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    # Save evaluation metrics
    val_acc = max(history.history.get("val_accuracy", [0.0]))
    train_acc = max(history.history.get("accuracy", [0.0]))
    val_loss = min(history.history.get("val_loss", [0.0]))
    
    metrics = {
        "model_name": "VGG16 Transfer Learning",
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "num_classes": num_classes,
        "epochs_completed": len(history.history["loss"]),
        "train_accuracy": round(float(train_acc) * 100, 2),
        "val_accuracy": round(float(val_acc) * 100, 2),
        "val_loss": round(float(val_loss), 4),
        "input_dimensions": list(config.INPUT_SHAPE),
        "status": "TRAINED_ON_DATASET"
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Training Complete! Saved weights to: {output_model_path}")
    print(f"Peak Validation Accuracy: {metrics['val_accuracy']}%")
    print(f"Classes Saved: {classes}")
    print("=" * 70)
    return True


def init_transfer_model(num_classes: int = 2, output_path: Path = config.DEFAULT_MODEL_PATH):
    """
    Initializes and saves a pre-configured VGG16 model architecture with ImageNet weights.
    Provides students and evaluators a working deep learning model out of the box.
    """
    print(f"Initializing VGG16 transfer model ({num_classes} classes)...")
    model = build_vgg16_transfer_model(num_classes=num_classes)
    
    # Save model weights
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    classes = ["Fire", "Normal"] if num_classes == 2 else ["Fire", "Smoke", "Normal"]
    class_map = {str(i): cname for i, cname in enumerate(classes)}
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    metrics = {
        "model_name": "VGG16 Transfer Learning",
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "num_classes": num_classes,
        "epochs_completed": 0,
        "train_accuracy": 0.0,
        "val_accuracy": 0.0,
        "val_loss": 0.0,
        "input_dimensions": list(config.INPUT_SHAPE),
        "status": "INITIALIZED_BASELINE"
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Initialized VGG16 model saved to: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIREGUARD AI - VGG16 Training Script")
    parser.add_argument("--data_dir", type=str, default=str(config.DATASET_DIR), help="Path to Kaggle FIRE dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--init_baseline", action="store_true", help="Initialize and save VGG16 model architecture")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes (2 for Fire/Normal, 3 for Fire/Smoke/Normal)")
    args = parser.parse_args()

    if args.init_baseline:
        init_transfer_model(num_classes=args.classes)
    else:
        dataset_path = Path(args.data_dir)
        if not dataset_path.exists() or len(list(dataset_path.glob("*/*"))) < 10:
            print("\n[NOTE] No full Kaggle dataset found in data/dataset/.")
            print("Run with '--init_baseline' to initialize the VGG16 model architecture for testing,")
            print("or place your Kaggle FIRE Dataset inside 'data/dataset/Fire' and 'data/dataset/Normal'.")
        else:
            train_on_dataset(dataset_path, epochs=args.epochs, batch_size=args.batch_size)
