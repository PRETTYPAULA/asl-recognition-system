"""
Fast CPU-friendly CNN trainer for ASL gesture recognition.
Uses a lightweight custom CNN instead of heavy VGG16.
Much faster training while maintaining good accuracy.
"""
import os
import importlib
import time


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.h5")

IMG_DIM = int(os.getenv("IMG_DIM", "224"))
IMG_SIZE = (IMG_DIM, IMG_DIM)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("EPOCHS", "30"))
TRAINING_TIME_LIMIT_SECONDS = int(os.getenv("TRAINING_TIME_LIMIT_SECONDS", "7200"))


def build_model(num_classes: int):
    """Build a lightweight CNN optimized for CPU training."""
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras_models = importlib.import_module("tensorflow.keras.models")
    keras_optimizers = importlib.import_module("tensorflow.keras.optimizers")
    keras_regularizers = importlib.import_module("tensorflow.keras.regularizers")

    model = keras_models.Sequential([
        # Block 1
        keras_layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                          input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        keras_layers.BatchNormalization(),
        keras_layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras_layers.BatchNormalization(),
        keras_layers.MaxPooling2D((2, 2)),
        keras_layers.Dropout(0.25),

        # Block 2
        keras_layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras_layers.BatchNormalization(),
        keras_layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras_layers.BatchNormalization(),
        keras_layers.MaxPooling2D((2, 2)),
        keras_layers.Dropout(0.25),

        # Block 3
        keras_layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras_layers.BatchNormalization(),
        keras_layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras_layers.BatchNormalization(),
        keras_layers.MaxPooling2D((2, 2)),
        keras_layers.Dropout(0.25),

        # Classifier
        keras_layers.Flatten(),
        keras_layers.Dense(256, activation='relu'),
        keras_layers.BatchNormalization(),
        keras_layers.Dropout(0.5),
        keras_layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras_optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    keras_image = importlib.import_module("tensorflow.keras.preprocessing.image")
    image_data_generator = keras_image.ImageDataGenerator
    keras_callbacks = importlib.import_module("tensorflow.keras.callbacks")
    
    class StopByTimeLimit(keras_callbacks.Callback):
        def __init__(self, limit_seconds: int) -> None:
            super().__init__()
            self.limit_seconds = limit_seconds
            self.start_time = 0.0

        def on_train_begin(self, logs=None) -> None:
            self.start_time = time.time()
            print(f"Training time limit: {self.limit_seconds / 60:.1f} minutes")

        def on_epoch_end(self, epoch, logs=None) -> None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.limit_seconds:
                print(f"\nStopping: reached time limit ({elapsed / 60:.1f} minutes).")
                self.model.stop_training = True

    tensorflow_module = importlib.import_module("tensorflow")

    tensorflow_module.config.optimizer.set_jit(True)
    tensorflow_module.config.threading.set_inter_op_parallelism_threads(0)
    tensorflow_module.config.threading.set_intra_op_parallelism_threads(0)

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")

    datagen = image_data_generator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False,
    )

    train_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    print(f"\n{'='*60}")
    print("FAST CNN TRAINER - Lightweight model for CPU")
    print(f"{'='*60}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Classes: {train_data.num_classes}")
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    print(f"{'='*60}\n")

    model = build_model(num_classes=train_data.num_classes)
    model.summary()

    callbacks = [
        StopByTimeLimit(TRAINING_TIME_LIMIT_SECONDS),
        keras_callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("\nStarting training... (this is much faster than VGG16!)\n")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Evaluate
    print("\nEvaluating on validation set...")
    loss, accuracy = model.evaluate(val_data, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
