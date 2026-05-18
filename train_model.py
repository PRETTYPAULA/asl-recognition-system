import os
import importlib
import time


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.h5")

IMG_DIM = int(os.getenv("IMG_DIM", "224"))
IMG_SIZE = (IMG_DIM, IMG_DIM)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
TRAINING_TIME_LIMIT_SECONDS = int(os.getenv("TRAINING_TIME_LIMIT_SECONDS", "7200"))


def build_model(num_classes: int):
    keras_applications = importlib.import_module("tensorflow.keras.applications")
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras_models = importlib.import_module("tensorflow.keras.models")
    keras_optimizers = importlib.import_module("tensorflow.keras.optimizers")

    base_model = keras_applications.VGG16(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    print("Unfroze top 20 VGG layers for fine-tuning")

    x = base_model.output
    x = keras_layers.GlobalAveragePooling2D()(x)
    x = keras_layers.Dense(256, activation="relu")(x)
    x = keras_layers.Dropout(0.5)(x)
    output = keras_layers.Dense(num_classes, activation="softmax")(x)
    model = keras_models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=keras_optimizers.Adam(learning_rate=1e-5),
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
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
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

    model = build_model(num_classes=train_data.num_classes)
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

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} - Higher confidence model!")


if __name__ == "__main__":
    main()

