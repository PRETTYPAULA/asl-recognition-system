import os
import importlib
from typing import Tuple


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_model.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_model() -> Tuple[float, float]:
    keras_models = importlib.import_module("tensorflow.keras.models")
    keras_image = importlib.import_module("tensorflow.keras.preprocessing.image")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")

    datagen = keras_image.ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    val_data = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    model = keras_models.load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(val_data, verbose=1)
    return loss, accuracy * 100


if __name__ == "__main__":
    _, accuracy_percentage = evaluate_model()
    print(f"Validation Accuracy: {accuracy_percentage:.2f}%")
