import os
import numpy as np
import pickle
import importlib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

LANDMARK_X_PATH = os.path.join(MODEL_DIR, "landmarks_X.npy")
LANDMARK_Y_PATH = os.path.join(MODEL_DIR, "landmarks_y.npy")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "landmark_asl_model.keras")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_landmark_model.py first.")

    X = np.load(LANDMARK_X_PATH)
    y = np.load(LANDMARK_Y_PATH)

    with open(LABEL_MAP_PATH, "rb") as f:
        labels = pickle.load(f)

    # Use the same validation split as training
    val_split = 0.2
    split_idx = int(len(X) * (1 - val_split))
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    keras_models = importlib.import_module("tensorflow.keras.models")
    model = keras_models.load_model(MODEL_PATH)

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc * 100:.2f}%\n")

    # Per-class accuracy
    predictions = np.argmax(model.predict(X_val, verbose=0), axis=1)
    from sklearn.metrics import classification_report, confusion_matrix

    print("Classification Report:")
    print(classification_report(y_val, predictions, target_names=labels))

    # Show per-class accuracy in a simple table
    print("Per-Class Accuracy:")
    for i, label in enumerate(labels):
        mask = y_val == i
        if np.sum(mask) == 0:
            continue
        class_acc = np.mean(predictions[mask] == i)
        print(f"  {label}: {class_acc * 100:.1f}% ({np.sum(mask)} samples)")


if __name__ == "__main__":
    main()

