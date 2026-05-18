import os
import numpy as np
import pickle
import importlib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = MODEL_DIR

LANDMARK_X_PATH = os.path.join(DATA_DIR, "landmarks_X.npy")
LANDMARK_Y_PATH = os.path.join(DATA_DIR, "landmarks_y.npy")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.pkl")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "landmark_asl_model.keras")

# Training hyperparameters
EPOCHS = 200
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def augment_landmarks(X, y, noise_std=0.02, rotation_std=0.1, scale_std=0.1, num_augment=2):
    """Apply landmark-specific augmentation: Gaussian noise, small rotation, scaling."""
    X_aug, y_aug = [], []
    for _ in range(num_augment):
        X_copy = X.copy()
        # Per-landmark Gaussian noise
        X_copy += np.random.normal(0, noise_std, X_copy.shape).astype(np.float32)

        # Small random rotation around Z axis (applied per sample)
        angles = np.random.normal(0, rotation_std, len(X_copy))
        scales = np.random.normal(1.0, scale_std, len(X_copy))

        for i in range(len(X_copy)):
            lm = X_copy[i].reshape(-1, 3)
            theta = angles[i]
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            lm = (R @ lm.T).T
            lm *= scales[i]
            X_copy[i] = lm.flatten()

        X_aug.append(X_copy)
        y_aug.append(y.copy())

    return np.concatenate([X] + X_aug), np.concatenate([y] + y_aug)


def build_model(input_dim, num_classes):
    keras_layers = importlib.import_module("tensorflow.keras.layers")
    keras_models = importlib.import_module("tensorflow.keras.models")
    keras_optimizers = importlib.import_module("tensorflow.keras.optimizers")
    keras_regularizers = importlib.import_module("tensorflow.keras.regularizers")

    model = keras_models.Sequential([
        keras_layers.Input(shape=(input_dim,)),
        keras_layers.Dense(256, activation="relu", kernel_regularizer=keras_regularizers.l2(1e-4)),
        keras_layers.BatchNormalization(),
        keras_layers.Dropout(0.4),
        keras_layers.Dense(128, activation="relu", kernel_regularizer=keras_regularizers.l2(1e-4)),
        keras_layers.BatchNormalization(),
        keras_layers.Dropout(0.3),
        keras_layers.Dense(64, activation="relu"),
        keras_layers.Dropout(0.2),
        keras_layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras_optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(LANDMARK_X_PATH) or not os.path.exists(LANDMARK_Y_PATH):
        raise FileNotFoundError("Landmark dataset not found. Run generate_landmarks.py first.")

    X = np.load(LANDMARK_X_PATH)
    y = np.load(LANDMARK_Y_PATH)

    with open(LABEL_MAP_PATH, "rb") as f:
        labels = pickle.load(f)

    print(f"Loaded dataset: {X.shape[0]} samples, {len(labels)} classes")
    print(f"Labels: {labels}")

    # Augment training data
    print("Augmenting dataset...")
    X, y = augment_landmarks(X, y, noise_std=0.015, rotation_std=0.08, scale_std=0.08, num_augment=3)
    print(f"After augmentation: {X.shape[0]} samples")

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Split
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

    keras_callbacks = importlib.import_module("tensorflow.keras.callbacks")

    callbacks = [
        keras_callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model = build_model(input_dim=X.shape[1], num_classes=len(labels))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Classification report
    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=labels))


if __name__ == "__main__":
    main()

