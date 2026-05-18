import os
import cv2
import numpy as np
import importlib
import mediapipe as mp
import pickle

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
HAND_TASK_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

# Try to load the task-based MediaPipe API (available in this environment)
MP_TASKS = None
MP_VISION = None
try:
    MP_TASKS = importlib.import_module("mediapipe.tasks.python")
    MP_VISION = importlib.import_module("mediapipe.tasks.python.vision")
except ModuleNotFoundError:
    pass

# Create a persistent HandLandmarker so we don't recreate it per-image
_hand_landmarker = None


def _get_hand_landmarker():
    global _hand_landmarker
    if _hand_landmarker is not None:
        return _hand_landmarker
    if MP_TASKS is None or MP_VISION is None:
        return None
    if not os.path.exists(HAND_TASK_MODEL_PATH):
        raise FileNotFoundError(f"Missing hand landmarker model: {HAND_TASK_MODEL_PATH}")
    options = MP_VISION.HandLandmarkerOptions(
        base_options=MP_TASKS.BaseOptions(model_asset_path=HAND_TASK_MODEL_PATH),
        running_mode=MP_VISION.RunningMode.IMAGE,
        num_hands=1,
    )
    _hand_landmarker = MP_VISION.HandLandmarker.create_from_options(options)
    return _hand_landmarker


def extract_landmarks(image_path):
    """Extract and normalize 21 MediaPipe hand landmarks from an image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  [ERROR] Failed to load image {image_path}: {e}")
        return None

    landmarker = _get_hand_landmarker()
    if landmarker is None:
        return None

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
    except Exception as e:
        print(f"  [ERROR] MediaPipe detection failed for {image_path}: {e}")
        return None

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # --- Anatomically-aware normalization ---
    # 1. Translate so wrist (landmark 0) is at origin
    wrist = landmarks_array[0].copy()
    landmarks_array -= wrist

    # 2. Scale by palm size (wrist -> middle-finger MCP, landmark 9)
    palm_size = np.linalg.norm(landmarks_array[9])

    if palm_size > 1e-6:
        landmarks_array /= palm_size
    else:
        max_dist = np.max(np.abs(landmarks_array))
        if max_dist > 1e-6:
            landmarks_array /= max_dist

    return landmarks_array.flatten()


def generate_dataset(max_images_per_class=None, report_every=100):
    """
    Generate landmark dataset from images.

    Args:
        max_images_per_class: Cap images per class (None = no limit).
        report_every: Print progress every N images processed.
    """
    X, y = [], []
    labels = []
    label_to_idx = {}
    total_processed = 0
    total_extracted = 0
    total_skipped = 0

    for label_dir in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(label_path):
            continue

        label = label_dir.upper()
        if label not in label_to_idx:
            label_to_idx[label] = len(labels)
            labels.append(label)

        count = 0
        skipped = 0
        image_files = sorted([
            f for f in os.listdir(label_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Apply cap if configured
        if max_images_per_class is not None:
            image_files = image_files[:max_images_per_class]

        for idx, img_file in enumerate(image_files, start=1):
            img_path = os.path.join(label_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_to_idx[label])
                count += 1
            else:
                skipped += 1

            total_processed += 1
            if report_every and total_processed % report_every == 0:
                print(f"  ... processed {total_processed} images total (extracted {total_extracted + count}, skipped {total_skipped + skipped})")

        total_extracted += count
        total_skipped += skipped
        print(f"[{label}] {count} extracted, {skipped} skipped (no hand) | total files: {len(image_files)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, 'landmarks_X.npy'), X)
    np.save(os.path.join(MODEL_DIR, 'landmarks_y.npy'), y)
    with open(os.path.join(MODEL_DIR, 'label_map.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    print(f"\nLandmark dataset ready: {X.shape[0]} samples, {len(labels)} classes")
    print(f"Feature vector length: {X.shape[1]} (21 landmarks × 3 coords)")
    print(f"Total processed: {total_processed}, extracted: {total_extracted}, skipped: {total_skipped}")
    print(f"Saved to: {MODEL_DIR}")
    return X, y, labels


if __name__ == '__main__':
    # Default: cap at 500 images per class for fast iteration
    MAX_IMAGES = int(os.environ.get("MAX_IMAGES_PER_CLASS", "500"))
    if MAX_IMAGES <= 0:
        MAX_IMAGES = None
    print(f"MAX_IMAGES_PER_CLASS={MAX_IMAGES}")
    generate_dataset(max_images_per_class=MAX_IMAGES)

