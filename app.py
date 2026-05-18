import os
import sqlite3
import threading
import time
import importlib
from datetime import datetime
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, url_for

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "asl_model.h5")
LANDMARK_MODEL_PATH = os.path.join(BASE_DIR, "model", "landmark_asl_model.keras")
LEARNING_IMAGE_DIR = os.path.join(BASE_DIR, "static", "images", "learning")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
HAND_TASK_MODEL_PATH = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

app = Flask(__name__, static_folder='static', template_folder='templates')

try:
    MP_HANDS = mp.solutions.hands
except AttributeError:
    try:
        MP_HANDS = importlib.import_module("mediapipe.solutions.hands")
    except ModuleNotFoundError:
        MP_HANDS = None

MP_TASKS = None
MP_VISION = None
try:
    MP_TASKS = importlib.import_module("mediapipe.tasks.python")
    MP_VISION = importlib.import_module("mediapipe.tasks.python.vision")
except ModuleNotFoundError:
    MP_TASKS = None
    MP_VISION = None


def init_database() -> None:
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_letter TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()


def log_detection(predicted_letter: str, confidence: float) -> None:
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO detection_logs (predicted_letter, confidence, timestamp)
        VALUES (?, ?, ?)
        """,
        (predicted_letter, confidence, datetime.now().isoformat(timespec="seconds")),
    )
    connection.commit()
    connection.close()


class CameraService:
    def __init__(self) -> None:
        self.capture: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_prediction: Dict[str, object] = {
            "letter": "-",
            "confidence": 0.0,
            "message": "Waiting for detection...",
            "status": "idle",
        }
        self.running = False
        self.lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None
        self.pred_worker: Optional[threading.Thread] = None


    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True

            # Try multiple camera indices
            for cam_id in range(4):  # Try 0,1,2,3
                capture = cv2.VideoCapture(cam_id)
                if capture.isOpened():
                    print(f"Using camera {cam_id}")
                    self.capture = capture
                    self.running = True
                    self.worker = threading.Thread(target=self._capture_loop, daemon=True)
                    self.worker.start()
                    # Prediction worker is started by main after DetectionEngine is created.
                    return True
                capture.release()

            print("No camera found on indices 0-3")
            return False


    def _capture_loop(self) -> None:
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.01)

    def set_prediction(self, prediction: Dict[str, object]) -> None:
        with self.lock:
            self.latest_prediction = prediction

    def get_prediction(self) -> Dict[str, object]:
        with self.lock:
            # Return a shallow copy to avoid race conditions
            return dict(self.latest_prediction)


    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            if self.capture is not None:
                self.capture.release()
            self.capture = None
            self.latest_frame = None
            # Keep last_prediction for UI stability; optionally reset:
            # self.latest_prediction = {"letter":"-","confidence":0.0,"message":"Waiting for detection...","status":"idle"}



class DetectionEngine:
    def __init__(self, model_path: str) -> None:
        self.model = None
        self.model_path = model_path
        self.landmark_model = None
        self.hands = None
        self.hand_landmarker = None
        self.hand_model_missing = False
        print(f"  [DEBUG] DetectionEngine init: MP_HANDS={MP_HANDS is not None}")
        if MP_HANDS is not None:
            print("  [DEBUG] Using legacy mp.solutions.hands")
            self.hands = MP_HANDS.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        else:
            print("  [DEBUG] MP_HANDS is None, trying task-based landmarker...")
            self._init_task_hand_landmarker()
        self.last_prediction = {
            "letter": "-",
            "confidence": 0.0,
            "message": "Waiting for detection...",
            "status": "idle",
        }
        self.last_logged_at = 0.0
        self.load_model_if_exists()
        self.load_landmark_model_if_exists()

    def load_model_if_exists(self) -> None:
        if os.path.exists(self.model_path):
            keras_models = importlib.import_module("tensorflow.keras.models")
            self.model = keras_models.load_model(self.model_path)
            print("CNN model loaded successfully")

    def load_landmark_model_if_exists(self) -> None:
        if os.path.exists(LANDMARK_MODEL_PATH):
            keras_models = importlib.import_module("tensorflow.keras.models")
            self.landmark_model = keras_models.load_model(LANDMARK_MODEL_PATH)
            print("Landmark model loaded successfully")

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def _init_task_hand_landmarker(self) -> None:
        print(f"  [DEBUG] MP_TASKS={MP_TASKS is not None}, MP_VISION={MP_VISION is not None}")
        if MP_TASKS is None or MP_VISION is None:
            print("  [DEBUG] Task modules not available")
            return
        if not os.path.exists(HAND_TASK_MODEL_PATH):
            self.hand_model_missing = True
            print(f"  [DEBUG] Task model missing: {HAND_TASK_MODEL_PATH}")
            return
        print(f"  [DEBUG] Task model exists, creating HandLandmarker...")
        try:
            options = MP_VISION.HandLandmarkerOptions(
                base_options=MP_TASKS.BaseOptions(model_asset_path=HAND_TASK_MODEL_PATH),
                running_mode=MP_VISION.RunningMode.IMAGE,
                num_hands=1,
            )
            self.hand_landmarker = MP_VISION.HandLandmarker.create_from_options(options)
            print("  [DEBUG] HandLandmarker created successfully!")
        except Exception as e:
            print(f"  [DEBUG] HandLandmarker creation failed: {e}")
            self.hand_landmarker = None

    def _has_hand(self, frame_rgb: np.ndarray) -> bool:
        if self.hands is not None:
            hand_results = self.hands.process(frame_rgb)
            return bool(hand_results.multi_hand_landmarks)
        if self.hand_landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.hand_landmarker.detect(mp_image)
            return bool(result.hand_landmarks)
        return False

    def _get_first_hand_landmarks(self, frame_rgb: np.ndarray):
        if self.hands is not None:
            hand_results = self.hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                return hand_results.multi_hand_landmarks[0].landmark
            return None

        if self.hand_landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.hand_landmarker.detect(mp_image)
            if result.hand_landmarks:
                return result.hand_landmarks[0]
        return None

    @staticmethod
    def normalize_landmarks(landmarks):
        """Normalize landmarks exactly like generate_landmarks.py:
        wrist (0) as origin, scale by palm size (wrist->middle-finger MCP)."""
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        wrist = arr[0].copy()
        arr -= wrist
        palm_size = np.linalg.norm(arr[9])
        if palm_size > 1e-6:
            arr /= palm_size
        else:
            max_dist = np.max(np.abs(arr))
            if max_dist > 1e-6:
                arr /= max_dist
        return arr.flatten()

    def draw_hand_skeleton(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self._get_first_hand_landmarks(frame_rgb)
        if not landmarks:
            return frame_bgr

        output = frame_bgr.copy()
        height, width, _ = output.shape
        points = []
        for lm in landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))

        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(output, points[start], points[end], (0, 255, 0), 2)

        for x, y in points:
            cv2.circle(output, (x, y), 4, (0, 120, 255), -1)

        return output

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, object]:
        if self.hands is None and self.hand_landmarker is None:
            if self.hand_model_missing:
                return {
                    "letter": "-",
                    "confidence": 0.0,
                    "message": "Missing local model: model/hand_landmarker.task",
                    "status": "error",
                }
            return {
                "letter": "-",
                "confidence": 0.0,
                "message": "MediaPipe Hands is not available in this environment.",
                "status": "error",
            }

        if self.landmark_model is None and self.model is None:
            return {
                "letter": "-",
                "confidence": 0.0,
                "message": "No model found. Train first (python train_landmark_model.py).",
                "status": "error",
            }

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not self._has_hand(frame_rgb):
            return {
                "letter": "-",
                "confidence": 0.0,
                "message": "No hand detected",
                "status": "no_hand",
            }

        landmarks = self._get_first_hand_landmarks(frame_rgb)
        if landmarks is None:
            return {
                "letter": "-",
                "confidence": 0.0,
                "message": "No landmarks found",
                "status": "no_landmarks",
            }

        # --- Landmark-based prediction (preferred) ---
        if self.landmark_model is not None:
            normalized = self.normalize_landmarks(landmarks)
            input_vector = np.expand_dims(normalized, axis=0)
            probabilities = self.landmark_model.predict(input_vector, verbose=0)[0]
            class_index = int(np.argmax(probabilities))
            confidence = float(probabilities[class_index] * 100)
            predicted_letter = CLASS_NAMES[class_index]
            return {
                "letter": predicted_letter,
                "confidence": round(confidence, 2),
                "message": "Hand detected (landmark model)",
                "status": "ok",
            }

        # --- Fallback: CNN on cropped ROI ---
        height, width = frame_bgr.shape[:2]
        min_x, min_y, max_x, max_y = width, height, 0, 0
        for lm in landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        hand_size = max((max_x - min_x), (max_y - min_y))
        pad = 0.2 * hand_size
        min_x = max(0, int(min_x - pad))
        min_y = max(0, int(min_y - pad))
        max_x = min(width, int(max_x + pad))
        max_y = min(height, int(max_y + pad))

        if min_x >= max_x or min_y >= max_y:
            return {
                "letter": "-",
                "confidence": 0.0,
                "message": "Invalid hand ROI",
                "status": "invalid_roi",
            }

        roi_frame = frame_bgr[min_y:max_y, min_x:max_x]
        processed = self.preprocess(roi_frame)
        probabilities = self.model.predict(processed, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index] * 100)
        predicted_letter = CLASS_NAMES[class_index]

        return {
            "letter": predicted_letter,
            "confidence": round(confidence, 2),
            "message": "Hand detected (CNN fallback)",
            "status": "ok",
        }


camera_service = CameraService()
detection_engine = DetectionEngine(MODEL_PATH)
init_database()

# Start prediction thread to avoid doing heavy inference inside /predict requests.
# This keeps UI smooth and reduces lag.

def prediction_loop() -> None:
    # Run at a capped rate (adjustable) to balance CPU usage.
    prediction_interval_sec = 0.15  # ~6.6 FPS predictions
    while True:
        if not camera_service.running:
            time.sleep(0.1)
            continue
        frame = camera_service.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        result = detection_engine.detect(frame)
        camera_service.set_prediction(result)
        time.sleep(prediction_interval_sec)

camera_service.pred_worker = threading.Thread(target=prediction_loop, daemon=True)
camera_service.pred_worker.start()

print(" * CNN model:", "OK" if os.path.exists(MODEL_PATH) else "MISSING - run python train_model.py")
print(" * Landmark model:", "OK" if os.path.exists(LANDMARK_MODEL_PATH) else "MISSING - run python train_landmark_model.py")
print(" * Running on http://0.0.0.0:5000 (press CTRL+C to quit)")



def frame_generator():
    camera_ready = camera_service.start()
    if not camera_ready:
        while True:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "No webcam found. Check connection.",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            success, encoded = cv2.imencode(".jpg", blank)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )
            time.sleep(0.03)
            continue

    while True:
        frame = camera_service.get_frame()
        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Camera warming up...",
                (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            success, encoded = cv2.imencode(".jpg", blank)
        else:
            frame_with_skeleton = detection_engine.draw_hand_skeleton(frame)  # Re-enable skeleton
            success, encoded = cv2.imencode(".jpg", frame_with_skeleton)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        )
        time.sleep(0.03)


def get_learning_item() -> dict:
    return {"letter": "Reference", "image_url": "/static/images/learning/asl.jpg"}


def get_letter_reference_path(letter: str) -> Optional[str]:
    """Look for a reference image in static/images/learning first, then fall back to dataset."""
    letter = letter.upper()
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(LEARNING_IMAGE_DIR, f"{letter}{ext}")
        if os.path.isfile(candidate):
            return candidate
    folder = os.path.join(DATASET_DIR, letter)
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return None
    files.sort()
    return os.path.join(folder, files[0])


@app.route("/reference_image/<letter>")
def reference_image(letter: str):
    from flask import send_file
    path = get_letter_reference_path(letter.upper())
    if path is None:
        return "Not found", 404
    ext = os.path.splitext(path)[1].lower()
    mimetype = "image/png" if ext == ".png" else "image/jpeg"
    return send_file(path, mimetype=mimetype)


@app.route("/health")
def health_check():
    cnn_ok = detection_engine.model is not None
    landmark_ok = detection_engine.landmark_model is not None
    hands_ok = detection_engine.hands is not None or detection_engine.hand_landmarker is not None
    camera_ok = camera_service.start()
    return jsonify({
        "status": "ok",
        "cnn_model_loaded": cnn_ok,
        "landmark_model_loaded": landmark_ok,
        "hands_detector": hands_ok,
        "camera_available": camera_ok,
        "message": "Server ready" if (cnn_ok or landmark_ok) and hands_ok and camera_ok else "Partial ready"
    })


@app.route("/")
def loading_page():
    return render_template("loading.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/learning")
def learning_page():
    letters = []
    for letter in CLASS_NAMES:
        path = get_letter_reference_path(letter)
        if path and path.startswith(LEARNING_IMAGE_DIR):
            # Serve from static folder
            filename = os.path.basename(path)
            image_url = url_for("static", filename=f"images/learning/{filename}")
        elif path:
            image_url = f"/reference_image/{letter}"
        else:
            image_url = ""
        letters.append({
            "letter": letter,
            "image_url": image_url,
        })
    return render_template("learning.html", letters=letters)


@app.route("/detection")
def detection_page():
    webcam_ready = camera_service.start()
    print("Detection page - Webcam:", "OK" if webcam_ready else "FAILED")
    return render_template("detection.html", webcam_ready=webcam_ready)


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predict")
def predict():
    webcam_ready = camera_service.start()
    if not webcam_ready:
        return jsonify({
            "letter": "-",
            "confidence": 0.0,
            "message": "Webcam not available. Check connection.",
            "status": "error",
        })

    # Return cached prediction computed by background thread.
    result = camera_service.get_prediction()

    # Log at most once per second when prediction is confident.
    current_ts = time.time()
    if result.get("status") == "ok" and current_ts - detection_engine.last_logged_at >= 1.0:
        log_detection(str(result.get("letter", "-")), float(result.get("confidence", 0.0)))
        detection_engine.last_logged_at = current_ts

    detection_engine.last_prediction = result
    return jsonify(result)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
