import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FALL_DETECTED = False


# ============================================
# LOAD MODEL
# ============================================

saved = joblib.load(
    "models/fall_detection_model.pkl"
)

model = saved["model"]
imputer = saved["imputer"]
scaler = saved["scaler"]

feature_names = saved["feature_names"]


# ============================================
# MEDIAPIPE
# ============================================

def create_landmarker():
    base_options = python.BaseOptions(
        model_asset_path="models/pose_landmarker_lite.task"
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)

_global_landmarker = None

def get_landmarker():
    global _global_landmarker
    if _global_landmarker is None:
        _global_landmarker = create_landmarker()
    return _global_landmarker


# ============================================
# FEATURE CREATION
# ============================================

def create_features(landmarks):

    x = np.array([p.x for p in landmarks])
    y = np.array([p.y for p in landmarks])
    z = np.array([p.z for p in landmarks])

    visibility = np.array(
        [p.visibility for p in landmarks]
    )

    features = np.column_stack([
        x,
        y,
        z,
        visibility
    ]).flatten().tolist()

    body_height = np.sqrt(
        (x[0] - x[24]) ** 2 +
        (y[0] - y[24]) ** 2
    )

    shoulder_width = np.sqrt(
        (x[11] - x[12]) ** 2 +
        (y[11] - y[12]) ** 2
    )

    hip_width = np.sqrt(
        (x[23] - x[24]) ** 2 +
        (y[23] - y[24]) ** 2
    )

    head_hip_distance = np.sqrt(
        (x[0] - x[24]) ** 2 +
        (y[0] - y[24]) ** 2
    )

    body_angle = np.degrees(
        np.arctan2(
            y[24] - y[0],
            x[24] - x[0]
        )
    )

    body_aspect_ratio = (
        body_height /
        (shoulder_width + 1e-6)
    )

    features.extend([
        body_height,
        shoulder_width,
        hip_width,
        head_hip_distance,
        body_angle,
        body_aspect_ratio
    ])

    return np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# ============================================
# BODY VISIBILITY
# ============================================

def is_full_body_visible(landmarks):
    """
    Fall-aware visibility check: anchors on core torso landmarks (nose, shoulders, hips)
    while tolerating occluded knees or ankles during floor falls.
    """
    core_landmarks = [
        0,          # Nose
        11, 12,     # Shoulders
        23, 24      # Hips
    ]
    required_landmarks = [
        0,          # Nose
        11, 12,     # Shoulders
        23, 24,     # Hips
        25, 26,     # Knees
        27, 28      # Ankles
    ]

    core_visible = sum(1 for index in core_landmarks if landmarks[index].visibility >= 0.50)
    total_visible = sum(1 for index in required_landmarks if landmarks[index].visibility >= 0.50)

    return (core_visible >= 4) and (total_visible >= 6)


# ============================================
# FALL POSTURE
# ============================================

def is_fall_posture(landmarks):
    """
    Multi-directional posture validation: uses the biomechanical spine vector
    (mid-shoulder to mid-hip) with fallback to nose-to-hip, plus a strict
    floor-collapse foreshortening check for forward/backward falls.
    """
    x = np.array([p.x for p in landmarks])
    y = np.array([p.y for p in landmarks])

    sh_x = (x[11] + x[12]) / 2.0
    sh_y = (y[11] + y[12]) / 2.0
    hip_x = (x[23] + x[24]) / 2.0
    hip_y = (y[23] + y[24]) / 2.0

    dx = hip_x - sh_x
    dy = hip_y - sh_y
    torso_angle = np.degrees(np.arctan2(dy, dx))
    horizontal_angle = min(abs(torso_angle), abs(abs(torso_angle) - 180))

    # Fallback nose-to-right-hip angle
    orig_angle = np.degrees(np.arctan2(y[24] - y[0], x[24] - x[0]))
    orig_horizontal_angle = min(abs(orig_angle), abs(abs(orig_angle) - 180))

    # Strict forward/backward floor-collapse check: shoulders, hips, and head all at floor elevation
    strict_fw_bw = (min(sh_y, hip_y) > 0.65) and (abs(dy) < 0.12) and (y[0] > 0.60)

    return (horizontal_angle <= 55) or (orig_horizontal_angle <= 55) or strict_fw_bw


# ============================================
# PREDICTION
# ============================================

def predict_fall(landmarks):

    if not is_full_body_visible(landmarks):
        return "BODY NOT FULLY VISIBLE", 0.0

    features = create_features(landmarks)
    features_df = pd.DataFrame(features, columns=feature_names)
    features_imp = imputer.transform(features_df)
    features_scl = scaler.transform(features_imp)

    probability = float(model.predict_proba(features_scl)[0][1])

    if (
        probability >= 0.5
        and is_fall_posture(landmarks)
    ):
        label = "FALL"
    else:
        label = "NORMAL"

    return label, probability


# ============================================
# WEBRTC FRAME PROCESSOR
# ============================================

import cv2
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase


class FallDetectionProcessor(VideoProcessorBase):

    def __init__(self):
        self.timestamp_ms = 0
        self.last_timestamp_ms = -1
        self.fall_count = 0
        self.fall_detected = False

        # Alarm state machine: "normal", "active", "silenced", "recovery"
        self.alarm_state = "normal"
        self.last_fall_time = 0.0
        self.recovery_duration = 3.0  # seconds in normal posture required to rearm

        self.FALL_THRESHOLD = 0.5
        self.REQUIRED_FRAMES = 12
        self.POSSIBLE_FALL_FRAMES = 4

        # Per-instance landmarker prevents multi-session and restart timestamp collisions
        self.landmarker = create_landmarker()

    def silence_alarm(self):
        """Immediately mutes the active alarm."""
        self.alarm_state = "silenced"

    def reset_alarm(self):
        """Manual reset to normal armed state."""
        self.alarm_state = "normal"
        self.fall_detected = False
        self.fall_count = 0

    def close(self):
        if hasattr(self, "landmarker") and self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def __del__(self):
        self.close()

    def recv(self, frame):
        global FALL_DETECTED

        image = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w, _ = image.shape

        # Adaptive inference scaling:
        # If camera resolution is high (> 1280px width), resize a separate copy for MediaPipe inference.
        # Original full-resolution image is preserved for display.
        if w > 1280:
            proc_w = 1280
            proc_h = int(h * (1280.0 / w))
            proc_rgb = cv2.resize(rgb, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        else:
            proc_rgb = rgb

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=proc_rgb
        )

        # Monotonically increasing timestamp based on frame arrival time
        current_time_ms = int(time.time() * 1000)
        if current_time_ms <= self.last_timestamp_ms:
            current_time_ms = self.last_timestamp_ms + 33
        self.last_timestamp_ms = current_time_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            current_time_ms
        )

        label = "NO PERSON"
        probability = 0.0

        # ====================================
        # PERSON DETECTED
        # ====================================

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            if is_full_body_visible(landmarks):
                # Exact 138-feature calculation
                features = create_features(landmarks)
                features_df = pd.DataFrame(features, columns=feature_names)
                features_imp = imputer.transform(features_df)
                features_scl = scaler.transform(features_imp)

                probability = float(model.predict_proba(features_scl)[0][1])
                posture_ok = is_fall_posture(landmarks)

                # Anti-jitter confirmation: decay gently instead of instant zeroing
                if probability >= self.FALL_THRESHOLD and posture_ok:
                    self.fall_count += 1
                else:
                    self.fall_count = max(0, self.fall_count - 2)

                # ====================================
                # FALL STATUS & ALARM STATE MACHINE
                # ====================================
                now = time.time()

                if self.fall_count >= self.REQUIRED_FRAMES:
                    self.last_fall_time = now
                    self.fall_detected = True
                    FALL_DETECTED = True

                    if self.alarm_state != "silenced":
                        self.alarm_state = "active"
                        label = "FALL DETECTED!"
                    else:
                        label = "FALL DETECTED (MUTED)"

                elif self.fall_count >= self.POSSIBLE_FALL_FRAMES:
                    if self.alarm_state == "active":
                        label = "FALL DETECTED!"
                        self.fall_detected = True
                        FALL_DETECTED = True
                    elif self.alarm_state == "silenced":
                        label = "FALL DETECTED (MUTED)"
                        self.fall_detected = True
                    else:
                        label = "POSSIBLE FALL"
                        self.fall_detected = False
                        FALL_DETECTED = False

                else:
                    # Normal posture: check for recovery & re-arming
                    if self.alarm_state in ["active", "silenced"]:
                        if now - self.last_fall_time >= self.recovery_duration:
                            self.alarm_state = "normal"
                            self.fall_detected = False
                            FALL_DETECTED = False
                            label = "NORMAL (REARMED)"
                        else:
                            if self.alarm_state == "active":
                                label = "FALL DETECTED!"
                                self.fall_detected = True
                                FALL_DETECTED = True
                            else:
                                label = "FALL DETECTED (MUTED)"
                                self.fall_detected = True
                    else:
                        label = "NORMAL"
                        self.fall_detected = False
                        FALL_DETECTED = False

                cv2.putText(
                    image,
                    f"Fall Probability: {probability:.2f}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    f"Fall Frames: {self.fall_count}/{self.REQUIRED_FRAMES}",
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:
                # Body not sufficiently visible
                now = time.time()
                self.fall_count = max(0, self.fall_count - 1)

                if self.alarm_state in ["active", "silenced"]:
                    if now - self.last_fall_time >= self.recovery_duration:
                        self.alarm_state = "normal"
                        self.fall_detected = False
                        FALL_DETECTED = False
                        label = "BODY NOT FULLY VISIBLE"
                        probability = 0.0
                    else:
                        if self.alarm_state == "active":
                            label = "FALL DETECTED!"
                            self.fall_detected = True
                            FALL_DETECTED = True
                        else:
                            label = "FALL DETECTED (MUTED)"
                            self.fall_detected = True
                else:
                    label = "BODY NOT FULLY VISIBLE"
                    self.fall_detected = False
                    FALL_DETECTED = False
                    probability = 0.0

            # ====================================
            # DRAW POSE LANDMARKS (Drawn for all detected persons)
            # ====================================

            h, w, _ = image.shape
            for landmark in landmarks:
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(
                        image,
                        (cx, cy),
                        4,
                        (0, 0, 255) if ("FALL" in label) else (0, 255, 0),
                        -1
                    )

        # ====================================
        # DISPLAY
        # ====================================

        cv2.putText(
            image,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255)
            if "FALL" in label
            else (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            f"Fall Probability: {probability:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )