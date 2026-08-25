import numpy as np
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

base_options = python.BaseOptions(
    model_asset_path="models/pose_landmarker_lite.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(
    options
)


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

    required_landmarks = [
        0,
        11, 12,
        23, 24,
        25, 26,
        27, 28
    ]

    visible_count = 0

    for index in required_landmarks:

        if landmarks[index].visibility >= 0.6:
            visible_count += 1

    return visible_count >= 8


# ============================================
# FALL POSTURE
# ============================================

def is_fall_posture(landmarks):

    x = np.array([p.x for p in landmarks])
    y = np.array([p.y for p in landmarks])

    body_angle = np.degrees(
        np.arctan2(
            y[24] - y[0],
            x[24] - x[0]
        )
    )

    horizontal_angle = min(
        abs(body_angle),
        abs(abs(body_angle) - 180)
    )

    return horizontal_angle <= 45


# ============================================
# PREDICTION
# ============================================

def predict_fall(landmarks):

    if not is_full_body_visible(landmarks):

        return "BODY NOT FULLY VISIBLE", 0.0

    features = create_features(
        landmarks
    )

    features = imputer.transform(
        features
    )

    features = scaler.transform(
        features
    )

    probability = model.predict_proba(
        features
    )[0][1]

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
        self.fall_count = 0
        self.fall_detected = False
        self.fall_detected_until=0

        self.FALL_THRESHOLD = 0.5
        self.REQUIRED_FRAMES = 12
        self.POSSIBLE_FALL_FRAMES = 4


    def recv(self, frame):
        global FALL_DETECTED

        image = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        label = "NO PERSON"
        probability = 0.0

        # ====================================
        # PERSON DETECTED
        # ====================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            if is_full_body_visible(landmarks):

                # Create exactly the same 138 features
                features = create_features(landmarks)

                features = imputer.transform(features)
                features = scaler.transform(features)

                probability = model.predict_proba(features)[0][1]



                # Check posture separately
                posture_ok = is_fall_posture(landmarks)

                # Same logic as your Mac detector
                if (
                    probability >= self.FALL_THRESHOLD
                    and posture_ok
                ):

                    self.fall_count += 1

                else:

                    self.fall_count = 0

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
                    f"Fall Frames: {self.fall_count}/12",
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Same confirmation rules
                if self.fall_count >= self.REQUIRED_FRAMES:

                    label = "FALL DETECTED!"
                    self.fall_detected = True
                    
                    FALL_DETECTED = True
                    self.fall_detected_until = time.time() + 5  # Reset after 5 seconds

                elif self.fall_count >= self.POSSIBLE_FALL_FRAMES:

                    label = "POSSIBLE FALL"
                    self.fall_detected = False
                    FALL_DETECTED = False

                else:

                    label = "NORMAL"
                    self.fall_detected = False
                    FALL_DETECTED = False
                    # ====================================
                    # KEEP FALL DETECTED ACTIVE
                    # ====================================

                    if time.time() < self.fall_detected_until:

                        label = "FALL DETECTED!"
                        self.fall_detected = True
                        FALL_DETECTED = True

                    else:

                        self.fall_detected = False
                        FALL_DETECTED = False

                    # ====================================
                    # DRAW POSE LANDMARKS
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
                                (0, 255, 0),
                                -1
                            )

            else:

                label = "BODY NOT FULLY VISIBLE"
                self.fall_detected = False
                FALL_DETECTED = False
                probability = 0.0
                self.fall_count = 0

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