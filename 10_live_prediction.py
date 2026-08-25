# ============================================
# STEP 21: LIVE FALL DETECTION + ALARM
# ============================================

import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
import subprocess

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

print("Expected features:", len(feature_names))


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
# ALARM SOUND
# ============================================

alarm_process = None

def play_alarm():

    global alarm_process

    # Don't start another alarm
    # while the current one is playing

    if alarm_process is not None:

        if alarm_process.poll() is None:
            return

    alarm_process = subprocess.Popen(
        [
            "afplay",
            "sounds/adhi_adhi_raat_meme.mp3"
        ]
    )

# ============================================
# POSE VISIBILITY CHECK
# ============================================

def is_full_body_visible(landmarks):

    # Important body landmarks
    required_landmarks = [
        0,          # Nose
        11, 12,     # Shoulders
        23, 24,     # Hips
        25, 26,     # Knees
        27, 28      # Ankles
    ]

    visible_count = 0

    for index in required_landmarks:

        if landmarks[index].visibility >= 0.6:
            visible_count += 1

    # Require at least 8 of the 9 important landmarks
    return visible_count >= 8

# ============================================
# FALL POSTURE CHECK
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
# CAMERA
# ============================================

cap = cv2.VideoCapture(0)

timestamp_ms = 0

# ============================================
# FALL DETECTION SETTINGS
# ============================================

FALL_THRESHOLD = 0.5
REQUIRED_FRAMES = 12
POSSIBLE_FALL_FRAMES = 4

fall_count = 0

alarm_active = False
alarm_start_time = 0


print("===================================")
print("LIVE FALL DETECTION")
print("===================================")
print("Press Q to quit")


# ============================================
# MAIN LOOP
# ============================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms += 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    label = "NO PERSON"
    probability = 0.0

    # ========================================
    # PERSON DETECTED
    # ========================================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # ====================================
        # CHECK BODY VISIBILITY
        # ====================================

        if is_full_body_visible(landmarks):

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

            # ====================================
            # FALL CONFIRMATION
            # ====================================

            if probability >= FALL_THRESHOLD and is_fall_posture(landmarks):

                fall_count += 1

            else:

                fall_count = 0

            # ====================================
            # FALL STATUS
            # ====================================

            if fall_count >= REQUIRED_FRAMES:

                label = "FALL DETECTED!"

                if not alarm_active:

                    alarm_active = True
                    alarm_start_time = time.time()

                    print("🚨 FALL DETECTED!")

                    play_alarm()

            elif fall_count >= POSSIBLE_FALL_FRAMES:

                label = "POSSIBLE FALL"

            else:

                label = "NORMAL"

        else:

            # Body not sufficiently visible
            label = "BODY NOT FULLY VISIBLE"

            probability = 0.0
            fall_count = 0
    

        
        # ====================================
        # DRAW LANDMARKS
        # ====================================

        h, w, _ = frame.shape

        for landmark in landmarks:

            cx = int(
                landmark.x * w
            )

            cy = int(
                landmark.y * h
            )

            cv2.circle(
                frame,
                (cx, cy),
                3,
                (0, 255, 0),
                -1
            )

    # ========================================
    # DISPLAY
    # ========================================

    cv2.putText(
        frame,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 255)
        if "FALL" in label
        else (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Fall Probability: {probability:.2f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if alarm_active:

        cv2.putText(
            frame,
            "🚨 ALARM ACTIVE 🚨",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Fall Detection System",
        frame
    )

    # ========================================
    # QUIT
    # ========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()