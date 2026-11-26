# ============================================
# STEP 21: LIVE FALL DETECTION + ALARM
# ============================================

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import time
import os
import platform
import subprocess
import argparse
import signal
import sys

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
# CROSS-PLATFORM ALARM SOUND ENGINE
# ============================================

class DesktopAudioAlert:
    """
    Cross-platform desktop audio player supporting macOS, Windows, and Linux.
    Primary: sounddevice + PyAV (PortAudio PCM in-memory stream).
    Fallback: OS-native CLI player (afplay on macOS, paplay/aplay on Linux, powershell on Windows).
    Asset fallback: Synthetic modulated siren if audio file is missing.
    """
    def __init__(self, audio_path="sounds/adhi_adhi_raat_meme.mp3"):
        self.audio_path = os.path.abspath(audio_path)
        self.is_playing = False
        self.audio_data = None
        self.sample_rate = 44100
        self.backend = "none"
        self.fallback_process = None
        self._load_audio()

    def _load_audio(self):
        try:
            import av
            import sounddevice as sd
            if os.path.exists(self.audio_path):
                container = av.open(self.audio_path)
                stream = container.streams.audio[0]
                self.sample_rate = stream.rate
                frames = []
                for frame in container.decode(audio=0):
                    frames.append(frame.to_ndarray())
                if frames:
                    raw_data = np.concatenate(frames, axis=1)
                    if raw_data.dtype != np.float32:
                        raw_data = raw_data.astype(np.float32) / 32768.0
                    self.audio_data = raw_data.T
                    self.backend = "sounddevice"
                    return
            else:
                self._generate_synthetic_tone()
                return
        except Exception:
            pass

        if os.path.exists(self.audio_path):
            self.backend = "os_native"
        else:
            self.backend = "fallback_silent"

    def _generate_synthetic_tone(self):
        t = np.linspace(0, 2.0, 44100 * 2, False, dtype=np.float32)
        mod = 0.5 * np.sin(2 * np.pi * 4 * t)
        tone = 0.4 * np.sin(2 * np.pi * (880 + 200 * mod) * t)
        self.audio_data = np.column_stack([tone, tone])
        self.sample_rate = 44100
        self.backend = "sounddevice_synth"

    def play(self):
        if self.is_playing:
            return  # Prevent duplicate / overlapping playback

        self.is_playing = True

        if self.backend in ["sounddevice", "sounddevice_synth"] and self.audio_data is not None:
            try:
                import sounddevice as sd
                sd.play(self.audio_data, self.sample_rate, loop=True)
                return
            except Exception:
                pass

        # OS Native Fallback
        current_os = platform.system()
        try:
            if current_os == "Darwin":
                self.fallback_process = subprocess.Popen(["afplay", self.audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif current_os == "Linux":
                for player in ["paplay", "aplay", "mpg123", "ffplay"]:
                    try:
                        self.fallback_process = subprocess.Popen([player, self.audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        break
                    except FileNotFoundError:
                        continue
            elif current_os == "Windows":
                cmd = f"(New-Object Media.SoundPlayer '{self.audio_path}').PlayLooping()"
                self.fallback_process = subprocess.Popen(["powershell", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def stop(self):
        self.is_playing = False
        if self.backend in ["sounddevice", "sounddevice_synth"]:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass

        if self.fallback_process is not None:
            try:
                self.fallback_process.terminate()
                self.fallback_process.wait(timeout=0.3)
            except Exception:
                try:
                    self.fallback_process.kill()
                except Exception:
                    pass
            self.fallback_process = None

alarm_player = DesktopAudioAlert("sounds/adhi_adhi_raat_meme.mp3")

# ============================================
# POSE VISIBILITY CHECK
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
# FALL POSTURE CHECK
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
# CLI ARGUMENTS & SIGNAL HANDLING
# ============================================

parser = argparse.ArgumentParser(description="Live Human Fall Detection System")
parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (0, 1, 2, ...)")
parser.add_argument("--width", type=int, default=1280, help="Target camera frame width")
parser.add_argument("--height", type=int, default=720, help="Target camera frame height")

running = True

def handle_exit_signal(signum, frame):
    global running
    print(f"\n🛑 Termination signal ({signum}) received. Shutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)


# ============================================
# CAMERA INITIALIZATION & BACKEND SELECTION
# ============================================

def open_camera(camera_index=0, target_width=1280, target_height=720):
    current_os = platform.system()
    backends_to_try = []

    if current_os == "Darwin":
        backends_to_try = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif current_os == "Windows":
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif current_os == "Linux":
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends_to_try = [cv2.CAP_ANY]

    cap = None
    used_backend = "UNKNOWN"

    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap is not None and cap.isOpened():
                used_backend = backend
                break
        except Exception:
            continue

    if cap is None or not cap.isOpened():
        if camera_index != 0:
            print(f"! Camera index {camera_index} failed to open. Retrying with default index 0...")
            for backend in backends_to_try:
                try:
                    cap = cv2.VideoCapture(0, backend)
                    if cap is not None and cap.isOpened():
                        used_backend = backend
                        camera_index = 0
                        break
                except Exception:
                    continue

    if cap is None or not cap.isOpened():
        print(f"❌ ERROR: Failed to open camera (Index: {camera_index}, OS: {current_os}).")
        print("Please verify camera permissions and check if another application is using the camera.")
        return None, 0, 0

    # Resolution negotiation
    if target_width > 0 and target_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera initialized: Index {camera_index}, Backend {used_backend}, Native resolution {actual_w}x{actual_h}")
    return cap, actual_w, actual_h


def main():
    args, _ = parser.parse_known_args()
    cap, cam_w, cam_h = open_camera(args.camera_index, args.width, args.height)
    if cap is None:
        sys.exit(1)

    timestamp_ms = 0

    # ============================================
    # FALL DETECTION SETTINGS
    # ============================================

    FALL_THRESHOLD = 0.5
    REQUIRED_FRAMES = 12
    POSSIBLE_FALL_FRAMES = 4

    fall_count = 0

    # Alarm state machine: "normal", "active", "silenced"
    alarm_state = "normal"
    last_fall_time = 0.0
    recovery_duration = 3.0  # seconds in normal posture required to rearm


    print("===================================")
    print("LIVE FALL DETECTION")
    print("===================================")
    print("Press Q to quit, S to silence alarm")


    # ============================================
    # MAIN LOOP
    # ============================================

    while running and cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("! Frame capture failed. Exiting main loop...")
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, _ = frame.shape
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

                features_df = pd.DataFrame(
                    features,
                    columns=feature_names
                )

                features_imp = imputer.transform(
                    features_df
                )

                features_scl = scaler.transform(
                    features_imp
                )

                probability = float(
                    model.predict_proba(
                        features_scl
                    )[0][1]
                )

                # ====================================
                # FALL CONFIRMATION
                # ====================================

                if probability >= FALL_THRESHOLD and is_fall_posture(landmarks):

                    fall_count += 1

                else:

                    fall_count = max(0, fall_count - 2)

                # ====================================
                # FALL STATUS & ALARM STATE MACHINE
                # ====================================
                now = time.time()

                if fall_count >= REQUIRED_FRAMES:
                    last_fall_time = now

                    if alarm_state in ["normal", "recovery"]:
                        alarm_state = "active"
                        print("🚨 FALL DETECTED! Alarm sounding (Press 'S' to silence)...")
                        alarm_player.play()

                    if alarm_state == "active":
                        label = "FALL DETECTED!"
                    else:
                        label = "FALL DETECTED (MUTED)"

                elif fall_count >= POSSIBLE_FALL_FRAMES:
                    if alarm_state == "active":
                        label = "FALL DETECTED!"
                    elif alarm_state == "silenced":
                        label = "FALL DETECTED (MUTED)"
                    else:
                        label = "POSSIBLE FALL"

                else:
                    # Normal posture: check for recovery & re-arming
                    if alarm_state in ["active", "silenced"]:
                        if now - last_fall_time >= recovery_duration:
                            if alarm_state == "active":
                                alarm_player.stop()
                            alarm_state = "normal"
                            print("🟢 Subject recovered. System REARMED for subsequent falls.")
                            label = "NORMAL (REARMED)"
                        else:
                            if alarm_state == "active":
                                label = "FALL DETECTED!"
                            else:
                                label = "FALL DETECTED (MUTED)"
                    else:
                        label = "NORMAL"

            else:
                # Body not sufficiently visible
                now = time.time()
                fall_count = max(0, fall_count - 1)

                if alarm_state in ["active", "silenced"]:
                    if now - last_fall_time >= recovery_duration:
                        if alarm_state == "active":
                            alarm_player.stop()
                        alarm_state = "normal"
                        label = "BODY NOT FULLY VISIBLE"
                        probability = 0.0
                    else:
                        if alarm_state == "active":
                            label = "FALL DETECTED!"
                        else:
                            label = "FALL DETECTED (MUTED)"
                else:
                    label = "BODY NOT FULLY VISIBLE"
                    probability = 0.0



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
                    (0, 0, 255) if ("FALL" in label) else (0, 255, 0),
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

        if alarm_state == "active":
            cv2.putText(
                frame,
                "🚨 ALARM ACTIVE (Press 'S' to Silence) 🚨",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        elif alarm_state == "silenced":
            cv2.putText(
                frame,
                "🔕 ALARM SILENCED (Waiting for recovery)",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "Fall Detection System",
            frame
        )

        # ========================================
        # KEYBOARD CONTROLS (Quit: Q, Silence: S)
        # ========================================

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            if alarm_state == "active":
                alarm_state = "silenced"
                alarm_player.stop()
                print("🔕 Alarm manually silenced by user.")


    # ============================================
    # CLEANUP
    # ============================================

    alarm_player.stop()
    cap.release()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)
    try:
        landmarker.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
