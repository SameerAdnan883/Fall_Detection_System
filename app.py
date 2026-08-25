import streamlit as st
import joblib
import subprocess
import platform

import base64
from streamlit_webrtc import webrtc_streamer

import web_detector


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 20px;
}

.status {
    font-size: 24px;
    font-weight: 600;
}

.metric-title {
    font-size: 15px;
    opacity: 0.7;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
}

.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)





# ============================================
# LOAD MODEL
# ============================================

saved = joblib.load(
    "models/fall_detection_model.pkl"
)

model = saved["model"]
feature_names = saved["feature_names"]


# ============================================
# HEADER
# ============================================

st.markdown(
    '<div class="main-title">🚨 Fall Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered real-time human fall detection '
    'using MediaPipe Pose and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================
# STATUS
# ============================================

st.markdown(
    '<div class="card">'
    '<div class="status">🟢 System Ready</div>'
    '<p>Fall detection model is loaded and ready.</p>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================
# METRICS
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        '<div class="card">'
        '<div class="metric-title">MODEL</div>'
        '<div class="metric-value">Random Forest</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        '<div class="card">'
        '<div class="metric-title">FEATURES</div>'
        f'<div class="metric-value">{len(feature_names)}</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        '<div class="card">'
        '<div class="metric-title">POSE MODEL</div>'
        '<div class="metric-value">MediaPipe</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        '<div class="card">'
        '<div class="metric-title">ALARM</div>'
        '<div class="metric-value">🔊 Enabled</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================
# MODEL PERFORMANCE
# ============================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Accuracy", "84.18%")

with col2:
    st.metric("Precision", "73.41%")

with col3:
    st.metric("Recall", "56.81%")

with col4:
    st.metric("F1 Score", "64.05%")

with col5:
    st.metric("ROC-AUC", "91.03%")

# ============================================
# ML PIPELINE
# ============================================

st.subheader("🧠 Detection Pipeline")

cols = st.columns(6)

steps = [
    ("📷", "Camera"),
    ("🧍", "Pose Detection"),
    ("📐", "Feature Engineering"),
    ("⚙️", "Preprocessing"),
    ("🌲", "Random Forest"),
    ("🚨", "Fall Alarm")
]

for col, (icon, name) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding:15px;
                border:1px solid #333;
                border-radius:10px;
            ">
                <div style="font-size:30px">{icon}</div>
                <b>{name}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================
# LIVE DETECTION CONTROL
# ============================================

st.subheader("🎥 Live Detection")

st.markdown("## 🎥 Live Fall Detection")

webrtc_streamer(
    key="fall_detection",
    video_processor_factory=web_detector.FallDetectionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

col1, col2 = st.columns(2)

with col1:
    start = st.button(
        "▶️ Start Fall Detection",
        use_container_width=True
    )

with col2:
    stop = st.button(
        "⏹️ Stop Fall Detection",
        use_container_width=True
    )



if start:

    if platform.system() == "Darwin":

        subprocess.Popen(
            ["python", "10_live_prediction.py"]
        )

        st.success(
            "🟢 Live fall detection started!"
        )

    else:

        st.info(
            "🌐 Live camera detection is available "
            "when running locally."
        )


if stop:

    if platform.system() == "Darwin":

        subprocess.run(
            ["pkill", "-f", "10_live_prediction.py"]
        )

        st.warning(
            "🛑 Live fall detection stopped."
        )

    else:

        st.info(
            "🌐 Cloud mode does not control a local camera process."
        )

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 90% 5%,
            rgba(255, 40, 40, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(30, 100, 200, 0.10),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #05070c,
            #0b111c,
            #05070c
        );
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 15px;
    transition: all 0.25s ease;
}

/* Hover effect */
div[data-testid="stMetric"]:hover {
    border-color: rgba(255,60,60,0.45);
    box-shadow: 0 0 22px rgba(255,40,40,0.12);
    transform: translateY(-2px);
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.15);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #ff3b3b;
    color: #ff5555;
    box-shadow: 0 0 18px rgba(255,50,50,0.20);
}

/* Headings */
h1 {
    letter-spacing: -1px;
}

h2, h3 {
    letter-spacing: -0.5px;
}

/* Info boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)




st.markdown("""
<style>
.camera-container {
    max-width: 800px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)







# ============================================
# HOW IT WORKS
# ============================================

st.subheader("🧠 How the System Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🦾")
    st.write("**Pose Detection**")
    st.caption("MediaPipe extracts body landmarks.")

with col2:
    st.markdown("### 🧮")
    st.write("**Feature Engineering**")
    st.caption("Body measurements and angles are calculated.")

with col3:
    st.markdown("### 🤖")
    st.write("**ML Prediction**")
    st.caption("Random Forest predicts fall probability.")

with col4:
    st.markdown("### 🚨")
    st.write("**Fall Confirmation**")
    st.caption("Posture + visibility + consecutive frames confirm the fall.")


# ============================================
# DETECTION SETTINGS
# ============================================

st.subheader("⚙️ Detection Settings")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Fall Threshold",
        "0.50"
    )

with col2:
    st.metric(
        "Required Frames",
        "12"
    )

with col3:
    st.metric(
        "Visibility Requirement",
        "8 / 9"
    )


# ============================================
# FOOTER
# ============================================

st.markdown(
    '<div class="footer">'
    'Fall Detection System • MediaPipe + Random Forest • '
    'AI Safety Project'
    '</div>',
    unsafe_allow_html=True
)