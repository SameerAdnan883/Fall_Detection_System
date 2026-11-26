import streamlit as st
import joblib
import platform
import subprocess
import sys
import os
import base64
import psutil
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer

import web_detector

def get_desktop_detector_pid():
    """Returns the PID of a running 10_live_prediction.py process, or None."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline') or []
            if any("10_live_prediction.py" in arg for arg in cmd):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def stop_desktop_detector():
    """Gracefully terminates 10_live_prediction.py and all child audio/subprocesses."""
    pid = get_desktop_detector_pid()
    if pid:
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs([parent], timeout=2.0)
            for p in alive:
                p.kill()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


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

<!-- Audio Interaction Unlocker (Bypasses Browser Autoplay Restrictions on Click/Touch) -->
<script>
(function() {
    function primeBrowserAudio() {
        try {
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                var ctx = new AudioCtx();
                ctx.resume();
            }
        } catch(e) {}
    }
    document.addEventListener('click', primeBrowserAudio, { once: true });
    document.addEventListener('touchstart', primeBrowserAudio, { once: true });
})();
</script>
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
# WEBRTC ICE SERVER CONFIGURATION (STUN / TURN)
# ============================================

def get_ice_servers():
    """
    Builds the WebRTC ICE servers configuration.
    Preserves public Google STUN and dynamically attaches secure dedicated TURN/TURNS
    credentials from Streamlit Secrets (st.secrets["turn"] or st.secrets["ice_servers"])
    to enable cloud firewall and symmetric NAT traversal on Streamlit Community Cloud.
    Uses exact URLs supplied by the user/dashboard without altering schemes or assuming ports.
    """
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]
    turn_configured = False
    try:
        if hasattr(st, "secrets"):
            # Option 1: Configured via [turn] table in secrets.toml
            if "turn" in st.secrets:
                turn_sec = st.secrets["turn"]
                raw_urls = turn_sec.get("urls") or turn_sec.get("url")
                username = turn_sec.get("username")
                credential = turn_sec.get("credential") or turn_sec.get("password")

                if raw_urls and username and credential:
                    if isinstance(raw_urls, str):
                        turn_urls = [raw_urls.strip()]
                    elif isinstance(raw_urls, (list, tuple)):
                        turn_urls = [str(u).strip() for u in raw_urls if str(u).strip()]
                    else:
                        turn_urls = []

                    if turn_urls:
                        ice_servers.append({
                            "urls": turn_urls,
                            "username": str(username),
                            "credential": str(credential),
                        })
                        if any(u.startswith(("turn:", "turns:")) for u in turn_urls):
                            turn_configured = True

            # Option 2: Configured via [[ice_servers]] list of tables in secrets.toml
            elif "ice_servers" in st.secrets or "iceServers" in st.secrets:
                raw_servers = st.secrets.get("ice_servers") or st.secrets.get("iceServers")
                if isinstance(raw_servers, (list, tuple)):
                    for s in raw_servers:
                        if isinstance(s, dict) and "urls" in s:
                            entry = {"urls": s["urls"]}
                            if s.get("username"):
                                entry["username"] = str(s["username"])
                            if s.get("credential") or s.get("password"):
                                entry["credential"] = str(s.get("credential") or s.get("password"))
                            ice_servers.append(entry)
                            urls_list = s["urls"] if isinstance(s["urls"], (list, tuple)) else [s["urls"]]
                            if any(str(u).startswith(("turn:", "turns:")) for u in urls_list):
                                turn_configured = True
    except Exception:
        pass

    return ice_servers, turn_configured


# ============================================
# BROWSER AUDIO ENGINE & UNLOCKER
# ============================================

@st.cache_data
def get_alarm_audio_b64():
    audio_path = "sounds/adhi_adhi_raat_meme.mp3"
    if os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    return ""

def render_browser_audio_player(should_play: bool, audio_b64: str):
    """
    Renders an embedded client-side audio player in the user's browser DOM.
    Runs on the client device (Android, iOS, Desktop) and uses Web Audio API fallback.
    """
    html_code = f"""
    <div id="audio-container" style="display:none;">
        <audio id="fall-alarm-audio" loop preload="auto" src="data:audio/mp3;base64,{audio_b64}"></audio>
    </div>
    <script>
    (function() {{
        var audio = document.getElementById("fall-alarm-audio");
        var shouldPlay = {'true' if should_play else 'false'};

        function playSound() {{
            if (audio && audio.src && audio.src.length > 50) {{
                var p = audio.play();
                if (p !== undefined) {{
                    p.catch(function(err) {{
                        console.warn("HTML5 audio playback blocked by browser policy, attempting Web Audio synthesizer fallback:", err);
                        playWebAudioSynth();
                    }});
                }}
            }} else {{
                playWebAudioSynth();
            }}
        }}

        function stopSound() {{
            if (audio) {{
                audio.pause();
                audio.currentTime = 0;
            }}
            if (window._synthOsc) {{
                try {{ window._synthOsc.stop(); }} catch(e) {{}}
                window._synthOsc = null;
            }}
        }}

        function playWebAudioSynth() {{
            if (window._synthOsc) return;
            try {{
                var AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (!window._audioCtx) {{
                    window._audioCtx = new AudioCtx();
                }}
                if (window._audioCtx.state === 'suspended') {{
                    window._audioCtx.resume();
                }}
                var osc = window._audioCtx.createOscillator();
                var gain = window._audioCtx.createGain();
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(880, window._audioCtx.currentTime);
                gain.gain.setValueAtTime(0.3, window._audioCtx.currentTime);
                osc.connect(gain);
                gain.connect(window._audioCtx.destination);
                osc.start();
                window._synthOsc = osc;
            }} catch(e) {{
                console.error("Web Audio fallback error:", e);
            }}
        }}

        if (shouldPlay) {{
            playSound();
        }} else {{
            stopSound();
        }}
    }})();
    </script>
    """
    components.html(html_code, height=0, width=0)

@st.fragment(run_every="500ms")
def browser_alarm_panel(webrtc_ctx, audio_b64):
    if webrtc_ctx and webrtc_ctx.video_processor:
        proc = webrtc_ctx.video_processor
        alarm_state = getattr(proc, "alarm_state", "normal")
        fall_count = getattr(proc, "fall_count", 0)

        should_play = (alarm_state == "active")
        render_browser_audio_player(should_play=should_play, audio_b64=audio_b64)

        if alarm_state == "active":
            st.error("🚨 **FALL DETECTED! Alarm is sounding on your device.**")
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.caption(f"Confirmed fall posture • Counter: {fall_count}/12 frames")
            with col_b:
                if st.button("🔕 Silence Alarm", key="silence_btn", use_container_width=True):
                    proc.silence_alarm()
                    st.rerun(scope="fragment")

        elif alarm_state == "silenced":
            st.warning("🔕 **Alarm Silenced** — Sound muted. Waiting for subject recovery to re-arm.")
            if st.button("🔄 Re-arm Now", key="rearm_btn", use_container_width=True):
                proc.reset_alarm()
                st.rerun(scope="fragment")

        elif alarm_state in ["normal", "recovery"]:
            if fall_count >= 4:
                st.warning(f"⚠️ **Possible Fall:** {fall_count}/12 frames detected")
            else:
                st.info("🟢 **System Armed & Ready** • Monitoring for falls")
    else:
        st.caption("ℹ️ Camera stream is starting...")


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
# LIVE FALL DETECTION
# ============================================

st.subheader("🎥 Live Fall Detection")

camera_mode = st.radio(
    "Select Camera Access Pipeline:",
    [
        "🌐 Browser Camera (Works on Android, iPhone, Windows, Mac)",
        "🖥️ Local Desktop Camera (Host OpenCV Window)"
    ],
    index=0,
    help="Browser Camera streams from your actual device through WebRTC. Local Desktop Camera runs on the machine hosting Streamlit."
)


# ============================================
# PIPELINE 1: DEVICE-ADAPTIVE BROWSER CAMERA
# ============================================

if camera_mode.startswith("🌐 Browser Camera"):

    # Camera Capability & Device Controls
    cam_col1, cam_col2 = st.columns(2)

    with cam_col1:
        facing_option = st.selectbox(
            "📷 Camera Selection (Mobile / Front & Rear):",
            [
                "User / Front Camera",
                "Environment / Rear Camera"
            ],
            index=0,
            help="Select between Front (selfie) and Rear (room monitoring) camera on Android and mobile devices."
        )

    with cam_col2:
        quality_option = st.selectbox(
            "⚙️ Camera Quality / Resolution:",
            [
                "Adaptive (Recommended - Best Device Quality)",
                "Full HD (1080p)",
                "Standard HD (720p)",
                "Smooth Performance (480p)"
            ],
            index=0,
            help="Negotiates best supported quality with your camera hardware without forced downscaling."
        )

    # Dynamic W3C MediaStreamConstraints Negotiation
    facing_mode_constraint = "user" if facing_option == "User / Front Camera" else {"ideal": "environment"}

    if quality_option == "Full HD (1080p)":
        video_constraints = {
            "facingMode": facing_mode_constraint,
            "width": {"ideal": 1920, "min": 1280},
            "height": {"ideal": 1080, "min": 720},
            "frameRate": {"ideal": 30}
        }
    elif quality_option == "Standard HD (720p)":
        video_constraints = {
            "facingMode": facing_mode_constraint,
            "width": {"ideal": 1280, "min": 640},
            "height": {"ideal": 720, "min": 480},
            "frameRate": {"ideal": 30}
        }
    elif quality_option == "Smooth Performance (480p)":
        video_constraints = {
            "facingMode": facing_mode_constraint,
            "width": {"ideal": 640, "min": 320},
            "height": {"ideal": 480, "min": 240},
            "frameRate": {"ideal": 30}
        }
    else:
        # Adaptive (Recommended - W3C ideal negotiation for diverse devices)
        video_constraints = {
            "facingMode": facing_mode_constraint,
            "width": {"ideal": 1280, "max": 1920},
            "height": {"ideal": 720, "max": 1080},
            "frameRate": {"ideal": 30, "max": 30}
        }

    if "browser_stream_running" not in st.session_state:
        st.session_state.browser_stream_running = False

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        start_browser = st.button("▶️ Start Browser Detection", use_container_width=True)
        if start_browser:
            st.session_state.browser_stream_running = True
            st.rerun()

    with btn_col2:
        stop_browser = st.button("⏹️ Stop Browser Detection", use_container_width=True)
        if stop_browser:
            st.session_state.browser_stream_running = False
            st.rerun()

    ice_servers, turn_configured = get_ice_servers()

    if not turn_configured:
        st.warning(
            "⚠️ **TURN Server Not Configured:** Remote cloud deployments on Streamlit Community Cloud "
            "require a dedicated TURN server to stream camera video across cloud firewalls and symmetric NATs. "
            "If your camera preview keeps loading without video, add your TURN credentials to "
            "Streamlit Cloud **App Settings → Secrets** (see setup guide below)."
        )
    else:
        st.caption("🔒 **TURN Relay Active:** Media stream configured for cloud NAT/firewall traversal.")

    webrtc_ctx = webrtc_streamer(
        key="fall_detection_adaptive",
        video_processor_factory=web_detector.FallDetectionProcessor,
        media_stream_constraints={
            "video": video_constraints,
            "audio": False
        },
        desired_playing_state=st.session_state.get(
            "browser_stream_running",
            False
        ),
        async_processing=True,
        rtc_configuration={
            "iceServers": ice_servers
        }
    )

    # Universal Browser Sound Alert & Live Status Panel
    browser_alarm_panel(webrtc_ctx, get_alarm_audio_b64())

    with st.expander("ℹ️ Camera Permissions & Mobile Browser Connection Guide"):
        st.markdown("""
        - **Desktop Browsers (Chrome / Safari / Edge):** Click **Allow** when prompted for camera permission.
        - **Android Mobile Browsers (Chrome / Firefox):**
          - Android requires a **Secure Context (HTTPS or localhost)** to access device cameras.
          - When accessing over local Wi-Fi (e.g. `http://192.168.x.x:8501`), open Chrome on Android, go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add your Streamlit URL (`http://<ip>:8501`), and relaunch Chrome.
          - Alternatively, connect via an HTTPS tunnel (e.g. Cloudflare Tunnel, ngrok, or Streamlit Cloud).
        - **Front vs Rear Camera:** Use the selector above to switch between selfie mode and room-monitoring mode.
        - **Camera In Use Error:** If the camera fails to start, ensure no other application (Zoom, FaceTime, Teams) is locking the camera hardware.

        ---
        #### 🌐 Streamlit Community Cloud TURN Server Configuration
        Streamlit Community Cloud containers block direct inbound UDP traffic. To stream live video from your browser to Streamlit Cloud, configure TURN credentials in **Streamlit Cloud Dashboard → Manage App → App Settings → Secrets**:
        ```toml
        [turn]
        urls = [
            "turn:<your-metered-domain>:80",
            "turn:<your-metered-domain>:80?transport=tcp",
            "turn:<your-metered-domain>:443",
            "turns:<your-metered-domain>:443",
            "turns:<your-metered-domain>:443?transport=tcp"
        ]
        username = "<your-metered-username>"
        credential = "<your-metered-credential>"
        ```
        *Or with a single TURNS (TURN over TLS) endpoint:*
        ```toml
        [turn]
        url = "turns:<your-metered-domain>:443?transport=tcp"
        username = "<your-metered-username>"
        credential = "<your-metered-credential>"
        ```
        *(Copy the exact ICE server URLs, username, and credential directly from your Metered.ca dashboard under ICE Servers).*
        """)


# ============================================
# PIPELINE 2: LOCAL DESKTOP OPENCV CAMERA
# ============================================

else:

    st.info("🖥️ **Local Desktop Mode:** Runs direct OpenCV camera capture and native windowing on the host computer.")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        desktop_cam_idx = st.selectbox(
            "📷 Hardware Camera Index:",
            [0, 1, 2, 3],
            index=0,
            help="Select camera index. 0 is built-in webcam; 1 or 2 for external USB cameras or Continuity Camera."
        )
    with d_col2:
        desktop_res_choice = st.selectbox(
            "⚙️ Target Desktop Resolution:",
            ["HD (720p - Recommended)", "Full HD (1080p)", "Standard (480p)"],
            index=0
        )

    res_map = {
        "HD (720p - Recommended)": (1280, 720),
        "Full HD (1080p)": (1920, 1080),
        "Standard (480p)": (640, 480)
    }
    target_w, target_h = res_map[desktop_res_choice]

    running_pid = get_desktop_detector_pid()
    if running_pid:
        st.success(f"🟢 **Detector Process Active** (PID: `{running_pid}`)")
    else:
        st.caption("ℹ️ Local desktop detector is currently stopped.")

    col1, col2 = st.columns(2)

    with col1:
        start_desktop = st.button(
            "▶️ Launch Desktop Detector",
            use_container_width=True,
            disabled=bool(running_pid)
        )

    with col2:
        stop_desktop = st.button(
            "⏹️ Stop Desktop Detector",
            use_container_width=True,
            disabled=not bool(running_pid)
        )

    if start_desktop:
        cmd = [
            sys.executable,
            "10_live_prediction.py",
            "--camera-index", str(desktop_cam_idx),
            "--width", str(target_w),
            "--height", str(target_h)
        ]
        proc = subprocess.Popen(cmd)
        st.session_state.desktop_pid = proc.pid
        st.success(f"🟢 Launched desktop fall detection (PID: `{proc.pid}`)!")
        st.rerun()

    if stop_desktop:
        stopped = stop_desktop_detector()
        if stopped:
            st.warning("🛑 Local desktop fall detector and child audio processes stopped cleanly.")
        else:
            st.info("Process was not active.")
        st.rerun()

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