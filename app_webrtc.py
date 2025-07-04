# FILE: app_webrtc.py

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
import pyttsx3
import queue
import time
from utils import PoseAnalyzer

# ----------------- Streamlit Page Config -----------------
st.set_page_config(
    page_title="Human Pose Estimation App",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Session State Init -----------------
def initialize_state():
    if 'pose_analyzer' not in st.session_state:
        st.session_state.pose_analyzer = PoseAnalyzer()
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False
    if 'last_feedback' not in st.session_state:
        st.session_state.last_feedback = ""
    if 'feedback_queue' not in st.session_state:
        st.session_state.feedback_queue = queue.Queue()
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'pose_detected' not in st.session_state:
        st.session_state.pose_detected = False
    if 'tts_thread' not in st.session_state:
        st.session_state.tts_thread = None

initialize_state()

# ----------------- Video Processor -----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        initialize_state()
        self.pose_analyzer = st.session_state.pose_analyzer
        self.activity_mode = "Free Pose"
        self.frame_count = 0
        self.skip_frames = 2

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            if self.frame_count % self.skip_frames != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            processed_frame, pose_data = self.pose_analyzer.process_frame(img, self.activity_mode)

            if pose_data['landmarks_detected']:
                st.session_state.pose_detected = True
                st.session_state.frame_count += 1

                if pose_data['squat_data']:
                    st.session_state.squat_data = pose_data['squat_data']
                    if (
                        st.session_state.voice_enabled and
                        pose_data['squat_data']['feedback'] != st.session_state.last_feedback and
                        self.frame_count % 30 == 0
                    ):
                        try:
                            st.session_state.feedback_queue.put_nowait(pose_data['squat_data']['feedback'])
                            st.session_state.last_feedback = pose_data['squat_data']['feedback']
                        except queue.Full:
                            pass
            else:
                st.session_state.pose_detected = False

            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        except Exception:
            return frame

# ----------------- Text-to-Speech -----------------
def text_to_speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        while True:
            try:
                feedback = st.session_state.feedback_queue.get(timeout=1)
                if feedback and st.session_state.voice_enabled:
                    clean_feedback = ''.join(char for char in feedback if char.isalnum() or char.isspace())
                    engine.say(clean_feedback)
                    engine.runAndWait()
            except queue.Empty:
                continue
    except Exception as e:
        print(f"TTS initialization failed: {e}")

# ----------------- Main Function -----------------
def main():
    st.title("🏃‍♂️ Real-time Human Pose Estimation")
    st.markdown("### WebRTC - Pose detection with feedback")

    with st.sidebar:
        st.header("⚙️ Settings")
        activity_mode = st.selectbox("Choose Activity:", ["Free Pose", "Squat Counter"])
        st.session_state.voice_enabled = st.toggle("🔊 Voice Feedback", value=st.session_state.voice_enabled)
        quality_mode = st.selectbox("Video Quality:", ["Low", "Medium", "High"], index=0)

    video_constraints = {
        "Low": {"width": 480, "height": 360},
        "Medium": {"width": 640, "height": 480},
        "High": {"width": 1280, "height": 720}
    }[quality_mode]

    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })

    video_processor = VideoProcessor()
    video_processor.activity_mode = activity_mode

    st.subheader("📹 Live Camera Feed")

    webrtc_ctx = webrtc_streamer(
        key="pose-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: video_processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": video_constraints, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.state.playing:
        st.success("🟢 Camera Connected")
    elif webrtc_ctx.state.signalling:
        st.info("🟡 Connecting...")
    else:
        st.error("🔴 Disconnected")

# ----------------- Launch TTS Background Thread -----------------
if st.session_state.voice_enabled and st.session_state.feedback_queue:
    if not st.session_state.tts_thread or not st.session_state.tts_thread.is_alive():
        st.session_state.tts_thread = threading.Thread(target=text_to_speech_worker, daemon=True)
        st.session_state.tts_thread.start()

if __name__ == "__main__":
    main()
