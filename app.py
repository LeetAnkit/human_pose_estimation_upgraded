# FILE: app.py

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
import pyttsx3
import queue
import time
import pandas as pd
import altair as alt
from utils import PoseAnalyzer, get_pose_landmarks_info
from firebase_helper import save_workout_to_firebase, get_all_workouts

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
    if 'tts_thread' not in st.session_state:
        st.session_state.tts_thread = None
    if 'pose_detected' not in st.session_state:
        st.session_state.pose_detected = False

initialize_state()

# ----------------- VideoProcessor Class -----------------
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

# ----------------- Main UI -----------------
def main():
    st.title("🏃‍♂️ Real-time Human Pose Estimation")
    st.markdown("### AI-powered pose detection with exercise tracking and feedback")

    with st.sidebar:
        st.header("⚙️ Settings")
        activity_mode = st.selectbox("Choose Activity:", ["Free Pose", "Squat Counter"])
        st.session_state.voice_enabled = st.toggle("🔊 Voice Feedback", value=st.session_state.voice_enabled)
        quality_mode = st.selectbox("Video Quality:", ["Low (Better Performance)", "Medium", "High (May Lag)"], index=0)
        st.divider()

        if activity_mode == "Squat Counter":
            st.header("📊 Workout Stats")
            if hasattr(st.session_state, 'squat_data'):
                squat_data = st.session_state.squat_data
                st.metric("Repetitions", squat_data['count'])
                stage_color = "🟢" if squat_data['stage'] == "up" else "🔴"
                st.metric("Current Stage", f"{stage_color} {squat_data['stage'].upper()}")
                st.metric("Knee Angle", f"{squat_data['angle']}°")
                accuracy = squat_data['accuracy']
                st.metric("Form Accuracy", f"{accuracy}%")
                st.progress(accuracy / 100)
                st.info(f"💬 {squat_data['feedback']}")
            else:
                st.info("Start exercising to see stats!")

            if st.button("🔄 Reset Counter", type="secondary"):
                st.session_state.pose_analyzer.reset_counter()
                st.rerun()

            if hasattr(st.session_state, 'squat_data') and st.session_state.squat_data['count'] > 0:
                if st.button("📂 Save Workout", type="primary"):
                    st.session_state.pose_analyzer.save_workout_data(
                        st.session_state.squat_data['count'],
                        st.session_state.squat_data['accuracy']
                    )
                    save_workout_to_firebase(
                        reps=st.session_state.squat_data['count'],
                        accuracy=st.session_state.squat_data['accuracy']
                    )
                    st.success("✅ Workout saved locally and to cloud!")

                try:
                    with open("workout_history.csv", "rb") as file:
                        st.download_button(
                            label="📥 Download Workout History (CSV)",
                            data=file,
                            file_name="workout_history.csv",
                            mime="text/csv"
                        )
                except FileNotFoundError:
                    st.warning("No workout history available yet.")

                try:
                    df = pd.read_csv("workout_history.csv")
                    if not df.empty:
                        st.subheader("📊 Workout Progress")

                        reps_chart = alt.Chart(df).mark_line(point=True).encode(
                            x='date:T',
                            y='reps:Q',
                            tooltip=['date', 'reps']
                        ).properties(title="📈 Reps Over Time")

                        accuracy_chart = alt.Chart(df).mark_line(point=True, color="green").encode(
                            x='date:T',
                            y='accuracy:Q',
                            tooltip=['date', 'accuracy']
                        ).properties(title="🎯 Accuracy Over Time")

                        st.altair_chart(reps_chart, use_container_width=True)
                        st.altair_chart(accuracy_chart, use_container_width=True)
                except Exception as e:
                    st.warning("⚠️ Could not load workout history.")

                # ☁️ Show Firebase (Cloud) History
                if st.toggle("🌐 Show Cloud History"):
                    try:
                        df_cloud = get_all_workouts()
                        st.dataframe(df_cloud)
                        st.success("✅ Synced with Firebase")
                    except Exception as e:
                        st.error("❌ Could not load Firebase data")

        st.divider()
        st.header("🔍 Diagnostics")
        st.metric("Frames Processed", st.session_state.frame_count)
        st.success("✅ Pose Detected" if st.session_state.pose_detected else "⚠️ No Pose Detected")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_constraints = {"width": 480, "height": 360} if quality_mode == "Low (Better Performance)" else \
                            {"width": 640, "height": 480} if quality_mode == "Medium" else \
                            {"width": 1280, "height": 720}

        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })

        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": video_constraints, "audio": False},
            async_processing=True
        )

        if webrtc_ctx.state.playing:
            st.success("🟢 Camera Connected")
        elif webrtc_ctx.state.signalling:
            st.info("🟡 Connecting to camera...")
        else:
            st.error("🔴 Camera Disconnected")

    with col2:
        st.subheader("🌟 Quick Tips")
        st.markdown("""**Perfect Squat Form:**  
        - Keep back straight  
        - Knees behind toes  
        - Thighs parallel to floor  
        - Push through heels to rise""")
        if st.button("🔄 Restart Camera", type="primary"):
            st.rerun()

# ----------------- Launch Background TTS -----------------
if st.session_state.voice_enabled and st.session_state.feedback_queue:
    if not st.session_state.tts_thread or not st.session_state.tts_thread.is_alive():
        st.session_state.tts_thread = threading.Thread(target=text_to_speech_worker, daemon=True)
        st.session_state.tts_thread.start()

if __name__ == "__main__":
    main()
