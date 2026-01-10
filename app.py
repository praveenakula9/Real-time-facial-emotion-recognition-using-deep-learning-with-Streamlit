import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# PAGE CONFIG
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Facial Emotion Recognition (FER)")

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_model.keras")

model = load_model()

# CONSTANTS
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = (96, 96)

# FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# PREPROCESS
def preprocess_face(face):
    face = clahe.apply(face)
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face

# EMOTION COLORS
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 165, 255),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Neutral': (255, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0)
}

# EMOTION PROCESSOR CLASS
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "Detecting..."
        self.confidence = 0.0
        self.frame_count = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Reduce resolution for speed
        img = cv2.resize(img, (640, 480))
        img = cv2.flip(img, 1)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        
        # Process every 5 frames
        if self.frame_count % 5 == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(64, 64)
            )
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                pad = int(0.2 * w)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(gray.shape[1], x + w + pad)
                y2 = min(gray.shape[0], y + h + pad)
                
                face = gray[y1:y2, x1:x2]
                preds = model.predict(preprocess_face(face), verbose=0)[0]
                
                self.emotion = CLASSES[np.argmax(preds)]
                self.confidence = float(np.max(preds))
                
                # Draw on frame
                color = emotion_colors.get(self.emotion, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    f"{self.emotion} ({self.confidence*100:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# SIDEBAR
st.sidebar.title("⚙️ Select Mode")
mode = st.sidebar.radio(
    "Choose input:",
    ["📹 Live Webcam", "🖼️ Upload Images"]
)

# =======================
# 📹 LIVE WEBCAM MODE
# =======================
if mode == "📹 Live Webcam":
    st.subheader("📹 Real-Time Webcam Emotion Detection")
    st.markdown("**Allow camera access when browser asks!**")
    
    # RTC Configuration for better compatibility
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "audio": False,
                "video": {"frameRate": {"ideal": 15}}
            },
            async_processing=True,
            video_processor_factory=EmotionProcessor,
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ Webcam is running!")
            st.info("🎥 Allow camera access when your browser asks")
            
            # Display stats
            if webrtc_ctx.video_processor:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("😊 Emotion", webrtc_ctx.video_processor.emotion)
                with col2:
                    st.metric("📊 Confidence", f"{webrtc_ctx.video_processor.confidence*100:.1f}%")
        else:
            st.info("👆 Click 'Start' above to begin webcam streaming")
    
    except Exception as e:
        st.error(f"❌ Webcam Error: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser (Chrome recommended)")

# =======================
# 🖼️ IMAGE UPLOAD MODE
# =======================
elif mode == "🖼️ Upload Images":
    st.subheader("🖼️ Upload Images for Emotion Detection")
    st.markdown("Upload face images (JPG, PNG) and get emotion predictions")
    
    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown("---")
        cols = st.columns(3)
        
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                image = Image.open(file).convert("RGB")
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                st.image(image, caption=f"📸 {file.name}", use_container_width=True)
                
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=6, minSize=(64, 64)
                )
                
                if len(faces) == 0:
                    st.warning("❌ No face detected")
                    continue
                
                x, y, w, h = faces[0]
                pad = int(0.2 * w)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(gray.shape[1], x + w + pad)
                y2 = min(gray.shape[0], y + h + pad)
                
                face = gray[y1:y2, x1:x2]
                preds = model.predict(preprocess_face(face), verbose=0)[0]
                
                emotion = CLASSES[np.argmax(preds)]
                confidence = np.max(preds)
                
                st.success(f"**{emotion}**")
                st.info(f"**{confidence*100:.1f}%**")

# FOOTER
st.markdown("---")
st.markdown(
    "<center>🔬 Facial Emotion Recognition v2.0 | Live Webcam Streaming</center>",
    unsafe_allow_html=True
)
