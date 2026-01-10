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
import threading
import time

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
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 165, 255),  # Orange
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 0),      # Green
    'Neutral': (255, 255, 0),  # Cyan
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 255, 0)  # Yellow
}

# SIDEBAR
st.sidebar.title("⚙️ Select Mode")
mode = st.sidebar.radio(
    "Choose input:",
    ["📹 Live Webcam Video", "🖼️ Upload Images"]
)

# =======================
# 📹 LIVE WEBCAM VIDEO MODE
# =======================
if mode == "📹 Live Webcam Video":
    st.subheader("📹 Real-Time Webcam Emotion Detection")
    st.markdown("**Allow camera access and start the stream**")
    
    # Start/Stop button
    col1, col2 = st.columns([1, 5])
    with col1:
        start_button = st.button("▶️ START", key="start_btn")
    with col2:
        stop_button = st.button("⏹️ STOP", key="stop_btn")
    
    # Placeholder for video stream
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if start_button or st.session_state.get('webcam_running', False):
        st.session_state.webcam_running = True
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Variables for smoothing
        smooth_emotion = None
        smooth_confidence = None
        frame_count = 0
        alpha = 0.3
        
        try:
            while st.session_state.get('webcam_running', True) and not stop_button:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Could not access webcam")
                    break
                
                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Process every 5 frames for performance
                frame_count += 1
                
                if frame_count % 5 == 0:
                    # Detect faces
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.2, 
                        minNeighbors=6, 
                        minSize=(64, 64)
                    )
                    
                    if len(faces) > 0:
                        # Get largest face
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        
                        # Add padding
                        pad = int(0.2 * w)
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(frame.shape[1], x + w + pad)
                        y2 = min(frame.shape[0], y + h + pad)
                        
                        # Extract and predict
                        face = gray[y1:y2, x1:x2]
                        preds = model.predict(preprocess_face(face), verbose=0)[0]
                        
                        emotion = CLASSES[np.argmax(preds)]
                        confidence = float(np.max(preds))
                        
                        # EMA smoothing
                        if smooth_emotion is None:
                            smooth_emotion = emotion
                            smooth_confidence = confidence
                        else:
                            # Smooth confidence
                            smooth_confidence = (alpha * confidence + 
                                                (1 - alpha) * smooth_confidence)
                        
                        # Draw rectangle around face
                        color = emotion_colors.get(emotion, (0, 255, 0))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw emotion label
                        label = f"{emotion} ({smooth_confidence*100:.1f}%)"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2
                        )
                        
                        # Draw confidence bar
                        bar_width = int(smooth_confidence * 150)
                        cv2.rectangle(frame, (x1, y2 + 10), (x1 + bar_width, y2 + 20), color, -1)
                        cv2.rectangle(frame, (x1, y2 + 10), (x1 + 150, y2 + 20), (255, 255, 255), 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True)
                
                # Display stats
                if smooth_emotion:
                    with stats_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("😊 Emotion", smooth_emotion)
                        with col2:
                            st.metric("📊 Confidence", f"{smooth_confidence*100:.1f}%")
                        with col3:
                            st.metric("🎬 Frames", frame_count)
                
                time.sleep(0.05)  # ~20 FPS
        
        finally:
            cap.release()
            st.session_state.webcam_running = False
            st.success("✅ Webcam stopped")

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
