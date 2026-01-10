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

# CONFIDENCE CHART
def plot_confidence(preds):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF6B6B' if i == np.argmax(preds) else '#4ECDC4' for i in range(len(CLASSES))]
    ax.barh(CLASSES, preds * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=12)
    ax.set_title("Emotion Prediction Confidence", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# SIDEBAR
st.sidebar.title("⚙️ Select Mode")
mode = st.sidebar.radio(
    "Choose input:",
    ["📹 Webcam", "🖼️ Test Images"]
)

# =======================
# 📹 WEBCAM MODE
# =======================
if mode == "📹 Webcam":
    st.subheader("📹 Real-Time Webcam Emotion Detection")
    st.markdown("Click **Start** to capture frames and detect emotions")
    
    # Webcam input
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        # Convert to numpy array
        image = Image.open(picture).convert("RGB")
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(64, 64)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📸 Captured Image", use_container_width=True)
        
        with col2:
            if len(faces) == 0:
                st.warning("❌ No face detected")
            else:
                # Process first face
                x, y, w, h = faces[0]
                pad = int(0.2 * w)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(gray.shape[1], x + w + pad)
                y2 = min(gray.shape[0], y + h + pad)
                
                face = gray[y1:y2, x1:x2]
                
                # Predict emotion
                preds = model.predict(preprocess_face(face), verbose=0)[0]
                
                emotion = CLASSES[np.argmax(preds)]
                confidence = np.max(preds)
                
                # Display results
                st.success(f"**😊 Emotion: {emotion}**")
                st.info(f"**Confidence: {confidence*100:.1f}%**")
                st.pyplot(plot_confidence(preds))

# =======================
# 🖼️ IMAGE TEST MODE
# =======================
elif mode == "🖼️ Test Images":
    st.subheader("🖼️ Upload Images for Emotion Testing")
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
                st.pyplot(plot_confidence(preds))

# FOOTER
st.markdown("---")
st.markdown(
    "<center>🔬 Facial Emotion Recognition v1.0 | Powered by TensorFlow</center>",
    unsafe_allow_html=True
)
