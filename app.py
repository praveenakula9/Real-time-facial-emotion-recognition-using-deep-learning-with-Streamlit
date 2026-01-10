import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')


import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Facial Emotion Recognition (FER)")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_model.keras")

model = load_model()

# ---------------- CONSTANTS ----------------
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = (96, 96)

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- CLAHE ----------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ---------------- ICE CONFIG ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------- PREPROCESS ----------------
def preprocess_face(face):
    face = clahe.apply(face)
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1,96,96,1)
    return face

# ---------------- CONFIDENCE CHART ----------------
def plot_confidence(preds):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(CLASSES, preds * 100)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    plt.tight_layout()
    return fig

# ---------------- WEBCAM PROCESSOR (EMA SMOOTHING) ----------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.smooth_preds = None
        self.alpha = 0.3
        self.frame_count = 0
        self.last_label = ""
        self.last_conf = 0.0
        self.last_box = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 🔥 Reduce resolution (HUGE speed gain)
        img = cv2.resize(img, (640, 480))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        # 🔥 Run heavy logic only every 5 frames
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
                face = clahe.apply(face)
                face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
                face = face.astype("float32") / 255.0
                face = np.expand_dims(face, axis=(0, -1))

                preds = model.predict(face, verbose=0)[0]

                # EMA smoothing
                if self.smooth_preds is None:
                    self.smooth_preds = preds
                else:
                    self.smooth_preds = (
                        self.alpha * preds +
                        (1 - self.alpha) * self.smooth_preds
                    )

                self.last_label = CLASSES[np.argmax(self.smooth_preds)]
                self.last_conf = np.max(self.smooth_preds)
                self.last_box = (x1, y1, x2, y2)

        # 🔥 Draw cached result (NO ML HERE)
        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{self.last_label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- SIDEBAR ----------------
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
    st.markdown("Allow **camera access** when prompted.")

    webrtc_streamer(
    key="fer-webcam",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": {"frameRate": {"max": 15}},
        "audio": False
    },
    async_processing=True,
)


# =======================
# 🖼️ IMAGE TEST MODE
# =======================
elif mode == "🖼️ Test Images":
    st.subheader("🖼️ Upload Images for Emotion Testing")

    uploaded_files = st.file_uploader(
        "Upload face images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        cols = st.columns(3)

        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                image = Image.open(file).convert("RGB")
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=6, minSize=(64, 64)
                )

                st.image(image, caption=file.name, use_container_width=True)

                if len(faces) == 0:
                    st.warning("No face detected")
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

                st.success(f"**{emotion}** ({confidence*100:.1f}%)")
                st.pyplot(plot_confidence(preds))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🔬 Facial Emotion Recognition </center>",
    unsafe_allow_html=True
)
