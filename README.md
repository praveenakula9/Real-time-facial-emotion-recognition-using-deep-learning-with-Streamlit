# 😊 Facial Emotion Recognition (FER)

Real-time facial emotion recognition web application using TensorFlow and Streamlit.

## Features

✨ **Real-Time Detection** - Live webcam emotion detection with EMA smoothing

📸 **Image Upload** - Test emotions on static images

📊 **Confidence Visualization** - Bar charts showing emotion probabilities

⚡ **Optimized** - Processes every 5 frames for smooth performance

🎨 **Beautiful UI** - Clean, intuitive interface

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Try It Online

🔗 **[Launch App](https://facial-emotion-recognition.streamlit.app)**

## Local Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/facial-emotion-recognition.git
cd facial-emotion-recognition
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure


├── app.py                 # Main Streamlit application

├── emotion_model.keras    # Emotion detection model

├── requirements.txt       # Python dependencies

├── .streamlit/config.toml # Streamlit configuration

└── README.md             # This file

## How It Works

1. **Face Detection** - Uses OpenCV's Haar Cascade Classifier
2. **Preprocessing** - CLAHE for contrast enhancement + normalization
3. **Emotion Classification** - TensorFlow model predicts emotion class
4. **Smoothing** - EMA (Exponential Moving Average) for stable predictions
5. **Visualization** - Real-time drawing + confidence charts

## Performance Optimizations

- ⚡ Processes every 5 frames (reduces compute load)
- 📉 Resolution reduced to 640x480
- 🎯 EMA smoothing for stable predictions
- 💾 Model cached using `@st.cache_resource`

## Author

Akula Durga Sri Praveen Kumar - 2026
