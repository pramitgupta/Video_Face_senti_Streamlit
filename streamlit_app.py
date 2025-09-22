import streamlit as st
import cv2
import tempfile
from deepface import DeepFace
import pandas as pd

st.title("🎥 Face Sentiment Analyzer (Framewise)")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    st.write(f"Video loaded: {frame_count} frames at {fps} FPS.")

    results = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # Run sentiment detection every N frames (to save time)
        if frame_number % int(fps) == 0:  # once per second
            try:
                analysis = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotions = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion']
                results.append({
                    "frame": frame_number,
                    "dominant_emotion": dominant_emotion,
                    **emotions
                })
            except Exception as e:
                st.write(f"Error at frame {frame_number}: {e}")

    cap.release()

    # Show results
    df = pd.DataFrame(results)
    st.dataframe(df)

    st.line_chart(df.set_index("frame")["dominant_emotion"].astype("category").cat.codes)
