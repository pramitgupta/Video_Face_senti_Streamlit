import streamlit as st
import cv2
import os
import pandas as pd
from deepface import DeepFace
import tempfile

# Set page config
st.set_page_config(page_title="Facial Sentiment Analysis", layout="wide")

st.markdown("### Upload a file and view its details and analysis results.")

# Initialize session state
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = None
if 'stats_df' not in st.session_state:
    st.session_state.stats_df = None
if 'preview_img' not in st.session_state:
    st.session_state.preview_img = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = []

# Tabs
tab1, tab2 = st.tabs(["User Input", "Face Sentiment"])

with tab1:
    st.header("User Input")

    input_file = st.file_uploader("Upload Image or Video", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])
    frame_interval = st.slider("Frame Interval (seconds)", min_value=1, max_value=30, value=10, step=1)

    if input_file:
        st.session_state.file_uploaded = input_file

    if st.button("Show File Stats") and st.session_state.file_uploaded:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.file_uploaded.name)[1]) as tmp_file:
            tmp_file.write(st.session_state.file_uploaded.getvalue())
            tmp_file_path = tmp_file.name

        # Get stats
        stats = {}
        file_size_kb = os.path.getsize(tmp_file_path) / 1024
        stats["File Size (KB)"] = round(file_size_kb, 2)

        if tmp_file_path.lower().endswith(('.png','.jpg','.jpeg')):
            img = cv2.imread(tmp_file_path)
            if img is None:
                stats["Resolution"] = "Error reading image"
                st.session_state.preview_img = None
            else:
                h, w = img.shape[:2]
                stats["Resolution"] = f"{w} x {h}"
                st.session_state.preview_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif tmp_file_path.lower().endswith(('.mp4','.avi','.mov')):
            cap = cv2.VideoCapture(tmp_file_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                stats["Resolution"] = f"{width} x {height}"
                ret, frame = cap.read()
                if ret:
                    st.session_state.preview_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    st.session_state.preview_img = None
                cap.release()
            else:
                stats["Resolution"] = "Error reading video"
                st.session_state.preview_img = None
        else:
            stats["Resolution"] = "Unsupported file type"
            st.session_state.preview_img = None

        st.session_state.stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])

        # Display stats and preview
        st.subheader("File Descriptive Stats")
        st.dataframe(st.session_state.stats_df)

        if st.session_state.preview_img is not None:
            st.subheader("Input Preview")
            st.image(st.session_state.preview_img, channels="RGB", use_column_width=True)
        else:
            st.warning("Could not generate preview.")

        # Cleanup temp file
        os.unlink(tmp_file_path)

    elif st.session_state.stats_df is not None:
        st.subheader("File Descriptive Stats")
        st.dataframe(st.session_state.stats_df)
        if st.session_state.preview_img is not None:
            st.subheader("Input Preview")
            st.image(st.session_state.preview_img, channels="RGB", use_column_width=True)

with tab2:
    st.header("Face Sentiment")

    if st.button("Process") and st.session_state.file_uploaded:
        with st.spinner("Processing... This may take a while."):

            # Save file again for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.file_uploaded.name)[1]) as tmp_file:
                tmp_file.write(st.session_state.file_uploaded.getvalue())
                tmp_file_path = tmp_file.name

            records_all = []
            annotated_images = []

            # Determine file type and process
            if tmp_file_path.lower().endswith(('.png','.jpg','.jpeg')):
                img = cv2.imread(tmp_file_path)
                faces = DeepFace.extract_faces(img_path=tmp_file_path, enforce_detection=False)
                annotated_img, sorted_faces = annotate_image(img, faces, frame_label="IMG")
                annotated_images.append(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                records = analyze_sentiment_for_faces(img, sorted_faces)
                records_all.extend([{"Frame": os.path.basename(tmp_file_path), **rec} for rec in records])

            elif tmp_file_path.lower().endswith(('.mp4','.avi','.mov')):
                cap = cv2.VideoCapture(tmp_file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_step = int(frame_interval * fps)
                frame_indices = range(0, frame_count, frame_step if frame_step > 0 else 1)

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_frame_file:
                        temp_frame_path = temp_frame_file.name
                        cv2.imwrite(temp_frame_path, frame)

                    faces = DeepFace.extract_faces(img_path=temp_frame_path, enforce_detection=False)
                    annotated_img, sorted_faces = annotate_image(frame, faces, frame_label=f"F{idx}")
                    annotated_images.append(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                    records = analyze_sentiment_for_faces(frame, sorted_faces)
                    records_all.extend([{"Frame": f"Frame {idx}", **rec} for rec in records])

                    os.unlink(temp_frame_path)  # Clean up temp frame

                cap.release()

            else:
                st.error("Unsupported file type.")
                os.unlink(tmp_file_path)
                st.stop()

            # Cleanup main temp file
            os.unlink(tmp_file_path)

            # Store results
            st.session_state.annotated_images = annotated_images
            st.session_state.sentiment_results = pd.DataFrame(records_all)

    # Display results if available
    if st.session_state.annotated_images:
        st.subheader("Annotated Frames")
        for i, img in enumerate(st.session_state.annotated_images):
            st.image(img, caption=f"Frame {i+1}", use_column_width=True)

    if st.session_state.sentiment_results is not None and not st.session_state.sentiment_results.empty:
        st.subheader("Sentiment Analysis Results")
        st.dataframe(st.session_state.sentiment_results)

# --- Helper Functions (same logic, no Gradio) ---

def annotate_image(img, faces, frame_label=""):
    """
    Annotate the image with bounding boxes. Label each box with
    both the frame label (e.g., F250) and the face index (e.g., -1).
    """
    sorted_faces = sorted(faces, key=lambda f: (f['facial_area']['y'], f['facial_area']['x']))
    annotated = img.copy()
    for i, face in enumerate(sorted_faces):
        x = face['facial_area']['x']
        y = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{frame_label}-{i+1}" if frame_label else str(i+1)
        cv2.putText(annotated, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        face['grid_marker'] = f"Face {i+1}"
    return annotated, sorted_faces

def analyze_sentiment_for_faces(img, faces):
    """Analyze sentiment (emotion) for each detected face in the image."""
    records = []
    sorted_faces = sorted(faces, key=lambda f: (f['facial_area']['y'], f['facial_area']['x']))
    for face in sorted_faces:
        x = face['facial_area']['x']
        y = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']
        face_img = img[y:y+h, x:x+w]
        try:
            analysis = DeepFace.analyze(img_path=face_img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            dominant_emotion = "Error"
        records.append({
            "Face": face.get('grid_marker', "Unknown"),
            "Sentiment": dominant_emotion
        })
    return records
