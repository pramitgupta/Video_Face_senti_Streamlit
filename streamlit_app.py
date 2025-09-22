# app.py
# Streamlit port of the Gradio app: Facial Sentiment Analysis
import os
import cv2
import shutil
import tempfile
import pandas as pd
import streamlit as st
from deepface import DeepFace

# ---------------------------
# Helpers
# ---------------------------
def get_session_tempdir() -> str:
    """Create one temp directory per session to store uploaded files/frames."""
    if "tmpdir" not in st.session_state:
        st.session_state.tmpdir = tempfile.mkdtemp(prefix="st_deepface_")
    return st.session_state.tmpdir

def save_uploaded_file(uploaded_file) -> str:
    """Persist Streamlit UploadedFile to disk and return its path."""
    tmpdir = get_session_tempdir()
    suffix = os.path.splitext(uploaded_file.name)[1]
    path = os.path.join(tmpdir, f"upload{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path

def bgr_to_rgb(img_bgr):
    """Convert OpenCV BGR image to RGB for Streamlit display."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def get_descriptive_stats(file_path: str) -> pd.DataFrame:
    """Return a dataframe of file metrics such as size and resolution."""
    stats = {}
    file_size_kb = os.path.getsize(file_path) / 1024
    stats["File Size (KB)"] = round(file_size_kb, 2)

    lower = file_path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(file_path)
        if img is None:
            stats["Resolution"] = "Error reading image"
        else:
            h, w = img.shape[:2]
            stats["Resolution"] = f"{w} x {h}"
    elif lower.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            stats["Resolution"] = f"{width} x {height}"
            cap.release()
        else:
            stats["Resolution"] = "Error reading video"
    else:
        stats["Resolution"] = "Unsupported file type"

    return pd.DataFrame(stats.items(), columns=["Metric", "Value"])

def preview_image_or_first_frame(file_path: str):
    """Return an RGB image suitable for preview (image itself or first video frame)."""
    lower = file_path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg")):
        img = cv2.imread(file_path)
        return bgr_to_rgb(img) if img is not None else None
    elif lower.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        return bgr_to_rgb(frame) if ret else None
    return None

def annotate_image(img_bgr, faces, frame_label=""):
    """
    Draw bounding boxes and labels on a BGR image.
    Returns annotated BGR image and a sorted faces list with 'grid_marker' labels attached.
    """
    sorted_faces = sorted(
        faces, key=lambda f: (int(f["facial_area"]["y"]), int(f["facial_area"]["x"]))
    )
    annotated = img_bgr.copy()
    for i, face in enumerate(sorted_faces):
        fa = face["facial_area"]
        x, y, w, h = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{frame_label}-{i+1}" if frame_label else str(i + 1)
        cv2.putText(annotated, label, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        face["grid_marker"] = f"Face {i+1}"
    return annotated, sorted_faces

def detect_faces_on_path(file_path: str):
    """Detect faces in an image file path using DeepFace."""
    faces = DeepFace.extract_faces(img_path=file_path, enforce_detection=False)
    return faces

def detect_faces_on_frame(frame_bgr):
    """Detect faces on an in-memory frame (numpy array)."""
    faces = DeepFace.extract_faces(img_path=frame_bgr, enforce_detection=False)
    return faces

def analyze_sentiment_for_faces(img_bgr, faces):
    """Analyze emotion for each detected face given the full image (BGR)."""
    records = []
    sorted_faces = sorted(
        faces, key=lambda f: (int(f["facial_area"]["y"]), int(f["facial_area"]["x"]))
    )
    for face in sorted_faces:
        fa = face["facial_area"]
        x, y, w, h = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
        x2, y2 = x + w, y + h
        face_img = img_bgr[max(0, y):max(0, y2), max(0, x):max(0, x2)]
        if face_img.size == 0:
            dominant_emotion = "n/a"
        else:
            analysis = DeepFace.analyze(
                img_path=face_img, actions=["emotion"], enforce_detection=False
            )
            # DeepFace.analyze can return a list or dict depending on version:
            if isinstance(analysis, list) and analysis:
                dominant_emotion = analysis[0].get("dominant_emotion", "unknown")
            elif isinstance(analysis, dict):
                dominant_emotion = analysis.get("dominant_emotion", "unknown")
            else:
                dominant_emotion = "unknown"

        records.append({"Face": face.get("grid_marker", "Unknown"),
                        "Sentiment": dominant_emotion})
    return records

def process_input(file_path: str, frame_interval_seconds: int):
    """
    Determine if the file is an image or video and extract frames accordingly.
    For video, sample frames every `frame_interval_seconds`.
    Returns:
      ("image", [(rgb_image, "IMG")]) or ("video", [(rgb_image, f"F{idx}")...])
    """
    lower = file_path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg")):
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return "unsupported", []
        return "image", [(bgr_to_rgb(img_bgr), "IMG", img_bgr)]
    elif lower.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return "unsupported", []

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Compute step safely
        step = int(max(1, round(frame_interval_seconds * fps))) if fps > 0 else int(1)

        frames_out = []
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frames_out.append((bgr_to_rgb(frame_bgr), f"F{idx}", frame_bgr))
        cap.release()
        return "video", frames_out
    else:
        return "unsupported", []

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Facial Sentiment Analysis", layout="wide")
st.title("Facial Sentiment Analysis")
st.caption("Upload an image or a video. View file stats, preview, and per-face emotion analysis.")

tab_input, tab_sentiment = st.tabs(["User Input", "Face Sentiment"])

with tab_input:
    st.subheader("Upload and Inspect")
    uploaded = st.file_uploader("Upload Image (.png/.jpg/.jpeg) or Video (.mp4/.avi/.mov)",
                                type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])
    frame_interval = st.slider("Frame Interval (seconds) for Video", min_value=1, max_value=30, value=10, step=1)
    show_stats = st.button("Show File Stats")

    if show_stats:
        if uploaded is None:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Saving & analyzing file..."):
                # Save to disk and compute stats + preview
                file_path = save_uploaded_file(uploaded)
                stats_df = get_descriptive_stats(file_path)
                st.markdown("**File Descriptive Stats**")
                st.dataframe(stats_df, use_container_width=True)
                preview = preview_image_or_first_frame(file_path)
                if preview is not None:
                    st.markdown("**Input Preview**")
                    st.image(preview, use_container_width=True)
                else:
                    st.info("Could not generate a preview for this file.")

with tab_sentiment:
    st.subheader("Detect Faces & Analyze Emotions")
    process_btn = st.button("Process")
    if process_btn:
        if uploaded is None:
            st.warning("Please upload a file in the *User Input* tab first.")
            st.stop()

        with st.spinner("Running face detection and emotion analysis..."):
            file_path = save_uploaded_file(uploaded)
            file_type, frames = process_input(file_path, frame_interval)

            if file_type == "unsupported" or len(frames) == 0:
                st.error("Unsupported file type or could not read frames.")
                st.stop()

            annotated_images_rgb = []
            captions = []
            all_records = []

            if file_type == "image":
                # frames: [(rgb, "IMG", bgr)]
                rgb_img, frame_label, bgr_img = frames[0]
                # Detect faces on path (a bit more robust for stills)
                faces = detect_faces_on_path(file_path)
                annotated_bgr, sorted_faces = annotate_image(bgr_img, faces, frame_label=frame_label)
                annotated_images_rgb.append(bgr_to_rgb(annotated_bgr))
                captions.append("Annotated IMG")
                records = analyze_sentiment_for_faces(bgr_img, sorted_faces)
                # add frame name
                for rec in records:
                    all_records.append({"Frame": os.path.basename(file_path), **rec})

            elif file_type == "video":
                # frames: list of (rgb, "F{idx}", bgr)
                for rgb_img, frame_label, bgr_img in frames:
                    faces = detect_faces_on_frame(bgr_img)
                    annotated_bgr, sorted_faces = annotate_image(bgr_img, faces, frame_label=frame_label)
                    annotated_images_rgb.append(bgr_to_rgb(annotated_bgr))
                    captions.append(f"Annotated {frame_label}")
                    records = analyze_sentiment_for_faces(bgr_img, sorted_faces)
                    for rec in records:
                        all_records.append({"Frame": f"Frame {frame_label[1:]}", **rec})

            # Show gallery (list display)
            st.markdown("**Annotated Frames**")
            st.image(annotated_images_rgb, caption=captions, use_container_width=True)

            # Results table
            df = pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=["Frame", "Face", "Sentiment"])
            st.markdown("**Sentiment Analysis Results**")
            st.dataframe(df, use_container_width=True)

# Optional: Cleanup button for temp directory
with st.expander("⚙️ Advanced"):
    if st.button("Clear session temp files"):
        if "tmpdir" in st.session_state and os.path.isdir(st.session_state.tmpdir):
            shutil.rmtree(st.session_state.tmpdir, ignore_errors=True)
            del st.session_state.tmpdir
            st.success("Cleared temporary files for this session.")
        else:
            st.info("No temp files to clear.")
