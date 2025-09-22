# streamlit_app.py
"""
Face Sentiment Analyzer (Streamlit) — framewise, multi-face

How to run locally:
  pip install -U streamlit pandas numpy opencv-python-headless deepface altair
  streamlit run streamlit_app.py

On Streamlit Cloud:
  - requirements.txt (recommended)
      streamlit
      pandas
      numpy
      altair
      opencv-python-headless==4.10.0.84
      deepface==0.0.93
      tensorflow-cpu==2.20.0
      tf-keras==2.20.0
  - packages.txt
      libgl1
      libglib2.0-0
      libsm6
      libxext6
      libxrender1
      ffmpeg
  - runtime.txt
      3.11
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # quiet TensorFlow/oneDNN logs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")  # avoid OpenMP issues in some envs

import math
import tempfile
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import streamlit as st

# ---- Robust import guard for OpenCV / TensorFlow / DeepFace ----
try:
    import cv2  # from opencv-python-headless
except Exception as e:
    st.error(
        f"OpenCV import failed: {e}\n\n"
        "Fix: use 'opencv-python-headless' in requirements.txt or add 'libgl1' to packages.txt."
    )
    st.stop()

# Optional: verify TF + tf-keras pairing when present (DeepFace on TF>=2.16 needs tf-keras)
try:
    import tensorflow as tf  # noqa: F401
    import tf_keras  # noqa: F401
except Exception as e:
    st.warning(
        f"TensorFlow/Keras shim not ready: {e}. If DeepFace import fails, make sure "
        "'tensorflow-cpu==2.20.0' and 'tf-keras==2.20.0' are in requirements.txt, and runtime.txt=3.11."
    )

try:
    from deepface import DeepFace
except Exception as e:
    st.error(
        f"DeepFace import failed: {e}\n\n"
        "Fix: pin DeepFace (e.g., deepface==0.0.93) and ensure TF/tf-keras versions match (2.20)."
    )
    st.stop()

import numpy as np
import pandas as pd

# Optional charts; degrade gracefully if Altair missing
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False


# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="Face Sentiment Analyzer", page_icon="🎥", layout="wide")
st.title("🎥 Face Sentiment Analyzer — framewise emotions")
st.caption("Upload a video, and the app will sample frames, detect faces, and estimate emotions like happy, sad, angry, neutral, etc.")

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")

    sample_every_sec: float = st.number_input(
        "Analyze every N seconds",
        min_value=0.1, max_value=10.0, value=0.5, step=0.1,
        help="Time step between analyzed frames. Smaller = more accurate, slower."
    )

    resize_width: int = st.slider(
        "Processing width (px)", min_value=320, max_value=1280, value=640, step=64,
        help="Frames are resized for faster processing. Higher = better accuracy but slower."
    )

    max_analyzed_frames: int = st.number_input(
        "Max frames to analyze",
        min_value=25, value=1500, step=25,
        help="Safety cap to prevent very long runs."
    )

    detector_backend: str = st.selectbox(
        "Face detector backend",
        options=["opencv", "retinaface", "ssd", "mtcnn"],
        index=0,
        help="Use 'opencv' for minimal deps (fastest on Streamlit Cloud)."
    )

    make_preview: bool = st.checkbox(
        "Produce annotated preview clip",
        value=True,
        help="Writes a short MP4 showing predicted emotions on faces."
    )
    preview_seconds: int = st.slider("Preview clip duration (s)", 3, 20, 8)

    st.divider()
    st.markdown("**Tip:** If faces are small/far, increase width or analyze more frequently.")

# ---------------------------
# Utils
# ---------------------------
def readable_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def ensure_list(analysis_result: Any) -> List[Dict[str, Any]]:
    """DeepFace.analyze may return a dict (single face) or a list. Normalize to list."""
    if analysis_result is None:
        return []
    if isinstance(analysis_result, list):
        return analysis_result
    return [analysis_result]


def resize_for_processing(frame: np.ndarray, target_w: int) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame, 1.0
    scale = target_w / float(w)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        img, text, (x, max(12, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )


# ---------------------------
# Uploader
# ---------------------------
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
)
if uploaded is None:
    st.info("⬆️ Upload an MP4/MOV/AVI/MKV to begin.")
    st.stop()

# Persist upload to a temp file for OpenCV
suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    video_path = tmp.name

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("Couldn't open the uploaded video. Try a different encoding/container.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or math.isnan(fps) or fps <= 0:
    fps = 25.0  # fallback

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
video_seconds = frame_count / fps if frame_count else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("FPS", f"{fps:.2f}")
m2.metric("Resolution", f"{width}×{height}")
m3.metric("Frames", f"{frame_count}")
m4.metric("Duration", readable_time(video_seconds))
st.divider()

# ---------------------------
# Processing (no manual model build; let DeepFace manage and cache internally)
# ---------------------------
step = max(int(round(sample_every_sec * fps)), 1)
progress = st.progress(0)
status = st.empty()
results: List[Dict[str, Any]] = []
preview_frames: List[np.ndarray] = []

current_index = 0
processed_frames = 0

while True:
    grabbed = cap.grab()
    if not grabbed:
        break

    current_index += 1
    if current_index % step != 0:
        continue

    ok, frame = cap.retrieve()
    if not ok:
        break

    timestamp_s = current_index / fps
    proc_frame, scale = resize_for_processing(frame, resize_width)

    try:
        analysis = DeepFace.analyze(
            img_path=proc_frame,          # ndarray supported
            actions=["emotion"],
            detector_backend=detector_backend,
            enforce_detection=False,      # don't crash if a face isn't found
            prog_bar=False,
            # NOTE: We purposefully do NOT pass a custom models dict here.
            # DeepFace will load & cache the correct model for this version.
        )
    except Exception as e:
        status.warning(f"Analysis skipped at {timestamp_s:.2f}s: {e}")
        continue

    faces = ensure_list(analysis)
    annotated = proc_frame.copy()

    for i, face in enumerate(faces):
        emotions: Dict[str, float] = face.get("emotion", {}) or {}
        dominant: str = str(face.get("dominant_emotion", "unknown"))
        region = face.get("region", {}) or {}
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        row: Dict[str, Any] = {
            "timestamp_s": round(timestamp_s, 3),
            "frame": current_index,
            "face_index": i,
            "dominant_emotion": dominant,
        }
        for k, v in emotions.items():
            try:
                row[k] = float(v)
            except Exception:
                pass

        results.append(row)

        if w > 0 and h > 0:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_label(annotated, dominant, x, y)

    if make_preview and timestamp_s <= preview_seconds:
        preview_frames.append(annotated)

    processed_frames += 1
    if processed_frames >= max_analyzed_frames:
        status.info("Reached max analyzed frames — stopping early.")
        break

    if frame_count:
        progress.progress(min(1.0, current_index / frame_count))
    else:
        progress.progress((processed_frames % 100) / 100.0)

cap.release()
progress.empty()

# ---------------------------
# Results & Visualization
# ---------------------------
if not results:
    st.warning("No faces/emotions detected in sampled frames. "
               "Try lowering the sampling interval or increasing the processing width.")
    st.stop()

st.subheader("Framewise emotions (per detected face)")
df = pd.DataFrame(results)
st.dataframe(df, use_container_width=True)

# Download CSV
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    data=csv_bytes,
    file_name="framewise_emotions.csv",
    mime="text/csv",
)

# Charts (if Altair available)
if ALT_AVAILABLE:
    st.subheader("Emotion timeline — dominant counts per timestamp")
    dom_counts = df.groupby(["timestamp_s", "dominant_emotion"]).size().reset_index(name="count")
    chart = (
        alt.Chart(dom_counts)
        .mark_area()
        .encode(
            x=alt.X("timestamp_s:Q", title="Time (s)"),
            y=alt.Y("count:Q", stack="normalize", title="Share of faces"),
            color=alt.Color("dominant_emotion:N", legend=alt.Legend(title="Emotion")),
            tooltip=["timestamp_s", "dominant_emotion", "count"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Average emotion scores over time")
    score_cols = [c for c in df.columns if c not in {"timestamp_s", "frame", "face_index", "dominant_emotion"}]
    if score_cols:
        melted = df.melt(id_vars=["timestamp_s"], value_vars=score_cols, var_name="emotion", value_name="score")
        avg_scores = melted.groupby(["timestamp_s", "emotion"]).score.mean().reset_index()
        line = (
            alt.Chart(avg_scores)
            .mark_line()
            .encode(
                x=alt.X("timestamp_s:Q", title="Time (s)"),
                y=alt.Y("score:Q", title="Avg score"),
                color=alt.Color("emotion:N", legend=alt.Legend(title="Emotion")),
                tooltip=["timestamp_s", "emotion", alt.Tooltip("score:Q", format=".2f")],
            )
            .interactive()
        )
        st.altair_chart(line, use_container_width=True)
else:
    st.info("Altair not available; charts are disabled. Add 'altair' to requirements.txt to enable charts.")

# Annotated preview clip
if make_preview and preview_frames:
    st.subheader("Annotated preview clip")
    h, w = preview_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4v is broadly supported
    out_path = os.path.join(tempfile.gettempdir(), "preview_annotated.mp4")
    playback_fps = max(4.0, min(fps / step, 24.0))
    vw = cv2.VideoWriter(out_path, fourcc, playback_fps, (w, h))
    for f in preview_frames:
        vw.write(f)
    vw.release()
    st.video(out_path)

st.divider()
st.caption(
    "Accuracy depends on video quality and lighting. Use responsibly and with consent. "
    "This demo runs on CPU; for heavy videos, increase the sampling interval."
)
