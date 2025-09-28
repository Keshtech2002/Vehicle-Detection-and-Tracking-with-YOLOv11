# streamlit_app.py
"""
Streamlit app for image & video vehicle detection + counting.

Place this file in the same folder as:
- image_counting.py   (contains `process_image` and helper)
- video_counting.py   (contains `process_video` and helper)

Run:
$ streamlit run streamlit_app.py

Requirements (example):
$ pip install streamlit ultralytics opencv-python-headless pillow pandas
"""

import streamlit as st
import tempfile
import os
import io
import pandas as pd
import cv2
from image_counting import process_image, read_image_to_bgr
from video_counting import process_video, process_video_stream
from ultralytics import YOLO


# -------------------------
# Helper utilities for this app
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "yolo11l.pt"):
    """
    Load and cache the YOLO model so it is not reloaded on every interaction.
    - model_path: path to a YOLO weights file (default: 'yolov11l.pt')
    """
    # st.write("Loading model...")  # uncomment for debug
    model = YOLO(model_path)
    return model

def format_counts_to_df(counts_dict):
    """
    Convert a counts dict (class -> count) into a pandas DataFrame for display.
    """
    if not counts_dict:
        return pd.DataFrame(columns=["class", "count"])
    rows = [{"class": k, "count": v} for k, v in counts_dict.items()]
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df

def make_download_link_bytes(data_bytes: bytes, filename: str, mime: str):
    """
    Return a tuple (bytes_io, filename) that Streamlit can use in st.download_button.
    We return the raw bytes and filename so the caller can pass them to download_button.
    """
    return data_bytes, filename, mime

# -------------------------
# Page layout & sidebar settings
# -------------------------
st.set_page_config(page_title="Vehicle Detection & Counting", layout="wide")

st.title("🚗 Vehicle Detection & Counting — Image & Video")
st.markdown(
    """
A simple Streamlit interface that uses an Ultralytics YOLO model to detect objects in images
and videos — and count objects crossing a line in videos (using tracking IDs).

**How to use**
1. Choose either *Image* or *Video* mode.
2. Upload the file.
3. Adjust confidence threshold and (for video) the counting line position.
4. Click **Run** to process.
"""
)

# Sidebar: model and options
st.sidebar.header("Model & Settings")

# Model selection: default to yolov8n. Advanced users can upload their own weights path.
model_path = st.sidebar.text_input("YOLO model path (local or name)", value="yolo11l.pt",
                                   help="Example: 'yolo11l.pt' or a local path to your weights file")

# Load model once and reuse
with st.sidebar.expander("Load model (status)"):
    try:
        model = load_model(model_path)
        st.success("Model loaded ✅")
        # Display class names available in the model
        names = model.names
        st.write("Model classes:", ", ".join([str(i) + ":" + n for i, n in names.items()]))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Global UI controls
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01,
                               help="Minimum detection confidence to consider")
# Allow users to enter class indices to filter (e.g., '2,3' or leave blank for all)
class_filter_str = st.sidebar.text_input("Class indices to keep (comma-separated)", value="",
                                         help="Enter class indices (numbers). Leave blank to keep all classes.")
if class_filter_str.strip() == "":
    class_filter = None
else:
    try:
        class_filter = [int(x.strip()) for x in class_filter_str.split(",") if x.strip() != ""]
    except ValueError:
        st.sidebar.error("Class indices must be integers separated by commas.")
        class_filter = None

# Mode selection
mode = st.radio("Mode", options=["Image", "Video"], index=0,
                help="Choose whether to process an image or video file")

# Split layout for input and output
input_col, output_col = st.columns([1, 2])

# -------------------------
# IMAGE mode UI
# -------------------------
if mode == "Image":
    with input_col:
        st.header("Image input")
        uploaded_image = st.file_uploader("Upload an image (jpg, png...)", type=["jpg", "jpeg", "png", "bmp"])
        resize_for_speed = st.checkbox("Resize large images to width 1024 for speed (recommended)", value=True)

        run_image = st.button("Run detection on image")

    with output_col:
        st.header("Image output")
        if run_image:
            if uploaded_image is None:
                st.warning("Please upload an image file first.")
            else:
                # Read bytes from uploaded_file and pass to your process_image function
                img_bytes = uploaded_image.getvalue()

                # Optionally resize before passing to model to speed up detection (but keep original for annotations)
                # We'll convert bytes -> numpy using read_image_to_bgr so your helper is used.
                frame_bgr = read_image_to_bgr(img_bytes)
                if frame_bgr is None:
                    st.error("Failed to read uploaded image.")
                else:
                    # Optional resize
                    if resize_for_speed:
                        h, w = frame_bgr.shape[:2]
                        max_w = 1024
                        if w > max_w:
                            scale = max_w / w
                            new_w, new_h = int(w * scale), int(h * scale)
                            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                            # Convert back to bytes for passing into process_image (which accepts bytes or path)
                            # Use cv2.imencode to get bytes
                            is_success, buffer = cv2.imencode(".jpg", frame_bgr)
                            img_bytes = buffer.tobytes()

                    # Run detection using provided helper
                    annotated_rgb, counts = process_image(img_bytes, model=model, classes=class_filter, conf_thres=confidence)

                    # Display annotated image
                    st.image(annotated_rgb, caption="Annotated image (RGB)", use_column_width=True)

                    # Show counts
                    df_counts = format_counts_to_df(counts)
                    st.subheader("Counts")
                    st.dataframe(df_counts)

                    # Offer download of annotated image (convert to bytes)
                    # Convert annotated RGB (numpy) back to BGR for cv2 encoding
                    annotated_bgr = annotated_rgb[:, :, ::-1]
                    is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
                    if is_success:
                        bytes_img = buffer.tobytes()
                        st.download_button(label="Download annotated image", data=bytes_img,
                                           file_name="annotated_image.jpg", mime="image/jpeg")
                    else:
                        st.error("Failed to encode annotated image for download.")

# -------------------------
# VIDEO mode UI
# -------------------------
# -------------------------
# VIDEO mode UI (streaming updates)
# -------------------------
else:
    with input_col:
        st.header("Video input (streaming)")
        uploaded_video = st.file_uploader("Upload a video file (mp4, mkv, avi...)", type=["mp4", "mov", "avi", "mkv"])
        st.markdown("**Video counting settings**")
        line_position = st.slider("Counting line vertical position (relative)", 0.0, 1.0, 0.75, 0.01,
                                  help="Where to draw the horizontal counting line (0=top, 1=bottom).")
        draw_counts_on_frame = st.checkbox("Overlay counts on video frames (basic)", value=True)
        frame_interval = st.number_input("Update every N frames in preview", min_value=1, max_value=60, value=5,
                                         help="How many frames to skip between UI updates. Smaller = smoother preview but slower.")
        run_video = st.button("Run detection & tracking (stream with updates)")

    with output_col:
        st.header("Video output (live preview)")
        # placeholders for live UI elements
        preview_image = st.empty()        # will hold current annotated frame
        progress_bar = st.progress(0.0)   # progress
        counts_area = st.empty()          # will hold the counts DataFrame
        status_text = st.empty()          # status messages

        if run_video:
            if uploaded_video is None:
                st.warning("Please upload a video file first.")
            else:
                # Save the uploaded video to a temp file (process_video_stream expects a path)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
                try:
                    tfile.write(uploaded_video.getvalue())
                    tfile.flush()
                    tfile.close()

                    status_text.info("Starting processing...")

                    # Define the callback that will be called by process_video_stream
                    def on_progress(frame_rgb, counts_snapshot, progress_frac):
                        """
                        This function will be called from process_video_stream every N frames.
                        We update the preview image, progress bar, and counts display here.
                        """
                        # Update preview image (Streamlit accepts numpy RGB arrays)
                        preview_image.image(frame_rgb, caption="Live annotated frame (RGB)", use_container_width=True)

                        # Build and display counts DataFrame
                        df_live = format_counts_to_df(counts_snapshot)
                        counts_area.dataframe(df_live)

                        # Update progress bar if we have a fraction
                        if progress_frac is not None:
                            # progress_frac might be between 0..1; st.progress expects 0..1 float or 0..100 int
                            progress_bar.progress(min(1.0, max(0.0, float(progress_frac))))

                    # Call stream-processing function; it will call on_progress periodically
                    annotated_path, final_counts = process_video_stream(
                        tfile.name,
                        model=model,
                        classes=class_filter,
                        conf_thres=confidence,
                        line_position=line_position,
                        frame_update_interval=int(frame_interval),
                        on_progress=on_progress,
                    )

                    status_text.success("Processing complete ✅")

                    # Show final counts
                    df_final = format_counts_to_df(final_counts)
                    st.subheader("Final counts")
                    st.dataframe(df_final)

                    # Read final annotated video bytes and show/download
                    with open(annotated_path, "rb") as f:
                        video_bytes = f.read()

                    st.video(video_bytes)
                    st.download_button("Download annotated video", data=video_bytes,
                                       file_name="annotated_video.mp4", mime="video/mp4")

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                finally:
                    # Clean up uploaded temp
                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass


# -------------------------
# Footer / help
# -------------------------
st.markdown("---")
st.markdown(
    """
**Tips & troubleshooting**

- First run may be slow while the YOLO model downloads (if using a model name like `yolov11l.pt`).
- If you run into memory issues, try smaller images or use the `resize` option.
- To restrict counting/detection to certain classes, enter their indices in the sidebar (e.g., `2,3`).
"""
)
