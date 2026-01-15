import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
import os
import gdown
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="PPE Detection App", layout="wide")

# Define the classes
CLASS_NAMES = {
    0: 'Boot',
    1: 'Goggles',
    2: 'Hardhat',
    3: 'Mask',
    4: 'No-Boot',
    5: 'No-Goggles',
    6: 'No-Hardhat',
    7: 'No-Mask',
    8: 'No-Safety Vest',
    9: 'Person',
    10: 'Safety Vest'
}

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.title("Model Settings")

# Confidence Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

# Class Selection
st.sidebar.subheader("Select Classes to Detect")
all_class_names = list(CLASS_NAMES.values())
selected_class_names = st.sidebar.multiselect(
    "Classes", 
    options=all_class_names, 
    default=all_class_names
)
selected_indices = [k for k, v in CLASS_NAMES.items() if v in selected_class_names]

# --- 3. MODEL LOADING (FROM GOOGLE DRIVE) ---
@st.cache_resource
def load_model_from_drive():
    output_path = "best.pt"
    if not os.path.exists(output_path):
        with st.spinner("Downloading model from secure storage..."):
            try:
                if "model_id" in st.secrets:
                    file_id = st.secrets["model_id"]
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, output_path, quiet=False)
                else:
                    st.error("Model ID not found in Secrets!")
                    return None
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None

    try:
        return YOLO(output_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_from_drive()

# --- 4. WEBRTC CALLBACK FUNCTION ---
# This function handles the video frames from the browser
def video_frame_callback(frame):
    # Convert frame to numpy array (OpenCV format)
    img = frame.to_ndarray(format="bgr24")
    
    # Run inference (using the global 'model' and 'conf_threshold')
    if model is not None:
        results = model.predict(
            source=img, 
            conf=conf_threshold,
            classes=selected_indices if selected_indices else None,
            verbose=False
        )
        # Plot results on the frame
        annotated_frame = results[0].plot()
        
        # Return the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    
    return frame

# --- 5. MAIN APP LOGIC ---
st.title("🚧 YOLOv8 Safety Equipment Detection")
st.write("Detect PPE (Boots, Hardhats, Vests, etc.) in images, videos, or live feed.")

source_option = st.selectbox(
    "Select Input Source", 
    ("Image", "Video", "Live Feed")
)

if model is not None:
    # --- IMAGE MODE ---
    if source_option == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Detect"):
                results = model.predict(image, conf=conf_threshold, classes=selected_indices)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Detection Result", use_container_width=True)

    # --- VIDEO MODE ---
    elif source_option == "Video":
        uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            stop_button = st.button("Stop Processing")
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret: break
                results = model.predict(frame, conf=conf_threshold, classes=selected_indices, verbose=False)
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st_frame.image(res_rgb, caption="Video Processing", use_container_width=True)
            cap.release()

    # --- LIVE FEED MODE (WEBRTC) ---
    elif source_option == "Live Feed":
        st.write("Click 'Start' to use your webcam.")
        
        # WEBRTC CONFIGURATION
        # STUN servers are necessary for the browser to connect to the cloud server
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="ppe-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

else:
    st.warning("Waiting for model to load...")
