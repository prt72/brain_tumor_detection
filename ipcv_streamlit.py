import os
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import torch
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# --- Page Config ---
st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="wide",
    page_icon="🧠"
)

# --- Background Image ---
def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_image("bgimage.jpg")

# --- Enhanced CSS Styling ---
STYLE = f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpeg;base64,{bg_image}") no-repeat center center fixed;
        background-size: cover;
        color: white;
        font-family: 'Poppins', sans-serif;
        padding: 2rem;
    }}

    [data-testid="stSidebar"] {{
        background: url("https://wallpaperaccess.com/full/396602.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 1rem;
        min-height: 100vh;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}

    .stButton>button {{
        background: #ffffff !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}

    .title {{
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}

    .result-container {{
        background: rgba(0, 0, 0, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }}

    .progress-bar {{
        height: 14px;
        background: #03DAC6;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: width 0.4s ease;
    }}

    .dropdown-container {{
        background: rgba(0,0,0,0.7);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #BB86FC;
        margin-top: 1rem;
    }}

    [data-testid="stDownloadButton"] > button {{
        background-color: #407c97 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
    }}

    [data-testid="stDownloadButton"] > button:hover {{
        background-color: #9A67EA !important;
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }}

    @media (max-width: 768px) {{
        .title {{ font-size: 2rem; }}
    }}

    footer {{ visibility: hidden; }}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# --- Class Definitions ---
CLASS_NAMES = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor"
}

EXPLANATIONS = {
    "Glioma Tumor": "A type of brain tumor originating from glial cells. Often malignant and located in the brain's white matter.",
    "Meningioma Tumor": "Benign tumor arising from the meninges. Grows slowly and usually near the brain's surface.",
    "Pituitary Tumor": "Tumor in the pituitary gland that can cause hormonal imbalances."
}

COLOR_MAP = {0: (255, 255, 0), 1: (255, 255, 0), 2: (255, 255, 0)}
TEXT_COLOR_MAP = {0: '#ffec00', 1: '#ffec00', 2: '#ffec00'}

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_resnet_model(path='brain_tumor_resnet.h5'):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"❌ Error loading ResNet model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_yolo_model(path='best.pt'):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        return None

# --- App Main ---
def main():
    st.markdown("<div class='title'>🧠 Brain Tumor Detection</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a brain scan image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image)

            # --- ResNet Classification ---
            with st.spinner("✨ Classifying with ResNet50..."):
                resnet = load_resnet_model()
                if resnet is None: st.stop()

                resized = image.resize((224, 224))
                preprocessed = tf.keras.applications.resnet50.preprocess_input(img_to_array(resized))
                prediction = resnet.predict(np.expand_dims(preprocessed, axis=0))
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                class_result = CLASS_NAMES.get(predicted_class, "Unknown")

            # --- YOLO Detection ---
            with st.spinner("🔎 Detecting tumors with YOLO..."):
                yolo = load_yolo_model()
                if yolo is None: st.stop()

                results = yolo(img_array)
                detections = results[0].boxes.data.tolist()
                final_detections = []

                if detections:
                    detections.sort(key=lambda x: -x[4])
                    for det in detections:
                        x1, y1, x2, y2, conf, cls = det[:6]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        final_detections.append([x1, y1, x2, y2, conf, cls])

                img_with_boxes = img_array.copy()
                for det in final_detections:
                    x1, y1, x2, y2, conf, cls = det
                    cls_int = int(cls)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), COLOR_MAP[cls_int], 2)
                    cv2.putText(
                        img_with_boxes,
                        f"{CLASS_NAMES[cls_int]} ({conf:.0%})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_MAP[cls_int], 2
                    )

            # --- Display Results ---
            st.markdown("<h3 style='color: #FFFFFF'>Results</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.image(img_with_boxes, caption="🧠 Detection Results", use_container_width=True)
                st.download_button(
                    label="⬇️ Download Image with Result",
                    data=cv2.imencode(".png", img_with_boxes)[1].tobytes(),
                    file_name="tumor_detection_result.png",
                    mime="image/png"
                )

                if final_detections:
                     st.markdown("<h4 style='color: #FFFFFF'>Detection Details</h4>", unsafe_allow_html=True)
                     for det in final_detections:
                         x1, y1, x2, y2, conf, cls = det
                         cls_int = int(cls)
                         st.markdown(f"""
                             <div class="detection">
                                 <p><strong>Class:</strong> <span style="color:{TEXT_COLOR_MAP[cls_int]}">{CLASS_NAMES[cls_int]}</span></p>
                                 <p><strong>Confidence:</strong> {conf:.2%}</p>
                                 <p><strong>Bounding Box:</strong> ({x1}, {y1}) to ({x2}, {y2})</p>
                             </div>
                         """, unsafe_allow_html=True)

            with col2:
                st.markdown("<h4>📊 Classification Result</h4>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="result-container">
                        <p style='font-size: 1.2em; color: {TEXT_COLOR_MAP[predicted_class]}'>{class_result}</p>
                        <div class="progress-bar" style="width: {confidence * 100}%"></div>
                        <p style="margin-top: 0.3rem">Confidence: {confidence:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='dropdown-container'><h4>💡 Learn More</h4>", unsafe_allow_html=True)
                selected_class = st.selectbox(
                    "", ["Select Tumor Type", *CLASS_NAMES.values()],
                    key="explanation_dropdown", label_visibility="collapsed"
                )
                if selected_class in EXPLANATIONS:
                    st.markdown(f"<p><strong>{selected_class}:</strong></p><p>{EXPLANATIONS[selected_class]}</p></div>", unsafe_allow_html=True)

            if not final_detections:
                st.info("📭 No tumors detected confidently in this scan.")

        except Exception as e:
            st.error(f"❗ Error processing the image: {e}")
    else:
        st.info("📸 Upload a brain scan image to begin detection.")

    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; font-size: 0.9rem; color: #cccccc'>
            🚨 Always consult a medical professional for diagnosis.
        </p>
    """, unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    main()
