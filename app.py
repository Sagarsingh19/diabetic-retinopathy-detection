import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from streamlit_app.gradcam_utils import generate_and_save_gradcam

# Set page config
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

# Title and intro
st.title("Diabetic Retinopathy Detection")
st.image("streamlit_app/assets/retina_cover.png", use_container_width=True)
st.markdown(
    "Upload a retina image below. The model will predict the Diabetic Retinopathy stage (0–4) "
    "and display a Grad-CAM heatmap showing where the model is focusing."
)

# Load model once
@st.cache_resource
def load_retinopathy_model():
    return load_model('streamlit_app/model/model.keras')

model = load_retinopathy_model()
last_conv_layer = 'last_conv'  # Update if using EfficientNet later

# Class label mappings
labels = {
    0: "No DR (Healthy)",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Sample image option
use_sample = st.button("Try Sample Retina Image")

if use_sample:
    image_path = "streamlit_app/assets/sample_retina.jpg"
    st.image(image_path, caption="Sample Retina Image", use_container_width=True)

    # Preprocess
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Grad-CAM
    gradcam_path, pred_class = generate_and_save_gradcam(
        model=model,
        img_array=img_array,
        last_conv_layer_name=last_conv_layer,
        save_path="streamlit_app/output_image/sample_gradcam.jpg"
    )

    st.markdown(
        f"<h2 style='color:#2E86AB; font-size:26px;'>Prediction: {labels[pred_class]} (Class {pred_class})</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:16px;'>Classes: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR</p>",
        unsafe_allow_html=True
    )

    st.image(gradcam_path, caption="Grad-CAM Heatmap", use_container_width=True)

# Upload section
st.markdown(
    "<h3 style='font-size:22px; font-weight:700;'>Upload a retina image (JPG/PNG)</h3>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Save uploaded image
    upload_dir = "streamlit_app/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="Uploaded Retina Image", use_container_width=True)

    # Preprocess
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Grad-CAM + Prediction
    gradcam_path, pred_class = generate_and_save_gradcam(
        model=model,
        img_array=img_array,
        last_conv_layer_name=last_conv_layer,
        save_path="streamlit_app/output_image/gradcam.jpg"
    )

    st.markdown(
        f"<h2 style='color:#2E86AB; font-size:26px;'>Prediction: {labels[pred_class]} (Class {pred_class})</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:16px;'>Classes: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR</p>",
        unsafe_allow_html=True
    )

    st.image(gradcam_path, caption="Grad-CAM Heatmap", use_container_width=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; font-size:14px;'>
        Developed by <strong>Sagar Singh</strong><br>
        <a href='https://github.com/Sagarsingh19' target='_blank'>GitHub</a> | 
        <a href='https://www.linkedin.com/in/sagarsingh19/' target='_blank'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
