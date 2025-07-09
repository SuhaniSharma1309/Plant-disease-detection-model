import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

import os

if not os.path.exists("Improved_Plant_disease_detection_model_pwp.keras"):
    import gdown
    url = "https://drive.google.com/uc?id=1xRJ9Bu_dtSp1i9pJAbcqQaj2Yl8pmL9z"
    gdown.download(url, "Improved_Plant_disease_detection_model_pwp.keras", quiet=False)

# Set page config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #e8f5e9;  /* Light green shade */
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
        .main { background-color: #abf7b1; }
        h1, h2, h3 {
            color: #2e7d32;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# Title and description
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a plant leaf image to detect the disease and get details like **cause** and **cure** using a deep learning model.")

# Load the trained model
model = tf.keras.models.load_model("Improved_Plant_disease_detection_model_pwp.keras", compile=False)

# Load disease info JSON
with open("plant_disease.json", "r") as f:
    disease_data = json.load(f)

# Extract class names
class_names = [d["name"] for d in disease_data]

# File uploader
uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Processing the image...")

    # Preprocessing
    #img = image.resize((160, 160))
    input_size = model.input_shape[1:3]  # e.g., (160, 160)
    img = image.resize(input_size)

    #img_array = np.array(img) / 255.0
    from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

    img_array = np.array(img)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)


    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # Disease info
    disease_info = disease_data[predicted_index]

    # Results
    st.success(f"🩺 **Predicted Disease:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")
    st.write(f"💡 **Cause:** {disease_info['cause']}")
    st.write(f"🧪 **Cure:** {disease_info['cure']}")