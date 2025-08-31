pip install tensorflow

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

MODEL_PATH = r"G:\My Drive\UDEMY\cat dog classification\my_model.h5"

@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

st.title("🐾 Cat vs Dog Image Classifier")

# Load / fail gracefully
if os.path.exists(MODEL_PATH):
    st.info("Loading pre-trained model...")
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.error(f"Model file not found at:\n{MODEL_PATH}")
    st.stop()

def predict_image(model, uploaded_file):
    """Predicts if the uploaded image contains a cat or a dog."""
    # Read and preprocess
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    pred = model.predict(img_array, verbose=0)[0]

    # Handle common output shapes robustly
    if np.ndim(pred) == 0:
        p_dog = float(pred)  # scalar
    elif pred.shape == (1,):
        p_dog = float(pred[0])  # sigmoid single unit
    elif pred.shape == (2,):
        p_dog = float(pred[1])  # softmax [cat, dog]
    else:
        # Fallback: take argmax as label
        idx = int(np.argmax(pred))
        label = "dog" if idx == 1 else "cat"
        conf = float(np.max(pred))
        return label, conf

    label = "dog" if p_dog >= 0.5 else "cat"
    conf = p_dog if label == "dog" else (1 - p_dog)
    return label, conf

st.markdown("Upload an image and let the model predict if it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Making a prediction..."):
            label, confidence = predict_image(model, uploaded_file)
            st.success(f"Prediction: **{label.upper()}**")
            st.write(f"Confidence: `{confidence:.2f}`")

st.markdown("---")
st.subheader("How it works")
st.markdown(
    """
1. **Model:** A CNN trained on cat/dog images and saved to `my_model.h5`.
2. **Preprocess:** Your image is resized to 150×150 and normalized.
3. **Predict:** The model outputs a probability; ≥ 0.5 ⇒ **dog**, else **cat**.
"""
)

st.markdown("Developed by Vishal")
