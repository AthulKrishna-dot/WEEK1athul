import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("waste_classification_model.h5")

# Class names (must match your dataset)
class_names = ["organic", "recycle", "non-organic"]

st.title("♻️ Waste Classification Web App")
st.write("Upload an image to classify it as Organic, Recyclable, or Non-organic.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", width=300)

    # Resize and preprocess
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    st.write("### 🧾 Predicted Category:", class_name.capitalize())
