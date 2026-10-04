import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf photo and the model will predict the disease.")

# PlantVillage classes in the same (alphabetical) order used during training
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5", compile=False)


model = load_model()

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), axis=0)
    preds = model.predict(x)[0]
    top = int(np.argmax(preds))

    st.success(f"Prediction: **{CLASS_NAMES[top].replace('_', ' ').strip()}**")
    st.write(f"Confidence: {preds[top] * 100:.2f}%")
