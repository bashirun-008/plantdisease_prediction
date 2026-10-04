from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ----------------------------------------------------------------- config
MODEL_PATH = Path(__file__).parent / "plant_disease_model.keras"
IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.60

# Same order as the training notebook (alphabetical folder names)
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# class -> (crop, condition, emoji, advice)
INFO = {
    "Pepper__bell___Bacterial_spot": ("Bell Pepper", "Bacterial Spot", "🫑",
        "Remove infected leaves, avoid overhead watering, use copper-based sprays, rotate crops and start with disease-free seed."),
    "Pepper__bell___healthy": ("Bell Pepper", "Healthy", "🫑",
        "Leaf looks healthy. Keep up regular watering, spacing and monitoring."),
    "Potato___Early_blight": ("Potato", "Early Blight", "🥔",
        "Remove affected lower leaves, mulch the soil, keep plants well fed, rotate crops and apply a protectant fungicide if it spreads."),
    "Potato___Late_blight": ("Potato", "Late Blight", "🥔",
        "Act fast: remove and destroy infected plants, avoid wet foliage, apply a protectant fungicide and destroy leftover tubers."),
    "Potato___healthy": ("Potato", "Healthy", "🥔",
        "Leaf looks healthy. Keep monitoring, especially in humid weather."),
    "Tomato_Bacterial_spot": ("Tomato", "Bacterial Spot", "🍅",
        "Remove infected leaves, water at the base, use copper-based sprays and avoid working with wet plants."),
    "Tomato_Early_blight": ("Tomato", "Early Blight", "🍅",
        "Prune lower infected leaves, mulch, improve airflow, rotate crops and use a suitable fungicide."),
    "Tomato_Late_blight": ("Tomato", "Late Blight", "🍅",
        "Remove and destroy infected plants quickly, keep foliage dry and apply a protectant fungicide."),
    "Tomato_Leaf_Mold": ("Tomato", "Leaf Mold", "🍅",
        "Improve ventilation, reduce humidity, remove affected leaves and avoid overhead watering."),
    "Tomato_Septoria_leaf_spot": ("Tomato", "Septoria Leaf Spot", "🍅",
        "Remove infected lower leaves, mulch, avoid wetting foliage and apply a fungicide if needed."),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider Mites", "🍅",
        "Spray leaves with water, use insecticidal soap or neem oil, and avoid drought stress and dusty conditions."),
    "Tomato__Target_Spot": ("Tomato", "Target Spot", "🍅",
        "Prune for airflow, clear plant debris, avoid overhead watering and use a fungicide if it spreads."),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomato", "Yellow Leaf Curl Virus", "🍅",
        "Spread by whiteflies. Control whiteflies, remove infected plants, use reflective mulch and resistant varieties."),
    "Tomato__Tomato_mosaic_virus": ("Tomato", "Mosaic Virus", "🍅",
        "No cure. Remove infected plants, disinfect tools, wash hands after handling and use resistant varieties."),
    "Tomato_healthy": ("Tomato", "Healthy", "🍅",
        "Leaf looks healthy. Keep up regular care and monitoring."),
}

# ----------------------------------------------------------------- page
st.set_page_config(page_title="LeafScan · Plant Disease Detector", page_icon="🌿", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #43a047 60%, #81c784 100%);
    border-radius: 24px; padding: 2.4rem 2rem; color: white; text-align: center;
    box-shadow: 0 10px 30px rgba(46,125,50,.25); margin-bottom: 1.5rem;
}
.hero h1 {font-size: 2.6rem; margin: 0; color: white;}
.hero p {font-size: 1.05rem; opacity: .92; margin: .5rem 0 0;}
.card {
    background: white; border-radius: 20px; padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 18px rgba(0,0,0,.07); border: 1px solid #e3eedf;
}
.badge {
    display: inline-block; padding: .3rem .9rem; border-radius: 999px;
    font-weight: 600; font-size: .85rem; margin-bottom: .6rem;
}
.ok {background:#e0f4e0; color:#1b5e20;}
.bad {background:#fdeaea; color:#b71c1c;}
.warn {background:#fff4d6; color:#8a5a00;}
.result-title {font-size: 1.9rem; font-weight: 700; margin: 0; color:#1b2e1f;}
.crop {color:#5b7260; font-size:1rem; margin-bottom:.8rem;}
.bar-row {margin: .55rem 0;}
.bar-label {display:flex; justify-content:space-between; font-size:.9rem; color:#33493a;}
.bar-bg {background:#e8f1e5; border-radius:8px; height:10px; overflow:hidden;}
.bar-fill {background:linear-gradient(90deg,#43a047,#81c784); height:10px; border-radius:8px;}
.advice {background:#f1f8ee; border-left:4px solid #43a047; border-radius:10px;
         padding:.9rem 1rem; margin-top:1rem; font-size:.95rem; color:#25402b;}
.placeholder {text-align:center; padding:3rem 1rem; color:#6d8472;}
.placeholder .big {font-size:3.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌿 LeafScan</h1>
  <p>Upload a photo of a leaf and get an instant AI diagnosis for pepper, potato and tomato plants.</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)  # raw 0-255; model rescales itself
    probs = load_model().predict(arr, verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top]


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.write("A MobileNetV2 model trained on the PlantVillage dataset (15 classes, ~86% validation accuracy).")
    st.subheader("Tips for best results")
    st.markdown("- One leaf, filling most of the frame\n- Good natural light\n- Plain background\n- Sharp, in-focus photo")
    st.subheader("Supported plants")
    st.markdown("🫑 Bell pepper  \n🥔 Potato  \n🍅 Tomato")
    st.caption("This tool is a guide, not a substitute for an agronomist.")

# ----------------------------------------------------------------- main
left, right = st.columns(2, gap="large")

with left:
    st.markdown("#### 📷 Your leaf")
    tab_up, tab_cam = st.tabs(["Upload", "Camera"])
    with tab_up:
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    with tab_cam:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
    source = up or cam
    if source:
        img = Image.open(source)
        st.image(img, use_container_width=True)

with right:
    st.markdown("#### 🔍 Diagnosis")
    if not source:
        st.markdown('<div class="card placeholder"><div class="big">🍃</div>'
                    'Upload or capture a leaf photo to see the result here.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analysing leaf…"):
            results = predict(img)
        cls, conf = results[0]
        crop, condition, emoji, advice = INFO[cls]
        healthy = condition == "Healthy"

        if conf < LOW_CONFIDENCE:
            badge = '<span class="badge warn">⚠️ Low confidence</span>'
        elif healthy:
            badge = '<span class="badge ok">✅ Healthy</span>'
        else:
            badge = '<span class="badge bad">🦠 Disease detected</span>'

        bars = "".join(
            f'<div class="bar-row"><div class="bar-label"><span>{INFO[c][0]} · {INFO[c][1]}</span>'
            f'<span>{p*100:.1f}%</span></div><div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%"></div></div></div>'
            for c, p in results
        )
        note = ""
        if conf < LOW_CONFIDENCE:
            note = ("<div class='advice' style='border-color:#e0a100;background:#fffaea'>"
                    "The model isn't sure. Try a clearer, closer photo of a single leaf. "
                    "Plants outside pepper, potato and tomato aren't supported.</div>")

        st.markdown(f"""
        <div class="card">
          {badge}
          <p class="result-title">{emoji} {condition}</p>
          <p class="crop">{crop} · {conf*100:.1f}% confidence</p>
          {bars}
          {note}
          <div class="advice"><b>What to do:</b> {advice}</div>
        </div>
        """, unsafe_allow_html=True)
