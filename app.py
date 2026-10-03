import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load trained model
MODEL = tf.keras.models.load_model("plant_disease_model.keras")

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Simple web interface using HTML/CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Disease Detector</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f4f9; }
        .card { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        input[type=file] { margin: 20px 0; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .result { margin-top: 20px; font-size: 18px; color: #333; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Plant Disease Detection</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br>
            <button type="submit">Predict Disease</button>
        </form>
        {% if prediction %}
        <div class="result">
            <p><strong>Prediction:</strong> {{ prediction }}</p>
            <p><strong>Confidence:</strong> {{ confidence }}%</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HTML_TEMPLATE)
    
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((224, 224))
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = MODEL.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(float(np.max(predictions)) * 100, 2)
    
    return render_template_string(HTML_TEMPLATE, prediction=predicted_class, confidence=confidence)

import os

# ... (rest of your app.py code)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
