import os
import io
import numpy as np
import pandas as pd
import pickle
import requests
from datetime import datetime
from PIL import Image

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from recommendations import (
    yield_recommendations,
    disease_recommendations,
    nutrient_recommendations
)

from translations import disease_names, nutrient_names

# -------------------------------------------------------
# APP INIT
# -------------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static"
)
CORS(app)


# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

disease_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "Models", "paddy_disease_mobilenet.h5")
)

nutrient_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "Models", "paddy_nutrient_model.h5")
)

yield_model = pickle.load(
    open(os.path.join(BASE_DIR, "Models", "yield_model.pkl"), "rb")
)

district_encoder = pickle.load(
    open(os.path.join(BASE_DIR, "Models", "district_encoder.pkl"), "rb")
)

season_encoder = pickle.load(
    open(os.path.join(BASE_DIR, "Models", "season_encoder.pkl"), "rb")
)

# -------------------------------------------------------
# CLASSES
# -------------------------------------------------------
disease_classes = [
    "Bacterial_leaf_blight","Brown_spot","Healthy_leaf",
    "Leaf_Blast","others","tungro"
]

nutrient_classes = ["Healthy","Nitrogen","Phosphorus","Potassium"]

API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/yield")
def yield_page():
    return render_template("yield.html")

@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html")

# -------------------------------------------------------
# YIELD PREDICTION
# -------------------------------------------------------
@app.route("/predict_yield", methods=["POST"])
def predict_yield():

    data = request.json
    lang = data.get("lang","en")

    district = data.get("district","").strip()
    season = data.get("season","").strip()
    area = float(data.get("area",0))

    if not district or not season or area <= 0:
        return jsonify({"error":"Invalid input"}),400

    # ---------------- GEO ----------------
    geo = requests.get(
        f"http://api.openweathermap.org/geo/1.0/direct?q={district},IN&limit=1&appid={API_KEY}"
    ).json()

    if not geo:
        return jsonify({"error":"Invalid district"}),400

    lat, lon = geo[0]["lat"], geo[0]["lon"]

    # ---------------- WEATHER ----------------
    forecast = requests.get(
        f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    ).json().get("list",[])

    if not forecast:
        return jsonify({"error":"Weather unavailable"}),400

    temp_avg = np.mean([d["main"]["temp"] for d in forecast])
    temp_max = np.max([d["main"]["temp_max"] for d in forecast])
    temp_min = np.min([d["main"]["temp_min"] for d in forecast])
    humidity = np.mean([d["main"]["humidity"] for d in forecast])
    rainfall = np.sum([d.get("rain",{}).get("3h",0) for d in forecast])
    solar = np.mean([100 - d["clouds"]["all"] for d in forecast])

    # ---------------- BASE DATA ----------------
    df = pd.DataFrame([{
        "District": district,
        "Season": season,
        "Year": datetime.now().year,
        "Area": area,
        "T2M": temp_avg,
        "T2M_MAX": temp_max,
        "T2M_MIN": temp_min,
        "RH2M": humidity,
        "Rainfall": rainfall,
        "Solar_Radiation": solar,
        "N": 90,
        "P": 45,
        "K": 120,
        "pH": 6.8
    }])

    # ---------------- ENCODING ----------------
    try:
        df["District_enc"] = district_encoder.transform(df["District"])
        df["Season_enc"] = season_encoder.transform(df["Season"])
    except:
        return jsonify({"error":"Encoding error"}),400

    # ---------------- FEATURE ENGINEERING ----------------
    df["Temp_Range"] = df["T2M_MAX"] - df["T2M_MIN"]
    df["Rainfall_per_Area"] = df["Rainfall"] / (df["Area"] + 1e-6)
    df["NPK_Sum"] = df["N"] + df["P"] + df["K"]
    df["Soil_Quality"] = df["NPK_Sum"] / (df["pH"] + 1e-6)

    df["District_Season_enc"] = df["District_enc"] * df["Season_enc"]

    # If your training used previous yield
    df["Prev_Yield"] = 0  # default (or improve later)

    # ---------------- FINAL FEATURES ----------------
    final_columns = [
        'District_enc', 'Season_enc', 'Year', 'Area',
        'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M',
        'Rainfall', 'Solar_Radiation', 'N', 'P', 'K', 'pH',
        'District_Season_enc', 'Prev_Yield', 'Temp_Range',
        'Rainfall_per_Area', 'NPK_Sum', 'Soil_Quality'
    ]

    df = df[final_columns]

    # ---------------- PREDICTION ----------------
    pred = float(yield_model.predict(df)[0])

    # ---------------- RECOMMENDATIONS ----------------
    recommendations = yield_recommendations(
        pred,
        {"avg_temperature":temp_avg,"rainfall_mm":rainfall},
        season,
        lang
    )

    return jsonify({
    "predicted_yield": float(round(pred, 2)),

    "weather": {
        "avg_temperature": float(round(temp_avg, 2)),
        "max_temperature": float(round(temp_max, 2)),
        "min_temperature": float(round(temp_min, 2)),
        "humidity": float(round(humidity, 2)),
        "rainfall": float(round(rainfall, 2)),
        "solar_radiation": float(round(solar, 2))
    },

    "recommendations": [str(r) for r in recommendations]
})

# -------------------------------------------------------
# ANALYZE LEAF
# -------------------------------------------------------
@app.route("/analyze_leaf", methods=["POST"])
def analyze_leaf():

    file = request.files.get("image")
    lang = request.form.get("lang","en")

    if not file:
        return jsonify({"error":"No image"}),400

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    # DISEASE
    disease_pred = disease_model.predict(img_array)
    d_conf = float(np.max(disease_pred))
    d_idx = int(np.argmax(disease_pred))
    disease = disease_classes[d_idx]

    if d_conf >= 0.6 and disease != "Healthy_leaf":

        res = disease_recommendations(disease,lang)
        product = res.get("product")

        if product:
            img_path = product.get("image","")
            if not img_path.startswith("/static"):
                product["image"] = f"/static/products/{img_path}"

        return jsonify({
            "type":"disease",
            "prediction":disease_names.get(disease,{}).get(lang,disease),
            "cause":res.get("cause",""),
            "actions":res.get("actions",[]),
            "prevention":res.get("prevention",[]),
            "product":product
        })

    # NUTRIENT
    nutrient_img = preprocess_input(img_array*255.0)
    n_pred = nutrient_model.predict(nutrient_img)
    n_idx = int(np.argmax(n_pred))
    nutrient = nutrient_classes[n_idx]

    res = nutrient_recommendations(nutrient,lang)
    product = res.get("product")

    if product:
        img_path = product.get("image","")
        if not img_path.startswith("/static"):
            product["image"] = f"/static/products/{img_path}"

    return jsonify({
        "type":"nutrient",
        "prediction":nutrient_names.get(nutrient,{}).get(lang,nutrient),
        "cause":res.get("cause",""),
        "actions":res.get("actions",[]),
        "prevention":res.get("prevention",[]),
        "product":product
    })

# -------------------------------------------------------
# RUN
# -------------------------------------------------------
if __name__ == "__main__":
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
