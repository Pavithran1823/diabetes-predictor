from flask import Flask, render_template, request
import numpy as np
from joblib import load
import os

app = Flask(__name__)

# Load model + scaler at startup
MODEL_PATH = os.path.join("model", "model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise RuntimeError("Model artifacts not found. Run: python model/train_model.py")

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# Field order must match training script columns
INPUT_FIELDS = [
    ("Pregnancies", "number", "0", "20", "0", "Integer (e.g., 2)"),
    ("Glucose", "number", "50", "300", "120", "mg/dL (e.g., 120)"),
    ("BloodPressure", "number", "30", "200", "70", "mm Hg (e.g., 70)"),
    ("SkinThickness", "number", "0", "100", "20", "mm (e.g., 20)"),
    ("Insulin", "number", "0", "900", "80", "mu U/ml (e.g., 80)"),
    ("BMI", "number", "10", "70", "28.0", "kg/m² (e.g., 28.0)"),
    ("DiabetesPedigreeFunction", "number", "0.05", "3.0", "0.5", "Relative risk (e.g., 0.5)"),
    ("Age", "number", "10", "100", "35", "Years (e.g., 35)"),
]

def validate_and_vectorize(form):
    values = []
    for (name, _type, _min, _max, _default, _placeholder) in INPUT_FIELDS:
        raw = form.get(name, "").strip()
        if raw == "":
            # allow default if blank
            raw = _default
        try:
            val = float(raw)
        except ValueError:
            raise ValueError(f"{name} must be a number.")
        values.append(val)
    return np.array(values, dtype=float).reshape(1, -1)

def risk_label(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.45:
        return "Moderate Risk"
    else:
        return "Low Risk"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=INPUT_FIELDS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = validate_and_vectorize(request.form)
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        label = risk_label(proba)
        return render_template(
            "result.html",
            probability=f"{proba*100:.1f}%",
            label=label,
            inputs=dict((k, request.form.get(k, "")) for (k, *_rest) in INPUT_FIELDS),
        )
    except Exception as e:
        return render_template("result.html", error=str(e)), 400

if __name__ == "__main__":
    app.run(debug=True)
