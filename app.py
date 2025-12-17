from flask import Flask, request, jsonify
import joblib
import pandas as pd
from features.build_features import preprocess_input

app = Flask(__name__)

# Load prediction model
prediction_model = joblib.load("models/donorschoose_model.pkl")

@app.route("/health", methods=["GET"])
def health():
    return {"status": "running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        X = preprocess_input(data)
        proba = prediction_model.predict_proba(X)[0][1]
        status = "Approved" if proba >= 0.7 else "Rejected"

        return jsonify({
            "id": data.get("id", None),
            "predicted_status": status,
            "approval_probability": round(float(proba), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
