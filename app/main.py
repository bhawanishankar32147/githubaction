from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
clf = joblib.load("app/iris.json")

@app.route("/")
def home():
    return jsonify({"message": "Iris classifier with Flask is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")
    if not features or not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Invalid input. Expecting a list of 4 numeric features."}), 400

    prediction = clf.predict([np.array(features)])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
