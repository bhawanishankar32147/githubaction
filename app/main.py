from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
clf = joblib.load("app/iris.json")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            f1 = float(request.form["sepal_length"])
            f2 = float(request.form["sepal_width"])
            f3 = float(request.form["petal_length"])
            f4 = float(request.form["petal_width"])
            prediction = clf.predict([[f1, f2, f3, f4]])
            iris_output = {0:"Setosa", 1:"Versicolor", 2:"Verginica"}
            result = f"Predicted Iris class: {iris_output[prediction[0]]}"
        except Exception as e:
            result = f"Error: {e}"

    return render_template_string(HTML_TEMPLATE, result=result)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Classifier 🌸</title>
    <style>
        body {
            background: #fdf6f0;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            padding: 40px;
            color: #333;
        }
        .container {
            background: #fff;
            border-radius: 15px;
            padding: 30px;
            width: 400px;
            margin: auto;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        input[type="number"] {
            width: 80%;
            padding: 10px;
            margin: 10px auto;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        button {
            padding: 10px 25px;
            border: none;
            background: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
        }
        button:hover {
            background: #45a049;
        }
        img {
            width: 150px;
            margin-bottom: 20px;
        }
        .result {
            margin-top: 20px;
            font-weight: bold;
            font-size: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Iris Flower Prediction 🌸</h2>
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" alt="Iris flower">
        <form method="POST">
            <input type="number" step="any" name="sepal_length" placeholder="Sepal Length" required><br>
            <input type="number" step="any" name="sepal_width" placeholder="Sepal Width" required><br>
            <input type="number" step="any" name="petal_length" placeholder="Petal Length" required><br>
            <input type="number" step="any" name="petal_width" placeholder="Petal Width" required><br>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
        <div class="result">{{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""




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
